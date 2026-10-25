"""Fused Triton kernels for the X28E slot-store tape head (phase 2, M1 engineering only).

Binding spec: ``experiments/expressivity_tasks/x28e_memory_head/REGISTRATION.md``
(the routing rule + engineering gates 1-5), the phase-1 eager implementation
``../tape_head.py`` (the parity oracle), and the supervisor gap resolutions
G1-G14 (pre-M2 bindings, to be audited at the freeze review):

  G1  pop = strength-decrement sigma <- sigma*(1-p), death at free_threshold;
  G2  reads weight live slots by (differentiable) strength;
  G3  write-enable = write-strength gate > write_enable_threshold; the same
      gate value doubles as the delta-rule write magnitude beta;
  G4  appended value scaled by the learned append gate; gate input includes
      the DETACHED write-error magnitude;
  G5  budget-full = no FREE slot; dead slots are reusable free capacity;
      total allocation never exceeds B;
  G6  match scoring = scaled dot product (q.k)/sqrt(K);
  G10 free-by-decay decisions JOIN the routing trace (int8 codes per token;
      replay consumes them, never recomputing a death decision);
  G13 usage counters are receipt-only (freeing is strength-driven);
  G14 placeholder constants: match_threshold=0.5, eps_residual=1e-6,
      pop_threshold=0.5, write_enable_threshold=0.5, CKPT_INTERVAL=16,
      decay_rate=0.999995, free_threshold=1e-3.

Two REAL fused ``@triton.jit`` kernels (no torch in the hot path), modelled on
``ndm/triton/mlp_mem_fused.py`` (sequential SRAM-resident scan + sparse
checkpoints; reverse-replay BPTT that forward-replays each segment FOLLOWING
THE TRACE then walks it in reverse threading the state gradient):

  * ``tape_fwd_kernel`` — one program per (batch, head). Sequential time loop.
    The (key, value, strength, live, age, usage) store is register/SRAM-
    resident up to the registered per-program slot tile (RESIDENT mode,
    S <= MAX_RESIDENT_SLOTS); larger stores spill to a per-program HBM scratch
    and are streamed tile-by-tile (SPILLED mode). Routing (the five cases +
    W-ON/W-OFF + pop) is computed IN-KERNEL; sparse checkpoints are written
    every CKPT_INTERVAL steps (full store + liveness/age/usage metadata) and
    the per-token routing trace — including the G10 free-by-decay codes — is
    written to the trace buffers; theta-final (the final store) is written at
    the end. The read pass streams an online softmax over the slot dimension
    so S may exceed any register-resident tile.

  * ``tape_bwd_kernel`` — reverse-segment replay per
    ``_mlp_mem_bwd_kernel``: reload the segment-entry checkpoint, FORWARD-REPLAY
    the segment FOLLOWING THE TRACE (decisions are inputs, never recomputed),
    cache per-step store states to per-program HBM scratch, then walk the steps
    in reverse applying the exact per-step VJPs (delta-rule correction adjoint,
    append adjoint, pop adjoint, decay adjoint, read-attention adjoint),
    threading the state gradient (d_keys, d_values, d_strength) across
    segments. The kernel also carries the gate instrumentation
    (REPLAY_OUT / ECHO / BOUNDARY constexpr flags) used by the
    replay-consistency gate: replayed read outputs, echoed trace decisions and
    replayed checkpoint-boundary stores.

Precision (registration gate 3): fp32 live slot state and kernel arithmetic;
bf16 sparse replay checkpoints (configurable — the gates run both bf16 and
fp32-exact modes); fp32 parameters and inputs. The per-step replay scratch
inside the backward kernel is fp32 (FLAGGED interpretation: the registration
groups "bf16 backward replay checkpoints/scratch"; the per-step cache is kept
fp32 because the delta-rule adjoints contain (1-beta)-style differences whose
bf16 rounding noise is not needed for the declared tolerance — receipts in
``receipts/kernel_match.json`` show gradient parity holds; the sparse
checkpoints, the declared rounding surface, are bf16).

No torch in the hot path: torch appears only in the launch wrappers
(allocation/glue), exactly as in ``mlp_mem_fused.MlpMemFn``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from ..routing_trace import (
    NO_SLOT, NO_WRITE, R_CAP, R_CONFLICT, R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL,
    FREED_NONE, FREED_POP, FREED_DECAY, RoutingTrace,
)
from ..tape_head import TapeHeadConfig, TapeState

# Soft mask value for dead/invalid slots (exact softmax zero via underflow).
NEG = tl.constexpr(-1.0e30)
# kernel-side constexpr instances of the architectural case codes (Triton
# requires globals referenced inside @triton.jit to be tl.constexpr instances;
# values are the routing_trace codes, one spelling of the concept)
K_NO_WRITE = tl.constexpr(NO_WRITE)
K_R_EMPTY = tl.constexpr(R_EMPTY)
K_R_NOMATCH = tl.constexpr(R_NOMATCH)
K_R_ZERO_RESIDUAL = tl.constexpr(R_ZERO_RESIDUAL)
K_R_CONFLICT = tl.constexpr(R_CONFLICT)
K_R_CAP = tl.constexpr(R_CAP)
K_NO_SLOT = tl.constexpr(NO_SLOT)
K_FREED_NONE = tl.constexpr(FREED_NONE)
K_FREED_POP = tl.constexpr(FREED_POP)
K_FREED_DECAY = tl.constexpr(FREED_DECAY)

DEFAULT_CKPT_INTERVAL = 16          # G14 placeholder (TO-BE-BOUND-AT-FREEZE)
DEFAULT_TILE_S = 64                 # registered per-program slot tile (spilled)
DEFAULT_MAX_RESIDENT_SLOTS = 64     # store is register-resident up to this


def _next_pow2(x: int, minimum: int = 16) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(minimum, p)


# ---------------------------------------------------------------------------
# Forward kernel — sequential scan, in-kernel routing, sparse checkpoints.
# One program per (b, head). RESIDENT: store in registers. SPILLED: store in
# per-program HBM scratch, streamed tile-by-tile.
# ---------------------------------------------------------------------------
@triton.jit
def tape_fwd_kernel(
    K_ptr, Q_ptr, V_ptr, Beta_ptr, Pop_ptr, Alog_ptr, Aec_ptr,     # [B,T,NH,*], Aec [NH]
    S0K_ptr, S0V_ptr, S0S_ptr, S0L_ptr, S0A_ptr, S0U_ptr,          # init state [B,NH,S,*]
    Out_ptr,                                                        # [B,T,NH,V]
    FK_ptr, FV_ptr, FS_ptr, FL_ptr, FA_ptr, FU_ptr,                # final state
    TCcase_ptr, TCmatch_ptr, TCPopf_ptr, TCPopt_ptr, TCAlloc_ptr, TCWon_ptr,
    TCFreed_ptr,                                                    # [B,T,NH] / [B,T,NH,S]
    CK_K_ptr, CK_V_ptr, CK_S_ptr, CK_L_ptr, CK_A_ptr, CK_U_ptr,    # ckpts [nck,B,NH,S,*]
    XK_ptr, XV_ptr, XS_ptr, XL_ptr, XA_ptr, XU_ptr, XW_ptr,        # spilled scratch
    S, match_thr, eps_res, pop_thr, won_thr, free_thr, decay, usage_decay, scale,
    B: tl.constexpr, T: tl.constexpr, NH: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    RESIDENT: tl.constexpr, TSR: tl.constexpr, TS: tl.constexpr,
    CKPT: tl.constexpr,
):
    pid = tl.program_id(0)
    pid64 = pid.to(tl.int64)
    b = pid // NH
    h = pid % NH
    kk = tl.arange(0, BK)
    vv = tl.arange(0, BV)
    k_m = kk < K
    v_m = vv < V
    pS = pid64 * S
    aec = tl.load(Aec_ptr + h).to(tl.float32)

    if RESIDENT:
        # ---- register-resident store (S <= TSR) ----
        sr = tl.arange(0, TSR)
        s_m = sr < S
        keys = tl.load(S0K_ptr + ((pS + sr[:, None]) * K + kk[None, :]),
                       mask=s_m[:, None] & k_m[None, :], other=0.0).to(tl.float32)
        values = tl.load(S0V_ptr + ((pS + sr[:, None]) * V + vv[None, :]),
                        mask=s_m[:, None] & v_m[None, :], other=0.0).to(tl.float32)
        strength = tl.load(S0S_ptr + pS + sr, mask=s_m, other=0.0).to(tl.float32)
        live = tl.load(S0L_ptr + pS + sr, mask=s_m, other=0) > 0
        age = tl.load(S0A_ptr + pS + sr, mask=s_m, other=0)
        usage = tl.load(S0U_ptr + pS + sr, mask=s_m, other=0.0).to(tl.float32)

        # checkpoint slot 0 = the initial store
        ck0 = (((0 * B + b) * NH + h) * S + sr)
        tl.store(CK_K_ptr + (ck0[:, None] * K + kk[None, :]),
                 keys.to(CK_K_ptr.dtype.element_ty), mask=s_m[:, None] & k_m[None, :])
        tl.store(CK_V_ptr + (ck0[:, None] * V + vv[None, :]),
                 values.to(CK_V_ptr.dtype.element_ty), mask=s_m[:, None] & v_m[None, :])
        tl.store(CK_S_ptr + ck0, strength.to(CK_S_ptr.dtype.element_ty), mask=s_m)
        tl.store(CK_L_ptr + ck0, live.to(tl.int8), mask=s_m)
        tl.store(CK_A_ptr + ck0, age, mask=s_m)
        tl.store(CK_U_ptr + ck0, usage.to(CK_U_ptr.dtype.element_ty), mask=s_m)

        for t in range(T):
            t64 = tl.full([1], t, tl.int64)
            tok = (b * T + t64) * NH + h
            k_t = tl.load(K_ptr + tok * K + kk, mask=k_m, other=0.0).to(tl.float32)
            q_t = tl.load(Q_ptr + tok * K + kk, mask=k_m, other=0.0).to(tl.float32)
            v_t = tl.load(V_ptr + tok * V + vv, mask=v_m, other=0.0).to(tl.float32)
            beta = tl.load(Beta_ptr + tok).to(tl.float32)
            pop = tl.load(Pop_ptr + tok).to(tl.float32)
            alog = tl.load(Alog_ptr + tok).to(tl.float32)

            # ---- READ (pre-pop, pre-write; soft-strength weighting, G2) ----
            sc = tl.sum(keys * q_t[None, :], axis=1) * scale
            scm = tl.where(live, sc, NEG)
            m = tl.max(scm, axis=0)
            e = tl.exp(scm - m)
            z = tl.sum(e, axis=0)
            lf = live.to(tl.float32)
            w = (e / z) * strength * lf
            out = tl.sum(w[:, None] * values, axis=0)
            tl.store(Out_ptr + tok * V + vv, out, mask=v_m)
            usage = usage * usage_decay + w
            any_live = tl.sum(live.to(tl.int32), axis=0) > 0

            # ---- POP (read-path; G1: strength decrement, death at free_thr) ----
            is_c = scm == m
            rbest = tl.min(tl.where(is_c, sr, TSR), axis=0)
            pop_f = (pop > pop_thr) & any_live
            pop_row = (sr == rbest) & pop_f
            strength = tl.where(pop_row, strength * (1.0 - pop), strength)
            freed1 = pop_row & (strength <= free_thr)
            live = live & ~freed1
            has_live = tl.sum(live.to(tl.int32), axis=0) > 0

            # ---- WRITE-ENABLE + five-case routing (G3/G5/G6) ----
            w_on = beta > won_thr
            sc2 = tl.sum(keys * k_t[None, :], axis=1) * scale
            scm2 = tl.where(live, sc2, NEG)
            bscore = tl.max(scm2, axis=0)
            is_c2 = scm2 == bscore
            wbest = tl.min(tl.where(is_c2, sr, TSR), axis=0)
            v_best = tl.sum(tl.where((sr == wbest)[:, None], values, 0.0), axis=0)
            resid = tl.sum((v_t - v_best) * (v_t - v_best), axis=0)
            matched_ok = has_live & w_on
            nomatch = matched_ok & (bscore < match_thr)
            mm = matched_ok & (~nomatch)
            zero_res = mm & (resid < eps_res)
            case = tl.where(w_on & (~has_live), K_R_EMPTY, K_NO_WRITE)
            case = tl.where(nomatch, K_R_NOMATCH, case)
            case = tl.where(zero_res, K_R_ZERO_RESIDUAL, case)
            case = tl.where(mm & (~zero_res), K_R_CONFLICT, case)
            append_req = (case == K_R_EMPTY) | (case == K_R_NOMATCH) | (case == K_R_ZERO_RESIDUAL)
            free_exists = tl.sum(tl.where(s_m & (~live), 1, 0), axis=0) > 0
            first_free = tl.min(tl.where(s_m & (~live), sr, TSR), axis=0)
            case = tl.where(append_req & (~free_exists), K_R_CAP, case)
            append_f = append_req & free_exists
            alloc = tl.where(append_f, first_free, K_NO_SLOT)

            # ---- append gate (G4: detached write-error magnitude input) ----
            err_mag = tl.where(has_live, tl.sqrt(tl.maximum(resid, 0.0)), 0.0)
            gate = tl.sigmoid(alog + aec * err_mag)

            # ---- APPEND (the only allocating op) ----
            arow = (sr == alloc) & append_f
            keys = tl.where(arow[:, None], k_t[None, :], keys)
            values = tl.where(arow[:, None], (gate * v_t)[None, :], values)
            strength = tl.where(arow, 1.0, strength)
            age = tl.where(arow, 0, age)
            usage = tl.where(arow, 0.0, usage)
            live = live | arow

            # ---- DELTA-RULE CORRECTION / MERGE (R-CONFLICT, R-CAP) ----
            cfire = (case == K_R_CONFLICT) | (case == K_R_CAP)
            v_corr = v_best + beta * (v_t - v_best)
            crow = (sr == wbest) & cfire
            values = tl.where(crow[:, None], v_corr[None, :], values)

            # ---- DECAY / AGE / FREE (end of token; G10 frees traced) ----
            strength = tl.where(live, strength * decay, strength)
            age = age + live.to(tl.int32)
            freed2 = live & (strength <= free_thr)
            live = live & ~freed2

            # ---- trace record (incl. free-by-decay, G10) ----
            matched_o = tl.where(has_live, wbest, K_NO_SLOT)
            pop_target_o = tl.where(pop_f, rbest, K_NO_SLOT)
            freed_code = tl.where(freed2, K_FREED_DECAY,
                                  tl.where(freed1, K_FREED_POP, K_FREED_NONE)).to(tl.int8)
            tl.store(TCcase_ptr + tok, case.to(tl.int32))
            tl.store(TCmatch_ptr + tok, matched_o.to(tl.int32))
            tl.store(TCPopf_ptr + tok, pop_f.to(tl.int8))
            tl.store(TCPopt_ptr + tok, pop_target_o.to(tl.int32))
            tl.store(TCAlloc_ptr + tok, alloc.to(tl.int32))
            tl.store(TCWon_ptr + tok, w_on.to(tl.int8))
            tl.store(TCFreed_ptr + tok * S + sr, freed_code, mask=s_m)

            is_ck = ((t + 1) % CKPT) == 0
            if is_ck:
                cks = tl.full([1], (t + 1) // CKPT, tl.int64)
                ckb = (((cks * B + b) * NH + h) * S + sr)
                tl.store(CK_K_ptr + (ckb[:, None] * K + kk[None, :]),
                         keys.to(CK_K_ptr.dtype.element_ty),
                         mask=s_m[:, None] & k_m[None, :])
                tl.store(CK_V_ptr + (ckb[:, None] * V + vv[None, :]),
                         values.to(CK_V_ptr.dtype.element_ty),
                         mask=s_m[:, None] & v_m[None, :])
                tl.store(CK_S_ptr + ckb, strength.to(CK_S_ptr.dtype.element_ty), mask=s_m)
                tl.store(CK_L_ptr + ckb, live.to(tl.int8), mask=s_m)
                tl.store(CK_A_ptr + ckb, age, mask=s_m)
                tl.store(CK_U_ptr + ckb, usage.to(CK_U_ptr.dtype.element_ty), mask=s_m)

        tl.store(FK_ptr + ((pS + sr[:, None]) * K + kk[None, :]), keys,
                 mask=s_m[:, None] & k_m[None, :])
        tl.store(FV_ptr + ((pS + sr[:, None]) * V + vv[None, :]), values,
                 mask=s_m[:, None] & v_m[None, :])
        tl.store(FS_ptr + pS + sr, strength, mask=s_m)
        tl.store(FL_ptr + pS + sr, live.to(tl.int8), mask=s_m)
        tl.store(FA_ptr + pS + sr, age, mask=s_m)
        tl.store(FU_ptr + pS + sr, usage, mask=s_m)

    else:
        # ---- spilled store: per-program HBM scratch, streamed tile-by-tile ----
        # initialize the scratch with the entry store
        for s0 in range(0, S, TS):
            sm = s0 + tl.arange(0, TS)
            vm = sm < S
            tl.store(XK_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                     tl.load(S0K_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                             mask=vm[:, None] & k_m[None, :], other=0.0),
                     mask=vm[:, None] & k_m[None, :])
            tl.store(XV_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                     tl.load(S0V_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                             mask=vm[:, None] & v_m[None, :], other=0.0),
                     mask=vm[:, None] & v_m[None, :])
            tl.store(XS_ptr + pS + sm,
                     tl.load(S0S_ptr + pS + sm, mask=vm, other=0.0), mask=vm)
            tl.store(XL_ptr + pS + sm,
                     tl.load(S0L_ptr + pS + sm, mask=vm, other=0), mask=vm)
            tl.store(XA_ptr + pS + sm,
                     tl.load(S0A_ptr + pS + sm, mask=vm, other=0), mask=vm)
            tl.store(XU_ptr + pS + sm,
                     tl.load(S0U_ptr + pS + sm, mask=vm, other=0.0), mask=vm)

        tl.debug_barrier()
        # checkpoint slot 0 = the initial store
        for s0 in range(0, S, TS):
            sm = s0 + tl.arange(0, TS)
            vm = sm < S
            ck0 = (((0 * B + b) * NH + h) * S + sm)
            tl.store(CK_K_ptr + (ck0[:, None] * K + kk[None, :]),
                     tl.load(XK_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                             mask=vm[:, None] & k_m[None, :], other=0.0)
                     .to(CK_K_ptr.dtype.element_ty),
                     mask=vm[:, None] & k_m[None, :])
            tl.store(CK_V_ptr + (ck0[:, None] * V + vv[None, :]),
                     tl.load(XV_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                             mask=vm[:, None] & v_m[None, :], other=0.0)
                     .to(CK_V_ptr.dtype.element_ty),
                     mask=vm[:, None] & v_m[None, :])
            tl.store(CK_S_ptr + ck0,
                     tl.load(XS_ptr + pS + sm, mask=vm, other=0.0)
                     .to(CK_S_ptr.dtype.element_ty), mask=vm)
            tl.store(CK_L_ptr + ck0,
                     tl.load(XL_ptr + pS + sm, mask=vm, other=0), mask=vm)
            tl.store(CK_A_ptr + ck0,
                     tl.load(XA_ptr + pS + sm, mask=vm, other=0), mask=vm)
            tl.store(CK_U_ptr + ck0,
                     tl.load(XU_ptr + pS + sm, mask=vm, other=0.0)
                     .to(CK_U_ptr.dtype.element_ty), mask=vm)

        for t in range(T):
            t64 = tl.full([1], t, tl.int64)
            tok = (b * T + t64) * NH + h
            k_t = tl.load(K_ptr + tok * K + kk, mask=k_m, other=0.0).to(tl.float32)
            q_t = tl.load(Q_ptr + tok * K + kk, mask=k_m, other=0.0).to(tl.float32)
            v_t = tl.load(V_ptr + tok * V + vv, mask=v_m, other=0.0).to(tl.float32)
            beta = tl.load(Beta_ptr + tok).to(tl.float32)
            pop = tl.load(Pop_ptr + tok).to(tl.float32)
            alog = tl.load(Alog_ptr + tok).to(tl.float32)

            # ---- READ: streaming online softmax over the slot dimension ----
            m = tl.full([1], NEG, tl.float32)
            z = tl.full([1], 0.0, tl.float32)
            acc = tl.zeros([BV], tl.float32)
            best_sc = tl.full([1], NEG, tl.float32)
            best_idx = tl.full([1], 0, tl.int32)
            n_live = tl.full([1], 0, tl.int32)
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                kt = tl.load(XK_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                             mask=vm[:, None] & k_m[None, :], other=0.0).to(tl.float32)
                st_t = tl.load(XS_ptr + pS + sm, mask=vm, other=0.0)
                lv = tl.load(XL_ptr + pS + sm, mask=vm, other=0) > 0
                sc = tl.sum(kt * q_t[None, :], axis=1) * scale
                scm = tl.where(lv, sc, NEG)
                tm_ = tl.max(scm, axis=0)
                m_new = tl.maximum(m, tm_)
                alpha = tl.exp(m - m_new)
                e = tl.exp(scm - m_new)
                z = z * alpha + tl.sum(tl.where(vm, e, 0.0), axis=0)
                vt_ = tl.load(XV_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                              mask=vm[:, None] & v_m[None, :], other=0.0).to(tl.float32)
                wf = e * st_t * lv.to(tl.float32)
                acc = acc * alpha + tl.sum(wf[:, None] * vt_, axis=0)
                m = m_new
                # store the masked scores: the running max can still rise on
                # later tiles, so the usage pass renormalizes with the final m
                tl.store(XW_ptr + pS + sm, scm, mask=vm)
                n_live += tl.sum(lv.to(tl.int32), axis=0)
                take = tm_ > best_sc
                is_c = (scm == tm_) & vm
                loc = tl.min(tl.where(is_c, tl.arange(0, TS), TS), axis=0)
                best_idx = tl.where(take, s0 + loc, best_idx)
                best_sc = tl.where(take, tm_, best_sc)
            out = acc / z
            tl.store(Out_ptr + tok * V + vv, out, mask=v_m)

            # ---- usage pass (G13: receipt-only bookkeeping) ----
            tl.debug_barrier()
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                scm_w = tl.load(XW_ptr + pS + sm, mask=vm, other=0.0)
                st_w = tl.load(XS_ptr + pS + sm, mask=vm, other=0.0)
                lv_w = tl.load(XL_ptr + pS + sm, mask=vm, other=0) > 0
                wu = (tl.exp(scm_w - m) / z) * st_w * lv_w.to(tl.float32)
                us = tl.load(XU_ptr + pS + sm, mask=vm, other=0.0)
                tl.store(XU_ptr + pS + sm, us * usage_decay + wu, mask=vm)

            # ---- POP ----
            pop_f = (pop > pop_thr) & (n_live > 0)
            pt = tl.minimum(best_idx, S)  # always a valid index (< S)
            srow = tl.load(XS_ptr + pS + pt)
            srow2 = tl.where(pop_f, srow * (1.0 - pop), srow)
            tl.store(XS_ptr + pS + pt, srow2)
            freed1 = pop_f & (srow2 <= free_thr)
            lrow = tl.load(XL_ptr + pS + pt)
            tl.store(XL_ptr + pS + pt, tl.where(freed1, 0, lrow).to(tl.int8))
            frow = tl.load(TCFreed_ptr + tok * S + pt)
            tl.store(TCFreed_ptr + tok * S + pt,
                     tl.where(freed1, K_FREED_POP, frow).to(tl.int8))
            # post-pop live count: the pop target was live, freed1 kills exactly it
            has_live = (n_live - freed1.to(tl.int32)) > 0

            # ---- write-routing pass (post-pop liveness) ----
            tl.debug_barrier()
            w_on = beta > won_thr
            best2_sc = tl.full([1], NEG, tl.float32)
            best2_idx = tl.full([1], 0, tl.int32)
            free_min = tl.full([1], S, tl.int32)
            free_any = tl.full([1], 0, tl.int32)
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                kt = tl.load(XK_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                             mask=vm[:, None] & k_m[None, :], other=0.0).to(tl.float32)
                lv = tl.load(XL_ptr + pS + sm, mask=vm, other=0) > 0
                sc2 = tl.sum(kt * k_t[None, :], axis=1) * scale
                scm2 = tl.where(lv, sc2, NEG)
                c2 = tl.max(scm2, axis=0)
                take2 = c2 > best2_sc
                is_c2 = (scm2 == c2) & vm
                loc2 = tl.min(tl.where(is_c2, tl.arange(0, TS), TS), axis=0)
                best2_idx = tl.where(take2, s0 + loc2, best2_idx)
                best2_sc = tl.where(take2, c2, best2_sc)
                nf = vm & (~lv)
                free_any = tl.maximum(free_any, tl.sum(nf.to(tl.int32), axis=0))
                free_min = tl.minimum(free_min,
                                      tl.min(tl.where(nf, sm, 1073741824), axis=0))
            bscore = best2_sc
            v_best = tl.load(XV_ptr + (pS + best2_idx) * V + vv, mask=v_m, other=0.0).to(tl.float32)
            resid = tl.sum((v_t - v_best) * (v_t - v_best), axis=0)
            matched_ok = has_live & w_on
            nomatch = matched_ok & (bscore < match_thr)
            mm = matched_ok & (~nomatch)
            zero_res = mm & (resid < eps_res)
            case = tl.where(w_on & (~has_live), K_R_EMPTY, K_NO_WRITE)
            case = tl.where(nomatch, K_R_NOMATCH, case)
            case = tl.where(zero_res, K_R_ZERO_RESIDUAL, case)
            case = tl.where(mm & (~zero_res), K_R_CONFLICT, case)
            append_req = (case == K_R_EMPTY) | (case == K_R_NOMATCH) | (case == K_R_ZERO_RESIDUAL)
            case = tl.where(append_req & (free_any == 0), K_R_CAP, case)
            append_f = append_req & (free_any > 0)
            alloc = tl.where(append_f, free_min, K_NO_SLOT)

            # ---- append gate (G4) ----
            err_mag = tl.where(has_live, tl.sqrt(tl.maximum(resid, 0.0)), 0.0)
            gate = tl.sigmoid(alog + aec * err_mag)

            # ---- APPEND: single-row stores into the scratch ----
            a_ok = alloc >= 0
            old_k = tl.load(XK_ptr + (pS + alloc) * K + kk, mask=k_m & a_ok, other=0.0)
            tl.store(XK_ptr + (pS + alloc) * K + kk,
                     tl.where(a_ok, k_t, old_k), mask=k_m & a_ok)
            old_v = tl.load(XV_ptr + (pS + alloc) * V + vv, mask=v_m & a_ok, other=0.0)
            tl.store(XV_ptr + (pS + alloc) * V + vv,
                     tl.where(a_ok, gate * v_t, old_v), mask=v_m & a_ok)
            old_s = tl.load(XS_ptr + pS + alloc, mask=a_ok, other=0.0)
            tl.store(XS_ptr + pS + alloc, tl.where(a_ok, 1.0, old_s), mask=a_ok)
            old_l = tl.load(XL_ptr + pS + alloc, mask=a_ok, other=0)
            tl.store(XL_ptr + pS + alloc, tl.where(a_ok, 1, old_l).to(tl.int8), mask=a_ok)
            old_a = tl.load(XA_ptr + pS + alloc, mask=a_ok, other=0)
            tl.store(XA_ptr + pS + alloc, tl.where(a_ok, 0, old_a), mask=a_ok)
            old_u = tl.load(XU_ptr + pS + alloc, mask=a_ok, other=0.0)
            tl.store(XU_ptr + pS + alloc, tl.where(a_ok, 0.0, old_u), mask=a_ok)

            # ---- DELTA-RULE CORRECTION / MERGE ----
            cfire = (case == K_R_CONFLICT) | (case == K_R_CAP)
            v_corr = v_best + beta * (v_t - v_best)
            old_cv = tl.load(XV_ptr + (pS + best2_idx) * V + vv, mask=v_m & cfire, other=0.0)
            tl.store(XV_ptr + (pS + best2_idx) * V + vv,
                     tl.where(cfire, v_corr, old_cv), mask=v_m & cfire)

            # ---- trace record (G10: freed row written in the decay pass) ----
            matched_o = tl.where(has_live, best2_idx, K_NO_SLOT)
            pop_target_o = tl.where(pop_f, best_idx, K_NO_SLOT)
            tl.store(TCcase_ptr + tok, case.to(tl.int32))
            tl.store(TCmatch_ptr + tok, matched_o.to(tl.int32))
            tl.store(TCPopf_ptr + tok, pop_f.to(tl.int8))
            tl.store(TCPopt_ptr + tok, pop_target_o.to(tl.int32))
            tl.store(TCAlloc_ptr + tok, alloc.to(tl.int32))
            tl.store(TCWon_ptr + tok, w_on.to(tl.int8))

            # ---- DECAY / AGE / FREE pass (end of token) ----
            tl.debug_barrier()
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                st_t = tl.load(XS_ptr + pS + sm, mask=vm, other=0.0)
                lv = tl.load(XL_ptr + pS + sm, mask=vm, other=0) > 0
                ag = tl.load(XA_ptr + pS + sm, mask=vm, other=0)
                st2 = tl.where(lv, st_t * decay, st_t)
                ag2 = ag + lv.to(tl.int32)
                fr2 = lv & (st2 <= free_thr)
                lv2 = lv & ~fr2
                fr_old = tl.load(TCFreed_ptr + tok * S + sm, mask=vm, other=0)
                fr_new = tl.where(fr2, K_FREED_DECAY, fr_old)
                tl.store(XS_ptr + pS + sm, st2, mask=vm)
                tl.store(XL_ptr + pS + sm, lv2.to(tl.int8), mask=vm)
                tl.store(XA_ptr + pS + sm, ag2, mask=vm)
                tl.store(TCFreed_ptr + tok * S + sm, fr_new.to(tl.int8), mask=vm)

            # ---- sparse checkpoint (every CKPT steps) ----
            tl.debug_barrier()
            is_ck = ((t + 1) % CKPT) == 0
            if is_ck:
                cks = tl.full([1], (t + 1) // CKPT, tl.int64)
                for s0 in range(0, S, TS):
                    sm = s0 + tl.arange(0, TS)
                    vm = sm < S
                    ckb = (((cks * B + b) * NH + h) * S + sm)
                    tl.store(CK_K_ptr + (ckb[:, None] * K + kk[None, :]),
                             tl.load(XK_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                                     mask=vm[:, None] & k_m[None, :], other=0.0)
                             .to(CK_K_ptr.dtype.element_ty),
                             mask=vm[:, None] & k_m[None, :])
                    tl.store(CK_V_ptr + (ckb[:, None] * V + vv[None, :]),
                             tl.load(XV_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                                     mask=vm[:, None] & v_m[None, :], other=0.0)
                             .to(CK_V_ptr.dtype.element_ty),
                             mask=vm[:, None] & v_m[None, :])
                    tl.store(CK_S_ptr + ckb,
                             tl.load(XS_ptr + pS + sm, mask=vm, other=0.0)
                             .to(CK_S_ptr.dtype.element_ty), mask=vm)
                    tl.store(CK_L_ptr + ckb,
                             tl.load(XL_ptr + pS + sm, mask=vm, other=0), mask=vm)
                    tl.store(CK_A_ptr + ckb,
                             tl.load(XA_ptr + pS + sm, mask=vm, other=0), mask=vm)
                    tl.store(CK_U_ptr + ckb,
                             tl.load(XU_ptr + pS + sm, mask=vm, other=0.0)
                             .to(CK_U_ptr.dtype.element_ty), mask=vm)

        # ---- final store ----
        for s0 in range(0, S, TS):
            sm = s0 + tl.arange(0, TS)
            vm = sm < S
            tl.store(FK_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                     tl.load(XK_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                             mask=vm[:, None] & k_m[None, :], other=0.0),
                     mask=vm[:, None] & k_m[None, :])
            tl.store(FV_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                     tl.load(XV_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                             mask=vm[:, None] & v_m[None, :], other=0.0),
                     mask=vm[:, None] & v_m[None, :])
            tl.store(FS_ptr + pS + sm,
                     tl.load(XS_ptr + pS + sm, mask=vm, other=0.0), mask=vm)
            tl.store(FL_ptr + pS + sm,
                     tl.load(XL_ptr + pS + sm, mask=vm, other=0), mask=vm)
            tl.store(FA_ptr + pS + sm,
                     tl.load(XA_ptr + pS + sm, mask=vm, other=0), mask=vm)
            tl.store(FU_ptr + pS + sm,
                     tl.load(XU_ptr + pS + sm, mask=vm, other=0.0), mask=vm)


# ---------------------------------------------------------------------------
# Backward kernel — reverse-segment replay BPTT following the trace.
# One program per (b, head). Reloads the segment-entry checkpoint, replays the
# segment FOLLOWING THE TRACE (decisions are inputs), caches per-step store
# states to per-program HBM scratch, then walks the segment in reverse applying
# the exact per-step VJPs, threading the state gradient in HBM (d_keys,
# d_values, d_strength). Gate instrumentation (REPLAY_OUT / ECHO / BOUNDARY)
# serves the replay-consistency gate (G10).
# ---------------------------------------------------------------------------
@triton.jit
def tape_bwd_kernel(
    K_ptr, Q_ptr, V_ptr, Beta_ptr, Pop_ptr, Alog_ptr, Aec_ptr,
    CK_K_ptr, CK_V_ptr, CK_S_ptr, CK_L_ptr, CK_A_ptr,          # sparse checkpoints
    TCcase_ptr, TCmatch_ptr, TCPopf_ptr, TCPopt_ptr, TCAlloc_ptr, TCFreed_ptr,
    dOut_ptr,                                                   # [B,T,NH,V]
    dFK_ptr, dFV_ptr, dFS_ptr,                                  # final-state grads
    dK_ptr, dQ_ptr, dV_ptr, dBeta_ptr, dPop_ptr, dAlog_ptr,     # input grads
    dS0K_ptr, dS0V_ptr, dS0S_ptr,                               # init-state grads
    dAecB_ptr,                                                   # [B,NH]
    Cc_K_ptr, Cc_V_ptr, Cc_S_ptr, Cc_L_ptr, Cc_A_ptr,            # per-step cache
    G_K_ptr, G_V_ptr, G_S_ptr, DP_ptr,                           # adjoints + scratch
    RO_ptr,                                                      # replay outputs (gate)
    EB_case_ptr, EB_match_ptr, EB_popt_ptr, EB_alloc_ptr, EB_popf_ptr,  # echo (gate)
    BD_K_ptr, BD_V_ptr, BD_S_ptr, BD_L_ptr, BD_A_ptr,            # boundary dumps (gate)
    S, free_thr, decay, scale,
    B: tl.constexpr, T: tl.constexpr, NH: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, TS: tl.constexpr,
    CKPT: tl.constexpr,
    REPLAY_OUT: tl.constexpr, ECHO: tl.constexpr, BOUNDARY: tl.constexpr,
):
    pid = tl.program_id(0)
    pid64 = pid.to(tl.int64)
    b = pid // NH
    h = pid % NH
    kk = tl.arange(0, BK)
    vv = tl.arange(0, BV)
    k_m = kk < K
    v_m = vv < V
    pS = pid64 * S
    pC = pid64 * (CKPT + 1) * S
    aec = tl.load(Aec_ptr + h).to(tl.float32)

    # thread the state adjoint: seed from the final-state gradient
    for s0 in range(0, S, TS):
        sm = s0 + tl.arange(0, TS)
        vm = sm < S
        tl.store(G_K_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                 tl.load(dFK_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                         mask=vm[:, None] & k_m[None, :], other=0.0),
                 mask=vm[:, None] & k_m[None, :])
        tl.store(G_V_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                 tl.load(dFV_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                         mask=vm[:, None] & v_m[None, :], other=0.0),
                 mask=vm[:, None] & v_m[None, :])
        tl.store(G_S_ptr + pS + sm,
                 tl.load(dFS_ptr + pS + sm, mask=vm, other=0.0), mask=vm)
    daec_acc = tl.full([1], 0.0, tl.float32)

    num_seg = (T + CKPT - 1) // CKPT
    for seg_rev in range(0, num_seg):
        seg = num_seg - 1 - seg_rev
        t0 = seg * CKPT
        seg_len = tl.minimum(CKPT, T - t0)
        # int64 checkpoint indexing (post-M2-review kernel fix): the segment
        # index is an int32 loop variable; at large S and T (e.g. the B0x16
        # arm at the 8,264-token per-arm smoke) the int32 element offset
        # (seg*B*NH + h)*S*V exceeds 2^31 and the checkpoint load goes out
        # of bounds. B0/B0x4 landed just under the limit; the per-arm gate-4
        # smokes (F5) exposed the defect. Every checkpoint offset below uses
        # this int64 segment index.
        seg64 = tl.full([1], seg, tl.int64)

        # ---- load the segment-entry checkpoint into cache slot 0 ----
        for s0 in range(0, S, TS):
            sm = s0 + tl.arange(0, TS)
            vm = sm < S
            ckb = (((seg64 * B + b) * NH + h) * S + sm)
            tl.store(Cc_K_ptr + ((pC + sm[:, None]) * K + kk[None, :]),
                     tl.load(CK_K_ptr + (ckb[:, None] * K + kk[None, :]),
                             mask=vm[:, None] & k_m[None, :], other=0.0)
                     .to(tl.float32),
                     mask=vm[:, None] & k_m[None, :])
            tl.store(Cc_V_ptr + ((pC + sm[:, None]) * V + vv[None, :]),
                     tl.load(CK_V_ptr + (ckb[:, None] * V + vv[None, :]),
                             mask=vm[:, None] & v_m[None, :], other=0.0)
                     .to(tl.float32),
                     mask=vm[:, None] & v_m[None, :])
            tl.store(Cc_S_ptr + pC + sm,
                     tl.load(CK_S_ptr + ckb, mask=vm, other=0.0).to(tl.float32),
                     mask=vm)
            tl.store(Cc_L_ptr + pC + sm,
                     tl.load(CK_L_ptr + ckb, mask=vm, other=0), mask=vm)
            tl.store(Cc_A_ptr + pC + sm,
                     tl.load(CK_A_ptr + ckb, mask=vm, other=0), mask=vm)
        tl.debug_barrier()

        # ---- forward-replay the segment FOLLOWING THE TRACE ----
        for j in range(0, seg_len):
            t = t0 + j
            t64 = tl.full([1], t, tl.int64)
            tok = (b * T + t64) * NH + h
            k_t = tl.load(K_ptr + tok * K + kk, mask=k_m, other=0.0).to(tl.float32)
            q_t = tl.load(Q_ptr + tok * K + kk, mask=k_m, other=0.0).to(tl.float32)
            v_t = tl.load(V_ptr + tok * V + vv, mask=v_m, other=0.0).to(tl.float32)
            beta = tl.load(Beta_ptr + tok).to(tl.float32)
            pop = tl.load(Pop_ptr + tok).to(tl.float32)
            alog = tl.load(Alog_ptr + tok).to(tl.float32)
            case = tl.load(TCcase_ptr + tok)
            matched = tl.load(TCmatch_ptr + tok)
            popf = tl.load(TCPopf_ptr + tok) > 0
            popt = tl.load(TCPopt_ptr + tok)
            alloc = tl.load(TCAlloc_ptr + tok)
            cb_j = pC + j * S
            cb_n = pC + (j + 1) * S

            # live pass: cache_L[j+1], post-pop liveness, has_live
            has_live = tl.full([1], 0, tl.int32)
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                lv = tl.load(Cc_L_ptr + cb_j + sm, mask=vm, other=0) > 0
                fr = tl.load(TCFreed_ptr + tok * S + sm, mask=vm, other=0)
                lvp = lv & ~(fr == K_FREED_POP)
                arow = sm == alloc
                lva = lvp | arow
                lvn = lva & ~(fr == K_FREED_DECAY)
                tl.store(Cc_L_ptr + cb_n + sm, lvn.to(tl.int8), mask=vm)
                has_live = tl.maximum(has_live, tl.sum(lvp.to(tl.int32), axis=0))

            # append gate (recomputed; err_mag from the pre-state at `matched`)
            m_ok = matched >= 0
            v_b = tl.load(Cc_V_ptr + (cb_j + matched) * V + vv, mask=v_m & m_ok, other=0.0)
            resid = tl.sum((v_t - v_b) * (v_t - v_b), axis=0)
            err_mag = tl.where(has_live > 0, tl.sqrt(tl.maximum(resid, 0.0)), 0.0)
            gate = tl.sigmoid(alog + aec * err_mag)

            # replayed read output (gate instrumentation only)
            if REPLAY_OUT:
                m = tl.full([1], NEG, tl.float32)
                z = tl.full([1], 0.0, tl.float32)
                acc = tl.zeros([BV], tl.float32)
                for s0 in range(0, S, TS):
                    sm = s0 + tl.arange(0, TS)
                    vm = sm < S
                    kt = tl.load(Cc_K_ptr + ((cb_j + sm[:, None]) * K + kk[None, :]),
                                 mask=vm[:, None] & k_m[None, :], other=0.0).to(tl.float32)
                    st_t = tl.load(Cc_S_ptr + cb_j + sm, mask=vm, other=0.0)
                    lv = tl.load(Cc_L_ptr + cb_j + sm, mask=vm, other=0) > 0
                    vt_ = tl.load(Cc_V_ptr + ((cb_j + sm[:, None]) * V + vv[None, :]),
                                  mask=vm[:, None] & v_m[None, :], other=0.0).to(tl.float32)
                    sc = tl.sum(kt * q_t, axis=1) * scale
                    scm = tl.where(lv, sc, NEG)
                    tm_ = tl.max(scm, axis=0)
                    m_new = tl.maximum(m, tm_)
                    alpha = tl.exp(m - m_new)
                    e = tl.exp(scm - m_new)
                    z = z * alpha + tl.sum(tl.where(vm, e, 0.0), axis=0)
                    wf = e * st_t * lv.to(tl.float32)
                    acc = acc * alpha + tl.sum(wf[:, None] * vt_, axis=0)
                    m = m_new
                tl.store(RO_ptr + tok * V + vv, acc / z, mask=v_m)

            # keys / values / strength+age passes: cache[j] -> cache[j+1]
            cfire = (case == K_R_CONFLICT) | (case == K_R_CAP)
            v_corr = v_b + beta * (v_t - v_b)
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                arow = sm == alloc
                crow = (sm == matched) & cfire
                kpre = tl.load(Cc_K_ptr + ((cb_j + sm[:, None]) * K + kk[None, :]),
                               mask=vm[:, None] & k_m[None, :], other=0.0)
                tl.store(Cc_K_ptr + ((cb_n + sm[:, None]) * K + kk[None, :]),
                         tl.where(arow[:, None], k_t[None, :], kpre),
                         mask=vm[:, None] & k_m[None, :])
                vpre = tl.load(Cc_V_ptr + ((cb_j + sm[:, None]) * V + vv[None, :]),
                               mask=vm[:, None] & v_m[None, :], other=0.0)
                vnew = tl.where(arow[:, None], (gate * v_t)[None, :],
                                tl.where(crow[:, None], v_corr[None, :], vpre))
                tl.store(Cc_V_ptr + ((cb_n + sm[:, None]) * V + vv[None, :]), vnew,
                         mask=vm[:, None] & v_m[None, :])
                lv = tl.load(Cc_L_ptr + cb_j + sm, mask=vm, other=0) > 0
                fr = tl.load(TCFreed_ptr + tok * S + sm, mask=vm, other=0)
                lvd = (lv & ~(fr == K_FREED_POP)) | arow
                spre = tl.load(Cc_S_ptr + cb_j + sm, mask=vm, other=0.0)
                spop = tl.where((sm == popt) & popf, spre * (1.0 - pop), spre)
                sall = tl.where(arow, 1.0, spop)
                snext = tl.where(lvd, sall * decay, sall)
                tl.store(Cc_S_ptr + cb_n + sm, snext, mask=vm)
                agpre = tl.load(Cc_A_ptr + cb_j + sm, mask=vm, other=0)
                agnew = tl.where(arow, 0, agpre) + lvd.to(tl.int32)
                tl.store(Cc_A_ptr + cb_n + sm, agnew, mask=vm)

            if ECHO:
                tl.store(EB_case_ptr + tok, case)
                tl.store(EB_match_ptr + tok, matched)
                tl.store(EB_popt_ptr + tok, popt)
                tl.store(EB_alloc_ptr + tok, alloc)
                tl.store(EB_popf_ptr + tok, popf.to(tl.int8))
            tl.debug_barrier()

        # replayed segment-boundary store (gate instrumentation)
        if BOUNDARY:
            if (seg + 1) * CKPT <= T:
                cb_n = pC + seg_len * S
                for s0 in range(0, S, TS):
                    sm = s0 + tl.arange(0, TS)
                    vm = sm < S
                    bdb = ((((seg64 + 1) * B + b) * NH + h) * S + sm)
                    tl.store(BD_K_ptr + (bdb[:, None] * K + kk[None, :]),
                             tl.load(Cc_K_ptr + ((cb_n + sm[:, None]) * K + kk[None, :]),
                                     mask=vm[:, None] & k_m[None, :], other=0.0),
                             mask=vm[:, None] & k_m[None, :])
                    tl.store(BD_V_ptr + (bdb[:, None] * V + vv[None, :]),
                             tl.load(Cc_V_ptr + ((cb_n + sm[:, None]) * V + vv[None, :]),
                                     mask=vm[:, None] & v_m[None, :], other=0.0),
                             mask=vm[:, None] & v_m[None, :])
                    tl.store(BD_S_ptr + bdb,
                             tl.load(Cc_S_ptr + cb_n + sm, mask=vm, other=0.0), mask=vm)
                    tl.store(BD_L_ptr + bdb,
                             tl.load(Cc_L_ptr + cb_n + sm, mask=vm, other=0), mask=vm)
                    tl.store(BD_A_ptr + bdb,
                             tl.load(Cc_A_ptr + cb_n + sm, mask=vm, other=0), mask=vm)
        tl.debug_barrier()

        # ---- reverse walk: exact per-step VJPs (decisions are trace inputs) ----
        for j_rev in range(0, seg_len):
            j = seg_len - 1 - j_rev
            t = t0 + j
            t64 = tl.full([1], t, tl.int64)
            tok = (b * T + t64) * NH + h
            q_t = tl.load(Q_ptr + tok * K + kk, mask=k_m, other=0.0).to(tl.float32)
            v_t = tl.load(V_ptr + tok * V + vv, mask=v_m, other=0.0).to(tl.float32)
            beta = tl.load(Beta_ptr + tok).to(tl.float32)
            pop = tl.load(Pop_ptr + tok).to(tl.float32)
            alog = tl.load(Alog_ptr + tok).to(tl.float32)
            dout = tl.load(dOut_ptr + tok * V + vv, mask=v_m, other=0.0).to(tl.float32)
            case = tl.load(TCcase_ptr + tok)
            matched = tl.load(TCmatch_ptr + tok)
            popf = tl.load(TCPopf_ptr + tok) > 0
            popt = tl.load(TCPopt_ptr + tok)
            alloc = tl.load(TCAlloc_ptr + tok)
            cb_j = pC + j * S

            dk_acc = tl.zeros([BK], tl.float32)
            dq_acc = tl.zeros([BK], tl.float32)
            dv_acc = tl.zeros([BV], tl.float32)
            dbeta_acc = tl.full([1], 0.0, tl.float32)
            dpop_acc = tl.full([1], 0.0, tl.float32)
            dalog_acc = tl.full([1], 0.0, tl.float32)

            # (a) decay VJP on the threaded strength adjoint (+ has_live recompute)
            has_live = tl.full([1], 0, tl.int32)
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                lv = tl.load(Cc_L_ptr + cb_j + sm, mask=vm, other=0) > 0
                fr = tl.load(TCFreed_ptr + tok * S + sm, mask=vm, other=0)
                lvp = lv & ~(fr == K_FREED_POP)
                lvd = lvp | (sm == alloc)
                g = tl.load(G_S_ptr + pS + sm, mask=vm, other=0.0)
                tl.store(G_S_ptr + pS + sm, tl.where(lvd, g * decay, g), mask=vm)
                has_live = tl.maximum(has_live, tl.sum(lvp.to(tl.int32), axis=0))
            tl.debug_barrier()

            # (b) delta-rule correction / merge adjoint (values at `matched`)
            cfire = (case == K_R_CONFLICT) | (case == K_R_CAP)
            ctg = tl.maximum(matched, 0)
            gvc = tl.load(G_V_ptr + (pS + ctg) * V + vv, mask=v_m & cfire, other=0.0)
            vbc = tl.load(Cc_V_ptr + (cb_j + ctg) * V + vv, mask=v_m & cfire, other=0.0)
            tl.store(G_V_ptr + (pS + ctg) * V + vv, (1.0 - beta) * gvc,
                     mask=v_m & cfire)
            dv_acc += tl.where(cfire, beta * gvc, 0.0)
            dbeta_acc += tl.where(cfire, tl.sum((v_t - vbc) * gvc), 0.0)

            # (c) append adjoint (keys, gate-scaled values, strength reset)
            a_ok = alloc >= 0
            gka = tl.load(G_K_ptr + (pS + alloc) * K + kk, mask=k_m & a_ok, other=0.0)
            dk_acc += tl.where(a_ok, gka, 0.0)
            tl.store(G_K_ptr + (pS + alloc) * K + kk, 0.0, mask=k_m & a_ok)
            gva = tl.load(G_V_ptr + (pS + alloc) * V + vv, mask=v_m & a_ok, other=0.0)
            m_ok = matched >= 0
            v_bm = tl.load(Cc_V_ptr + (cb_j + matched) * V + vv, mask=v_m & m_ok, other=0.0)
            resid_m = tl.sum((v_t - v_bm) * (v_t - v_bm), axis=0)
            err_mag_m = tl.where(has_live > 0, tl.sqrt(tl.maximum(resid_m, 0.0)), 0.0)
            gate_m = tl.sigmoid(alog + aec * err_mag_m)
            dv_acc += tl.where(a_ok, gate_m * gva, 0.0)
            dgate = tl.where(a_ok, tl.sum(v_t * gva), 0.0)
            dzg = dgate * gate_m * (1.0 - gate_m)
            dalog_acc += dzg
            daec_acc += dzg * err_mag_m
            tl.store(G_V_ptr + (pS + alloc) * V + vv, 0.0, mask=v_m & a_ok)
            tl.store(G_S_ptr + pS + alloc, 0.0, mask=a_ok)

            # (d) pop adjoint (strength decrement at the pop target)
            pt_ok = popf & (popt >= 0)
            spre_p = tl.load(Cc_S_ptr + cb_j + popt, mask=pt_ok, other=0.0)
            gp = tl.load(G_S_ptr + pS + popt, mask=pt_ok, other=0.0)
            dpop_acc += tl.where(pt_ok, -spre_p * gp, 0.0)
            tl.store(G_S_ptr + pS + popt, (1.0 - pop) * gp, mask=pt_ok)
            tl.debug_barrier()

            # (e) read adjoint: out = sum_i softmax(s)_i * sigma_i * live_i * v_i
            m_r = tl.full([1], NEG, tl.float32)
            z_r = tl.full([1], 0.0, tl.float32)
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                kt = tl.load(Cc_K_ptr + ((cb_j + sm[:, None]) * K + kk[None, :]),
                             mask=vm[:, None] & k_m[None, :], other=0.0).to(tl.float32)
                lv = tl.load(Cc_L_ptr + cb_j + sm, mask=vm, other=0) > 0
                sc = tl.sum(kt * q_t[None, :], axis=1) * scale
                scm = tl.where(lv, sc, NEG)
                tm_ = tl.max(scm, axis=0)
                m_new = tl.maximum(m_r, tm_)
                alpha = tl.exp(m_r - m_new)
                e = tl.exp(scm - m_new)
                z_r = z_r * alpha + tl.sum(tl.where(vm, e, 0.0), axis=0)
                m_r = m_new
            S_p = tl.full([1], 0.0, tl.float32)
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                kt = tl.load(Cc_K_ptr + ((cb_j + sm[:, None]) * K + kk[None, :]),
                             mask=vm[:, None] & k_m[None, :], other=0.0).to(tl.float32)
                vt_ = tl.load(Cc_V_ptr + ((cb_j + sm[:, None]) * V + vv[None, :]),
                              mask=vm[:, None] & v_m[None, :], other=0.0).to(tl.float32)
                st_t = tl.load(Cc_S_ptr + cb_j + sm, mask=vm, other=0.0)
                lv = tl.load(Cc_L_ptr + cb_j + sm, mask=vm, other=0) > 0
                lf = lv.to(tl.float32)
                sc = tl.sum(kt * q_t[None, :], axis=1) * scale
                scm = tl.where(lv, sc, NEG)
                p = tl.exp(scm - m_r) / z_r
                dw = tl.sum(vt_ * dout[None, :], axis=1)
                dp = st_t * lf * dw
                S_p += tl.sum(p * dp, axis=0)
                gw = p * st_t * lf
                gvr = tl.load(G_V_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                              mask=lv[:, None] & v_m[None, :], other=0.0)
                tl.store(G_V_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                         gvr + gw[:, None] * dout[None, :],
                         mask=lv[:, None] & v_m[None, :])
                gsr = tl.load(G_S_ptr + pS + sm, mask=lv, other=0.0)
                tl.store(G_S_ptr + pS + sm, gsr + p * lf * dw, mask=lv)
                tl.store(DP_ptr + pS + sm, dp, mask=vm)
            for s0 in range(0, S, TS):
                sm = s0 + tl.arange(0, TS)
                vm = sm < S
                kt = tl.load(Cc_K_ptr + ((cb_j + sm[:, None]) * K + kk[None, :]),
                             mask=vm[:, None] & k_m[None, :], other=0.0).to(tl.float32)
                lv = tl.load(Cc_L_ptr + cb_j + sm, mask=vm, other=0) > 0
                lf = lv.to(tl.float32)
                dp = tl.load(DP_ptr + pS + sm, mask=vm, other=0.0)
                sc = tl.sum(kt * q_t[None, :], axis=1) * scale
                scm = tl.where(lv, sc, NEG)
                p = tl.exp(scm - m_r) / z_r
                ds = lf * p * (dp - S_p)
                gkr = tl.load(G_K_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                              mask=lv[:, None] & k_m[None, :], other=0.0)
                tl.store(G_K_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                         gkr + ds[:, None] * (scale * q_t)[None, :],
                         mask=lv[:, None] & k_m[None, :])
                dq_acc += tl.sum(ds[:, None] * kt, axis=0) * scale
            tl.debug_barrier()

            tl.store(dK_ptr + tok * K + kk, dk_acc, mask=k_m)
            tl.store(dQ_ptr + tok * K + kk, dq_acc, mask=k_m)
            tl.store(dV_ptr + tok * V + vv, dv_acc, mask=v_m)
            tl.store(dBeta_ptr + tok, dbeta_acc)
            tl.store(dPop_ptr + tok, dpop_acc)
            tl.store(dAlog_ptr + tok, dalog_acc)

    # ---- initial-state adjoint (threaded to t=0) + per-head aec grad ----
    for s0 in range(0, S, TS):
        sm = s0 + tl.arange(0, TS)
        vm = sm < S
        tl.store(dS0K_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                 tl.load(G_K_ptr + ((pS + sm[:, None]) * K + kk[None, :]),
                         mask=vm[:, None] & k_m[None, :], other=0.0),
                 mask=vm[:, None] & k_m[None, :])
        tl.store(dS0V_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                 tl.load(G_V_ptr + ((pS + sm[:, None]) * V + vv[None, :]),
                         mask=vm[:, None] & v_m[None, :], other=0.0),
                 mask=vm[:, None] & v_m[None, :])
        tl.store(dS0S_ptr + pS + sm,
                 tl.load(G_S_ptr + pS + sm, mask=vm, other=0.0), mask=vm)
    tl.store(dAecB_ptr + pid, tl.sum(daec_acc, axis=0))


# ---------------------------------------------------------------------------
# Launch wrappers (allocation/glue only — no torch math in the scan path).
# ---------------------------------------------------------------------------
@dataclass
class FusedBuffers:
    """Raw launch outputs (fp32/int8/int32 device tensors + trace buffers)."""
    out: torch.Tensor            # [B,T,NH,V]
    f_keys: torch.Tensor         # [B,NH,S,K]
    f_values: torch.Tensor       # [B,NH,S,V]
    f_strength: torch.Tensor     # [B,NH,S]
    f_live: torch.Tensor         # int8 [B,NH,S]
    f_age: torch.Tensor          # int32 [B,NH,S]
    f_usage: torch.Tensor        # [B,NH,S]
    t_case: torch.Tensor         # int32 [B,T,NH]
    t_matched: torch.Tensor      # int32 [B,T,NH]
    t_popf: torch.Tensor         # int8  [B,T,NH]
    t_popt: torch.Tensor         # int32 [B,T,NH]
    t_alloc: torch.Tensor        # int32 [B,T,NH]
    t_won: torch.Tensor          # int8  [B,T,NH]
    t_freed: torch.Tensor        # int8  [B,T,NH,S]
    ck_keys: torch.Tensor        # [nck,B,NH,S,K] (checkpoint_dtype)
    ck_values: torch.Tensor      # [nck,B,NH,S,V]
    ck_strength: torch.Tensor    # [nck,B,NH,S]
    ck_live: torch.Tensor        # int8 [nck,B,NH,S]
    ck_age: torch.Tensor         # int32 [nck,B,NH,S]
    ck_usage: torch.Tensor       # [nck,B,NH,S]
    resident: bool
    num_ck: int


def _fused_fwd_launch(k, q, v, beta, pop, alog, aec, s0, cfg: TapeHeadConfig,
                      ckpt_interval: int, checkpoint_dtype: torch.dtype,
                      tile_s: int, max_resident: int, num_warps: int) -> FusedBuffers:
    B, T, NH, K = k.shape
    V = v.shape[-1]
    S = cfg.slots
    assert 1 <= ckpt_interval
    dev = k.device
    nck = T // ckpt_interval + 1
    resident = S <= max_resident
    tsr = _next_pow2(S) if resident else 16
    ts = tile_s

    buf = FusedBuffers(
        out=torch.empty((B, T, NH, V), device=dev, dtype=torch.float32),
        f_keys=torch.empty((B, NH, S, K), device=dev, dtype=torch.float32),
        f_values=torch.empty((B, NH, S, V), device=dev, dtype=torch.float32),
        f_strength=torch.empty((B, NH, S), device=dev, dtype=torch.float32),
        f_live=torch.empty((B, NH, S), device=dev, dtype=torch.int8),
        f_age=torch.empty((B, NH, S), device=dev, dtype=torch.int32),
        f_usage=torch.empty((B, NH, S), device=dev, dtype=torch.float32),
        t_case=torch.zeros((B, T, NH), device=dev, dtype=torch.int32),
        t_matched=torch.full((B, T, NH), NO_SLOT, device=dev, dtype=torch.int32),
        t_popf=torch.zeros((B, T, NH), device=dev, dtype=torch.int8),
        t_popt=torch.full((B, T, NH), NO_SLOT, device=dev, dtype=torch.int32),
        t_alloc=torch.full((B, T, NH), NO_SLOT, device=dev, dtype=torch.int32),
        t_won=torch.zeros((B, T, NH), device=dev, dtype=torch.int8),
        t_freed=torch.zeros((B, T, NH, S), device=dev, dtype=torch.int8),
        ck_keys=torch.empty((nck, B, NH, S, K), device=dev, dtype=checkpoint_dtype),
        ck_values=torch.empty((nck, B, NH, S, V), device=dev, dtype=checkpoint_dtype),
        ck_strength=torch.empty((nck, B, NH, S), device=dev, dtype=checkpoint_dtype),
        ck_live=torch.empty((nck, B, NH, S), device=dev, dtype=torch.int8),
        ck_age=torch.empty((nck, B, NH, S), device=dev, dtype=torch.int32),
        ck_usage=torch.empty((nck, B, NH, S), device=dev, dtype=checkpoint_dtype),
        resident=resident, num_ck=nck,
    )
    P = B * NH
    if resident:
        scratch = [buf.ck_keys] * 7  # unused; dummy pointers
        xk, xv, xs, xl, xa, xu, xw = scratch
    else:
        xk = torch.empty((P, S, K), device=dev, dtype=torch.float32)
        xv = torch.empty((P, S, V), device=dev, dtype=torch.float32)
        xs = torch.empty((P, S), device=dev, dtype=torch.float32)
        xl = torch.empty((P, S), device=dev, dtype=torch.int8)
        xa = torch.empty((P, S), device=dev, dtype=torch.int32)
        xu = torch.empty((P, S), device=dev, dtype=torch.float32)
        xw = torch.empty((P, S), device=dev, dtype=torch.float32)
    warps = num_warps if num_warps > 0 else (8 if resident else 4)
    import math
    scale = 1.0 / math.sqrt(K)
    tape_fwd_kernel[(P,)](
        k, q, v, beta, pop, alog, aec,
        s0.keys, s0.values, s0.strength,
        s0.live.to(torch.int8), s0.age.to(torch.int32), s0.usage,
        buf.out,
        buf.f_keys, buf.f_values, buf.f_strength, buf.f_live, buf.f_age, buf.f_usage,
        buf.t_case, buf.t_matched, buf.t_popf, buf.t_popt, buf.t_alloc, buf.t_won,
        buf.t_freed,
        buf.ck_keys, buf.ck_values, buf.ck_strength, buf.ck_live, buf.ck_age,
        buf.ck_usage,
        xk, xv, xs, xl, xa, xu, xw,
        S, cfg.match_threshold, cfg.eps_residual, cfg.pop_threshold,
        cfg.write_enable_threshold, cfg.free_threshold, cfg.decay_rate,
        cfg.usage_decay, scale,
        B=B, T=T, NH=NH, K=K, V=V,
        BK=_next_pow2(K), BV=_next_pow2(V),
        RESIDENT=resident, TSR=tsr, TS=ts,
        CKPT=ckpt_interval, num_warps=warps,
    )
    return buf


def _fused_bwd_launch(k, q, v, beta, pop, alog, aec, buf: FusedBuffers,
                      cfg: TapeHeadConfig, ckpt_interval: int,
                      d_out: torch.Tensor, d_final: dict, tile_s: int,
                      num_warps: int, gate_mode: bool = False):
    """Launch tape_bwd_kernel. In gate_mode the VJP seeds are zero and the
    kernel additionally writes replayed outputs / echoed trace / boundary
    dumps (the replay-consistency gate, G10)."""
    B, T, NH, K = k.shape
    V = v.shape[-1]
    S = cfg.slots
    dev = k.device
    P = B * NH
    zeros_scalar = torch.zeros((), device=dev, dtype=torch.float32)
    if d_out is None:
        d_out = torch.zeros((B, T, NH, V), device=dev, dtype=torch.float32)
    if d_final is None:
        d_final = {}
    dfk = d_final.get("keys")
    dfv = d_final.get("values")
    dfs = d_final.get("strength")
    if dfk is None:
        dfk = torch.zeros((B, NH, S, K), device=dev, dtype=torch.float32)
    if dfv is None:
        dfv = torch.zeros((B, NH, S, V), device=dev, dtype=torch.float32)
    if dfs is None:
        dfs = torch.zeros((B, NH, S), device=dev, dtype=torch.float32)

    out = {}
    out["dk"] = torch.empty_like(k)
    out["dq"] = torch.empty_like(q)
    out["dv"] = torch.empty_like(v)
    out["dbeta"] = torch.empty_like(beta)
    out["dpop"] = torch.empty_like(pop)
    out["dalog"] = torch.empty_like(alog)
    out["ds0_keys"] = torch.zeros((B, NH, S, K), device=dev, dtype=torch.float32)
    out["ds0_values"] = torch.zeros((B, NH, S, V), device=dev, dtype=torch.float32)
    out["ds0_strength"] = torch.zeros((B, NH, S), device=dev, dtype=torch.float32)
    out["daec_b"] = torch.empty((B, NH), device=dev, dtype=torch.float32)

    cache_k = torch.empty((P, ckpt_interval + 1, S, K), device=dev, dtype=torch.float32)
    cache_v = torch.empty((P, ckpt_interval + 1, S, V), device=dev, dtype=torch.float32)
    cache_s = torch.empty((P, ckpt_interval + 1, S), device=dev, dtype=torch.float32)
    cache_l = torch.empty((P, ckpt_interval + 1, S), device=dev, dtype=torch.int8)
    cache_a = torch.empty((P, ckpt_interval + 1, S), device=dev, dtype=torch.int32)
    g_k = torch.empty((P, S, K), device=dev, dtype=torch.float32)
    g_v = torch.empty((P, S, V), device=dev, dtype=torch.float32)
    g_s = torch.empty((P, S), device=dev, dtype=torch.float32)
    dp_scr = torch.empty((P, S), device=dev, dtype=torch.float32)

    if gate_mode:
        ro = torch.zeros((B, T, NH, V), device=dev, dtype=torch.float32)
        eb_case = torch.zeros((B, T, NH), device=dev, dtype=torch.int32)
        eb_match = torch.zeros((B, T, NH), device=dev, dtype=torch.int32)
        eb_popt = torch.zeros((B, T, NH), device=dev, dtype=torch.int32)
        eb_alloc = torch.zeros((B, T, NH), device=dev, dtype=torch.int32)
        eb_popf = torch.zeros((B, T, NH), device=dev, dtype=torch.int8)
        nck = buf.num_ck
        bd_k = torch.zeros((nck, B, NH, S, K), device=dev, dtype=torch.float32)
        bd_v = torch.zeros((nck, B, NH, S, V), device=dev, dtype=torch.float32)
        bd_s = torch.zeros((nck, B, NH, S), device=dev, dtype=torch.float32)
        bd_l = torch.zeros((nck, B, NH, S), device=dev, dtype=torch.int8)
        bd_a = torch.zeros((nck, B, NH, S), device=dev, dtype=torch.int32)
    else:
        ro = buf.out
        eb_case = buf.t_case; eb_match = buf.t_matched; eb_popt = buf.t_popt
        eb_alloc = buf.t_alloc; eb_popf = buf.t_popf
        bd_k = buf.ck_keys; bd_v = buf.ck_values; bd_s = buf.ck_strength
        bd_l = buf.ck_live; bd_a = buf.ck_age

    import math
    scale = 1.0 / math.sqrt(K)
    tape_bwd_kernel[(P,)](
        k, q, v, beta, pop, alog, aec,
        buf.ck_keys, buf.ck_values, buf.ck_strength, buf.ck_live, buf.ck_age,
        buf.t_case, buf.t_matched, buf.t_popf, buf.t_popt, buf.t_alloc, buf.t_freed,
        d_out, dfk, dfv, dfs,
        out["dk"], out["dq"], out["dv"], out["dbeta"], out["dpop"], out["dalog"],
        out["ds0_keys"], out["ds0_values"], out["ds0_strength"],
        out["daec_b"],
        cache_k, cache_v, cache_s, cache_l, cache_a,
        g_k, g_v, g_s, dp_scr,
        ro, eb_case, eb_match, eb_popt, eb_alloc, eb_popf,
        bd_k, bd_v, bd_s, bd_l, bd_a,
        S, cfg.free_threshold, cfg.decay_rate, scale,
        B=B, T=T, NH=NH, K=K, V=V,
        BK=_next_pow2(K), BV=_next_pow2(V), TS=tile_s,
        CKPT=ckpt_interval,
        REPLAY_OUT=gate_mode, ECHO=gate_mode, BOUNDARY=gate_mode,
        num_warps=num_warps if num_warps > 0 else 4, num_stages=1,
    )
    if gate_mode:
        out["replay_out"] = ro
        out["echo_case"] = eb_case
        out["echo_matched"] = eb_match
        out["echo_popt"] = eb_popt
        out["echo_alloc"] = eb_alloc
        out["echo_popf"] = eb_popf
        out["bd_keys"] = bd_k
        out["bd_values"] = bd_v
        out["bd_strength"] = bd_s
        out["bd_live"] = bd_l
        out["bd_age"] = bd_a
    return out


def fused_fwd_raw(k, q, v, write_strength, pop_strength, append_logit, state,
                  cfg: TapeHeadConfig, append_err_coef=None,
                  ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
                  checkpoint_dtype: torch.dtype = torch.bfloat16,
                  tile_s: int = DEFAULT_TILE_S,
                  max_resident: int = DEFAULT_MAX_RESIDENT_SLOTS,
                  num_warps: int = -1) -> FusedBuffers:
    """Raw fused forward (no autograd). Returns all kernel buffers (used by the
    gates/tests; the training path uses :func:`fused_tape_scan`)."""
    k = k.contiguous().float(); q = q.contiguous().float()
    v = v.contiguous().float()
    beta = write_strength.contiguous().float()
    pop = pop_strength.contiguous().float()
    alog = append_logit.contiguous().float()
    assert k.is_cuda, "fused tape kernels require CUDA"
    B, T, NH, K = k.shape
    assert NH == cfg.n_heads and K == cfg.key_dim and v.shape[-1] == cfg.value_dim
    assert state.keys.shape == (B, NH, cfg.slots, K)
    if append_err_coef is None:
        aec = torch.zeros(NH, device=k.device, dtype=torch.float32)
    else:
        aec = append_err_coef.detach().contiguous().float()
    return _fused_fwd_launch(k, q, v, beta, pop, alog, aec, state, cfg,
                             ckpt_interval, checkpoint_dtype, tile_s,
                             max_resident, num_warps)


def fused_replay_raw(fwd: FusedBuffers, k, q, v, write_strength, pop_strength,
                     append_logit, cfg: TapeHeadConfig, append_err_coef=None,
                     ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
                     tile_s: int = DEFAULT_TILE_S, num_warps: int = -1):
    """Trace-following segment replay via the backward kernel with zero VJP
    seeds and full gate instrumentation (replayed outputs, echoed trace,
    checkpoint-boundary store dumps). The replay-consistency gate (G10)."""
    k = k.contiguous().float(); q = q.contiguous().float()
    v = v.contiguous().float()
    beta = write_strength.contiguous().float()
    pop = pop_strength.contiguous().float()
    alog = append_logit.contiguous().float()
    B, T, NH, K = k.shape
    if append_err_coef is None:
        aec = torch.zeros(NH, device=k.device, dtype=torch.float32)
    else:
        aec = append_err_coef.detach().contiguous().float()
    return _fused_bwd_launch(k, q, v, beta, pop, alog, aec, fwd, cfg,
                             ckpt_interval, None, None, tile_s, num_warps,
                             gate_mode=True)


def buffers_to_trace(buf: FusedBuffers) -> RoutingTrace:
    """Build the integral RoutingTrace (incl. G10 freed codes) from the raw
    kernel buffers (int64/bool parity dtypes, matching the eager trace)."""
    return RoutingTrace(
        case=buf.t_case.long(),
        matched=buf.t_matched.long(),
        pop_fired=buf.t_popf > 0,
        pop_target=buf.t_popt.long(),
        alloc_target=buf.t_alloc.long(),
        w_on=buf.t_won > 0,
        freed=buf.t_freed,
    )


def buffers_to_state(buf: FusedBuffers) -> TapeState:
    return TapeState(
        keys=buf.f_keys, values=buf.f_values, strength=buf.f_strength,
        live=buf.f_live > 0, age=buf.f_age.long(), usage=buf.f_usage,
    )


# ---------------------------------------------------------------------------
# autograd.Function gluing the two fused kernels.
# ---------------------------------------------------------------------------
class TapeFusedFunction(torch.autograd.Function):
    """Fused sequential tape-head scan: fwd + bwd are both @triton.jit kernels."""

    @staticmethod
    def forward(ctx, k, q, v, beta, pop, alog, aec,
                s0_keys, s0_values, s0_strength, s0_live, s0_age, s0_usage,
                cfg, ckpt_interval, checkpoint_dtype, tile_s, max_resident,
                collect_trace):
        B, T, NH, K = k.shape
        assert k.is_cuda, "fused tape kernels require CUDA"
        V = v.shape[-1]
        S = s0_keys.shape[2]
        assert NH == cfg.n_heads and K == cfg.key_dim and V == cfg.value_dim
        assert S == cfg.slots
        state0 = TapeState(keys=s0_keys, values=s0_values, strength=s0_strength,
                           live=s0_live > 0, age=s0_age.long(), usage=s0_usage)
        buf = _fused_fwd_launch(k, q, v, beta, pop, alog, aec, state0, cfg,
                               ckpt_interval, checkpoint_dtype, tile_s,
                               max_resident, -1)
        ctx.save_for_backward(k, q, v, beta, pop, alog, aec,
                              buf.ck_keys, buf.ck_values, buf.ck_strength,
                              buf.ck_live, buf.ck_age,
                              buf.t_case, buf.t_matched, buf.t_popf, buf.t_popt,
                              buf.t_alloc, buf.t_freed)
        ctx.cfg = cfg
        ctx.ckpt_interval = ckpt_interval
        ctx.tile_s = tile_s
        ctx.collect_trace = collect_trace
        ctx.mark_non_differentiable(buf.f_live, buf.f_age, buf.f_usage)
        trace = buffers_to_trace(buf) if collect_trace else None
        return (buf.out, buf.f_keys, buf.f_values, buf.f_strength,
                buf.f_live, buf.f_age, buf.f_usage, trace)

    @staticmethod
    def backward(ctx, dout, dfk, dfv, dfs, *rest):
        (k, q, v, beta, pop, alog, aec,
         ck_keys, ck_values, ck_strength, ck_live, ck_age,
         t_case, t_matched, t_popf, t_popt, t_alloc, t_freed) = ctx.saved_tensors
        cfg = ctx.cfg
        B, T, NH, K = k.shape
        V = v.shape[-1]
        S = cfg.slots
        if dout.shape[1] != T:
            raise ValueError("dOut token length mismatch")
        dout = dout.contiguous().float()
        if dfk is None:
            dfk = torch.zeros((B, NH, S, K), device=k.device, dtype=torch.float32)
        else:
            dfk = dfk.contiguous().float()
        if dfv is None:
            dfv = torch.zeros((B, NH, S, V), device=k.device, dtype=torch.float32)
        else:
            dfv = dfv.contiguous().float()
        if dfs is None:
            dfs = torch.zeros((B, NH, S), device=k.device, dtype=torch.float32)
        else:
            dfs = dfs.contiguous().float()
        # a minimal FusedBuffers view over the saved checkpoint/trace tensors
        # (only ck_* / t_* / num_ck are read by the backward launcher)
        buf = FusedBuffers(
            out=dout, f_keys=dfk, f_values=dfv, f_strength=dfs,
            f_live=t_popf, f_age=t_popf, f_usage=t_popf,
            t_case=t_case, t_matched=t_matched, t_popf=t_popf, t_popt=t_popt,
            t_alloc=t_alloc, t_won=t_popf, t_freed=t_freed,
            ck_keys=ck_keys, ck_values=ck_values, ck_strength=ck_strength,
            ck_live=ck_live, ck_age=ck_age, ck_usage=ck_live,
            resident=False, num_ck=T // ctx.ckpt_interval + 1,
        )
        grads = _fused_bwd_launch(k, q, v, beta, pop, alog, aec, buf, cfg,
                                  ctx.ckpt_interval, dout,
                                  {"keys": dfk, "values": dfv, "strength": dfs},
                                  ctx.tile_s, -1)
        daec = grads["daec_b"].sum(0) if ctx.needs_input_grad[6] else None
        return (grads["dk"], grads["dq"], grads["dv"], grads["dbeta"],
                grads["dpop"], grads["dalog"], daec,
                grads["ds0_keys"], grads["ds0_values"], grads["ds0_strength"],
                None, None, None, None, None, None, None, None, None)


def fused_tape_scan(k, q, v, write_strength, pop_strength, append_logit, state,
                    cfg: TapeHeadConfig, append_err_coef=None,
                    ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
                    checkpoint_dtype: torch.dtype = torch.bfloat16,
                    collect_trace: bool = True,
                    tile_s: int = DEFAULT_TILE_S,
                    max_resident: int = DEFAULT_MAX_RESIDENT_SLOTS):
    """Autograd-enabled fused sequential tape-head scan (the phase-2 hot path).

    Mirrors the eager ``tape_head.tape_scan`` signature/semantics (G1-G14
    bindings). Inputs are fp32 CUDA tensors shaped [B,T,NH,*]; ``state`` is the
    entry :class:`TapeState` (fresh zero state or a caller-provided store —
    gradients flow to its keys/values/strength). Returns
    ``(read_out, final_state, trace)``.

    FLAGGED (reported): ``trace_override`` (eager replay mode) is NOT
    supported on the fused forward — trace-following replay lives inside
    ``tape_bwd_kernel`` (and its gate-instrumented form
    :func:`fused_replay_raw`); the eager oracle retains the trace-override
    API for the eager-side gates.
    """
    k = k.contiguous().float(); q = q.contiguous().float()
    v = v.contiguous().float()
    beta = write_strength.contiguous().float()
    pop = pop_strength.contiguous().float()
    alog = append_logit.contiguous().float()
    B, T, NH, K = k.shape
    if append_err_coef is None:
        aec = torch.zeros(NH, device=k.device, dtype=torch.float32)
    else:
        aec = append_err_coef.contiguous().float()
    s0_keys = state.keys.contiguous().float()
    s0_values = state.values.contiguous().float()
    s0_strength = state.strength.contiguous().float()
    s0_live = state.live.to(torch.int8).contiguous()
    s0_age = state.age.to(torch.int32).contiguous()
    s0_usage = state.usage.contiguous().float()
    out, fk, fv, fs, flive, fage, fusage, trace = TapeFusedFunction.apply(
        k, q, v, beta, pop, alog, aec,
        s0_keys, s0_values, s0_strength, s0_live, s0_age, s0_usage,
        cfg, ckpt_interval, checkpoint_dtype, tile_s, max_resident, collect_trace)
    final_state = TapeState(keys=fk, values=fv, strength=fs,
                            live=flive > 0, age=fage.long(), usage=fusage)
    return out, final_state, trace


__all__ = [
    "tape_fwd_kernel", "tape_bwd_kernel", "TapeFusedFunction", "fused_tape_scan",
    "fused_fwd_raw", "fused_replay_raw", "buffers_to_trace", "buffers_to_state",
    "FusedBuffers",
    "DEFAULT_CKPT_INTERVAL", "DEFAULT_TILE_S", "DEFAULT_MAX_RESIDENT_SLOTS",
]
