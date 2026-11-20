"""K15 -- THE COMPOSITION: the two proven-positive quality levers of the
session, composed.  The K13 ADDITIVE READ (the banked score channel verbatim,
softmax removed, strength dropped, the g_proj read gate zero-born at exact 1)
on the K11B p3 PER-CHANNEL GATE STACK (the per-channel input-dependent DECAY
via the f_proj analog, the per-channel write admission via the b_proj analog,
the per-channel read gate via the g_proj analog).

Why this file exists
--------------------
K13 receipted the additive read as the session's largest single lever
(post-cooldown 1.3928/1.3845 vs the banked 1.4338, ~60% of the GDN2 gap
closed, |w|-mass effective rows 3.6x, no collapse).  K11B receipted the p3
gate stack as seed-stable on the SOFTMAX read (+0.0103/+0.0104 vs the banked
anchor).  The two levers touch DIFFERENT sub-mechanisms of the same store --
the read form (a value-channel change that touches neither the store nor the
ops) and the write-side gate granularity (DECAY/write channels that scale
payload magnitudes, never op selection) -- so the only question is whether
they COMPOSE.  This lane is that composition, at the matched-exposure probe
protocol under the corrected scoring, both seeds.

The semantics (``PREREGISTRATION_COMPOSITION.md``, committed BEFORE any probe
measurement)
--------------------------------------------------------------------------------
* **The read is K13's, VERBATIM** (rulings R13-1/R13-2): ``w_j =
  read_scale * cos(q_key(t), K_j)`` -- the banked score channel -- raw and
  live-gated; NO softmax, NO ``||S_j||`` strength factor (magnitude rides
  ``v_j = S[j]`` itself); the read is the K6 floor-channel rank sum; the
  k11b ``g_proj`` read gate (R13-3) applies OUTSIDE the scan.
* **The gate stack is K11B p3's, VERBATIM** (its committed construction):
  the per-channel DECAY ``lam_vec = clamp(lam_inc * (1 +
  m_f*sigmoid(f_proj(x))), floor, 1-1e-6)`` with the clamp bounds in the
  incumbent's carrier dtype (the falsifier's corner, receipt 57e1f942); the
  per-channel write admission ``g_vec = 1 - m_w*sigmoid(b_proj(x))``
  (APPEND admits ``g_vec (.) p``, CORRECT blends ``g_vec (.) p +
  (1-g_vec) (.) (lam_vec (.) v_pre)``); the per-channel read gate
  ``r_vec = 1 + m_o*sigmoid(g_proj(x))`` -- the SAME construction K13's
  additive machine already carries (the composition instantiates it ONCE;
  the parameter delta is p3's own 8,657,040 at depth 24, never
  double-added).
* **Birth discipline**: the K11B gates are born at the registered scalar
  equivalents -- bit-equal at birth through the ``channel_decay`` wrapper
  (the committed construction, ``lam_vec == lam_inc`` per channel,
  ``g_vec == 1``, ``r_vec == 1``) -- so at birth THE COMPOSITION IS THE K13
  ADDITIVE MACHINE bit for bit (receipted engine-side: every channel of the
  K15 forward launch vs the K13 kernel's own at identical inputs).  The
  additive read has no birth parity vs the BANKED winner (a different
  function -- the K13 decisions-bit-identical receipt stands: 0 flips in
  315M lanes, the read cannot feed op selection); the composition's step-1
  is NOT banked-parity and none is asserted -- this is a machine-variant arm
  like K13 was.  The step-1 loss is asserted bit-identical to the K13
  additive arm's own same-seed row instead (the composition's nearest kin:
  the same function at birth), with the K11-receipted -0.0 corner the only
  legal escape, escalated not excused if it fires.
* **P2 (decisions inviolable), verbatim**: no gate value, scalar or
  per-channel, and no read form feeds op selection.  The discrete op
  sequence is bit-invariant to every gate value AND to the read form at
  fixed inputs.

The kernels (the two committed engines, imported; ``k10_*``/``k11b_*``/
``k13_*`` sources are imported, never edited)
--------------------------------------------------------------------------------
``_k15_seq_fwd_kernel``     THE ONE NEW KERNEL: K11B's persistent
                            whole-window walk with the read block replaced by
                            K13's additive block (``w = where(live, score,
                            0)``; the softmax's exp/max/denominator GONE).
                            Everything else is K11B's verbatim: the
                            per-channel DECAY load, the gated lane-0
                            arithmetic, the 2-lane recorded schedule.
``_k13_bwd_prep_kernel``   IMPORTED VERBATIM (the additive read adjoint's
                            coefficients: ``dscores = dw`` on live rows, the
                            strength channel EXACT ZEROS, the clamp-select
                            key-side identities -- the decomposition is
                            form-agnostic).
``_k11b_bwd_value_walk_kernel`` IMPORTED VERBATIM (the per-channel DECAY and
                            gate adjoints plus the read adjoint's value
                            terms; the strength path dies by the prep's
                            zero ``dstrength`` channel -- IEEE add of 0.0 is
                            exact, the K13 receipt carried over).
``_k11b_traj_kernel``      IMPORTED VERBATIM via ``channel_persistent_trajectory``
                            (the per-channel segment value replay at K4's
                            defined-order spec).
``_k10_key_traj_kernel``   IMPORTED VERBATIM (the key lane lives on the key
                            channel; keys are never decayed, never gated).
``_k10_bwd_key_walk_kernel`` IMPORTED VERBATIM (the key-table adjoint,
                            consuming the prep's c/nk channels unchanged).

Doctrine, unchanged: decisions are recorded constants in the backward.  The
read weights (raw scores), payload lanes, the PER-CHANNEL DECAY vector, the
PER-CHANNEL write gate and the KEY CHANNEL carry gradient to their inputs;
the cross-axis reductions are autograd-ordered on the oracle side, so the
gradient gates are floor-discipline (fp64 gradcheck of the oracle THROUGH
THE COMPOSED PATH first, fp32-vs-fp64 floors, ``NOISE_FACTOR`` x), stated,
not hidden.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import triton
import triton.language as tl

from .algebra import (APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule)
from .k1_schedule_scan import _add_rn, _mul_rn, _sub_rn
from .k6_sequential_matrix import (_div_rn32, _fold_axis2, _fold_vec, _fp32,
                                   _log2_up, _norm_defined, _pow2,
                                   _round_f32, _seq_warps, _sqrt_rn32,
                                   tree_fold)
from .k10_separated_keys import (_key_selection, DEFAULT_CONTEXT,
                                 DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS,
                                 MAX_KEY_TILE, MIMETIC_SEED,
                                 SeparatedSequentialMixer,
                                 _k10_bwd_key_walk_kernel,
                                 _k10_key_traj_kernel, _split_fold_axis2,
                                 separated_available, span_key_std)
from .k11b_channel_gates import (LANES, ChannelGatedSeparatedMixer,
                                 _k11b_bwd_value_walk_kernel,
                                 arm_channels, arm_parameter_delta,
                                 channel_persistent_trajectory)
from .k13_additive_read import (BT_BWD, BV_BWD, _k13_bwd_prep_kernel)
from .key_separation_probe import causal_context
from .scan import RoutedResult, ScanStats
from .schedule import EPS, RoutingPolicy

_EPS = tl.constexpr(EPS)
_BIG_INDEX = tl.constexpr(1 << 30)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))


def composed_parameter_delta(depth: int, width: int, heads: int,
                            value_dim: int) -> int:
    """The registration's parameter accounting: the composition's gate stack
    is p3's own -- the per-channel DECAY (f_proj), write (b_proj) and read
    (g_proj) channels, the read channel being the SAME construction the K13
    additive machine already carries (instantiated once, never
    double-added); the additive read itself is parameter-free.  Computed by
    the committed k11b accounting, not hardcoded."""
    return arm_parameter_delta("p3", depth, width, heads, value_dim)


# ------------------------------------------------------------- the oracle


def composed_oracle_scan(payload, write_keys, query_keys, decay, gates, *,
                         policy: RoutingPolicy, gated: bool = True,
                         rank=None, rank_max=None, initial_state=None,
                         initial_live=None, initial_keys=None, collect=True):
    """The composition's eager per-token oracle: the parity fixed point,
    differentiable through the key channel, the per-channel DECAY vector,
    the per-channel write gate and the ADDITIVE read.

    K11B's oracle walk (decisions from the DETACHED key channel, the MERGE
    blend from the PRE-decay rows, the per-channel DECAY and the gated
    lane-0 arithmetic, the per-lane-guarded MERGE-source close -- the
    K12-flagged multi-lane corner of the K10/K11/K13 oracles is NOT copied)
    with the read block replaced by K13's: ``w_j = score_j`` live-gated (the
    raw banked score; NO softmax, NO ``||S_j||`` factor -- magnitude rides
    ``S[j]`` itself), the read the same defined-order rank fold.  Every
    recorded channel is the committed engines' own, bit for bit at fixed
    inputs."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape[:-1] != payload.shape[:-1]:
        raise ValueError("keys must share the payload's (batch, tokens, heads)")
    if decay.shape != payload.shape:
        raise ValueError(f"the per-channel decay must share the payload's "
                         f"shape {tuple(payload.shape)}; got "
                         f"{tuple(decay.shape)}")
    if gated and gates.shape != payload.shape:
        raise ValueError(f"the per-channel gates must share the payload's "
                         f"shape {tuple(payload.shape)}; got "
                         f"{tuple(gates.shape)}")
    if not policy.ops:
        raise ValueError("the append-only control is refused by the "
                          "composition registration; it is a chunk-dial "
                          "control")
    if policy.grow:
        raise ValueError("growth is refused by the composition registration: "
                         "growth is a chunk-boundary concept")
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    if int(rank) != int(rank_max):
        raise ValueError("the composition registration requires rank == "
                         "rank_max (the registered shape); a rank prefix is "
                         "refused by name")
    device = payload.device
    compute_dtype = torch.promote_types(_fp32(payload).dtype, torch.float32)
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=compute_dtype,
                         device=device) if initial_state is None
             else initial_state)
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
            if initial_live is None else initial_live.clone())
    keys = (torch.zeros(batch, heads, rank, key_dim, dtype=compute_dtype,
                        device=device) if initial_keys is None
            else initial_keys)
    p32 = _fp32(payload)
    wk32 = _fp32(write_keys)
    qk32 = _fp32(query_keys)
    d32 = _fp32(decay)
    g32 = _fp32(gates)
    RP, VP, KP = _pow2(rank), _pow2(value_dim), _pow2(key_dim)
    tau_match, tau_merge, tau_pop = (_round_f32(policy.tau_match),
                                     _round_f32(policy.tau_merge),
                                     _round_f32(policy.tau_pop))
    tie = _round_f32(policy.tie_break)
    scale = _round_f32(policy.read_scale)
    lanes = LANES
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=compute_dtype,
                        device=device)
    fields = dict(
        slot=torch.zeros(batch, tokens, heads, lanes, dtype=torch.int64,
                         device=device),
        erase=torch.zeros(batch, tokens, heads, lanes, dtype=torch.bool,
                          device=device),
        payload=torch.zeros(batch, tokens, heads, lanes, value_dim,
                            dtype=compute_dtype, device=device),
        op=torch.zeros(batch, tokens, heads, lanes, dtype=torch.int64,
                       device=device),
        live_after=torch.zeros(batch, tokens, heads, lanes, dtype=torch.bool,
                               device=device))
    weights_out = (torch.zeros(batch, tokens, heads, rank,
                               dtype=compute_dtype, device=device)
                   if collect else None)
    live_pre_out = (torch.zeros(batch, tokens, heads, rank, dtype=torch.bool,
                                device=device) if collect else None)
    b_index = torch.arange(batch, device=device)[:, None]
    h_index = torch.arange(heads, device=device)[None, :]
    for t in range(tokens):
        if collect:
            live_pre_out[:, t] = live
        p_t, wk_t, qk_t = p32[:, t], wk32[:, t], qk32[:, t]
        lam_t = d32[:, t]                                 # (b, h, V)
        # ---- the read (K13's block, VERBATIM): the banked score channel,
        # the raw score live-gated -- NO softmax, NO strength.  An all-dead
        # store reads EXACTLY zero (w is zero on every row).
        nk = _norm_defined(keys, -1, KP)                       # (b, h, rank)
        nqk = _norm_defined(qk_t, -1, KP)                      # (b, h)
        uqk = qk_t / nqk.clamp_min(EPS).unsqueeze(-1)
        scores = (scale * tree_fold(uqk.unsqueeze(2) * keys, -1, KP)) \
            / nk.clamp_min(EPS)
        w = scores.masked_fill(~live, 0.0)
        reads[:, t] = tree_fold(w.unsqueeze(-1) * state, -2, RP)
        if collect:
            weights_out[:, t] = w
        # ---- the decisions: the registered chain on the DETACHED key channel
        # (K10 verbatim -- P2: no gate value, no read form feeds op
        # selection).
        keys_det = keys.detach()
        nkd = _norm_defined(keys_det, -1, KP)
        wk_det = wk_t.detach()
        uk = wk_det / _norm_defined(wk_det, -1, KP).clamp_min(EPS).unsqueeze(-1)
        sim = tree_fold(uk.unsqueeze(2) * keys_det, -1, KP) \
            / nkd.clamp_min(EPS)                               # (b, h, rank)
        sim = sim.masked_fill(~live, -2.0)
        sel = _key_selection(sim, live, tau_match, tau_merge, tau_pop, tie,
                             rank, RP)
        # ---- the MERGE blend from the PRE-decay rows (K10 verbatim: the
        # registered detached strengths, K6's own blend weights).
        strength = _norm_defined(state.detach(), -1, VP) * live.to(compute_dtype)
        blend = None
        if bool(sel["merge"].any()):
            rows = state[b_index.expand(batch, heads),
                         h_index.expand(batch, heads), sel["best_index"]]
            rows2 = state[b_index.expand(batch, heads),
                          h_index.expand(batch, heads), sel["second_index"]]
            fw = strength.gather(-1, sel["best_index"].unsqueeze(-1)
                                 ).clamp_min(EPS).squeeze(-1)
            sw = strength.gather(-1, sel["second_index"].unsqueeze(-1)
                                 ).clamp_min(EPS).squeeze(-1)
            total = fw + sw
            blend = ((fw / total).detach().unsqueeze(-1) * rows
                     + (sw / total).detach().unsqueeze(-1) * rows2)
        # ---- DECAY (values only, PER CHANNEL), then the gated lane-0 value
        # (K11B verbatim: the CORRECT blend is the PRE-decay per-channel
        # form g*p + (1-g)*(lam*v_pre) -- the identical expression the
        # kernel implements, so the parity bar is torch.equal).
        rows_pre = state[b_index.expand(batch, heads),
                         h_index.expand(batch, heads), sel["slot0"]]
        state = state * lam_t.unsqueeze(2)                # (b,h,R,V)*(b,h,1,V)
        rows_at = state[b_index.expand(batch, heads),
                        h_index.expand(batch, heads), sel["slot0"]]
        if gated:
            gate_t = g32[:, t]                            # (b, h, V)
            blend_correct = gate_t * p_t + (1.0 - gate_t) * (lam_t * rows_pre)
            admission = gate_t * p_t
            lane0 = torch.where(sel["correct"].unsqueeze(-1), blend_correct,
                                torch.where(sel["carries"].unsqueeze(-1),
                                            admission,
                                            torch.zeros_like(p_t)))
        else:
            lane0 = torch.where(sel["carries"].unsqueeze(-1), p_t,
                                torch.zeros_like(p_t))
        if blend is not None:
            lane0 = torch.where(sel["merge"].unsqueeze(-1), blend, lane0)
        # ---- the key lane: K10 verbatim.
        key_lane = torch.where(sel["op0"].eq(POP).unsqueeze(-1),
                               torch.zeros_like(wk_t), wk_t)
        # ---- record lanes 0/1 (K10's 2-lane layout) and apply.
        fields["slot"][:, t, :, 0] = sel["slot0"]
        fields["slot"][:, t, :, 1] = sel["second_index"]
        fields["op"][:, t, :, 0] = sel["op0"]
        fields["op"][:, t, :, 1] = torch.where(sel["merge"],
                                               torch.full_like(sel["op0"], MERGE),
                                               torch.full_like(sel["op0"], NONE))
        fields["erase"][:, t, :, 0] = sel["erase0"]
        fields["erase"][:, t, :, 1] = sel["merge"]
        fields["live_after"][:, t, :, 0] = sel["live0"]
        fields["live_after"][:, t, :, 1] = False
        fields["payload"][:, t, :, 0, :] = lane0
        written = torch.where(sel["erase0"].unsqueeze(-1), lane0,
                              rows_at + lane0)
        state = state.index_put((b_index.expand(batch, heads),
                                 h_index.expand(batch, heads),
                                 sel["slot0"]), written)
        keys_at = keys[b_index.expand(batch, heads),
                       h_index.expand(batch, heads), sel["slot0"]]
        keys = keys.index_put(
            (b_index.expand(batch, heads),
             h_index.expand(batch, heads),
             sel["slot0"]),
            torch.where(sel["op0"].ne(NONE).unsqueeze(-1), key_lane, keys_at))
        live = live.index_put((b_index.expand(batch, heads),
                               h_index.expand(batch, heads),
                               sel["slot0"]),
                              torch.where(sel["op0"].ne(NONE), sel["live0"],
                                          live.gather(-1,
                                                      sel["slot0"].unsqueeze(-1)
                                                      ).squeeze(-1)))
        # the MERGE-source close, guarded PER LANE (K11B's committed fix of
        # the K12-flagged multi-lane corner -- the kernel's own
        # ``hit2 = merge & (second == ranks)`` semantics, bit for bit)
        if bool(sel["merge"].any()):
            second_rows = state[b_index.expand(batch, heads),
                                 h_index.expand(batch, heads),
                                 sel["second_index"]]
            state = state.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.where(sel["merge"].unsqueeze(-1),
                            torch.zeros(batch, heads, value_dim,
                                        dtype=compute_dtype, device=device),
                            second_rows))
            second_keys = keys[b_index.expand(batch, heads),
                                h_index.expand(batch, heads),
                                sel["second_index"]]
            keys = keys.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.where(sel["merge"].unsqueeze(-1),
                            torch.zeros(batch, heads, key_dim,
                                        dtype=compute_dtype, device=device),
                            second_keys))
            second_live = live.gather(-1, sel["second_index"].unsqueeze(-1)
                                       ).squeeze(-1)
            live = live.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.where(sel["merge"], torch.zeros(batch, heads,
                                                      dtype=torch.bool,
                                                      device=live.device),
                            second_live))
    schedule = MatrixSchedule(**fields)
    return dict(reads=reads, state=state, live=live, keys=keys,
                schedule=schedule, weights=weights_out, live_pre=live_pre_out,
                rank=int(rank))


# ------------------------------------------------------------- the forward kernel


@triton.jit(do_not_specialize=['T', 'SEG'])
def _k15_seq_fwd_kernel(
    payload, write_keys, query_keys, decay, gates, state_in, live_in, keys_in,
    reads, state_out, live_out, keys_out,
    slot_out, op_out, erase_out, live_after_out, payload_out, live_pre,
    ckpt_state, ckpt_live, ckpt_keys, weights_out,
    T, R, ROWS, SEG, KD,
    pbs, pts, phs, pvs,
    wbs, wts, whs, wks,
    qbs, qts, qhs, qks,
    cbs, cts, chs, cvs,
    gbs, gts, ghs, gvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, KP: tl.constexpr,
    RP: tl.constexpr, L: tl.constexpr,
    LOGV: tl.constexpr, LOGR: tl.constexpr, LOGK: tl.constexpr,
    TAU_MATCH: tl.constexpr, TAU_MERGE: tl.constexpr, TAU_POP: tl.constexpr,
    TIE: tl.constexpr, SCALE: tl.constexpr, COLLECT: tl.constexpr,
    GATED: tl.constexpr,
    STAGES: tl.constexpr,
):
    """THE ONE NEW KERNEL of the composition: K11B's persistent whole-window
    walk with the read block replaced by K13's additive block.  One program
    per (batch*head) row; the value tile ``acc`` and the key tile ``kacc``
    are register-resident for the whole window.  Per token: the defined-order
    value and key norms, the KEY-channel read scores, the ADDITIVE read
    weights (the raw score, live-gated; the softmax's exp/max/denominator are
    GONE), the read (value side, the K6 floor channel), the KEY-channel
    decision similarity, the five-op selection, the MERGE blend, then the
    PER-CHANNEL DECAY (values only), the gated lane 0 and the key lane."""
    row = tl.program_id(0)
    batch_index = row // H
    head_index = row % H

    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = tl.arange(0, VP)
    keep_v = values < V
    kdims = tl.arange(0, KP)
    keep_k = kdims < KD
    keep = keep_r[:, None] & keep_v[None, :]
    keep_rk = keep_r[:, None] & keep_k[None, :]

    acc = tl.load(state_in + row.to(tl.int64) * R * V
                  + ranks[:, None] * V + values[None, :], mask=keep, other=0.0)
    kacc = tl.load(keys_in + row.to(tl.int64) * R * KD
                   + ranks[:, None] * KD + kdims[None, :], mask=keep_rk,
                   other=0.0)
    live = tl.load(live_in + row * R + ranks, mask=keep_r, other=0) != 0

    payload_row = payload + batch_index * pbs + head_index * phs
    wk_row = write_keys + batch_index * wbs + head_index * whs
    qk_row = query_keys + batch_index * qbs + head_index * qhs
    decay_row = decay + batch_index * cbs + head_index * chs
    gates_row = gates + batch_index * gbs + head_index * ghs
    reads_base = reads + batch_index.to(tl.int64) * T * H * V + head_index * V
    live_pre_base = (live_pre + batch_index.to(tl.int64) * T * H * R
                     + head_index * R)
    slot_base = slot_out + batch_index.to(tl.int64) * T * H * L + head_index * L
    op_base = op_out + batch_index.to(tl.int64) * T * H * L + head_index * L
    erase_base = (erase_out + batch_index.to(tl.int64) * T * H * L
                  + head_index * L)
    live_after_base = (live_after_out + batch_index.to(tl.int64) * T * H * L
                       + head_index * L)
    payload_base = (payload_out + batch_index.to(tl.int64) * T * H * L * V
                    + head_index * L * V)
    weights_base = (weights_out + batch_index.to(tl.int64) * T * H * R
                    + head_index * R)

    for t in tl.range(0, T, num_stages=STAGES):
        # ---- the state and key table as this token's read and decisions see
        # them (pre-DECAY, pre-lanes).
        tl.store(live_pre_base + t * H * R + ranks, live.to(tl.int8),
                 mask=keep_r)
        if t % SEG == 0:
            seg = t // SEG
            ck = (seg * ROWS + row).to(tl.int64)
            tl.store(ckpt_state + ck * R * V + ranks[:, None] * V
                     + values[None, :], acc, mask=keep)
            tl.store(ckpt_live + ck * R + ranks, live.to(tl.int8),
                     mask=keep_r)
            tl.store(ckpt_keys + ck * R * KD + ranks[:, None] * KD
                     + kdims[None, :], kacc, mask=keep_rk)
        p = tl.load(payload_row + t * pts + values * pvs, mask=keep_v,
                    other=0.0).to(tl.float32)
        wk = tl.load(wk_row + t * wts + kdims * wks, mask=keep_k,
                     other=0.0).to(tl.float32)
        qk = tl.load(qk_row + t * qts + kdims * qks, mask=keep_k,
                     other=0.0).to(tl.float32)
        # the PER-CHANNEL decay vector (K11B's registered replacement)
        lam = tl.load(decay_row + t * cts + values * cvs, mask=keep_v,
                      other=0.0).to(tl.float32)

        # ---- the value norms (the MERGE blend's registered detached
        # strengths -- K6 verbatim; K13: no longer part of the read weights)
        # and the ADDITIVE read weights on the KEY channel.
        nr = tl.reshape(_sqrt_rn32(_fold_axis2(_mul_rn(acc, acc), VP, LOGV)),
                        (RP,))
        strength = _mul_rn(nr, tl.where(live & keep_r, 1.0, 0.0))
        nk = tl.reshape(_sqrt_rn32(_split_fold_axis2(_mul_rn(kacc, kacc), KP,
                                                     LOGK)), (RP,))
        nqk = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(qk, qk), KP, LOGK)), (1,))
        uqk = _div_rn32(qk, tl.broadcast_to(tl.maximum(nqk, _EPS), (KP,)))
        scores = _div_rn32(
            _mul_rn(tl.reshape(_split_fold_axis2(_mul_rn(uqk[None, :], kacc),
                                                 KP, LOGK), (RP,)), SCALE),
            tl.maximum(nk, _EPS))
        live_r = live & keep_r
        live_any = tl.sum(live_r.to(tl.int32), 0) > 0
        # K13: the additive read -- the raw score, live-gated.  NO softmax,
        # NO strength factor: magnitude rides v_j itself (R13-1/R13-2).  An
        # all-dead store reads EXACTLY zero.
        w = tl.where(live_r, scores, 0.0)
        # the read: the K6 floor channel (a native rank-axis sum -- the read
        # feeds no decision, so no flip can come of it).
        read_o = tl.sum(_mul_rn(w[:, None], acc), 0)
        tl.store(reads_base + t * H * V + values, read_o, mask=keep_v)
        if COLLECT:
            tl.store(weights_base + t * H * R + ranks, w, mask=keep_r)

        # ---- the decisions: the registered selection chain on the KEY
        # channel (K10 verbatim -- P2: no gate value feeds this chain).
        nwk = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(wk, wk), KP, LOGK)), (1,))
        uk = _div_rn32(wk, tl.broadcast_to(tl.maximum(nwk, _EPS), (KP,)))
        sim = _div_rn32(
            tl.reshape(_split_fold_axis2(_mul_rn(uk[None, :], kacc), KP, LOGK),
                       (RP,)),
            tl.maximum(nk, _EPS))
        sim = tl.where(live_r, sim, -2.0)
        penalty = _mul_rn(ranks.to(tl.float32), TIE)
        score = tl.where(keep_r, sim - penalty, -3.0)
        best_top = tl.max(score, 0)
        best = tl.min(tl.where(score == best_top, ranks, _BIG_INDEX), 0)
        best_value = tl.sum(tl.where(ranks == best, sim, 0.0), 0)
        second_score = tl.where(ranks == best, -3.0, score)
        second_top = tl.max(second_score, 0)
        second = tl.min(tl.where(second_score == second_top, ranks,
                                 _BIG_INDEX), 0)
        second_value = tl.sum(tl.where(ranks == second, sim, 0.0), 0)
        worst_score = tl.where(keep_r, score, float('inf'))
        worst_top = tl.min(worst_score, 0)
        worst = tl.min(tl.where(worst_score == worst_top, ranks, _BIG_INDEX),
                       0)
        worst_value = tl.sum(tl.where(ranks == worst, sim, 0.0), 0)

        correct = best_value >= TAU_MATCH
        merge = correct & (second_value >= TAU_MERGE)
        free = keep_r & (~live)
        free_count = tl.sum(free.to(tl.int32), 0)
        need = ~correct
        appendable = need & (free_count > 0)
        free_found = tl.min(tl.where(free, ranks, _BIG_INDEX), 0)
        free_slot = tl.where(free_found < _BIG_INDEX, free_found, 0)
        full = free_count <= 0
        pop = (~correct) & (~appendable) & full & (worst_value <= TAU_POP) \
            & live_any
        op0 = tl.where(correct, tl.where(merge, _MERGE_CODE, _CORRECT_CODE),
                       tl.where(appendable, _APPEND_CODE,
                                tl.where(pop, _POP_CODE, _NONE_CODE)))
        slot0 = tl.where(correct, best,
                         tl.where(appendable, free_slot, worst))
        erase0 = (op0 == _CORRECT_CODE) | (op0 == _MERGE_CODE) \
            | (op0 == _POP_CODE)
        live0 = op0 != _POP_CODE
        # ---- the MERGE blend from the PRE-decay rows (K10 verbatim), and
        # the gated lane 0 (K11B verbatim: the pre-decay per-channel CORRECT
        # form -- at the unity-gate birth the incumbent's arithmetic bit for
        # bit).
        fw = tl.maximum(tl.sum(tl.where(ranks == best, strength, 0.0), 0),
                        _EPS)
        sw = tl.maximum(tl.sum(tl.where(ranks == second, strength, 0.0), 0),
                        _EPS)
        total = fw + sw
        a_mix = _div_rn32(fw, total)
        b_mix = _div_rn32(sw, total)
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        hit = keep_r & (slot0 == ranks)
        if merge:
            coef = tl.where(ranks == best, a_mix,
                            tl.where(ranks == second, b_mix, 0.0))
            blend = tl.sum(_mul_rn(coef[:, None], acc), 0)
            lane0 = blend
        else:
            if GATED:
                gt = tl.load(gates_row + t * gts + values * gvs, mask=keep_v,
                             other=0.0).to(tl.float32)
                if op0 == _CORRECT_CODE:
                    # the delta-rule blend in the pre-decay form, per channel
                    old_pre = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)
                    lane0 = _add_rn(
                        _mul_rn(gt, p),
                        _mul_rn(_sub_rn(1.0, gt), _mul_rn(lam, old_pre)))
                else:
                    # the admission form (gate*value) on the carrying ops --
                    # at a free row the old content is exact zero, so this
                    # is g*p on a fresh row; POP/NONE carry no content
                    lane0 = tl.where(carries, _mul_rn(gt, p), 0.0)
            else:
                lane0 = tl.where(carries, p, 0.0)

        # ---- apply: DECAY (values only, PER CHANNEL), then the value lanes
        # in order (K10 verbatim), then the key lane.
        acc = _mul_rn(acc, tl.broadcast_to(lam[None, :], (RP, VP)))
        old_row = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)

        # ---- record the decision channel (the 2-lane (b, T, h, L) layout).
        tl.store(slot_base + t * H * L + 0, slot0.to(tl.int64))
        tl.store(slot_base + t * H * L + 1, second.to(tl.int64))
        tl.store(op_base + t * H * L + 0, op0.to(tl.int64))
        tl.store(op_base + t * H * L + 1,
                 tl.where(merge, _MERGE_CODE, _NONE_CODE).to(tl.int64))
        tl.store(erase_base + t * H * L + 0, erase0.to(tl.int8))
        tl.store(erase_base + t * H * L + 1, merge.to(tl.int8))
        tl.store(live_after_base + t * H * L + 0, live0.to(tl.int8))
        tl.store(live_after_base + t * H * L + 1,
                 tl.zeros((), dtype=tl.int8))
        tl.store(payload_base + t * H * L * V + 0 * V + values, lane0,
                 mask=keep_v)
        tl.store(payload_base + t * H * L * V + 1 * V + values,
                 tl.zeros((VP,), dtype=tl.float32), mask=keep_v)

        # ---- apply lane 0, then the MERGE-source close (K10 verbatim).
        if erase0:
            new_row = lane0
        else:
            new_row = _add_rn(old_row, lane0)
        acc = tl.where(hit[:, None], new_row[None, :], acc)
        live = tl.where(hit, live0, live)
        hit2 = merge & keep_r & (second == ranks)
        acc = tl.where(hit2[:, None], 0.0, acc)
        live = tl.where(hit2, False, live)
        # ---- the key lane: K10 verbatim (gates never touch it).
        carries_key = op0 != _NONE_CODE
        pop_gate = tl.where(op0 == _POP_CODE, 0.0, 1.0)
        key_lane = _mul_rn(wk, pop_gate)
        hitk = carries_key & keep_r & (slot0 == ranks)
        kacc = tl.where(hitk[:, None], key_lane[None, :], kacc)
        hit2k = merge & keep_r & (second == ranks)
        kacc = tl.where(hit2k[:, None], 0.0, kacc)

    tl.store(state_out + row.to(tl.int64) * R * V + ranks[:, None] * V
             + values[None, :], acc, mask=keep)
    tl.store(live_out + row * R + ranks, live.to(tl.int8), mask=keep_r)
    tl.store(keys_out + row.to(tl.int64) * R * KD + ranks[:, None] * KD
             + kdims[None, :], kacc, mask=keep_rk)


# ------------------------------------------------------------ the forward wrapper


def _check_composed_shapes(payload, write_keys, query_keys, decay, gates,
                           state, live, keys, rank, segment_tokens, gated):
    """Refuse anything the composed path cannot express, before any launch
    (K11B's checks verbatim)."""
    if not (payload.is_cuda and write_keys.is_cuda and query_keys.is_cuda
            and decay.is_cuda):
        raise ValueError("the composed scan is a device kernel; inputs must "
                         "be on CUDA")
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape != (batch, tokens, heads, key_dim):
        raise ValueError(f"keys must be (batch, tokens, heads, key_dim); got "
                         f"{tuple(write_keys.shape)}")
    if decay.shape != payload.shape:
        raise ValueError(f"the per-channel decay must share the payload's "
                         f"shape {tuple(payload.shape)}; got "
                         f"{tuple(decay.shape)}")
    if gated and gates.shape != payload.shape:
        raise ValueError(f"the per-channel gates must share the payload's "
                         f"shape {tuple(payload.shape)}; got "
                         f"{tuple(gates.shape)}")
    if state.shape != (batch, heads, rank, value_dim):
        raise ValueError(f"state {tuple(state.shape)} is not (batch, heads, "
                         f"rank, value_dim) = {(batch, heads, rank, value_dim)}")
    if keys.shape != (batch, heads, rank, key_dim):
        raise ValueError(f"keys {tuple(keys.shape)} is not (batch, heads, "
                         f"rank, key_dim) = {(batch, heads, rank, key_dim)}")
    if live.shape != (batch, heads, rank) or live.dtype != torch.bool:
        raise ValueError("live must be a bool (batch, heads, rank) tensor")
    for name, tensor in (("state", state), ("keys", keys)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be fp32; the algebra is fp32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if tokens < 1 or segment_tokens < 1:
        raise ValueError("tokens and segment_tokens must be positive")


def _k15_forward_launch(payload, write_keys, query_keys, decay, gates,
                        state, live, keys, policy_f32, rank, segment_tokens,
                        gated, collect, warps, stages):
    """The persistent walk: one launch, every output allocated contiguous
    (K11B's launch verbatim, the K15 kernel)."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    lanes = LANES
    reads = torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device)
    out_state = torch.empty_like(state)
    out_live = torch.empty_like(live)
    out_keys = torch.empty_like(keys)
    shape = (batch, tokens, heads, lanes)
    slot = torch.empty(shape, dtype=torch.int64, device=device)
    op = torch.empty(shape, dtype=torch.int64, device=device)
    erase = torch.empty(shape, dtype=torch.bool, device=device)
    live_after = torch.empty(shape, dtype=torch.bool, device=device)
    payload_field = torch.empty((*shape, value_dim), dtype=torch.float32,
                                device=device)
    live_pre = torch.empty((batch, tokens, heads, rank), dtype=torch.bool,
                           device=device)
    nseg = -(-tokens // int(segment_tokens))
    ckpt_state = torch.empty((nseg, batch, heads, rank, value_dim),
                             dtype=torch.float32, device=device)
    ckpt_live = torch.empty((nseg, batch, heads, rank), dtype=torch.bool,
                            device=device)
    ckpt_keys = torch.empty((nseg, batch, heads, rank, key_dim),
                            dtype=torch.float32, device=device)
    weights = (torch.empty((batch, tokens, heads, rank), dtype=torch.float32,
                           device=device) if collect
              else torch.empty((1,), dtype=torch.float32, device=device))
    tau_match, tau_merge, tau_pop, tie, scale = policy_f32
    handle = _k15_seq_fwd_kernel[(batch * heads,)](
        payload, write_keys, query_keys, decay, gates, state,
        live.view(torch.int8), keys,
        reads, out_state, out_live.view(torch.int8), out_keys,
        slot, op, erase.view(torch.int8), live_after.view(torch.int8),
        payload_field, live_pre.view(torch.int8),
        ckpt_state, ckpt_live.view(torch.int8), ckpt_keys, weights,
        tokens, rank, batch * heads, int(segment_tokens), key_dim,
        *payload.stride(), *write_keys.stride(), *query_keys.stride(),
        *decay.stride(), *gates.stride(),
        H=heads, V=value_dim, VP=_pow2(value_dim), KP=_pow2(key_dim),
        RP=_pow2(rank), L=lanes,
        LOGV=_log2_up(_pow2(value_dim)), LOGR=_log2_up(_pow2(rank)),
        LOGK=_log2_up(_pow2(key_dim)),
        TAU_MATCH=tau_match, TAU_MERGE=tau_merge, TAU_POP=tau_pop,
        TIE=tie, SCALE=scale, COLLECT=1 if collect else 0,
        GATED=1 if gated else 0,
        STAGES=stages, num_warps=warps)
    return dict(reads=reads, state=out_state, live=out_live, keys=out_keys,
                slot=slot, op=op, erase=erase, live_after=live_after,
                payload=payload_field, live_pre=live_pre,
                ckpt_state=ckpt_state, ckpt_live=ckpt_live,
                ckpt_keys=ckpt_keys, weights=weights, kernel=handle)


# --------------------------------------------------- the backward body


def _k15_backward_body(payload, write_keys, query_keys, decay, gates,
                       dreads, schedule, live_pre, ckpt_state, ckpt_live,
                       ckpt_keys, rank, segment_tokens, scale, gated):
    """The whole window's adjoint: the COMPOSED segment loop -- the
    per-channel value replay (K11B's, imported), the key replay (K10's,
    imported -- the K13 backward's own construction), the ADDITIVE prep
    (K13's, imported verbatim: ``dscores = dw`` on live rows, the strength
    channel EXACT ZEROS), the PER-CHANNEL value walk (K11B's, imported
    verbatim: the read adjoint's value terms consume the additive weights
    and the strength path dies by the zero channel -- IEEE add of 0.0 is
    exact, the K13 receipt carried over), and the key walk (K10's,
    imported).  Decisions are recorded constants (the standing doctrine)."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    R, SEG = int(rank), int(segment_tokens)
    rows = batch * heads
    # the contiguous (rows, T, V) per-channel decay view the walk indexes
    decay32 = (decay.to(torch.float32).permute(0, 2, 1, 3).contiguous()
               .view(rows, tokens, value_dim))
    dreads_c = dreads.contiguous()
    dpayload = torch.empty_like(payload)
    dwrite_keys = torch.empty_like(write_keys)
    dquery_keys = torch.empty_like(query_keys)
    ddecay_ch = torch.zeros((rows, tokens, value_dim), dtype=torch.float32,
                            device=device)
    dgates_ch = (torch.zeros((rows, tokens, value_dim), dtype=torch.float32,
                             device=device) if gated
                 else torch.zeros((rows, tokens, value_dim),
                                  dtype=torch.float32, device=device))
    RP, VP, KP = _pow2(R), _pow2(value_dim), _pow2(key_dim)
    LOGV, LOGR, LOGK = _log2_up(VP), _log2_up(RP), _log2_up(KP)
    nseg = -(-tokens // SEG)
    carry = torch.zeros(batch, heads, R, value_dim, dtype=torch.float32,
                        device=device)
    carry_keys = torch.zeros(batch, heads, R, key_dim, dtype=torch.float32,
                             device=device)
    nblocks = triton.cdiv(value_dim, BV_BWD)
    for seg in reversed(range(nseg)):
        start = seg * SEG
        stop = min(start + SEG, tokens)
        segt = stop - start
        # 1. the segment's value trajectory (K11B's per-channel replay,
        #    imported verbatim).
        _, _, _, traj = channel_persistent_trajectory(
            ckpt_state[seg], ckpt_live[seg],
            decay32.view(batch, heads, tokens, value_dim)[:, :, start:stop],
            schedule.chunk(start, stop))
        # 2. the segment's key trajectory (K10's key-lane replay, imported
        #    verbatim -- the K13 backward's own construction).
        ktraj = torch.empty((rows, segt, R, key_dim), dtype=torch.float32,
                            device=device)
        _k10_key_traj_kernel[(rows,)](
            ckpt_keys[seg], write_keys, schedule.slot, schedule.op, ktraj,
            segt, tokens, start, R, key_dim,
            *write_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        # 3. prep: the per-token weights and adjoint coefficients, both sides
        #    (the K13 ADDITIVE form, imported verbatim; dstrength is EXACT
        #    ZEROS).
        nr_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        nk_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        w_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dscores_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                                 device=device)
        dstrength_ch = torch.zeros((rows, segt, R), dtype=torch.float32,
                                   device=device)
        z_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dc_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dnk_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                             device=device)
        scal = torch.empty((rows, segt, 2), dtype=torch.float32, device=device)
        grid = (rows, triton.cdiv(segt, BT_BWD))
        _k13_bwd_prep_kernel[grid](
            traj, ktraj, query_keys, dreads_c, live_pre.view(torch.int8),
            nr_ch, nk_ch, w_ch, dscores_ch, dstrength_ch, z_ch, dc_ch,
            dnk_ch, scal,
            segt, tokens, start, R, key_dim,
            *query_keys.stride(), *dreads_c.stride(),
            H=heads, V=value_dim, VP=VP, KP=KP, RP=RP, LOGV=LOGV, LOGR=LOGR,
            LOGK=LOGK, BT=BT_BWD, SCALE=_round_f32(scale), num_warps=8)
        # 4. the value walk (K11B's, imported verbatim: the PER-CHANNEL
        #    decay and gate adjoints plus the read adjoint's value terms;
        #    the strength path dies by the prep's zero dstrength channel).
        dstate_seg = torch.empty((batch, heads, R, value_dim),
                                 dtype=torch.float32, device=device)
        _k11b_bwd_value_walk_kernel[(rows, nblocks)](
            traj, payload, gates, dreads_c, decay32, live_pre.view(torch.int8),
            carry,
            nr_ch, w_ch, dstrength_ch,
            schedule.slot, schedule.op, schedule.erase.view(torch.int8),
            dpayload, ddecay_ch, dgates_ch, dstate_seg,
            segt, tokens, start, R,
            *payload.stride(), *gates.stride(), *dreads_c.stride(),
            *schedule.slot.stride(),
            H=heads, V=value_dim, VP=VP, RP=RP, LOGV=LOGV, BT=BT_BWD,
            BV=BV_BWD, GATED=1 if gated else 0,
            num_warps=4)
        # 5. the key walk (K10's, imported verbatim -- the K13 backward's
        #    own construction).
        dkeys_seg = torch.empty((batch, heads, R, key_dim),
                                dtype=torch.float32, device=device)
        _k10_bwd_key_walk_kernel[(rows,)](
            ktraj, write_keys, query_keys, carry_keys,
            dc_ch, dnk_ch, nk_ch, scal,
            schedule.slot, schedule.op,
            dwrite_keys, dquery_keys, dkeys_seg,
            segt, tokens, start, R, key_dim,
            *write_keys.stride(), *query_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        carry = dstate_seg
        carry_keys = dkeys_seg
    dstate = carry
    dkeys = carry_keys
    ddecay = ddecay_ch.view(batch, heads, tokens, value_dim).permute(0, 2, 1, 3)
    dgates = dgates_ch.view(batch, heads, tokens, value_dim).permute(0, 2, 1, 3)
    return (dpayload, dwrite_keys, dquery_keys, ddecay, dgates, dstate,
            dkeys)


# ------------------------------------------------------------ the op pair


@torch.library.custom_op("x45::k15_scan_fwd", mutates_args=())
def _k15_scan_fwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor, gates: torch.Tensor,
    state: torch.Tensor, live: torch.Tensor, keys: torch.Tensor, rank: int,
    segment_tokens: int, tau_match: float, tau_merge: float, tau_pop: float,
    tie_break: float, scale: float, gated: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """The whole (layer, window) composed scan as ONE opaque op (the
    K4/K6/K10/K11/K11B/K13 hosting pattern)."""
    _check_composed_shapes(payload, write_keys, query_keys, decay, gates,
                           state, live, keys, int(rank), int(segment_tokens),
                           bool(gated))
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _k15_forward_launch(payload, write_keys, query_keys, decay, gates,
                              state, live, keys, pol32, int(rank),
                              int(segment_tokens), bool(gated),
                              collect=False,
                              warps=_seq_warps(int(rank), payload.shape[-1]),
                              stages=3)
    return (out["reads"], out["state"], out["live"], out["keys"], out["slot"],
            out["op"], out["erase"], out["live_after"], out["payload"],
            out["live_pre"], out["ckpt_state"], out["ckpt_live"],
            out["ckpt_keys"])


@_k15_scan_fwd_op.register_fake
def _(payload, write_keys, query_keys, decay, gates, state, live, keys,
      rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale,
      gated):
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    nseg = -(-tokens // int(segment_tokens))
    shape = (batch, tokens, heads, LANES)
    return (torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty_like(state),
            torch.empty_like(live),
            torch.empty_like(keys),
            torch.empty(shape, dtype=torch.int64, device=device),
            torch.empty(shape, dtype=torch.int64, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty((*shape, value_dim), dtype=torch.float32, device=device),
            torch.empty((batch, tokens, heads, rank), dtype=torch.bool,
                        device=device),
            torch.empty((nseg, batch, heads, rank, value_dim),
                        dtype=torch.float32, device=device),
            torch.empty((nseg, batch, heads, rank), dtype=torch.bool,
                        device=device),
            torch.empty((nseg, batch, heads, rank, key_dim),
                        dtype=torch.float32, device=device))


def _k15_scan_setup(ctx, inputs, output):
    (payload, write_keys, query_keys, decay, gates, state, live, keys,
     rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale,
     gated) = inputs
    ctx.save_for_backward(payload, write_keys, query_keys, decay, gates,
                          state, live, keys, *output[4:])
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)
    ctx.gated = bool(gated)


@torch.library.custom_op("x45::k15_scan_bwd", mutates_args=())
def _k15_scan_bwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor, gates: torch.Tensor,
    dreads: torch.Tensor, slot: torch.Tensor, op: torch.Tensor,
    erase: torch.Tensor, live_after: torch.Tensor, payload_field: torch.Tensor,
    live_pre: torch.Tensor, ckpt_state: torch.Tensor, ckpt_live: torch.Tensor,
    ckpt_keys: torch.Tensor, rank: int, segment_tokens: int, scale: float,
    gated: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole window's adjoint as its own opaque op: the COMPOSED segment
    loop (per-channel value replay -> key replay -> the additive prep ->
    both walks) runs inside, decisions as recorded constants (the standing
    doctrine)."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _k15_backward_body(payload, write_keys, query_keys, decay, gates,
                              dreads, schedule, live_pre, ckpt_state,
                              ckpt_live, ckpt_keys, int(rank),
                              int(segment_tokens), float(scale), bool(gated))


@_k15_scan_bwd_op.register_fake
def _(payload, write_keys, query_keys, decay, gates, dreads, slot, op, erase,
      live_after, payload_field, live_pre, ckpt_state, ckpt_live, ckpt_keys,
      rank, segment_tokens, scale, gated):
    device = payload.device
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    return (torch.empty_like(payload),
            torch.empty_like(write_keys),
            torch.empty_like(query_keys),
            torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, key_dim), dtype=torch.float32,
                        device=device))


def _k15_scan_backward(ctx, dreads, dstate_out, dlive, dkeys_out, dslot, dop,
                        derase, dlive_after, dpayload_field, dlive_pre,
                        dckpt_state, dckpt_live, dckpt_keys):
    """The adjoint: dpayload, dwrite_keys, dquery_keys, ddecay (per channel),
    dgates (per channel), dstate, dkeys; the decision channel and the
    liveness carry no gradient by construction."""
    (payload, write_keys, query_keys, decay, gates, state, live, keys,
     slot, op, erase, live_after, payload_field, live_pre, ckpt_state,
     ckpt_live, ckpt_keys) = ctx.saved_tensors
    batch, tokens, heads, value_dim = payload.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=payload.device)
    (dpayload, dwrite_keys, dquery_keys, ddecay, dgates, dstate,
     dkeys) = _k15_scan_bwd_op(payload, write_keys, query_keys, decay, gates,
                               dreads, slot, op, erase, live_after,
                               payload_field, live_pre, ckpt_state,
                               ckpt_live, ckpt_keys, ctx.rank,
                               ctx.segment_tokens, ctx.scale, ctx.gated)
    return (dpayload, dwrite_keys, dquery_keys, ddecay, dgates,
            dstate, None, dkeys, None, None, None, None, None, None, None,
            None)


torch.library.register_autograd("x45::k15_scan_fwd", _k15_scan_backward,
                                setup_context=_k15_scan_setup)


# --------------------------------------------------------------- the wrapper


def composed_scan(payload, write_keys, query_keys, decay, gates, *,
                  policy: RoutingPolicy, gated: bool = True,
                  rank=None, rank_max=None,
                  segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                  initial_state=None, initial_live=None, initial_keys=None,
                  collect: bool = False) -> RoutedResult:
    """One whole (layer, window) composed scan at PER-TOKEN granularity.

    The K11B wrapper's drop-in for the composition: the registered five-op
    policy evaluated against the CURRENT state and key table at every
    token, addressing on the key channel, the ADDITIVE read (K13's form) as
    the value channel, with the per-channel DECAY vector and the
    per-channel write gate as schedule data.  The kernel path is ONE opaque
    op pair per scan; every refusal takes the EAGER ORACLE -- the same
    semantics, the parity fixed point -- never a silent change:

    * ``policy.ops=False`` (the append-only control) and ``policy.grow`` are
      refused outright (chunk-dial concepts); a rank prefix is refused by
      name;
    * a store the persistent walk cannot hold (``separated_available``), or
      non-CUDA inputs, take the oracle verbatim.
    """
    batch, tokens, heads, value_dim = payload.shape
    device = payload.device
    key_dim = write_keys.shape[-1]
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    rank, rank_max = int(rank), int(rank_max)
    if not policy.ops:
        raise ValueError("the append-only control is refused by the "
                          "composition registration; it is a chunk-dial "
                          "control")
    if policy.grow:
        raise ValueError("growth is refused by the composition registration: "
                         "growth is a chunk-boundary concept")
    if rank != rank_max:
        raise ValueError("the composition registration requires rank == "
                         "rank_max (the registered shape); a rank prefix is "
                         "refused by name")
    state = (torch.zeros(batch, heads, rank_max, value_dim,
                         dtype=torch.float32, device=device)
             if initial_state is None else initial_state)
    live = (torch.zeros(batch, heads, rank_max, dtype=torch.bool, device=device)
            if initial_live is None else initial_live)
    keys = (torch.zeros(batch, heads, rank_max, key_dim, dtype=torch.float32,
                        device=device) if initial_keys is None
            else initial_keys)
    if collect:
        pol32 = (_round_f32(policy.tau_match), _round_f32(policy.tau_merge),
                 _round_f32(policy.tau_pop), _round_f32(policy.tie_break),
                 _round_f32(policy.read_scale))
        with torch.no_grad():
            out = _k15_forward_launch(
                payload, write_keys, query_keys, decay, gates, state, live,
                keys, pol32, rank, segment_tokens, gated, collect=True,
                warps=_seq_warps(rank, value_dim), stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        return RoutedResult(out["reads"], out["state"], out["live"], rank,
                            ScanStats(), schedule, out["weights"])
    if (payload.is_cuda and separated_available(rank, value_dim, key_dim)):
        reads, state_out, live_out, _keys, _slot, _op, _erase, _live_after, \
            _payload, _live_pre, _ckpt_state, _ckpt_live, _ckpt_keys = \
            _k15_scan_fwd_op(payload, write_keys, query_keys, decay, gates,
                             state, live, keys, rank,
                             int(segment_tokens), policy.tau_match,
                             policy.tau_merge, policy.tau_pop,
                             policy.tie_break, policy.read_scale,
                             bool(gated))
        return RoutedResult(reads, state_out, live_out, rank, ScanStats(),
                            None, None)
    # the eager oracle: the same semantics, the parity fixed point
    ref = composed_oracle_scan(payload, write_keys, query_keys, decay, gates,
                               policy=policy, gated=gated, rank=rank,
                               rank_max=rank_max,
                               initial_state=state, initial_live=live,
                               initial_keys=keys, collect=False)
    return RoutedResult(ref["reads"], ref["state"], ref["live"], rank,
                        ScanStats(), None, None)


# ------------------------------------------------------- the mixer and the model


class ComposedReadMixer(ChannelGatedSeparatedMixer):
    """The K15 mixer: the K11B p3 mixer (the same projections, the same
    per-channel gate channels, the same committed ``channel_decay`` birth)
    with the scan swapped to the COMPOSED one -- the additive read under the
    per-channel decay and write gates, the per-channel read gate applied to
    the reads before the output projection, OUTSIDE the scan (the
    K11/K11B/K13 read-out convention, instantiated once).  At birth the
    mixer IS the K13 additive machine bit for bit: ``lam_vec == lam_inc``
    per channel, ``g_vec == 1`` per channel and ``r_vec == 1`` per channel
    exactly (the committed k11b birth, P5), and the additive read is K13's
    own committed form."""

    def __init__(self, width: int, heads: int, value_dim: int, rank: int,
                 rank_max: int, key_dim: int, policy: RoutingPolicy,
                 ctx: int = DEFAULT_CONTEXT,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
        super().__init__(width, heads, value_dim, rank, rank_max, key_dim,
                        policy, arm="p3", ctx=ctx,
                        segment_tokens=segment_tokens)

    def mix(self, x: torch.Tensor):
        batch, tokens, _ = x.shape
        context = causal_context(x, self.ctx)          # (b, T, ctx, width)
        context = context.reshape(batch, tokens, self.ctx * self.width)
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
        write_keys = self.key(context).view(batch, tokens, self.heads,
                                            self.key_dim)
        query_keys = self.query_key(context).view(batch, tokens, self.heads,
                                                  self.key_dim)
        lam_vec, _lam_inc = self.channel_decay(x)
        gated = self.channels["write"]
        if gated:
            gates = 1.0 - self.write_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(self.b_proj(x)).view(
                    batch, tokens, self.heads, self.value_dim)
        else:
            # a 4-d dummy so the kernel's stride arguments keep their arity
            # (the tensor is never loaded: GATED=0)
            gates = payload.new_ones((1, 1, 1, 1))
        result = composed_scan(payload, write_keys, query_keys, lam_vec,
                               gates, policy=self.policy, gated=gated,
                               rank=self.rank, rank_max=self.rank_max,
                               segment_tokens=self.segment_tokens)
        reads = result.reads
        if self.channels["read"]:
            r_vec = 1.0 + self.read_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(self.g_proj(x)).view(
                    batch, tokens, self.heads, self.value_dim)
            reads = reads * r_vec
        return (self.out(reads.reshape(batch, tokens,
                                       self.heads * self.value_dim)),
                result)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mix(x)
        return out


def convert_model_to_composition(model,
                                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
    """Swap every ``SeparatedSequentialMixer`` for the composed one, sharing
    parameters (the K10/K11/K11B/K13 conversion pattern: the ``nn.Linear``
    modules the optimizer already saw are the very same objects), adding
    the p3 gate channels at the mimetic-neutral birth.  The key/query-key
    projections and every value-side module are carried over UNTOUCHED, so
    at birth the converted model IS the K13 additive machine bit for bit
    (the gates at the registered scalar equivalents; the additive read is
    K13's own committed form) and the discrete schedule is the banked
    winner's bit for bit at fixed inputs.  ``k10_separated_keys.py``,
    ``k11b_channel_gates.py`` and ``k13_additive_read.py`` are not edited;
    this builds on the K10 conversion's output.  The parameter DELTA is the
    registration's (p3's own 8,657,040 at depth 24 -- the read channel is
    the SAME construction the K13 machine already carries, instantiated
    once; stated, asserted by the probe, not hidden)."""
    channels = arm_channels("p3")
    for block in model.blocks:
        mixer = block.mixer
        if not isinstance(mixer, SeparatedSequentialMixer):
            raise ValueError("convert_model_to_composition expects a "
                             "K10-converted model (run "
                             "convert_model_to_separated_keys first); got "
                             f"{type(mixer).__name__}")
        sep = ComposedReadMixer.__new__(ComposedReadMixer)
        torch.nn.Module.__init__(sep)
        sep.__dict__.update(mixer.__dict__)
        sep._modules = dict(mixer._modules)     # own copy; the old mixer keeps its
        sep.arm = "p3"
        sep.channels = channels
        sep.segment_tokens = int(segment_tokens)
        sep.width = int(mixer.value.weight.shape[1])
        value_dim = int(mixer.value_dim)
        heads = int(mixer.heads)
        # the per-channel DECAY channel (the committed k11b construction)
        sep.f_proj = nn.Sequential(
            nn.Linear(sep.width, value_dim, bias=False),
            nn.Linear(value_dim, heads * value_dim, bias=False))
        sep.decay_scalar = nn.Parameter(torch.zeros(heads))
        # the per-channel write admission channel
        sep.b_proj = nn.Linear(sep.width, heads * value_dim, bias=False)
        sep.write_scalar = nn.Parameter(torch.zeros(heads))
        # the per-channel read gate channel (K13's own committed g_proj
        # construction -- instantiated ONCE, never double-added)
        sep.g_proj = nn.Sequential(
            nn.Linear(sep.width, value_dim, bias=False),
            nn.Linear(value_dim, heads * value_dim, bias=True))
        sep.read_scalar = nn.Parameter(torch.zeros(heads))
        sep.apply_gate_birth()
        block.mixer = sep
    return model


__all__ = ["ComposedReadMixer", "convert_model_to_composition",
           "composed_parameter_delta", "composed_scan",
           "composed_oracle_scan", "_k15_forward_launch",
           "_k15_backward_body", "separated_available", "span_key_std",
           "mimetic_key_family", "DEFAULT_KEY_DIM", "DEFAULT_CONTEXT",
           "DEFAULT_SEGMENT_TOKENS", "MIMETIC_SEED", "MAX_KEY_TILE"]
