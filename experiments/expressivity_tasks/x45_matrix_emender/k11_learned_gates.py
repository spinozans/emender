"""K11 -- LEARNED GATES on the emender: input-dependent write-strength,
read-out and dense-write gates as SCHEDULE DATA inside the certified algebra.

Why this file exists
--------------------
The side-by-side dissection (``RESULT_SIDE_BY_SIDE_0239.md`` 5) receipted the
winner's residual as C5, the store's write/read FORM: byte-identity CORRECT on
0.979/token, reads over ~5.6 effective rows of ~27 live, 17/24 stores
near-decorative at eval.  GDN2's answer is an admission controller -- ~18% of
its parameters are gate matrices (erase/write/decay/output) that take a dense,
channel-gated rank-1 delta at EVERY token from step 1.  Our machine's write
strengths are fixed constants of the arithmetic; its only gates are the 24 DECAY
projections (27,648 parameters, 0.018%).

The enabling fact (``PREREGISTRATION_LEARNED_GATES.md`` 0): gates are SCHEDULE
DATA.  K1's injection list already carries a gate scalar per write; the Lean
algebra's APPEND writes ``(key, gate*value)`` and MERGE is
``old + beta*(new-old)`` with all gate magnitudes part of the fixed
``TokenDecision``; the affine-in-values scan theorems hold for ANY supplied
gate values.  Making the magnitudes input-dependent adds ZERO new math.

The semantics (the registration's 1, pre-stated; the supervisor's P1-P5
rulings cited inline)
--------------------------------------------------------------------------------
* **Arm G1 -- the write-strength gate** (P1(a): a per-token per-head scalar
  projection of the mixer's input; P4's simplest form).  Consumed at the two
  content-carrying ops through the existing write/injection channel:
  APPEND admits ``g*p`` (the Lean APPEND's ``gate*value`` verbatim -- a free
  row's content is exact zero), CORRECT blends
  ``v + g*(p - v)`` (the Lean MERGE arithmetic at the CORRECT write enable --
  the delta rule on the dominant op).  MERGE/POP/NONE stay VERBATIM.  Plus the
  read-out gate (P4: it costs four mixer lines outside the scan):
  ``reads' = reads * g_out`` before the output projection.
* **Arm G2 -- the dense-write channel** (the direct GDN2 move): every token
  deposits ``m_d * sigmoid(w_d*x + b_d) * p`` on its matched row, as schedule
  LANE 2.  **P3, verbatim: the dense contribution goes to the BEST-MATCH LIVE
  ROW (argmax over live, regardless of whether it clears tau_match); if NO live
  row exists, the dense contribution goes nowhere (zero).  Dense writes touch
  live rows only; structure creation stays the ops' job.**  The row must also
  be live AFTER lanes 0/1 (a POP can kill the best row in the same token) --
  the edge-case completion that keeps the dead-rows-are-exactly-zero invariant.
  A zero deposit is a NO-OP (lane recorded NONE).
* **P2 -- decisions are inviolable**: no gate value feeds op selection.  The
  decision chain is the K10 verbatim on the key channel; the gates scale
  payload magnitudes and read outputs ONLY.  The discrete op sequence is
  bit-invariant to every gate value and every deposit at fixed inputs (the
  0-flip receipt).
* **P5 -- mimetic-neutral birth**: every gate channel carries a per-head
  zero-born scalar (``g_write = 1 - m_w*sigmoid(...)``,
  ``g_out = 1 + m_o*sigmoid(...)``, ``dense = m_d*sigmoid(...)*p`` with
  ``m_*`` born at EXACT zero), so at birth ``g_write == 1``, ``g_out == 1``
  and ``dense == 0`` exactly: the arm is born as the banked winner and the
  gates grow by training.

The arithmetic spec is K10's verbatim (defined-order tree folds, fp32,
correctly-rounded IEEE ops, libdevice exp, ``mul.rn``/``add.rn`` -- plus K1's
``sub.rn`` for the blend), extended by the two gate scalars and the third
lane.  The recorded lane-0 payload is the BLENDED lane value and the deposit
is an ordinary scheduled additive write, so K4's
``persistent_scan_trajectory`` (imported verbatim, lane-generic) replays the
whole 3-lane schedule bit-exactly: the gates never appear in the replay.

The kernels (K10's walk, extended; ``k10_*``/``k6_*``/``k1_*``/``k4_*``
sources are imported, never edited)
--------------------------------------------------------------------------------
``_k11_seq_fwd_kernel``      K10's persistent whole-window walk with the
                             gated lane-0 arithmetic (``GATED`` constexpr: 0 =
                             the incumbent's lane verbatim, for the pure-G2
                             arm), the dense lane (``DENSE`` constexpr), and
                             the 3-lane recorded schedule.
``_k11_key_traj_kernel``     K10's key-lane replay with the channel strides
                             passed (K10's hardcodes the 2-lane layout).
``_k10_bwd_prep_kernel``    IMPORTED VERBATIM (read weights recomputed from
                             the trajectory -- gate-agnostic).
``_k11_bwd_value_walk_kernel`` K10's reverse value walk plus the gate adjoint
                             (``dpayload = g*dpay0``, the per-block
                             ``dgates = <dpay0, p - v>`` partials, the
                             ``(1-g)`` old-row path at CORRECT, the dense
                             lane's ``ddense = dS[best]``) and L-generic
                             channel strides.
``_k11_bwd_key_walk_kernel`` K10's key-table adjoint walk, stride-generalized.

Doctrine, unchanged: decisions are recorded constants in the backward.  The
read weights, payload lanes, DECAY gate, gates and the KEY CHANNEL carry
gradient to their inputs; the cross-axis reductions are autograd-ordered on
the oracle side, so the gradient gates are floor-discipline (fp64 gradcheck of
the oracle first, fp32-vs-fp64 floors, ``NOISE_FACTOR`` x), stated, not hidden.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .algebra import (APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule)
from .k1_schedule_scan import NONE_CODE, _add_rn, _mul_rn, _sub_rn
from .k4_fused_backward import persistent_scan_trajectory
from .k6_sequential_matrix import (_div_rn32, _fold_axis2, _fold_vec, _fp32,
                                   _log2_up, _norm_defined, _pow2, _round_f32,
                                   _seq_warps, _sqrt_rn32, tree_fold)
from .k10_separated_keys import (_key_selection, _k10_bwd_prep_kernel,
                                _split_fold_axis2, DEFAULT_CONTEXT,
                                DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS,
                                MIMETIC_SEED, MAX_KEY_TILE,
                                separated_available)
from .key_separation_probe import causal_context
from .model import DECAY_FLOOR, DECAY_HORIZON_TOKENS
from .scan import RoutedResult, ScanStats
from .schedule import EPS, RoutingPolicy

# The recorded schedule's lane count for every k11 scan: lanes 0/1 are K10's
# (the main write and the MERGE-source close), lane 2 is the dense deposit.
LANES = 3
# The backward's walk blocking (K10's own constants).
BT_BWD = 16
BV_BWD = 16
# The arms (the registration's 1): which gate channels each arm carries.
ARMS = ("g1", "g2", "g3")
ARM_CHANNELS = {
    "g1": dict(write=True, readout=True, dense=False),
    "g2": dict(write=False, readout=False, dense=True),
    "g3": dict(write=True, readout=True, dense=True),
}


def arm_channels(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unregistered arm: {arm!r} (expected one of {ARMS})")
    return dict(ARM_CHANNELS[arm])


# ------------------------------------------------------------- the oracle


def gated_oracle_scan(payload, dense, write_keys, query_keys, decay, gates, *,
                       policy: RoutingPolicy, gated: bool = True,
                       use_dense: bool = True, rank=None, rank_max=None,
                       initial_state=None, initial_live=None, initial_keys=None,
                       collect=True):
    """The learned-gate arm's eager per-token oracle: the parity fixed point,
    differentiable through the key channel, the gates and the dense lane.

    K10's oracle walk (decisions from the DETACHED key channel, read weights
    from the ATTACHED state and key table, the MERGE blend from the PRE-decay
    rows, DECAY then lanes in the registered order) extended with: the gated
    lane-0 arithmetic (APPEND admits ``g*p``, CORRECT blends
    ``v + g*(p-v)`` against the POST-decay row), and the dense lane (P3's
    best-match live row, a no-op at zero).  The recorded schedule carries
    THREE lanes; the lane-0 payload field is the blended lane value, so K4's
    replay consumes it verbatim.
    """
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape[:-1] != payload.shape[:-1]:
        raise ValueError("keys must share the payload's (batch, tokens, heads)")
    if use_dense and dense.shape != payload.shape:
        raise ValueError("dense must share the payload's shape")
    if gated and gates.shape != (batch, tokens, heads):
        raise ValueError(f"gates must be {(batch, tokens, heads)}")
    if not policy.ops:
        raise ValueError("the append-only control is refused by the learned-"
                          "gate registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the learned-gate registration: "
                          "it is a chunk-boundary concept")
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    if int(rank) != int(rank_max):
        raise ValueError("the learned-gate registration requires rank == "
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
    dn32 = _fp32(dense)
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
        dn_t = dn32[:, t] if use_dense else None
        lam_t = d32[:, t]                                 # (b, h)
        # ---- the read: certified weights on the KEY channel (K10 verbatim).
        nr = _norm_defined(state, -1, VP)
        nk = _norm_defined(keys, -1, KP)
        nqk = _norm_defined(qk_t, -1, KP)
        uqk = qk_t / nqk.clamp_min(EPS).unsqueeze(-1)
        scores = (scale * tree_fold(uqk.unsqueeze(2) * keys, -1, KP)) \
            / nk.clamp_min(EPS)
        live_f = live.to(compute_dtype)
        any_live = live.any(-1, keepdim=True)
        masked = scores.masked_fill(~live, float("-inf"))
        safe = torch.where(any_live, masked, torch.zeros_like(masked))
        pmax = safe.max(-1).values
        expo = torch.exp(safe - pmax.unsqueeze(-1))
        denom = tree_fold(expo * live_f, -1, RP)
        any_live_flat = live.any(-1)
        denom_safe = torch.where(any_live_flat, denom, torch.ones_like(denom))
        probs = torch.where(any_live, expo / denom_safe.unsqueeze(-1),
                            torch.zeros_like(expo))
        w = probs * nr * live_f
        reads[:, t] = tree_fold(w.unsqueeze(-1) * state, -2, RP)
        if collect:
            weights_out[:, t] = w
        # ---- the decisions: the registered chain on the DETACHED key channel
        # (P2: no gate value feeds op selection).
        keys_det = keys.detach()
        nkd = _norm_defined(keys_det, -1, KP)
        wk_det = wk_t.detach()
        uk = wk_det / _norm_defined(wk_det, -1, KP).clamp_min(EPS).unsqueeze(-1)
        sim = tree_fold(uk.unsqueeze(2) * keys_det, -1, KP) \
            / nkd.clamp_min(EPS)
        sim = sim.masked_fill(~live, -2.0)
        sel = _key_selection(sim, live, tau_match, tau_merge, tau_pop, tie,
                             rank, RP)
        any_live_pre = live.any(-1)                       # (b, h): P3's guard
        # ---- the MERGE blend from the PRE-decay rows (K10 verbatim).
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
        # ---- DECAY (values only), then the lane-0 value (K10's registered
        # order).  The CORRECT blend is the PRE-decay form
        # g*p + (1-g)*lam*v_pre (== v_dec + g*(p - v_dec) in exact
        # arithmetic) -- the identical expression the kernel implements, so
        # the parity bar is torch.equal; at g == 1 it degenerates to the
        # incumbent's overwrite bit for bit.
        rows_pre = state[b_index.expand(batch, heads),
                         h_index.expand(batch, heads), sel["slot0"]]
        state = state * lam_t.unsqueeze(-1).unsqueeze(-1)
        rows_at = state[b_index.expand(batch, heads),
                        h_index.expand(batch, heads), sel["slot0"]]
        if gated:
            gate_t = g32[:, t].unsqueeze(-1)               # (b, h, 1)
            blend_correct = gate_t * p_t + (1.0 - gate_t) * (
                lam_t.unsqueeze(-1) * rows_pre)
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
        # ---- record lanes 0/1 (K10 verbatim) and apply.
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
            (b_index.expand(batch, heads), h_index.expand(batch, heads),
             sel["slot0"]),
            torch.where(sel["op0"].ne(NONE).unsqueeze(-1), key_lane, keys_at))
        live = live.index_put((b_index.expand(batch, heads),
                               h_index.expand(batch, heads), sel["slot0"]),
                              torch.where(sel["op0"].ne(NONE), sel["live0"],
                                          live.gather(-1,
                                                      sel["slot0"].unsqueeze(-1)
                                                      ).squeeze(-1)))
        if bool(sel["merge"].any()):
            state = state.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.zeros(batch, heads, value_dim, dtype=compute_dtype,
                            device=device))
            keys = keys.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.zeros(batch, heads, key_dim, dtype=compute_dtype,
                            device=device))
            live = live.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.zeros(batch, heads, dtype=torch.bool,
                            device=live.device))
        # ---- lane 2: the dense deposit (P3 verbatim).  The target is the
        # BEST-MATCH LIVE ROW (argmax over live at decision time); if no live
        # row exists at decision time the deposit goes nowhere; the row must
        # also still be live after lanes 0/1 (a POP can kill it in-token).
        # The deposit's VALUE is the gate's output (born at exact zero -- the
        # identity by IEEE arithmetic, x + -+0 = x, the -0.0 corner receipted
        # by the tests), so the lane fires on the live guards alone and the
        # gradient reaches the zero-born scalar at step 0 (a value-dependent
        # no-op would make the recorded op code a function of the deposit
        # and cut the birth gradient -- the dead-arm failure the
        # registration refuses).
        fire = torch.zeros(batch, heads, dtype=torch.bool, device=device)
        if use_dense:
            live_best = live.gather(-1, sel["best_index"].unsqueeze(-1)
                                    ).squeeze(-1)
            fire = any_live_pre & live_best
        fields["slot"][:, t, :, 2] = sel["best_index"]
        fields["op"][:, t, :, 2] = torch.where(
            fire, torch.full_like(sel["op0"], APPEND),
            torch.full_like(sel["op0"], NONE))
        fields["erase"][:, t, :, 2] = False
        fields["live_after"][:, t, :, 2] = True
        if use_dense:
            fields["payload"][:, t, :, 2, :] = torch.where(
                fire.unsqueeze(-1), dn_t, torch.zeros_like(dn_t))
        if bool(fire.any()):
            rows_best = state[b_index.expand(batch, heads),
                              h_index.expand(batch, heads),
                              sel["best_index"]]
            state = state.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["best_index"]),
                torch.where(fire.unsqueeze(-1), rows_best + dn_t, rows_best))
    schedule = MatrixSchedule(**fields)
    return dict(reads=reads, state=state, live=live, keys=keys,
                schedule=schedule, weights=weights_out, live_pre=live_pre_out,
                rank=int(rank))


# ------------------------------------------------------------- the forward kernel

_EPS = tl.constexpr(EPS)
_BIG_INDEX = tl.constexpr(1 << 30)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))


@triton.jit(do_not_specialize=['T', 'SEG'])
def _k11_seq_fwd_kernel(
    payload, dense, write_keys, query_keys, decay, gates, state_in, live_in,
    keys_in,
    reads, state_out, live_out, keys_out,
    slot_out, op_out, erase_out, live_after_out, payload_out, live_pre,
    ckpt_state, ckpt_live, ckpt_keys, weights_out,
    T, R, ROWS, SEG, KD,
    pbs, pts, phs, pvs,
    dbs, dts, dhs, dvs,
    wbs, wts, whs, wks,
    qbs, qts, qhs, qks,
    cbs, cts, chs,
    gbs, gts, ghs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, KP: tl.constexpr,
    RP: tl.constexpr, L: tl.constexpr,
    LOGV: tl.constexpr, LOGR: tl.constexpr, LOGK: tl.constexpr,
    TAU_MATCH: tl.constexpr, TAU_MERGE: tl.constexpr, TAU_POP: tl.constexpr,
    TIE: tl.constexpr, SCALE: tl.constexpr, COLLECT: tl.constexpr,
    GATED: tl.constexpr, DENSE: tl.constexpr,
    STAGES: tl.constexpr,
):
    """One program per (batch*head) row: K10's whole-window walk with the
    gated lane-0 arithmetic, the dense lane, and the 3-lane recorded schedule.
    The value tile ``acc`` and the key tile ``kacc`` are register-resident for
    the whole window; per token the K10-verbatim read/decision/blend chain,
    then DECAY, the gated lane 0, the MERGE-source close, and the dense
    deposit at the best-match live row (P3)."""
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
    dense_row = dense + batch_index * dbs + head_index * dhs
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
        lam = tl.load(decay_row + t * cts).to(tl.float32)

        # ---- the value norms and the read weights on the KEY channel (K10).
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
        masked = tl.where(live_r, scores, float('-inf'))
        safe = tl.where(live_any, masked, 0.0)
        pmax = tl.max(safe, 0)
        expo = libdevice.exp(safe - pmax)
        denom = tl.reshape(_fold_vec(tl.where(live_r, expo, 0.0), RP, LOGR),
                           (1,))
        probs = tl.where(live_any,
                         _div_rn32(expo, tl.broadcast_to(denom, (RP,))), 0.0)
        w = _mul_rn(probs, strength)
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
        # the gated lane 0.  The CORRECT blend is implemented in the
        # PRE-decay form  g*p + (1-g)*lam*v_pre  -- the same value as
        # v_dec + g*(p - v_dec) in exact arithmetic (v_dec = lam*v_pre), so
        # the register walk keeps K10's registered lane order (the blend
        # chain before DECAY, the lanes after) AND the unity-gate birth is
        # the incumbent's overwrite bit for bit (g=1 gives add_rn(p, +-0)).
        # The oracle implements the identical expression, so the parity bar
        # stays torch.equal.
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
                gt = tl.load(gates_row + t * gts).to(tl.float32)
                if op0 == _CORRECT_CODE:
                    # the delta-rule blend in the pre-decay form
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

        # ---- apply: DECAY (values only), then the value lanes in order (K10
        # verbatim), then the key lane.
        acc = _mul_rn(acc, lam)
        old_row = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)

        # ---- record the decision channel (the 3-lane (b, T, h, L) layout).
        tl.store(slot_base + t * H * L + 0, slot0.to(tl.int64))
        tl.store(slot_base + t * H * L + 1, second.to(tl.int64))
        tl.store(slot_base + t * H * L + 2, best.to(tl.int64))
        tl.store(op_base + t * H * L + 0, op0.to(tl.int64))
        tl.store(op_base + t * H * L + 1,
                 tl.where(merge, _MERGE_CODE, _NONE_CODE).to(tl.int64))
        tl.store(erase_base + t * H * L + 0, erase0.to(tl.int8))
        tl.store(erase_base + t * H * L + 1, merge.to(tl.int8))
        tl.store(erase_base + t * H * L + 2,
                 tl.zeros((), dtype=tl.int8))
        tl.store(live_after_base + t * H * L + 0, live0.to(tl.int8))
        tl.store(live_after_base + t * H * L + 1,
                 tl.zeros((), dtype=tl.int8))
        tl.store(live_after_base + t * H * L + 2,
                 tl.full((), 1, tl.int8))
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
        # ---- lane 2: the dense deposit (P3 verbatim).  Target the BEST-MATCH
        # LIVE ROW (argmax over live at decision time, `best`); zero if no
        # live row existed at decision time; the row must still be live after
        # lanes 0/1 (a POP can kill it in-token).  The deposit's VALUE is the
        # gate's output -- born at exact zero, which is the identity by IEEE
        # arithmetic (x + ±0 = x, the -0.0 corner receipted by the tests) --
        # so the lane fires on the live guards alone and the gradient reaches
        # the zero-born scalar at step 0 (a value-dependent no-op would make
        # the recorded op code a function of the deposit and cut the birth
        # gradient -- the dead-arm failure the registration refuses).
        if DENSE:
            dn = tl.load(dense_row + t * dts + values * dvs, mask=keep_v,
                         other=0.0).to(tl.float32)
            live_best = tl.sum(tl.where(keep_r & (best == ranks),
                                        live.to(tl.float32), 0.0), 0) > 0.0
            fire = live_any & live_best
            hitd = fire & keep_r & (best == ranks)
            # the application is an UNCONDITIONAL select (no runtime branch
            # reassigning the loop-carried tile: the pipeliner miscompiles
            # that -- receipted by the replay-parity test)
            old2 = tl.sum(tl.where(hitd[:, None], acc, 0.0), 0)
            acc = tl.where(hitd[:, None],
                           _add_rn(old2, dn)[None, :], acc)
            tl.store(op_base + t * H * L + 2,
                     tl.where(fire, _APPEND_CODE, _NONE_CODE).to(tl.int64))
            tl.store(payload_base + t * H * L * V + 2 * V + values,
                     tl.where(fire, dn, tl.zeros((VP,), dtype=tl.float32)),
                     mask=keep_v)
        else:
            tl.store(op_base + t * H * L + 2,
                     tl.full((), _NONE_CODE, tl.int64))
            tl.store(payload_base + t * H * L * V + 2 * V + values,
                     tl.zeros((VP,), dtype=tl.float32), mask=keep_v)
        # ---- the key lane: K10 verbatim (gates and deposits never touch it).
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


def _check_gated_shapes(payload, dense, write_keys, query_keys, decay, gates,
                        state, live, keys, rank, segment_tokens, gated,
                        use_dense):
    """Refuse anything the gated path cannot express, before any launch."""
    if not (payload.is_cuda and write_keys.is_cuda and query_keys.is_cuda
            and decay.is_cuda):
        raise ValueError("the gated scan is a device kernel; inputs must be "
                         "on CUDA")
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape != (batch, tokens, heads, key_dim):
        raise ValueError(f"keys must be (batch, tokens, heads, key_dim); got "
                         f"{tuple(write_keys.shape)}")
    if decay.shape != (batch, tokens, heads):
        raise ValueError(f"decay must be {(batch, tokens, heads)}, got "
                         f"{tuple(decay.shape)}")
    if gated and gates.shape != (batch, tokens, heads):
        raise ValueError(f"gates must be {(batch, tokens, heads)}, got "
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


def _k11_forward_launch(payload, dense, write_keys, query_keys, decay, gates,
                        state, live, keys, policy_f32, rank, segment_tokens,
                        gated, use_dense, collect, warps, stages):
    """The persistent walk: one launch, every output allocated contiguous."""
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
    handle = _k11_seq_fwd_kernel[(batch * heads,)](
        payload, dense, write_keys, query_keys, decay, gates, state,
        live.view(torch.int8), keys,
        reads, out_state, out_live.view(torch.int8), out_keys,
        slot, op, erase.view(torch.int8), live_after.view(torch.int8),
        payload_field, live_pre.view(torch.int8),
        ckpt_state, ckpt_live.view(torch.int8), ckpt_keys, weights,
        tokens, rank, batch * heads, int(segment_tokens), key_dim,
        *payload.stride(), *dense.stride(), *write_keys.stride(),
        *query_keys.stride(), *decay.stride(), *gates.stride(),
        H=heads, V=value_dim, VP=_pow2(value_dim), KP=_pow2(key_dim),
        RP=_pow2(rank), L=lanes,
        LOGV=_log2_up(_pow2(value_dim)), LOGR=_log2_up(_pow2(rank)),
        LOGK=_log2_up(_pow2(key_dim)),
        TAU_MATCH=tau_match, TAU_MERGE=tau_merge, TAU_POP=tau_pop,
        TIE=tie, SCALE=scale, COLLECT=1 if collect else 0,
        GATED=1 if gated else 0, DENSE=1 if use_dense else 0,
        STAGES=stages, num_warps=warps)
    return dict(reads=reads, state=out_state, live=out_live, keys=out_keys,
                slot=slot, op=op, erase=erase, live_after=live_after,
                payload=payload_field, live_pre=live_pre,
                ckpt_state=ckpt_state, ckpt_live=ckpt_live,
                ckpt_keys=ckpt_keys, weights=weights, kernel=handle)


# --------------------------------------------------------- the backward kernels


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k11_key_traj_kernel(
    keys_in, write_keys, slot, op, traj_out,
    SEGT, T, T0, R, KD,
    sbs, sts, shs, sls,
    wbs, wts, whs, wks,
    H: tl.constexpr, KP: tl.constexpr, RP: tl.constexpr,
):
    """The key table's schedule replay (K10's kernel with the channel strides
    passed -- K10's hardcodes the 2-lane layout): the key tile walked forward
    over the segment applying the key lane (copies only), storing the PRE-lane
    table at every token.  The key lane lives on lanes 0/1 only."""
    row = tl.program_id(0)
    batch_index = row // H
    head_index = row % H
    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    kdims = tl.arange(0, KP)
    keep_k = kdims < KD
    keep_rk = keep_r[:, None] & keep_k[None, :]
    kacc = tl.load(keys_in + row.to(tl.int64) * R * KD + ranks[:, None] * KD
                   + kdims[None, :], mask=keep_rk, other=0.0)
    wk_row = write_keys + batch_index * wbs + head_index * whs
    channel_base = batch_index.to(tl.int64) * sbs + head_index * shs
    traj_base = traj_out + row.to(tl.int64) * SEGT * R * KD
    for tt in tl.range(0, SEGT):
        tl.store(traj_base + tt * R * KD + ranks[:, None] * KD
                 + kdims[None, :], kacc, mask=keep_rk)
        t = T0 + tt
        op0 = tl.load(op + channel_base + t * sts + 0 * sls)
        slot0 = tl.load(slot + channel_base + t * sts + 0 * sls).to(tl.int32)
        second = tl.load(slot + channel_base + t * sts + 1 * sls).to(tl.int32)
        merge = tl.load(op + channel_base + t * sts + 1 * sls) != _NONE_CODE
        wk = tl.load(wk_row + t * wts + kdims * wks, mask=keep_k,
                     other=0.0).to(tl.float32)
        carries_key = op0 != _NONE_CODE
        pop_gate = tl.where(op0 == _POP_CODE, 0.0, 1.0)
        key_lane = _mul_rn(wk, pop_gate)
        hitk = carries_key & keep_r & (slot0 == ranks)
        kacc = tl.where(hitk[:, None], key_lane[None, :], kacc)
        hit2k = merge & keep_r & (second == ranks)
        kacc = tl.where(hit2k[:, None], 0.0, kacc)


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k11_bwd_value_walk_kernel(
    traj, payload, gates, dout, decay32, live_pre, carry_in,
    nr_ch, w_ch, dstrength_ch,
    slot, op, erase,
    dpayload, ddense, dlam_partial, dgate_partial, dstate_seg,
    SEGT, T, T0, R, NB,
    pbs, pts, phs, pvs,
    gbs, gts, ghs,
    xbs, xts, xhs, xvs,
    sbs, sts, shs, sls,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, RP: tl.constexpr,
    LOGV: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    GATED: tl.constexpr, DENSE: tl.constexpr,
):
    """K10's value-blocked reverse walk plus the gate adjoint.  Lanes last
    first: the dense lane's ``ddense = dS[best]`` (additive, ds passes
    through), the MERGE-source zeroing, then lane 0 -- under the gate
    ``dpayload = g*dpay0``, ``dgates = <dpay0, p - v>`` (a per-block partial,
    like dlam's), and the CORRECT blend's old-row path
    ``dS_dec := (1-g)*dS`` instead of the overwrite's termination -- then the
    DECAY adjoint, the MERGE-blend scatter, and the read adjoint (K10
    verbatim).  """
    row = tl.program_id(0)
    vblock = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = vblock * BV + tl.arange(0, BV)
    keep_v = values < V
    keep = keep_r[:, None] & keep_v[None, :]

    ds = tl.load(carry_in + row.to(tl.int64) * R * V + ranks[:, None] * V
                 + values[None, :], mask=keep, other=0.0)
    traj_base = traj + row.to(tl.int64) * SEGT * R * V
    payload_row = payload + batch_index * pbs + head_index * phs
    gates_row = gates + batch_index * gbs + head_index * ghs
    dout_row = dout + batch_index.to(tl.int64) * xbs + head_index * xhs
    decay_row = decay32 + row * T
    live_base = (live_pre + batch_index.to(tl.int64) * T * H * R
                 + head_index * R)
    nr_row = nr_ch + row.to(tl.int64) * SEGT * R
    w_row = w_ch + row.to(tl.int64) * SEGT * R
    dst_row = dstrength_ch + row.to(tl.int64) * SEGT * R
    dp_row = dpayload + batch_index.to(tl.int64) * T * H * V + head_index * V
    dd_row = ddense + batch_index.to(tl.int64) * T * H * V + head_index * V
    ddl_row = dlam_partial + row.to(tl.int64) * NB * SEGT
    dgate_row = dgate_partial + row.to(tl.int64) * NB * SEGT
    channel_base = batch_index.to(tl.int64) * sbs + head_index * shs

    for i in tl.range(0, SEGT):
        tt = SEGT - 1 - i
        t = T0 + tt
        op0 = tl.load(op + channel_base + t * sts + 0 * sls)
        slot0 = tl.load(slot + channel_base + t * sts + 0 * sls).to(tl.int32)
        second = tl.load(slot + channel_base + t * sts + 1 * sls).to(tl.int32)
        erase0 = tl.load(erase + channel_base + t * sts + 0 * sls) != 0
        merge = tl.load(op + channel_base + t * sts + 1 * sls) != _NONE_CODE
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        op2 = tl.load(op + channel_base + t * sts + 2 * sls)
        slot2 = tl.load(slot + channel_base + t * sts + 2 * sls).to(tl.int32)
        lam = tl.load(decay_row + t)
        live = tl.load(live_base + t * H * R + ranks, mask=keep_r,
                       other=0) != 0
        acc = tl.load(traj_base + tt * R * V + ranks[:, None] * V
                      + values[None, :], mask=keep, other=0.0)
        nr = tl.load(nr_row + tt * R + ranks, mask=keep_r, other=0.0)
        w = tl.load(w_row + tt * R + ranks, mask=keep_r, other=0.0)
        dstrength = tl.load(dst_row + tt * R + ranks, mask=keep_r, other=0.0)
        g = tl.load(dout_row + t * xts + values * xvs, mask=keep_v,
                    other=0.0)

        # 1. the lanes, last lane first (the exact reverse of the forward).
        # 1a. lane 2 (the dense deposit): additive -- the full incoming
        #     adjoint at the best row IS ddense; ds passes through.
        if DENSE:
            if op2 != _NONE_CODE:
                hitd = keep_r & (slot2 == ranks)
                ddense_v = tl.sum(tl.where(hitd[:, None], ds, 0.0), 0)
            else:
                ddense_v = tl.zeros((BV,), dtype=tl.float32)
            tl.store(dd_row + t * (H * V) + values,
                     ddense_v.to(ddense.dtype.element_ty), mask=keep_v)
        # 1b. lane 1 (the MERGE source close): the source row's path
        #     terminates.
        hit2 = merge & keep_r & (second == ranks)
        if merge:
            ds = tl.where(hit2[:, None], 0.0, ds)
        # 1c. lane 0.
        hit = keep_r & (slot0 == ranks)
        dpay0 = tl.sum(tl.where(hit[:, None], ds, 0.0), 0)
        if GATED:
            gt = tl.load(gates_row + t * gts).to(tl.float32)
            if carries:
                # the admitted payload's adjoint: dpayload = g * dpay0
                dpay = _mul_rn(gt, dpay0)
                # the gate's adjoint: dgates = <dpay0, p - v> with v the
                # POST-decay row the forward blended against (traj is the
                # PRE-decay state; the forward's acc was mul_rn(acc, lam))
                p_v = tl.load(payload_row + t * pts + values * pvs,
                              mask=keep_v, other=0.0).to(tl.float32)
                old_pre = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)
                v_dec = _mul_rn(old_pre, lam)
                tl.store(dgate_row + vblock * SEGT + tt,
                         tl.sum(_mul_rn(dpay0, _sub_rn(p_v, v_dec)), 0))
            else:
                dpay = tl.zeros((BV,), dtype=tl.float32)
                tl.store(dgate_row + vblock * SEGT + tt, 0.0)
            if erase0:
                if op0 == _CORRECT_CODE:
                    # the blend's old-row path: dS_dec := (1-g) * dS
                    ds = tl.where(hit[:, None],
                                  _mul_rn(_sub_rn(1.0, gt), ds), ds)
                else:
                    # POP/MERGE: the incoming content is discarded
                    ds = tl.where(hit[:, None], 0.0, ds)
        else:
            dpay = tl.where(carries, dpay0, 0.0)
            if erase0:
                ds = tl.where(hit[:, None], 0.0, ds)
        tl.store(dp_row + t * (H * V) + values,
                 dpay.to(dpayload.dtype.element_ty), mask=keep_v)
        # 2. the DECAY adjoint: dlam = <ds, S_t> (a per-block partial), then
        #    ds <- lam * ds.
        tl.store(ddl_row + vblock * SEGT + tt,
                 tl.sum(tl.sum(_mul_rn(ds, acc), axis=1), axis=0))
        ds = _mul_rn(ds, lam)
        # 3. the MERGE-blend scatter against the pre-decay rows (K10 verbatim).
        if merge:
            strength = _mul_rn(nr, tl.where(live, 1.0, 0.0))
            fw = tl.maximum(tl.sum(tl.where(ranks == slot0, strength, 0.0),
                                    0), _EPS)
            sw = tl.maximum(tl.sum(tl.where(ranks == second, strength, 0.0),
                                   0), _EPS)
            total = fw + sw
            a_mix = _div_rn32(fw, total)
            b_mix = _div_rn32(sw, total)
            ds = tl.where(hit[:, None],
                          _add_rn(ds, _mul_rn(a_mix, dpay0)[None, :]), ds)
            ds = tl.where(hit2[:, None],
                          _add_rn(ds, _mul_rn(b_mix, dpay0)[None, :]), ds)
        # 4. the read adjoint's value terms: the direct term and the strength
        #    path only (the score path lives on the key channel -- K10).
        ds = _add_rn(ds, _mul_rn(w[:, None], g[None, :]))
        dnr = _mul_rn(dstrength, tl.where(live, 1.0, 0.0))
        unit_r = tl.where(nr[:, None] > 0, _div_rn32(acc, nr[:, None]), 0.0)
        ds = _add_rn(ds, _mul_rn(dnr[:, None], unit_r))

    tl.store(dstate_seg + row.to(tl.int64) * R * V + ranks[:, None] * V
             + values[None, :], ds, mask=keep)


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k11_bwd_key_walk_kernel(
    ktraj, write_keys, query_keys, carry_in,
    dc_ch, dnk_ch, nk_ch, scal,
    slot, op,
    dwrite_keys, dquery_keys, dkeys_seg,
    SEGT, T, T0, R, KD,
    sbs, sts, shs, sls,
    wbs, wts, whs, wks,
    qbs, qts, qhs, qks,
    H: tl.constexpr, KP: tl.constexpr, RP: tl.constexpr,
):
    """The key-table adjoint (K10's walk with the channel strides passed; the
    key channel is untouched by gates and deposits): the lane adjoint emits
    ``dwrite_keys`` at re-file events, the read adjoint's c-term and key-norm
    term accumulate into the table, and the query-key adjoint's clamp-select
    identity produces ``dquery_keys``.  Keys are not decayed."""
    row = tl.program_id(0)
    batch_index = row // H
    head_index = row % H
    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    kdims = tl.arange(0, KP)
    keep_k = kdims < KD
    keep_rk = keep_r[:, None] & keep_k[None, :]

    dK = tl.load(carry_in + row.to(tl.int64) * R * KD + ranks[:, None] * KD
                 + kdims[None, :], mask=keep_rk, other=0.0)
    ktraj_base = ktraj + row.to(tl.int64) * SEGT * R * KD
    wk_row = write_keys + batch_index * wbs + head_index * whs
    qk_row = query_keys + batch_index * qbs + head_index * qhs
    dc_row = dc_ch + row.to(tl.int64) * SEGT * R
    dnk_row = dnk_ch + row.to(tl.int64) * SEGT * R
    nk_row = nk_ch + row.to(tl.int64) * SEGT * R
    scal_row = scal + row.to(tl.int64) * SEGT * 2
    dwk_base = (dwrite_keys + batch_index.to(tl.int64) * T * H * KD
                + head_index * KD)
    dqk_base = (dquery_keys + batch_index.to(tl.int64) * T * H * KD
                + head_index * KD)
    channel_base = batch_index.to(tl.int64) * sbs + head_index * shs

    for i in tl.range(0, SEGT):
        tt = SEGT - 1 - i
        t = T0 + tt
        op0 = tl.load(op + channel_base + t * sts + 0 * sls)
        slot0 = tl.load(slot + channel_base + t * sts + 0 * sls).to(tl.int32)
        second = tl.load(slot + channel_base + t * sts + 1 * sls).to(tl.int32)
        merge = tl.load(op + channel_base + t * sts + 1 * sls) != _NONE_CODE
        wk = tl.load(wk_row + t * wts + kdims * wks, mask=keep_k,
                     other=0.0).to(tl.float32)
        qk = tl.load(qk_row + t * qts + kdims * qks, mask=keep_k,
                     other=0.0).to(tl.float32)
        # 1. the key-lane adjoint (K10 verbatim).
        refiles = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE) \
            | (op0 == _MERGE_CODE)
        carries_key = op0 != _NONE_CODE
        hitk = keep_r & (slot0 == ranks)
        emit = tl.sum(tl.where(hitk[:, None], dK, 0.0), 0)
        dwk = tl.where(refiles, emit, 0.0)
        tl.store(dwk_base + t * H * KD + kdims,
                 dwk.to(dwrite_keys.dtype.element_ty), mask=keep_k)
        dK = tl.where((carries_key & hitk)[:, None], 0.0, dK)
        hit2k = merge & keep_r & (second == ranks)
        if merge:
            dK = tl.where(hit2k[:, None], 0.0, dK)
        # 2. the read adjoint's key terms (K10 verbatim).
        kacc = tl.load(ktraj_base + tt * R * KD + ranks[:, None] * KD
                       + kdims[None, :], mask=keep_rk, other=0.0)
        dc = tl.load(dc_row + tt * R + ranks, mask=keep_r, other=0.0)
        dnk = tl.load(dnk_row + tt * R + ranks, mask=keep_r, other=0.0)
        nk = tl.load(nk_row + tt * R + ranks, mask=keep_r, other=0.0)
        nqk = tl.load(scal_row + tt * 2 + 0)
        duu = tl.load(scal_row + tt * 2 + 1)
        cq = tl.maximum(nqk, _EPS)
        uqk = _div_rn32(qk, tl.broadcast_to(cq, (KP,)))
        dK = _add_rn(dK, _mul_rn(dc[:, None], uqk[None, :]))
        unit_key = tl.where(nk[:, None] > 0, _div_rn32(kacc, nk[:, None]),
                            0.0)
        dK = _add_rn(dK, _mul_rn(dnk[:, None], unit_key))
        # 3. dquery_keys (K10 verbatim).
        duqk = tl.sum(_mul_rn(dc[:, None], kacc), 0)
        dqk = _div_rn32(duqk, tl.broadcast_to(cq, (KP,))) \
            - _mul_rn(uqk, tl.where(nqk >= _EPS,
                                   tl.broadcast_to(_div_rn32(duu, nqk),
                                                   (KP,)), 0.0))
        tl.store(dqk_base + t * H * KD + kdims,
                 dqk.to(dquery_keys.dtype.element_ty), mask=keep_k)

    tl.store(dkeys_seg + row.to(tl.int64) * R * KD + ranks[:, None] * KD
             + kdims[None, :], dK, mask=keep_rk)


# ------------------------------------------------- the backward body


def _k11_backward_body(payload, dense, write_keys, query_keys, decay, gates,
                        dreads, schedule, live_pre, ckpt_state, ckpt_live,
                        ckpt_keys, rank, segment_tokens, scale, gated,
                        use_dense):
    """The whole window's adjoint: segments in reverse, each recomputed from
    its entry checkpoints (the gate-C doctrine) -- K4's value replay (imported
    verbatim, lane-generic: the blended lane-0 payload and the dense lane are
    ordinary schedule data to it), the key replay, K10's token-parallel prep
    (imported verbatim), then the two reverse walks."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    R, SEG = int(rank), int(segment_tokens)
    rows = batch * heads
    decay32 = decay.to(torch.float32).permute(0, 2, 1).contiguous()  # (b,h,T)
    dreads_c = dreads.contiguous()
    dpayload = torch.empty_like(payload)
    ddense = torch.empty_like(dense) if use_dense \
        else torch.zeros_like(dense)
    dwrite_keys = torch.empty_like(write_keys)
    dquery_keys = torch.empty_like(query_keys)
    ddecay = torch.empty((batch, heads, tokens), dtype=torch.float32,
                         device=device)
    dgates = torch.empty((batch, heads, tokens), dtype=torch.float32,
                         device=device)
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
        # 1. the segment's value trajectory (K4's replay, imported verbatim).
        zeros_w = torch.zeros(batch, heads, segt, R, dtype=torch.float32,
                              device=device)
        _, _, _, traj = persistent_scan_trajectory(
            ckpt_state[seg], ckpt_live[seg], zeros_w,
            decay32[:, :, start:stop], schedule.chunk(start, stop))
        # 2. the segment's key trajectory (the key-lane replay).
        ktraj = torch.empty((rows, segt, R, key_dim), dtype=torch.float32,
                            device=device)
        _k11_key_traj_kernel[(rows,)](
            ckpt_keys[seg], write_keys, schedule.slot, schedule.op, ktraj,
            segt, tokens, start, R, key_dim,
            *schedule.slot.stride(),
            *write_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        # 3. prep: the per-token weights and adjoint coefficients (K10's
        # kernel, imported verbatim -- it recomputes the read weights from the
        # trajectory and the key channel, gate-agnostic).
        nr_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        nk_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        w_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dscores_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                                 device=device)
        dstrength_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                                   device=device)
        z_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dc_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dnk_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                             device=device)
        scal = torch.empty((rows, segt, 2), dtype=torch.float32, device=device)
        grid = (rows, triton.cdiv(segt, BT_BWD))
        _k10_bwd_prep_kernel[grid](
            traj, ktraj, query_keys, dreads_c, live_pre.view(torch.int8),
            nr_ch, nk_ch, w_ch, dscores_ch, dstrength_ch, z_ch, dc_ch,
            dnk_ch, scal,
            segt, tokens, start, R, key_dim,
            *query_keys.stride(), *dreads_c.stride(),
            H=heads, V=value_dim, VP=VP, KP=KP, RP=RP, LOGV=LOGV, LOGR=LOGR,
            LOGK=LOGK, BT=BT_BWD, SCALE=_round_f32(scale), num_warps=8)
        # 4. the value walk (value-blocked reverse adjoint, gate-aware).
        dstate_seg = torch.empty((batch, heads, R, value_dim),
                                 dtype=torch.float32, device=device)
        dlam_partial = torch.empty((rows, nblocks, segt), dtype=torch.float32,
                                   device=device)
        dgate_partial = (torch.empty((rows, nblocks, segt),
                                     dtype=torch.float32, device=device)
                         if gated else
                         torch.zeros((rows, nblocks, segt),
                                    dtype=torch.float32, device=device))
        _k11_bwd_value_walk_kernel[(rows, nblocks)](
            traj, payload, gates, dreads_c, decay32, live_pre.view(torch.int8),
            carry,
            nr_ch, w_ch, dstrength_ch,
            schedule.slot, schedule.op, schedule.erase.view(torch.int8),
            dpayload, ddense, dlam_partial, dgate_partial, dstate_seg,
            segt, tokens, start, R, nblocks,
            *payload.stride(), *gates.stride(), *dreads_c.stride(),
            *schedule.slot.stride(),
            H=heads, V=value_dim, VP=VP, RP=RP, LOGV=LOGV, BT=BT_BWD,
            BV=BV_BWD, GATED=1 if gated else 0, DENSE=1 if use_dense else 0,
            num_warps=4)
        # 5. the key walk (the key-table adjoint tile; K10's, stride-general).
        dkeys_seg = torch.empty((batch, heads, R, key_dim),
                                dtype=torch.float32, device=device)
        _k11_bwd_key_walk_kernel[(rows,)](
            ktraj, write_keys, query_keys, carry_keys,
            dc_ch, dnk_ch, nk_ch, scal,
            schedule.slot, schedule.op,
            dwrite_keys, dquery_keys, dkeys_seg,
            segt, tokens, start, R, key_dim,
            *schedule.slot.stride(), *write_keys.stride(),
            *query_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        ddecay[:, :, start:stop] = dlam_partial.sum(dim=1).view(
            batch, heads, segt)
        dgates[:, :, start:stop] = dgate_partial.sum(dim=1).view(
            batch, heads, segt)
        carry = dstate_seg
        carry_keys = dkeys_seg
    dstate = carry
    dkeys = carry_keys
    return (dpayload, ddense, dwrite_keys, dquery_keys, ddecay.permute(0, 2, 1),
            dgates.permute(0, 2, 1), dstate, dkeys)


# ------------------------------------------------------------ the op pair


@torch.library.custom_op("x45::k11_scan_fwd", mutates_args=())
def _k11_scan_fwd_op(
    payload: torch.Tensor, dense: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor, gates: torch.Tensor,
    state: torch.Tensor, live: torch.Tensor, keys: torch.Tensor, rank: int,
    segment_tokens: int, tau_match: float, tau_merge: float, tau_pop: float,
    tie_break: float, scale: float, gated: bool, use_dense: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """The whole (layer, window) gated scan as ONE opaque op: the persistent
    walk's every channel as its outputs, so autograd itself keeps the
    backward's saves alive (the K4/K6/K10 hosting pattern)."""
    _check_gated_shapes(payload, dense, write_keys, query_keys, decay, gates,
                        state, live, keys, int(rank), int(segment_tokens),
                        bool(gated), bool(use_dense))
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _k11_forward_launch(payload, dense, write_keys, query_keys, decay,
                              gates, state, live, keys, pol32, int(rank),
                              int(segment_tokens), bool(gated),
                              bool(use_dense), collect=False,
                              warps=_seq_warps(int(rank), payload.shape[-1]),
                              stages=3)
    return (out["reads"], out["state"], out["live"], out["keys"], out["slot"],
            out["op"], out["erase"], out["live_after"], out["payload"],
            out["live_pre"], out["ckpt_state"], out["ckpt_live"],
            out["ckpt_keys"])


@_k11_scan_fwd_op.register_fake
def _(payload, dense, write_keys, query_keys, decay, gates, state, live, keys,
      rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale,
      gated, use_dense):
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


def _k11_scan_setup(ctx, inputs, output):
    (payload, dense, write_keys, query_keys, decay, gates, state, live, keys,
     rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale,
     gated, use_dense) = inputs
    ctx.save_for_backward(payload, dense, write_keys, query_keys, decay,
                          gates, state, live, keys, *output[4:])
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)
    ctx.gated = bool(gated)
    ctx.use_dense = bool(use_dense)


@torch.library.custom_op("x45::k11_scan_bwd", mutates_args=())
def _k11_scan_bwd_op(
    payload: torch.Tensor, dense: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor, gates: torch.Tensor,
    dreads: torch.Tensor, slot: torch.Tensor, op: torch.Tensor,
    erase: torch.Tensor, live_after: torch.Tensor, payload_field: torch.Tensor,
    live_pre: torch.Tensor, ckpt_state: torch.Tensor, ckpt_live: torch.Tensor,
    ckpt_keys: torch.Tensor, rank: int, segment_tokens: int, scale: float,
    gated: bool, use_dense: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole window's adjoint as its own opaque op: the segment loop
    (value replay -> key replay -> prep -> both walks) runs inside, decisions
    as recorded constants (the standing doctrine)."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _k11_backward_body(payload, dense, write_keys, query_keys, decay,
                              gates, dreads, schedule, live_pre, ckpt_state,
                              ckpt_live, ckpt_keys, int(rank),
                              int(segment_tokens), float(scale), bool(gated),
                              bool(use_dense))


@_k11_scan_bwd_op.register_fake
def _(payload, dense, write_keys, query_keys, decay, gates, dreads, slot, op,
      erase, live_after, payload_field, live_pre, ckpt_state, ckpt_live,
      ckpt_keys, rank, segment_tokens, scale, gated, use_dense):
    device = payload.device
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    return (torch.empty_like(payload), torch.empty_like(dense),
            torch.empty_like(write_keys), torch.empty_like(query_keys),
            torch.empty((batch, tokens, heads), dtype=torch.float32,
                        device=device),
            torch.empty((batch, tokens, heads), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, key_dim), dtype=torch.float32,
                        device=device))


def _k11_scan_backward(ctx, dreads, dstate_out, dlive, dkeys_out, dslot, dop,
                        derase, dlive_after, dpayload_field, dlive_pre,
                        dckpt_state, dckpt_live, dckpt_keys):
    """The adjoint: dpayload, ddense, dwrite_keys, dquery_keys, ddecay,
    dgates, dstate, dkeys; the decision channel and the liveness carry no
    gradient by construction."""
    (payload, dense, write_keys, query_keys, decay, gates, state, live, keys,
     slot, op, erase, live_after, payload_field, live_pre, ckpt_state,
     ckpt_live, ckpt_keys) = ctx.saved_tensors
    batch, tokens, heads, value_dim = payload.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=payload.device)
    (dpayload, ddense, dwrite_keys, dquery_keys, ddecay, dgates, dstate,
     dkeys) = _k11_scan_bwd_op(payload, dense, write_keys, query_keys, decay,
                               gates, dreads, slot, op, erase, live_after,
                               payload_field, live_pre, ckpt_state, ckpt_live,
                               ckpt_keys, ctx.rank, ctx.segment_tokens,
                               ctx.scale, ctx.gated, ctx.use_dense)
    return (dpayload, ddense, dwrite_keys, dquery_keys, ddecay, dgates,
            dstate, None, dkeys, None, None, None, None, None, None, None,
            None, None)


torch.library.register_autograd("x45::k11_scan_fwd", _k11_scan_backward,
                                setup_context=_k11_scan_setup)


# --------------------------------------------------------------- the wrapper


def gated_scan(payload, dense, write_keys, query_keys, decay, gates, *,
               policy: RoutingPolicy, gated: bool = True, use_dense: bool = True,
               rank=None, rank_max=None,
               segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
               initial_state=None, initial_live=None, initial_keys=None,
               collect: bool = False) -> RoutedResult:
    """One whole (layer, window) gated scan at PER-TOKEN granularity.

    The K10 wrapper's drop-in for the learned-gate arms: the registered
    five-op policy evaluated against the CURRENT state and key table at every
    token, addressing on the key channel, with the gate scalars and the dense
    deposit as schedule data.  The kernel path is ONE opaque op pair per scan;
    every refusal takes the EAGER ORACLE -- the same semantics, the parity
    fixed point -- never a silent change:

    * ``policy.ops=False`` (the append-only control) and ``policy.grow`` are
      refused outright (chunk-dial concepts); a rank prefix is refused by name;
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
        raise ValueError("the append-only control is refused by the learned-"
                          "gate registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the learned-gate registration: "
                          "growth is a chunk-boundary concept")
    if rank != rank_max:
        raise ValueError("the learned-gate registration requires rank == "
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
            out = _k11_forward_launch(
                payload, dense, write_keys, query_keys, decay, gates, state,
                live, keys, pol32, rank, segment_tokens, gated, use_dense,
                collect=True, warps=_seq_warps(rank, value_dim), stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        return RoutedResult(out["reads"], out["state"], out["live"], rank,
                            ScanStats(), schedule, out["weights"])
    if (payload.is_cuda and separated_available(rank, value_dim, key_dim)):
        reads, state_out, live_out, _keys, _slot, _op, _erase, _live_after, \
            _payload, _live_pre, _ckpt_state, _ckpt_live, _ckpt_keys = \
            _k11_scan_fwd_op(payload, dense, write_keys, query_keys, decay,
                             gates, state, live, keys, rank,
                             int(segment_tokens), policy.tau_match,
                             policy.tau_merge, policy.tau_pop,
                             policy.tie_break, policy.read_scale, bool(gated),
                             bool(use_dense))
        return RoutedResult(reads, state_out, live_out, rank, ScanStats(),
                            None, None)
    # the eager oracle: the same semantics, the parity fixed point
    ref = gated_oracle_scan(payload, dense, write_keys, query_keys, decay,
                            gates, policy=policy, gated=gated,
                            use_dense=use_dense, rank=rank, rank_max=rank_max,
                            initial_state=state, initial_live=live,
                            initial_keys=keys, collect=False)
    return RoutedResult(ref["reads"], ref["state"], ref["live"], rank,
                        ScanStats(), None, None)


# ------------------------------------------------------- the mixer and the model


class GatedSeparatedMixer(nn.Module):
    """The K11 mixer: the K10 separated mixer's projections and DECAY gate
    with the learned gate channels added, all born mimetic-neutral (P5: every
    channel carries a per-head zero-born scalar, so at birth the mixer IS the
    banked winner exactly).  The arm selects which channels exist:

    * ``write``   the write-strength gate ``g_write = 1 - m_w*sigmoid(...)``:
                  APPEND admits ``g*p`` (the Lean ``gate*value``), CORRECT
                  blends ``v + g*(p - v)`` (the Lean MERGE arithmetic).
    * ``readout`` the read-out gate ``g_out = 1 + m_o*sigmoid(...)``: the
                  g_proj analog, scaling the read before the out projection.
    * ``dense``   the dense-write channel ``m_d*sigmoid(...)*p``: every token
                  deposits on its best-match live row (P3), as schedule lane 2.
    """

    def __init__(self, width: int, heads: int, value_dim: int, rank: int,
                 rank_max: int, key_dim: int, policy: RoutingPolicy,
                 arm: str = "g1", ctx: int = DEFAULT_CONTEXT,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
        super().__init__()
        self.heads, self.value_dim = heads, value_dim
        self.rank, self.rank_max = rank, rank_max
        self.policy = policy
        self.key_dim, self.ctx = int(key_dim), int(ctx)
        self.segment_tokens = int(segment_tokens)
        self.width = width
        self.arm = arm
        self.channels = arm_channels(arm)
        self.value = nn.Linear(width, heads * value_dim, bias=False)
        self.key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.query_key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.out = nn.Linear(heads * value_dim, width, bias=False)
        self.decay = nn.Linear(width, heads, bias=True)
        with torch.no_grad():
            lam0 = math.exp(-1.0 / DECAY_HORIZON_TOKENS)
            self.decay.bias.fill_(math.log(lam0 / (1.0 - lam0)))
        if self.channels["write"]:
            self.write_gate = nn.Linear(width, heads, bias=True)
            self.write_gate_scalar = nn.Parameter(torch.zeros(heads))
        if self.channels["readout"]:
            self.readout_gate = nn.Linear(width, heads, bias=True)
            self.readout_scalar = nn.Parameter(torch.zeros(heads))
        if self.channels["dense"]:
            self.dense_gate = nn.Linear(width, heads, bias=True)
            self.dense_scalar = nn.Parameter(torch.zeros(heads))
        self.apply_gate_birth()

    def apply_gate_birth(self):
        """P5's mimetic-neutral birth: every gate projection is born at ZERO
        (weight and bias) and every channel's per-head scalar at EXACT zero,
        so ``g_write == 1``, ``g_out == 1`` and ``dense == 0`` at step 0 --
        the arm is born as the banked winner, bit for bit on the discrete
        schedule, and the gates grow by training (the gradient reaches the
        scalar at step 0; the projections train once the scalar moves)."""
        with torch.no_grad():
            for name in ("write_gate", "readout_gate", "dense_gate"):
                projection = getattr(self, name, None)
                if projection is not None:
                    projection.weight.zero_()
                    if projection.bias is not None:
                        projection.bias.zero_()
            for name in ("write_gate_scalar", "readout_scalar",
                         "dense_scalar"):
                scalar = getattr(self, name, None)
                if scalar is not None:
                    scalar.zero_()

    def mix(self, x: torch.Tensor):
        batch, tokens, _ = x.shape
        context = causal_context(x, self.ctx)          # (b, T, ctx, width)
        context = context.reshape(batch, tokens, self.ctx * self.width)
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
        write_keys = self.key(context).view(batch, tokens, self.heads,
                                            self.key_dim)
        query_keys = self.query_key(context).view(batch, tokens, self.heads,
                                                  self.key_dim)
        decay = torch.sigmoid(self.decay(x)).clamp(min=DECAY_FLOOR,
                                                    max=1.0 - 1e-6)
        gated = self.channels["write"]
        use_dense = self.channels["dense"]
        if gated:
            gates = 1.0 - self.write_gate_scalar * torch.sigmoid(
                self.write_gate(x))                    # (b, T, h)
        else:
            # a 3-d dummy so the kernel's stride arguments keep their arity
            # (the tensor is never loaded: GATED=0)
            gates = payload.new_ones((1, 1, 1))
        if use_dense:
            dense = self.dense_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(self.dense_gate(x)).unsqueeze(-1) * payload
        else:
            # a 4-d dummy, likewise never loaded (DENSE=0)
            dense = payload.new_zeros((1, 1, 1, 1))
        result = gated_scan(payload, dense, write_keys, query_keys, decay,
                            gates, policy=self.policy, gated=gated,
                            use_dense=use_dense, rank=self.rank,
                            rank_max=self.rank_max,
                            segment_tokens=self.segment_tokens)
        reads = result.reads
        if self.channels["readout"]:
            g_out = 1.0 + self.readout_scalar * torch.sigmoid(
                self.readout_gate(x))                  # (b, T, h)
            reads = reads * g_out.unsqueeze(-1)
        return (self.out(reads.reshape(batch, tokens,
                                       self.heads * self.value_dim)),
                result)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mix(x)
        return out


def convert_model_to_gated_writes(model, arm: str = "g1",
                                  segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
    """Swap every ``SeparatedSequentialMixer`` for the gated one, sharing
    parameters (the K10 conversion pattern: the ``nn.Linear`` modules the
    optimizer already saw are the very same objects), adding the arm's gate
    projections at the mimetic-neutral birth.  The key/query-key projections
    and every value-side module are carried over UNTOUCHED, so at birth the
    converted model's discrete schedule is the banked winner's bit for bit.
    ``k10_separated_keys.py`` is not edited; this builds on its conversion's
    output.  The parameter DELTA is the registration's 1.5 (stated, asserted
    by the probe, not hidden)."""
    channels = arm_channels(arm)
    for block in model.blocks:
        mixer = block.mixer
        sep = GatedSeparatedMixer.__new__(GatedSeparatedMixer)
        torch.nn.Module.__init__(sep)
        sep.__dict__.update(mixer.__dict__)
        sep._modules = dict(mixer._modules)     # own copy; the old mixer keeps its
        sep.arm = arm
        sep.channels = channels
        sep.segment_tokens = int(segment_tokens)
        sep.width = int(mixer.value.weight.shape[1])
        if channels["write"]:
            sep.write_gate = nn.Linear(sep.width, mixer.heads, bias=True)
            sep.write_gate_scalar = nn.Parameter(torch.zeros(mixer.heads))
        if channels["readout"]:
            sep.readout_gate = nn.Linear(sep.width, mixer.heads, bias=True)
            sep.readout_scalar = nn.Parameter(torch.zeros(mixer.heads))
        if channels["dense"]:
            sep.dense_gate = nn.Linear(sep.width, mixer.heads, bias=True)
            sep.dense_scalar = nn.Parameter(torch.zeros(mixer.heads))
        sep.apply_gate_birth()
        block.mixer = sep
    return model


def arm_parameter_delta(arm: str, depth: int, width: int, heads: int) -> int:
    """The registration's 1.5 parameter accounting, computed not hardcoded:
    per channel one ``nn.Linear(width, heads, bias=True)`` plus one per-head
    zero-born scalar."""
    channels = arm_channels(arm)
    per_channel = width * heads + heads + heads
    return depth * per_channel * sum(bool(channels[k])
                                     for k in ("write", "readout", "dense"))


__all__ = ["ARMS", "ARM_CHANNELS", "LANES", "GatedSeparatedMixer",
           "arm_channels", "arm_parameter_delta",
           "convert_model_to_gated_writes", "gated_scan", "gated_oracle_scan",
           "separated_available", "_k11_forward_launch", "_k11_backward_body",
           "RoutedResult", "ScanStats", "MatrixSchedule", "RoutingPolicy",
           "EPS"]
