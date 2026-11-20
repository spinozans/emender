"""K13 -- THE ADDITIVE READ on the emender: GDN2's read form transplanted
into the K10 winner -- the row-softmax read replaced by the additive sum of
the banked score channel's raw similarities, gated by the committed g_proj
read gate.

Why this file exists
--------------------
The side-by-side dissection (``RESULT_SIDE_BY_SIDE_0239.md`` 5) receipted the
winner's residual as C5, the store's write/read FORM: byte-identity CORRECT
on 0.979/token, reads over ~5.6 effective rows of ~27 live -- the
read-dilution finding -- and 17/24 stores near-decorative at eval.  Our read
is COMPETITIVE: ``softmax(read_scale*cos(q_key, K_j) over live) * ||S_j||`` --
every row competes for a probability simplex.  GDN2's read is ADDITIVE
(``out_t = S_t^T q_t`` with an output gate): every matching direction
contributes, magnitude carries information, nothing competes.  The follow-up
levers (K11 scalar gates, K11B per-channel gates, K12 placement) all receipted
honest nulls; the one C5 sub-lever never isolated is the read's NORMALIZATION
LAW itself.  This lane is that transplant: a value-channel change that touches
neither the store nor the ops.

The semantics (``PREREGISTRATION_ADDITIVE_READ.md``, committed before any
probe measurement; the supervisor's R13-1..R13-5 rulings cited inline)
--------------------------------------------------------------------------------
* **R13-1 -- the score form: the banked score channel VERBATIM.**
  ``w_j = read_scale * cos(q_key(t), K_j)`` on the key channel, live-gated,
  softmax REMOVED, nothing else in the score path changed.  ``read_scale``
  2.0 is INHERITED-FROM-BANKED, not re-tuned: under an additive read it is a
  gain, absorbed by the downstream learned projection within steps.
* **R13-2 -- strength-drop.**  The banked ``||S_j||`` multiplier goes WITH the
  softmax: magnitude enters through ``v_j = S[j]`` itself (GDN2-faithful).
  A POP'd row contributes exactly 0 through ``S[j] = 0``; the live mask
  still gates ``w_j`` (dead rows' keys are zeroed).  The MERGE blend's own
  DETACHED strength channel (K6's registered blend weights) is UNTOUCHED.
* **R13-3 -- the read gate: the committed k11b ``g_proj`` pattern verbatim.**
  ``r_vec = 1 + m_o*sigmoid(g_proj(x))`` with
  ``g_proj = Sequential(Linear(width, d_v, bias=False),
  Linear(d_v, heads*d_v, bias=True))`` zero-born and ``m_o`` a per-head
  scalar born at EXACT zero, applied ``reads (.) r_vec`` before the output
  projection, OUTSIDE the scan (K11/K11B's read-out convention).
* **P2 -- decisions are inviolable, verbatim**: the read form feeds NOTHING
  but the read output.  The decision chain is the K10 verbatim on the
  untempered key similarity; the recorded op sequence is bit-invariant to the
  read form at fixed inputs (receipted as the DECISIONS-BIT-IDENTICAL birth
  receipt -- NO birth parity is possible, a different function: this is a
  machine-variant arm in the K10 tradition, and the step-1 loss is a
  different function's value, disclosed).
* **The in-kernel accumulation**: the K7 full-population read law governed
  the softmax's in-kernel normalisation (ONE division by the full-population
  exact mass, never a per-block renormalisation).  The additive read has NO
  denominator at all -- a plain live-gated sum over the register-resident
  whole store (the K10 walk holds the full population in registers), so the
  K7 obstruction cannot arise by construction.  The read keeps the K6
  floor-channel discipline (the read feeds no decision, so no flip can come
  of it).

The registered read, stated exactly once::

    score_j = read_scale * cos(q_key(t), K_j)   # the banked score channel, VERBATIM
    w_j     = score_j  if slot j is live else 0  # raw score; NO softmax, NO ||S_j||
    y_raw   = sum_j w_j * S[j]                   # the floor-channel rank sum
    y       = r_vec (.) y_raw                     # the g_proj gate, outside the scan

The kernels (K10's walks, extended; ``k10_*``/``k6_*``/``k1_*``/``k4_*``
sources are imported, never edited)
--------------------------------------------------------------------------------
``_k13_seq_fwd_kernel``     K10's persistent whole-window walk with the read
                            block replaced: ``w = where(live, score, 0)``, the
                            same native rank-axis floor sum; the softmax's
                            exp/max/denominator are GONE.
``_k13_bwd_prep_kernel``    K10's token-parallel prep with the read adjoint's
                            coefficients replaced: ``dscores = dw`` on live
                            rows (the identity -- ``w_j = score_j``), the
                            strength channel written as EXACT ZEROS, ``z`` /
                            ``duu`` / ``dc`` / ``dnk`` on the new ``dscores``
                            (the same clamp-select identities; the
                            decomposition is form-agnostic).
``_k10_key_traj_kernel``   IMPORTED VERBATIM (the key table's schedule replay
                            is read-form-agnostic).
``_k10_bwd_value_walk_kernel`` IMPORTED VERBATIM -- the strength term dies by
                            the zero ``dstrength`` channel (IEEE add of 0.0
                            is exact); the direct term consumes the score
                            weights.
``_k10_bwd_key_walk_kernel`` IMPORTED VERBATIM -- the c-term and key-norm
                            term consume the prep's channels unchanged.

Doctrine, unchanged: decisions are recorded constants in the backward.  The
read weights (raw scores), payload lanes, DECAY gate and the KEY CHANNEL carry
gradient to their inputs; the cross-axis reductions are autograd-ordered on
the oracle side, so the gradient gates are floor-discipline (fp64 gradcheck of
the oracle first, fp32-vs-fp64 floors, ``NOISE_FACTOR`` x), stated, not
hidden.
"""
from __future__ import annotations

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import triton
import triton.language as tl

from .algebra import (APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule)
from .k1_schedule_scan import _add_rn, _mul_rn
from .k4_fused_backward import persistent_scan_trajectory
from .k6_sequential_matrix import (_div_rn32, _fold_axis2, _fold_vec, _fp32,
                                   _log2_up, _norm_defined, _pow2, _round_f32,
                                   _seq_warps, _sqrt_rn32, tree_fold)
from .k10_separated_keys import (_key_selection, DEFAULT_CONTEXT,
                                 DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS,
                                 MAX_KEY_TILE, MIMETIC_SEED,
                                 SeparatedSequentialMixer,
                                 _k10_bwd_key_walk_kernel,
                                 _k10_bwd_value_walk_kernel,
                                 _k10_key_traj_kernel, _split_fold_axis2,
                                 convert_model_to_separated_keys,
                                 mimetic_key_family, separated_available,
                                 separated_scan,
                                 separated_sequential_oracle_scan, span_key_std)
from .key_separation_probe import causal_context
from .model import DECAY_FLOOR
from .scan import RoutedResult, ScanStats
from .schedule import EPS, RoutingPolicy

# The backward's walk blocking (K10's own constants, imported by value).
BT_BWD = 16
BV_BWD = 16

_EPS = tl.constexpr(EPS)
_BIG_INDEX = tl.constexpr(1 << 30)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))


def read_gate_parameter_delta(depth: int, width: int, heads: int,
                              value_dim: int) -> int:
    """The registration's 1.5 parameter accounting, the k11b read channel's
    own formula computed not hardcoded: per mixer, ``g_proj`` =
    ``width*d_v`` + ``d_v*heads*d_v`` + ``heads*d_v`` (the second layer's
    bias) plus one per-head zero-born scalar."""
    return depth * (width * value_dim
                    + value_dim * heads * value_dim
                    + heads * value_dim + heads)


# ------------------------------------------------------------- the oracle


def additive_oracle_scan(payload, write_keys, query_keys, decay, *,
                         policy: RoutingPolicy, rank=None, rank_max=None,
                         initial_state=None, initial_live=None,
                         initial_keys=None, collect=True):
    """The additive-read arm's eager per-token oracle: the parity fixed
    point, differentiable through the key channel and the read.

    K10's oracle VERBATIM with the read block replaced per the registration:
    ``w_j = score_j`` live-gated (the raw banked score; NO softmax, NO
    ``||S_j||`` factor -- magnitude rides ``S[j]`` itself), the read the same
    defined-order rank fold.  The decisions, the MERGE blend's detached
    strengths, DECAY (values only), the key lane and every recorded channel
    are the K10 winner's, bit for bit at fixed inputs.
    """
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape[:-1] != payload.shape[:-1]:
        raise ValueError("keys must share the payload's (batch, tokens, heads)")
    if not policy.ops:
        raise ValueError("the append-only control is refused by the additive "
                          "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the additive registration: "
                         "growth is a chunk-boundary concept")
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    if int(rank) != int(rank_max):
        raise ValueError("the additive registration requires rank == "
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
    RP, VP, KP = _pow2(rank), _pow2(value_dim), _pow2(key_dim)
    tau_match, tau_merge, tau_pop = (_round_f32(policy.tau_match),
                                     _round_f32(policy.tau_merge),
                                     _round_f32(policy.tau_pop))
    tie = _round_f32(policy.tie_break)
    scale = _round_f32(policy.read_scale)
    lanes = 2 if policy.ops else 1
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
        lam_t = d32[:, t]                                     # (b, h)
        # ---- the read (the registration's 1): the banked score channel
        # VERBATIM, the raw score live-gated -- NO softmax, NO strength.
        # An all-dead store reads EXACTLY zero (w is zero on every row).
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
        # (K10 verbatim -- the read form cannot feed op selection).
        keys_det = keys.detach()
        nkd = _norm_defined(keys_det, -1, KP)
        wk_det = wk_t.detach()
        uk = wk_det / _norm_defined(wk_det, -1, KP).clamp_min(EPS).unsqueeze(-1)
        sim = tree_fold(uk.unsqueeze(2) * keys_det, -1, KP) \
            / nkd.clamp_min(EPS)                               # (b, h, rank)
        sim = sim.masked_fill(~live, -2.0)
        sel = _key_selection_imported(sim, live, tau_match, tau_merge, tau_pop,
                                      tie, rank, RP)
        # ---- the strength of the DETACHED value state (the MERGE blend's
        # registered detached weights, unchanged from K6).
        strength = _norm_defined(state.detach(), -1, VP) * live.to(compute_dtype)
        # ---- the payload lane: incoming value (attached), or the MERGE blend
        # (content attached, weights detached -- the registered construction).
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
        lane0 = torch.where(sel["carries"].unsqueeze(-1), p_t,
                            torch.zeros_like(p_t))
        if blend is not None:
            lane0 = torch.where(sel["merge"].unsqueeze(-1), blend, lane0)
        # ---- the key lane: APPEND/CORRECT/MERGE re-file the row under the new
        # write context (attached -- keys are parameters); POP clears it; NONE
        # leaves it untouched.
        key_lane = torch.where(sel["op0"].eq(POP).unsqueeze(-1),
                               torch.zeros_like(wk_t), wk_t)
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
        # ---- apply: DECAY (values only), then the lanes in the registered order.
        state = state * lam_t.unsqueeze(-1).unsqueeze(-1)
        rows_at = state[b_index.expand(batch, heads),
                        h_index.expand(batch, heads), sel["slot0"]]
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
    schedule = MatrixSchedule(**fields)
    return dict(reads=reads, state=state, live=live, keys=keys,
                schedule=schedule, weights=weights_out, live_pre=live_pre_out,
                rank=int(rank))


def _key_selection_imported(sim, live, tau_match, tau_merge, tau_pop, tie,
                            rank, RP):
    """K10's registered selection chain, imported by name (never retyped): the
    oracle is K10's verbatim with ONE block replaced."""
    return _key_selection(sim, live, tau_match, tau_merge, tau_pop, tie,
                          rank, RP)


# ------------------------------------------------------------- the forward kernel


@triton.jit(do_not_specialize=['T', 'SEG'])
def _k13_seq_fwd_kernel(
    payload, write_keys, query_keys, decay, state_in, live_in, keys_in,
    reads, state_out, live_out, keys_out,
    slot_out, op_out, erase_out, live_after_out, payload_out, live_pre,
    ckpt_state, ckpt_live, ckpt_keys, weights_out,
    T, R, ROWS, SEG, KD,
    pbs, pts, phs, pvs,
    wbs, wts, whs, wks,
    qbs, qts, qhs, qks,
    dbs, dts, dhs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, KP: tl.constexpr,
    RP: tl.constexpr, L: tl.constexpr,
    LOGV: tl.constexpr, LOGR: tl.constexpr, LOGK: tl.constexpr,
    TAU_MATCH: tl.constexpr, TAU_MERGE: tl.constexpr, TAU_POP: tl.constexpr,
    TIE: tl.constexpr, SCALE: tl.constexpr, COLLECT: tl.constexpr,
    STAGES: tl.constexpr,
):
    """One program per (batch*head) row: K10's whole-window walk with the
    read block replaced (the registration's 1).  The value tile ``acc`` and
    the key tile ``kacc`` are register-resident for the whole window; per
    token the defined-order value and key norms, the KEY-channel read
    scores, the ADDITIVE read weights (the raw score, live-gated; the
    softmax's exp/max/denominator are GONE -- there is no renormalisation
    anywhere in the form, so the K7 obstruction cannot arise), the read
    (value side, the K6 floor channel), the KEY-channel decision
    similarity, the five-op selection, the MERGE blend, then DECAY (values
    only) and both lanes."""
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
    decay_row = decay + batch_index * dbs + head_index * dhs
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
        lam = tl.load(decay_row + t * dts).to(tl.float32)

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
        # K13: the additive read -- the raw score, live-gated.  NO softmax, NO
        # strength factor: magnitude rides v_j itself (the registration's 1,
        # rulings R13-1/R13-2).  An all-dead store reads EXACTLY zero.
        w = tl.where(live_r, scores, 0.0)
        # the read: the K6 floor channel (a native rank-axis sum -- the read
        # feeds no decision, so no flip can come of it).
        read_o = tl.sum(_mul_rn(w[:, None], acc), 0)
        tl.store(reads_base + t * H * V + values, read_o, mask=keep_v)
        if COLLECT:
            tl.store(weights_base + t * H * R + ranks, w, mask=keep_r)

        # ---- the decisions: the registered selection chain on the KEY channel.
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
        # ---- the MERGE blend from the CURRENT value rows (K6 verbatim).
        fw = tl.maximum(tl.sum(tl.where(ranks == best, strength, 0.0), 0),
                        _EPS)
        sw = tl.maximum(tl.sum(tl.where(ranks == second, strength, 0.0), 0),
                        _EPS)
        total = fw + sw
        a_mix = _div_rn32(fw, total)
        b_mix = _div_rn32(sw, total)
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        if merge:
            coef = tl.where(ranks == best, a_mix,
                           tl.where(ranks == second, b_mix, 0.0))
            blend = tl.sum(_mul_rn(coef[:, None], acc), 0)
            lane0 = blend
        else:
            lane0 = tl.where(carries, p, 0.0)

        # ---- record the decision channel (the scan's (b, T, h, lanes) layout).
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

        # ---- apply: DECAY (values only), then the value lanes in order (K6
        # verbatim), then the KEY lane.
        acc = _mul_rn(acc, lam)
        hit = keep_r & (slot0 == ranks)
        if erase0:
            new_row = lane0
        else:
            old_row = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)
            new_row = _add_rn(old_row, lane0)
        acc = tl.where(hit[:, None], new_row[None, :], acc)
        live = tl.where(hit, live0, live)
        hit2 = merge & keep_r & (second == ranks)
        acc = tl.where(hit2[:, None], 0.0, acc)
        live = tl.where(hit2, False, live)
        # the key lane: every op but NONE re-files the target row (POP clears
        # the address); MERGE zeroes the source row's key with its value.
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


def _k13_forward_launch(payload, write_keys, query_keys, decay, state, live,
                        keys, policy_f32, rank, segment_tokens, collect, warps,
                        stages):
    """The persistent walk: one launch, every output allocated contiguous
    (K10's launch verbatim, the K13 kernel)."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    lanes = 2
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
    handle = _k13_seq_fwd_kernel[(batch * heads,)](
        payload, write_keys, query_keys, decay, state, live.view(torch.int8),
        keys, reads, out_state, out_live.view(torch.int8), out_keys,
        slot, op, erase.view(torch.int8), live_after.view(torch.int8),
        payload_field, live_pre.view(torch.int8),
        ckpt_state, ckpt_live.view(torch.int8), ckpt_keys, weights,
        tokens, rank, batch * heads, int(segment_tokens), key_dim,
        *payload.stride(), *write_keys.stride(), *query_keys.stride(),
        *decay.stride(),
        H=heads, V=value_dim, VP=_pow2(value_dim), KP=_pow2(key_dim),
        RP=_pow2(rank), L=lanes,
        LOGV=_log2_up(_pow2(value_dim)), LOGR=_log2_up(_pow2(rank)),
        LOGK=_log2_up(_pow2(key_dim)),
        TAU_MATCH=tau_match, TAU_MERGE=tau_merge, TAU_POP=tau_pop,
        TIE=tie, SCALE=scale, COLLECT=1 if collect else 0,
        STAGES=stages, num_warps=warps)
    return dict(reads=reads, state=out_state, live=out_live, keys=out_keys,
                slot=slot, op=op, erase=erase, live_after=live_after,
                payload=payload_field, live_pre=live_pre,
                ckpt_state=ckpt_state, ckpt_live=ckpt_live,
                ckpt_keys=ckpt_keys, weights=weights, kernel=handle)


def _check_additive_shapes(payload, write_keys, query_keys, decay, state,
                           live, keys, rank, segment_tokens):
    """Refuse anything the additive path cannot express, before any launch
    (K10's checks verbatim)."""
    if not (payload.is_cuda and write_keys.is_cuda and query_keys.is_cuda
            and decay.is_cuda):
        raise ValueError("the additive scan is a device kernel; inputs must "
                         "be on CUDA")
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


# --------------------------------------------------------- the backward kernels


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k13_bwd_prep_kernel(
    traj, ktraj, query_keys, dout, live_pre,
    nr_ch, nk_ch, w_ch, dscores_ch, dstrength_ch, z_ch, dc_ch, dnk_ch, scal,
    SEGT, T, T0, R, KD,
    qbs, qts, qhs, qks,
    gbs, gts, ghs, gvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, KP: tl.constexpr,
    RP: tl.constexpr, LOGV: tl.constexpr, LOGR: tl.constexpr,
    LOGK: tl.constexpr, BT: tl.constexpr, SCALE: tl.constexpr,
):
    """Token-block parallel, K10's prep with the read adjoint's coefficients
    replaced for the additive form: ``w = where(live, score, 0)`` (bit-
    identical to the forward), ``dscores = dw`` on live rows (the identity --
    ``w_j = score_j``), the strength channel written as EXACT ZEROS (the
    strength path is GONE with the softmax; K10's value walk consumes the
    zeros verbatim -- IEEE add of 0.0 is exact), ``z``/``duu`` on the new
    ``dscores``, and the same clamp-select key-side identities (the
    decomposition is form-agnostic)."""
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    kdims = tl.arange(0, KP)
    keep_k = kdims < KD
    keep_rk = keep_r[:, None] & keep_k[None, :]

    traj_base = traj + row.to(tl.int64) * SEGT * R * V
    ktraj_base = ktraj + row.to(tl.int64) * SEGT * R * KD
    qk_row = query_keys + batch_index * qbs + head_index * qhs
    dout_row = dout + batch_index * gbs + head_index * ghs
    live_base = (live_pre + batch_index.to(tl.int64) * T * H * R
                 + head_index * R)
    nr_row = nr_ch + row.to(tl.int64) * SEGT * R
    nk_row = nk_ch + row.to(tl.int64) * SEGT * R
    w_row = w_ch + row.to(tl.int64) * SEGT * R
    ds_row = dscores_ch + row.to(tl.int64) * SEGT * R
    dst_row = dstrength_ch + row.to(tl.int64) * SEGT * R
    z_row = z_ch + row.to(tl.int64) * SEGT * R
    dc_row = dc_ch + row.to(tl.int64) * SEGT * R
    dnk_row = dnk_ch + row.to(tl.int64) * SEGT * R
    scal_row = scal + row.to(tl.int64) * SEGT * 2
    values = tl.arange(0, VP)
    keep_v = values < V

    for i in tl.range(0, BT):
        local = block * BT + i
        if local < SEGT:
            t = T0 + local
            live = tl.load(live_base + t * H * R + ranks, mask=keep_r,
                           other=0) != 0
            live_r = live & keep_r
            qk = tl.load(qk_row + t * qts + kdims * qks, mask=keep_k,
                         other=0.0).to(tl.float32)
            g = tl.load(dout_row + t * gts + values * gvs, mask=keep_v,
                        other=0.0)
            acc = tl.load(traj_base + local * R * V + ranks[:, None] * V
                          + values[None, :],
                          mask=keep_r[:, None] & keep_v[None, :], other=0.0)
            kacc = tl.load(ktraj_base + local * R * KD + ranks[:, None] * KD
                           + kdims[None, :], mask=keep_rk, other=0.0)
            nr = tl.reshape(_sqrt_rn32(_fold_axis2(_mul_rn(acc, acc), VP,
                                                   LOGV)), (RP,))
            nk = tl.reshape(_sqrt_rn32(_split_fold_axis2(_mul_rn(kacc, kacc),
                                                         KP, LOGK)), (RP,))
            nqk = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(qk, qk), KP, LOGK)),
                             (1,))
            uqk = _div_rn32(qk, tl.broadcast_to(tl.maximum(nqk, _EPS),
                                                (KP,)))
            scores = _div_rn32(
                _mul_rn(tl.reshape(_split_fold_axis2(_mul_rn(uqk[None, :], kacc),
                                                      KP, LOGK), (RP,)), SCALE),
                tl.maximum(nk, _EPS))
            # K13: the additive read's weights and adjoint coefficients.
            w = tl.where(live_r, scores, 0.0)
            dw = tl.reshape(_fold_axis2(_mul_rn(g[None, :], acc), VP, LOGV),
                            (RP,))
            # dscores is the identity on live rows (w_j == score_j); the
            # strength channel is EXACT ZEROS (the path is gone with the
            # softmax; K10's value walk consumes it verbatim).
            dscores = tl.where(live_r, dw, 0.0)
            z = _mul_rn(dscores, scores)
            duu = tl.reshape(_fold_vec(z, RP, LOGR), (1,))
            # the KEY-side adjoint coefficients: scores = SCALE*c/max(nk,EPS)
            # with c = fold(uqk, K_j), so dc = dscores*SCALE/max(nk,EPS) and
            # dnk = -z/max(nk,EPS) inside the clamp region (the K6 identity,
            # form-agnostic).
            crk = tl.maximum(nk, _EPS)
            dc = _div_rn32(_mul_rn(dscores, SCALE), crk)
            dnk = tl.where(nk >= _EPS, _div_rn32(_mul_rn(z, -1.0), crk), 0.0)
            tl.store(nr_row + local * R + ranks, nr, mask=keep_r)
            tl.store(nk_row + local * R + ranks, nk, mask=keep_r)
            tl.store(w_row + local * R + ranks, w, mask=keep_r)
            tl.store(ds_row + local * R + ranks, dscores, mask=keep_r)
            tl.store(dst_row + local * R + ranks,
                     tl.zeros((RP,), dtype=tl.float32), mask=keep_r)
            tl.store(z_row + local * R + ranks, z, mask=keep_r)
            tl.store(dc_row + local * R + ranks, dc, mask=keep_r)
            tl.store(dnk_row + local * R + ranks, dnk, mask=keep_r)
            tl.store(scal_row + local * 2 + tl.arange(0, 1),
                     tl.reshape(nqk, (1,)))
            tl.store(scal_row + local * 2 + 1 + tl.arange(0, 1),
                     tl.reshape(duu, (1,)))


# --------------------------------------------------- the backward body


def _k13_backward_body(payload, write_keys, query_keys, decay, dreads,
                      schedule, live_pre, ckpt_state, ckpt_live, ckpt_keys,
                      rank, segment_tokens, scale):
    """The whole window's adjoint: K10's segment loop VERBATIM with the K13
    prep kernel and an EXACT-ZERO ``dstrength`` channel -- K10's key-traj
    replay, value walk and key walk kernels are IMPORTED VERBATIM (the
    strength term dies by the zero channel, IEEE add of 0.0 is exact; the
    direct term consumes the score weights; the key walk's c/nk terms
    consume the prep's channels).  Decisions are recorded constants (the
    standing doctrine)."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    R, SEG = int(rank), int(segment_tokens)
    rows = batch * heads
    decay32 = decay.to(torch.float32).permute(0, 2, 1).contiguous()  # (b,h,T)
    dreads_c = dreads.contiguous()
    dpayload = torch.empty_like(payload)
    dwrite_keys = torch.empty_like(write_keys)
    dquery_keys = torch.empty_like(query_keys)
    ddecay = torch.empty((batch, heads, tokens), dtype=torch.float32,
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
        # 2. the segment's key trajectory (K10's key-lane replay, verbatim).
        ktraj = torch.empty((rows, segt, R, key_dim), dtype=torch.float32,
                            device=device)
        _k10_key_traj_kernel[(rows,)](
            ckpt_keys[seg], write_keys, schedule.slot, schedule.op, ktraj,
            segt, tokens, start, R, key_dim,
            *write_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        # 3. prep: the per-token weights and adjoint coefficients, both sides
        #    (the K13 form; dstrength is EXACT ZEROS).
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
        # 4. the value walk (K10's, imported verbatim: the strength term dies
        #    by the zero dstrength channel).
        dstate_seg = torch.empty((batch, heads, R, value_dim),
                                 dtype=torch.float32, device=device)
        dlam_partial = torch.empty((rows, nblocks, segt), dtype=torch.float32,
                                   device=device)
        _k10_bwd_value_walk_kernel[(rows, nblocks)](
            traj, dreads_c, decay32, live_pre.view(torch.int8), carry,
            nr_ch, w_ch, dstrength_ch,
            schedule.slot, schedule.op, schedule.erase.view(torch.int8),
            dpayload, dlam_partial, dstate_seg,
            segt, tokens, start, R, nblocks,
            *dreads_c.stride(),
            H=heads, V=value_dim, VP=VP, RP=RP, LOGV=LOGV, BT=BT_BWD,
            BV=BV_BWD, num_warps=4)
        # 5. the key walk (K10's, imported verbatim).
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
        ddecay[:, :, start:stop] = dlam_partial.sum(dim=1).view(
            batch, heads, segt)
        carry = dstate_seg
        carry_keys = dkeys_seg
    dstate = carry
    dkeys = carry_keys
    return (dpayload, dwrite_keys, dquery_keys, ddecay.permute(0, 2, 1),
            dstate, dkeys)


# ------------------------------------------------------------ the op pair


@torch.library.custom_op("x45::k13_scan_fwd", mutates_args=())
def _k13_scan_fwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor, state: torch.Tensor,
    live: torch.Tensor, keys: torch.Tensor, rank: int, segment_tokens: int,
    tau_match: float, tau_merge: float, tau_pop: float, tie_break: float,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """The whole (layer, window) additive-read scan as ONE opaque op (the
    K4/K6/K10 hosting pattern)."""
    _check_additive_shapes(payload, write_keys, query_keys, decay, state,
                           live, keys, int(rank), int(segment_tokens))
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _k13_forward_launch(payload, write_keys, query_keys, decay, state,
                              live, keys, pol32, int(rank), int(segment_tokens),
                              collect=False,
                              warps=_seq_warps(int(rank), payload.shape[-1]),
                              stages=3)
    return (out["reads"], out["state"], out["live"], out["keys"], out["slot"],
            out["op"], out["erase"], out["live_after"], out["payload"],
            out["live_pre"], out["ckpt_state"], out["ckpt_live"],
            out["ckpt_keys"])


@_k13_scan_fwd_op.register_fake
def _(payload, write_keys, query_keys, decay, state, live, keys, rank,
      segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale):
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    nseg = -(-tokens // int(segment_tokens))
    shape = (batch, tokens, heads, 2)
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


def _k13_scan_setup(ctx, inputs, output):
    payload, write_keys, query_keys, decay, state, live, keys, rank, \
        segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale = inputs
    ctx.save_for_backward(payload, write_keys, query_keys, decay, state, live,
                          keys, *output[4:])        # slot..ckpt_keys
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)


@torch.library.custom_op("x45::k13_scan_bwd", mutates_args=())
def _k13_scan_bwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor, query_keys: torch.Tensor,
    decay: torch.Tensor, dreads: torch.Tensor, slot: torch.Tensor,
    op: torch.Tensor, erase: torch.Tensor, live_after: torch.Tensor,
    payload_field: torch.Tensor, live_pre: torch.Tensor,
    ckpt_state: torch.Tensor, ckpt_live: torch.Tensor,
    ckpt_keys: torch.Tensor, rank: int, segment_tokens: int, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """The whole window's adjoint as its own opaque op: the segment loop
    (value replay -> key replay -> K13 prep -> K10's walks) runs inside,
    decisions as recorded constants (the standing doctrine)."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _k13_backward_body(payload, write_keys, query_keys, decay, dreads,
                              schedule, live_pre, ckpt_state, ckpt_live,
                              ckpt_keys, int(rank), int(segment_tokens),
                              float(scale))


@_k13_scan_bwd_op.register_fake
def _(payload, write_keys, query_keys, decay, dreads, slot, op, erase,
      live_after, payload_field, live_pre, ckpt_state, ckpt_live,
      ckpt_keys, rank, segment_tokens, scale):
    device = payload.device
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    return (torch.empty_like(payload), torch.empty_like(write_keys),
            torch.empty_like(query_keys),
            torch.empty((batch, tokens, heads), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, key_dim), dtype=torch.float32,
                        device=device))


def _k13_scan_backward(ctx, dreads, dstate_out, dlive, dkeys_out, dslot, dop,
                        derase, dlive_after, dpayload_field, dlive_pre,
                        dckpt_state, dckpt_live, dckpt_keys):
    """The adjoint: dpayload, dwrite_keys, dquery_keys, ddecay, dstate,
    dkeys; the decision channel and the liveness carry no gradient by
    construction."""
    payload, write_keys, query_keys, decay, state, live, keys, slot, op, \
        erase, live_after, payload_field, live_pre, ckpt_state, ckpt_live, \
        ckpt_keys = ctx.saved_tensors
    batch, tokens, heads, value_dim = payload.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=payload.device)
    dpayload, dwrite_keys, dquery_keys, ddecay, dstate, dkeys = \
        _k13_scan_bwd_op(payload, write_keys, query_keys, decay, dreads, slot,
                         op, erase, live_after, payload_field, live_pre,
                         ckpt_state, ckpt_live, ckpt_keys, ctx.rank,
                         ctx.segment_tokens, ctx.scale)
    return (dpayload, dwrite_keys, dquery_keys, ddecay, dstate, None, dkeys,
            None, None, None, None, None, None, None)


torch.library.register_autograd("x45::k13_scan_fwd", _k13_scan_backward,
                                setup_context=_k13_scan_setup)


# --------------------------------------------------------------- the wrapper


def additive_scan(payload, write_keys, query_keys, decay, *,
                  policy: RoutingPolicy, rank=None, rank_max=None,
                  segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                  initial_state=None, initial_live=None, initial_keys=None,
                  collect: bool = False) -> RoutedResult:
    """One whole (layer, window) additive-read scan at PER-TOKEN granularity.

    The K10 wrapper's drop-in with the additive read: the registered five-op
    policy evaluated against the CURRENT state and key table at every token,
    addressing on the key channel, the read the additive sum of the banked
    score channel's raw similarities.  The kernel path is ONE opaque op pair
    per scan; every refusal takes the EAGER ORACLE -- the same semantics, the
    parity fixed point -- never a silent change:

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
        raise ValueError("the append-only control is refused by the additive "
                          "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the additive registration: "
                         "growth is a chunk-boundary concept")
    if rank != rank_max:
        raise ValueError("the additive registration requires rank == "
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
            out = _k13_forward_launch(
                payload, write_keys, query_keys, decay, state, live, keys,
                pol32, rank, segment_tokens, collect=True,
                warps=_seq_warps(rank, value_dim), stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        return RoutedResult(out["reads"], out["state"], out["live"], rank,
                            ScanStats(), schedule, out["weights"])
    if (payload.is_cuda and separated_available(rank, value_dim, key_dim)):
        reads, state_out, live_out, _keys, _slot, _op, _erase, _live_after, \
            _payload, _live_pre, _ckpt_state, _ckpt_live, _ckpt_keys = \
            _k13_scan_fwd_op(payload, write_keys, query_keys, decay, state,
                             live, keys, rank, int(segment_tokens),
                             policy.tau_match, policy.tau_merge,
                             policy.tau_pop, policy.tie_break,
                             policy.read_scale)
        return RoutedResult(reads, state_out, live_out, rank, ScanStats(),
                            None, None)
    # the eager oracle: the same semantics, the parity fixed point
    ref = additive_oracle_scan(payload, write_keys, query_keys,
                               decay, policy=policy, rank=rank,
                               rank_max=rank_max,
                               initial_state=state, initial_live=live,
                               initial_keys=keys, collect=False)
    return RoutedResult(ref["reads"], ref["state"], ref["live"], rank,
                        ScanStats(), None, None)


# ------------------------------------------------------- the mixer and the model


class AdditiveReadMixer(SeparatedSequentialMixer):
    """The K13 mixer: the K10 separated mixer VERBATIM (same projections, same
    DECAY gate, same mimetic key birth) with the scan swapped to the additive
    one and the k11b ``g_proj`` read gate added -- ``r_vec = 1 +
    m_o*sigmoid(g_proj(x))`` applied to the reads before the output
    projection, OUTSIDE the scan, ``g_proj`` zero-born and ``m_o`` a per-head
    scalar born at EXACT zero, so at birth ``r_vec == 1`` exactly and the arm
    is the PURE additive read (the registration's 1, ruling R13-3)."""

    def __init__(self, width: int, heads: int, value_dim: int, rank: int,
                 rank_max: int, key_dim: int, policy: RoutingPolicy,
                 ctx: int = DEFAULT_CONTEXT,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
        super().__init__(width, heads, value_dim, rank, rank_max, key_dim,
                         policy, ctx=ctx, segment_tokens=segment_tokens)
        self.g_proj = nn.Sequential(
            nn.Linear(width, value_dim, bias=False),
            nn.Linear(value_dim, heads * value_dim, bias=True))
        self.read_scalar = nn.Parameter(torch.zeros(heads))
        self.apply_gate_birth()

    def apply_gate_birth(self):
        """P5's birth half (the k11b pattern verbatim): the gate projection is
        born at ZERO (weight and bias) and the per-head scalar at EXACT zero,
        so ``r_vec == 1`` at step 0 -- the arm is born as the pure additive
        read, and the gate grows by training (the gradient reaches the scalar
        at step 0 through sigmoid(0) = 0.5; the projection wakes one step
        later -- the K11 two-step wake-up pattern, receipted so the step-0
        zero is not read as a dead channel)."""
        with torch.no_grad():
            for module in self.g_proj.modules():
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
            self.read_scalar.zero_()

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
        result = additive_scan(payload, write_keys, query_keys, decay,
                              policy=self.policy, rank=self.rank,
                              rank_max=self.rank_max,
                              segment_tokens=self.segment_tokens)
        reads = result.reads
        # the g_proj read gate (the committed k11b pattern, R13-3): zero-born,
        # exact 1 at birth, outside the scan.
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


def convert_model_to_additive_read(model,
                                   segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
    """Swap every ``SeparatedSequentialMixer`` for the additive-read one,
    sharing parameters (the K10/K11/K11B conversion pattern: the ``nn.Linear``
    modules the optimizer already saw are the very same objects), adding the
    ``g_proj`` read gate at the zero-born identity.  The key/query-key
    projections and every value-side module are carried over UNTOUCHED, so
    the converted model's discrete schedule is the banked winner's bit for
    bit at fixed inputs (the decisions-bit-identical receipt's engine-side
    fact).  ``k10_separated_keys.py`` is not edited; this builds on the K10
    conversion's output.  The parameter DELTA is the registration's 1.5
    (stated, asserted by the probe, not hidden)."""
    for block in model.blocks:
        mixer = block.mixer
        if not isinstance(mixer, SeparatedSequentialMixer):
            raise ValueError("convert_model_to_additive_read expects a "
                             "K10-converted model (run "
                             "convert_model_to_separated_keys first); got "
                             f"{type(mixer).__name__}")
        sep = AdditiveReadMixer.__new__(AdditiveReadMixer)
        torch.nn.Module.__init__(sep)
        sep.__dict__.update(mixer.__dict__)
        sep._modules = dict(mixer._modules)     # own copy; the old mixer keeps its
        width = int(mixer.value.weight.shape[1])
        heads, value_dim = int(mixer.heads), int(mixer.value_dim)
        sep.g_proj = nn.Sequential(
            nn.Linear(width, value_dim, bias=False),
            nn.Linear(value_dim, heads * value_dim, bias=True))
        sep.read_scalar = nn.Parameter(torch.zeros(heads))
        sep.apply_gate_birth()
        block.mixer = sep
    return model


__all__ = ["AdditiveReadMixer", "convert_model_to_additive_read",
           "read_gate_parameter_delta", "additive_scan",
           "additive_oracle_scan", "_k13_forward_launch",
           "_k13_backward_body", "separated_available", "span_key_std",
           "mimetic_key_family", "DEFAULT_KEY_DIM", "DEFAULT_CONTEXT",
           "DEFAULT_SEGMENT_TOKENS", "MIMETIC_SEED", "MAX_KEY_TILE"]
