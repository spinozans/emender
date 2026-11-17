"""K10 -- THE SEPARATED-KEY ARM: the receipted SEP-KV key fix applied to the
untangled per-token machine at 150 M scale.

Why this file exists
--------------------
The 150 M board arm's residual was diagnosed as the identity-cache regime
(``CMAES_TUNING.md``: CORRECT 60-71 % of decisions, firing mostly as exact-match
identity rewriting over repeated byte-level payload directions) -- a store whose
keys ARE its contents.  SEP-KV proved the cause at probe scale
(``RESULT_KEY_SEPARATION.md``: key=value fusion is mechanically role-blind, the
fused arm pinned at the derived 0.5 set-retrieval ceiling across all 48 depth-1
cells) and receipted the fix (small separated write-context keys, key_dim 4-16
all 1.000; the free-key collapse basin killed by mimetic birth,
``RESULT_WHEELER_CAPACITY.md`` 4b).  The fix has never touched real text at
scale; this module is that application, registered in
``PREREGISTRATION_SEPARATED_KEYS.md`` (a DRAFT until the operator signs it off).

The semantics (the registration's 1, pre-stated)
------------------------------------------------
1. A small key table ``K in R^{rank x key_dim}`` (``key_dim`` 16) per head, one
   row per slot, updated by THE SAME SCHEDULE'S WRITE EVENTS: APPEND/CORRECT/
   MERGE re-file the target row's key to the write's emitted key ``k_t`` (a
   learned projection of the last 4 positions of the mixer's input -- the SEP-KV
   receipted minimal write-context form), MERGE additionally zeroes the source
   row's key with its value and liveness, POP clears the address, NONE leaves
   it untouched.  A fresh row's key is zero until first written and matching
   against it is liveness-masked like the value channel.
2. Routing similarity AND read scores switch from ``cos(q, S_j)`` to
   ``cos(q_key, K_j)``; values, payloads, DECAY, the five ops' arithmetic and
   the MERGE blend (detached VALUE strengths) are UNCHANGED.  The value-space
   content query is REPLACED by the key-space query -- the mixer's ``query``
   projection is removed, not retained dead (a parameter-neutral swap at the
   registered shape: 147,456 out for 147,456 in per mixer).
3. Keys are NOT decayed (an address is categorical; the registration's first
   open question).
4. Mimetic birth, free drift: the key/query-key projections are initialised AT
   the fixed orthonormal family -- only the CURRENT-token context position is
   non-zero, a fixed seeded QR-orthonormal map per head applied to the position,
   IDENTICAL for the write-key and query-key projections -- so keys are born
   spread (an invariant of the embeddings) and the birth machine mimics the
   registered fused content-addressing (the read score at birth is the 16-d
   sketch of the content cosine).  Training then drifts the projections freely.

The arithmetic spec is K6's verbatim (balanced binary tree over each reduced
axis, fp32 decision/read arithmetic, correctly-rounded IEEE ops, libdevice exp,
``mul.rn``/``add.rn`` -- no FMA contraction), extended to the key axis: the
key norms and the key-channel score/similarity are defined-order folds over
``key_dim``.  The eager per-token oracle below DEFINES the order; the kernels
implement the same definition, so decisions, payload lanes, read weights, the
value state, the key table, liveness and pre-decision liveness are bit-exact
against the oracle by construction (the read OUTPUT keeps K6's floor channel --
its rank-axis reduction is a native sum, the K1/K23 precedent).

The kernels
------------
``_k10_seq_fwd_kernel``  K6's persistent whole-window walk with the key-match
                        side: the key tile (``rank x key_dim`` fp32) sits in
                        registers beside the value tile; per token the defined-
                        order key norms, read scores, softmax, read, decision
                        similarity, five-op selection, MERGE blend, then DECAY
                        (values only) and BOTH lanes.  Records K6's decision
                        channel shapes (``MatrixSchedule``-compatible, so K4's
                        ``persistent_scan_trajectory`` replays the VALUE side
                        verbatim) plus segment-entry checkpoints of state,
                        liveness AND the key table.
``_k10_key_traj_kernel`` the key table's own schedule replay (the backward's
                        key-side input): one program per row walks the segment
                        from its key checkpoint applying the key lane -- copies
                        only, bit-exact against the forward trivially.
``_k10_bwd_prep_kernel`` token-parallel, K6's prep extended to both channels:
                        the per-token read weights (bit-identical to the
                        forward) and their adjoint coefficients, including the
                        KEY-side coefficients ``dc`` and ``dnk``.
``_k10_bwd_value_walk_kernel``  K6's reverse value walk with the score-path
                        state term REMOVED (the read scores no longer touch the
                        value state; only the direct and strength terms
                        remain): ``dpayload``, ``dlam`` partials, ``dstate``.
``_k10_bwd_key_walk_kernel``   the key-table adjoint tile (``rank x key_dim``)
                        walked in reverse: the lane adjoint emits
                        ``dwrite_keys`` at re-file events (the pre-write key's
                        gradient terminates at its last re-file -- the POP/
                        MERGE-source analog of K4's case-iii), the read
                        adjoint's c-term and key-norm term accumulate into the
                        table, and the query-key adjoint's clamp-select
                        identity produces ``dquery_keys``.

Doctrine, unchanged: decisions are recorded constants in the backward.  The
read weights, payload lanes, DECAY gate, and now the KEY CHANNEL carry
gradient to their inputs (keys are parameters/inputs); the cross-axis
reductions are autograd-ordered on the oracle side, so the gradient gates are
floor-discipline (fp64 gradcheck of the oracle first, fp32-vs-fp64 floors,
``NOISE_FACTOR`` x), stated, not hidden.

The wiring: ONE opaque ``torch.library`` op pair per (layer, window) scan
(``x45::k10_scan_fwd``/``x45::k10_scan_bwd``, the K4/K6 hosting pattern).
Refusals by name, the K6 pattern: ``ops=False`` and ``grow`` refused outright;
a rank prefix refused; a store the persistent walk cannot hold takes the EAGER
ORACLE verbatim.  No ``k6_*``/``k7_*``/``k8_*``/``k9_*`` source is edited; K6's
spec helpers, K4's trajectory replay and the SEP-KV context helper are IMPORTED
(the K8/K9 composition stays at the API level: the key lane is a schedule-
driven lane over the same recorded ``MatrixSchedule`` the growth re-dispatch
and the distributed walk already consume).
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
from .k1_schedule_scan import NONE_CODE, _add_rn, _mul_rn
from .k4_fused_backward import persistent_scan_trajectory
from .k6_sequential_matrix import (_div_rn32, _fold_axis2, _fold_vec, _fp32,
                                   _log2_up, _norm_defined, _pow2, _round_f32,
                                   _seq_warps, _sqrt_rn32, tree_fold)
from .key_separation_probe import causal_context
from .model import DECAY_FLOOR, DECAY_HORIZON_TOKENS
from .scan import RoutedResult, ScanStats
from .schedule import EPS, RoutingPolicy

# The registration's constants: key_dim 16 (the SEP-KV key-dim curve's
# receipted-sufficient size), the 4-position write context (the SEP-KV receipted
# minimal form), the K6 coarse-checkpoint granularity.
DEFAULT_KEY_DIM = 16
DEFAULT_CONTEXT = 4
DEFAULT_SEGMENT_TOKENS = 64
MIMETIC_SEED = 3401
# The key tile's register gate (rank x key_dim fp32), beside K6's value gate.
MAX_KEY_TILE = 8192
# The backward's walk blocking (K6's own constants).
BT_BWD = 16
BV_BWD = 16


def separated_available(rank: int, value_dim: int, key_dim: int,
                        value_limit: int = 16384,
                        key_limit: int = MAX_KEY_TILE) -> bool:
    """Can one program hold the whole store AND its key table?"""
    return (_pow2(rank) * _pow2(value_dim) <= value_limit
            and _pow2(rank) * _pow2(key_dim) <= key_limit)


# ------------------------------------------------------------- the oracle


def _key_selection(sim, live, tau_match, tau_merge, tau_pop, tie, rank, RP):
    """The registered selection chain (K6's ``_oracle_selection`` logic) on one
    token's KEY-channel similarity ``(b, h, rank)``.  ``sim`` must already be
    the DETACHED routing similarity with dead rows at -2.  Everything
    downstream of the similarity is a comparison, so nothing here is
    differentiable."""
    penalty = torch.arange(rank, dtype=torch.float32,
                           device=sim.device) * tie
    score = sim - penalty
    best_index = score.argmax(-1)
    best_value = sim.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    second_score = score.scatter(-1, best_index.unsqueeze(-1), -3.0)
    second_index = second_score.argmax(-1)
    second_value = sim.gather(-1, second_index.unsqueeze(-1)).squeeze(-1)
    worst_index = score.argmin(-1)
    worst_value = sim.gather(-1, worst_index.unsqueeze(-1)).squeeze(-1)
    correct = best_value >= tau_match
    merge = correct & (second_value >= tau_merge)
    free = ~live
    free_count = free.sum(-1)
    need = ~correct
    appendable = need & (free_count > 0)
    free_slot = free.to(torch.int64).argmax(-1)     # first free index (chunk-1)
    full = free_count <= 0
    any_live = live.any(-1)
    pop = (~correct) & (~appendable) & full & (worst_value <= tau_pop) & any_live
    op0 = torch.where(correct,
                      torch.where(merge, torch.full_like(best_index, MERGE),
                                  torch.full_like(best_index, CORRECT)),
                      torch.where(appendable, torch.full_like(best_index, APPEND),
                                  torch.where(pop, torch.full_like(best_index, POP),
                                              torch.full_like(best_index, NONE))))
    slot0 = torch.where(correct, best_index,
                        torch.where(appendable, free_slot, worst_index))
    erase0 = (op0 == CORRECT) | (op0 == MERGE) | (op0 == POP)
    live0 = op0 != POP
    return dict(best_index=best_index, second_index=second_index,
                correct=correct, merge=merge, op0=op0, slot0=slot0,
                erase0=erase0, live0=live0,
                carries=(op0 == APPEND) | (op0 == CORRECT))


def separated_sequential_oracle_scan(payload, write_keys, query_keys, decay, *,
                                     policy: RoutingPolicy, rank=None,
                                     rank_max=None, initial_state=None,
                                     initial_live=None, initial_keys=None,
                                     collect=True):
    """The separated-key arm's eager per-token oracle: the parity fixed point,
    differentiable through the key channel.

    Walks the window token by token in torch, vectorised over (batch, head):
    per token the READ (the certified weights on the KEY channel, from the
    ATTACHED current state and key table), the DECISIONS (the registered
    five-op chain on the DETACHED key similarity), then DECAY (values only --
    keys are addresses and do not age) and the lanes in the registered order.
    The stored key is re-filed on every APPEND/CORRECT/MERGE (the row's address
    becomes the new write context) and cleared on POP; a MERGE's source row
    loses value, key and liveness.  All reductions are the K6 defined tree
    folds, so the kernel path is bit-equal to this by construction.
    """
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape[:-1] != payload.shape[:-1]:
        raise ValueError("keys must share the payload's (batch, tokens, heads)")
    if not policy.ops:
        raise ValueError("the append-only control is refused by the separated "
                          "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the separated registration: "
                          "it is a chunk-boundary concept")
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    if int(rank) != int(rank_max):
        raise ValueError("the separated registration requires rank == "
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
        # ---- the read: certified weights on the KEY channel (attached).
        # score_j = scale*fold(uqk * K_j)/max(nk_j, EPS) -- the K6 defined
        # order (the division after the fold), the value strength unchanged.
        nr = _norm_defined(state, -1, VP)                      # (b, h, rank)
        nk = _norm_defined(keys, -1, KP)                       # (b, h, rank)
        nqk = _norm_defined(qk_t, -1, KP)                      # (b, h)
        uqk = qk_t / nqk.clamp_min(EPS).unsqueeze(-1)
        scores = (scale * tree_fold(uqk.unsqueeze(2) * keys, -1, KP)) \
            / nk.clamp_min(EPS)
        live_f = live.to(compute_dtype)
        any_live = live.any(-1, keepdim=True)
        masked = scores.masked_fill(~live, float("-inf"))
        safe = torch.where(any_live, masked, torch.zeros_like(masked))
        pmax = safe.max(-1).values
        expo = torch.exp(safe - pmax.unsqueeze(-1))
        denom = tree_fold(expo * live_f, -1, RP)               # (b, h)
        any_live_flat = live.any(-1)
        denom_safe = torch.where(any_live_flat, denom, torch.ones_like(denom))
        probs = torch.where(any_live, expo / denom_safe.unsqueeze(-1),
                            torch.zeros_like(expo))
        w = probs * nr * live_f
        reads[:, t] = tree_fold(w.unsqueeze(-1) * state, -2, RP)
        if collect:
            weights_out[:, t] = w
        # ---- the decisions: the registered chain on the DETACHED key channel.
        keys_det = keys.detach()
        nkd = _norm_defined(keys_det, -1, KP)
        wk_det = wk_t.detach()
        uk = wk_det / _norm_defined(wk_det, -1, KP).clamp_min(EPS).unsqueeze(-1)
        sim = tree_fold(uk.unsqueeze(2) * keys_det, -1, KP) \
            / nkd.clamp_min(EPS)                               # (b, h, rank)
        sim = sim.masked_fill(~live, -2.0)
        sel = _key_selection(sim, live, tau_match, tau_merge, tau_pop, tie,
                             rank, RP)
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


# ------------------------------------------------------------- the forward kernel

_EPS = tl.constexpr(EPS)
_BIG_INDEX = tl.constexpr(1 << 30)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))


@triton.jit
def _split_fold_axis2(x, W: tl.constexpr, LOG: tl.constexpr):
    """The DEFINED balanced tree over the last axis of a ``(R, W)`` tile via
    ``tl.split``: pure elementwise ``add.rn`` pair-sums -- the same pairings
    and the same single IEEE adds as torch's ``tree_fold`` (bit-identical),
    but without ``tl.sum``'s reduction-layout machinery, which collapses this
    stack's register allocator when a small ``(rank, key_dim)`` tile is folded
    beside the live whole-store value tile (the 30k-spill receipt in the gate-D
    development notes; the split form holds 0 spills at the registered shape)."""
    for i in tl.static_range(LOG):
        if W >= (2 << i):
            a, b = tl.split(tl.reshape(x, (x.shape[0], W // (2 << i), 2)))
            x = _add_rn(a, b)
    return x


@triton.jit(do_not_specialize=['T', 'SEG'])
def _k10_seq_fwd_kernel(
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
    """One program per (batch*head) row: K6's whole-window walk with the
    key-match side.  The value tile ``acc`` and the key tile ``kacc`` are
    register-resident for the whole window; per token the defined-order value
    and key norms, the KEY-channel read scores and softmax, the read (value
    side, the K6 floor channel), the KEY-channel decision similarity, the
    five-op selection, the MERGE blend, then DECAY (values only) and both
    lanes."""
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

        # ---- the value norms and the read weights on the KEY channel.
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


def _check_separated_shapes(payload, write_keys, query_keys, decay, state,
                            live, keys, rank, segment_tokens):
    """Refuse anything the separated path cannot express, before any launch."""
    if not (payload.is_cuda and write_keys.is_cuda and query_keys.is_cuda
            and decay.is_cuda):
        raise ValueError("the separated scan is a device kernel; inputs must "
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


def _k10_forward_launch(payload, write_keys, query_keys, decay, state, live,
                        keys, policy_f32, rank, segment_tokens, collect, warps,
                        stages):
    """The persistent walk: one launch, every output allocated contiguous."""
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
    handle = _k10_seq_fwd_kernel[(batch * heads,)](
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


# --------------------------------------------------------- the backward kernels


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k10_key_traj_kernel(
    keys_in, write_keys, slot, op, traj_out,
    SEGT, T, T0, R, KD,
    wbs, wts, whs, wks,
    H: tl.constexpr, KP: tl.constexpr, RP: tl.constexpr,
):
    """The key table's schedule replay, one program per (batch*head) row: the
    key tile walked forward over the segment applying the key lane (copies
    only -- bit-exact against the forward trivially), storing the PRE-lane
    table at every token (the table as that token's read saw it)."""
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
    channel_base = batch_index.to(tl.int64) * T * H * 2 + head_index * 2
    traj_base = traj_out + row.to(tl.int64) * SEGT * R * KD
    for tt in tl.range(0, SEGT):
        tl.store(traj_base + tt * R * KD + ranks[:, None] * KD
                 + kdims[None, :], kacc, mask=keep_rk)
        t = T0 + tt
        op0 = tl.load(op + channel_base + t * H * 2 + 0)
        slot0 = tl.load(slot + channel_base + t * H * 2 + 0).to(tl.int32)
        second = tl.load(slot + channel_base + t * H * 2 + 1).to(tl.int32)
        merge = tl.load(op + channel_base + t * H * 2 + 1) != _NONE_CODE
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
def _k10_bwd_prep_kernel(
    traj, ktraj, query_keys, dout, live_pre,
    nr_ch, nk_ch, w_ch, dscores_ch, dstrength_ch, z_ch, dc_ch, dnk_ch, scal,
    SEGT, T, T0, R, KD,
    qbs, qts, qhs, qks,
    gbs, gts, ghs, gvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, KP: tl.constexpr,
    RP: tl.constexpr, LOGV: tl.constexpr, LOGR: tl.constexpr,
    LOGK: tl.constexpr, BT: tl.constexpr, SCALE: tl.constexpr,
):
    """Token-block parallel, K6's prep extended to both channels: recompute the
    per-token read weights (bit-identically to the forward -- the same
    defined-order expressions) and their adjoint coefficients, value side and
    key side.  Stores eight rank vectors (``nr``, ``nk``, ``w``, ``dscores``,
    ``dstrength``, ``z``, ``dc``, ``dnk``) and two scalars (``nqk``, ``duu``)
    per token."""
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
            live_any = tl.sum(live_r.to(tl.int32), 0) > 0
            masked = tl.where(live_r, scores, float('-inf'))
            safe = tl.where(live_any, masked, 0.0)
            pmax = tl.max(safe, 0)
            expo = libdevice.exp(safe - pmax)
            denom = tl.reshape(_fold_vec(tl.where(live_r, expo, 0.0), RP,
                                         LOGR), (1,))
            probs = tl.where(live_any,
                             _div_rn32(expo, tl.broadcast_to(denom, (RP,))),
                             0.0)
            strength = _mul_rn(nr, tl.where(live_r, 1.0, 0.0))
            w = _mul_rn(probs, strength)
            dw = tl.reshape(_fold_axis2(_mul_rn(g[None, :], acc), VP, LOGV),
                            (RP,))
            dprobs = _mul_rn(dw, strength)
            dsum = tl.reshape(_fold_vec(_mul_rn(dprobs, probs), RP, LOGR), (1,))
            dscores = _mul_rn(probs, dprobs - tl.broadcast_to(dsum, (RP,)))
            dstrength = _mul_rn(dw, probs)
            z = _mul_rn(dscores, scores)
            duu = tl.reshape(_fold_vec(z, RP, LOGR), (1,))
            # the KEY-side adjoint coefficients: scores = SCALE*c/max(nk,EPS)
            # with c = fold(uqk, K_j), so dc = dscores*SCALE/max(nk,EPS) and
            # dnk = -z/max(nk,EPS) inside the clamp region (the K6 identity).
            crk = tl.maximum(nk, _EPS)
            dc = _div_rn32(_mul_rn(dscores, SCALE), crk)
            dnk = tl.where(nk >= _EPS, _div_rn32(_mul_rn(z, -1.0), crk), 0.0)
            tl.store(nr_row + local * R + ranks, nr, mask=keep_r)
            tl.store(nk_row + local * R + ranks, nk, mask=keep_r)
            tl.store(w_row + local * R + ranks, w, mask=keep_r)
            tl.store(ds_row + local * R + ranks, dscores, mask=keep_r)
            tl.store(dst_row + local * R + ranks, dstrength, mask=keep_r)
            tl.store(z_row + local * R + ranks, z, mask=keep_r)
            tl.store(dc_row + local * R + ranks, dc, mask=keep_r)
            tl.store(dnk_row + local * R + ranks, dnk, mask=keep_r)
            tl.store(scal_row + local * 2 + tl.arange(0, 1),
                     tl.reshape(nqk, (1,)))
            tl.store(scal_row + local * 2 + 1 + tl.arange(0, 1),
                     tl.reshape(duu, (1,)))


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k10_bwd_value_walk_kernel(
    traj, dout, decay32, live_pre, carry_in,
    nr_ch, w_ch, dstrength_ch,
    slot, op, erase,
    dpayload, dlam_partial, dstate_seg,
    SEGT, T, T0, R, NB,
    gbs, gts, ghs, gvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, RP: tl.constexpr,
    LOGV: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
):
    """K6's value-blocked reverse walk with the score-path state term REMOVED:
    the read scores live on the key channel and no longer touch the value
    state, so the read adjoint's value terms are the direct term and the
    strength term only.  Lanes last-first, the DECAY adjoint, the MERGE-blend
    scatter, ``dpayload``, ``dlam`` partials, ``dstate`` -- K6 verbatim."""
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
    dout_row = dout + batch_index * gbs + head_index * ghs
    decay_row = decay32 + row * T
    live_base = (live_pre + batch_index.to(tl.int64) * T * H * R
                 + head_index * R)
    nr_row = nr_ch + row.to(tl.int64) * SEGT * R
    w_row = w_ch + row.to(tl.int64) * SEGT * R
    dst_row = dstrength_ch + row.to(tl.int64) * SEGT * R
    dp_row = dpayload + batch_index.to(tl.int64) * T * H * V + head_index * V
    ddl_row = dlam_partial + row.to(tl.int64) * NB * SEGT
    channel_base = batch_index.to(tl.int64) * T * H * 2 + head_index * 2

    for i in tl.range(0, SEGT):
        tt = SEGT - 1 - i
        t = T0 + tt
        op0 = tl.load(op + channel_base + t * H * 2 + 0)
        slot0 = tl.load(slot + channel_base + t * H * 2 + 0).to(tl.int32)
        second = tl.load(slot + channel_base + t * H * 2 + 1).to(tl.int32)
        erase0 = tl.load(erase + channel_base + t * H * 2 + 0) != 0
        merge = tl.load(op + channel_base + t * H * 2 + 1) != _NONE_CODE
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        lam = tl.load(decay_row + t)
        live = tl.load(live_base + t * H * R + ranks, mask=keep_r,
                       other=0) != 0
        acc = tl.load(traj_base + tt * R * V + ranks[:, None] * V
                     + values[None, :], mask=keep, other=0.0)
        nr = tl.load(nr_row + tt * R + ranks, mask=keep_r, other=0.0)
        w = tl.load(w_row + tt * R + ranks, mask=keep_r, other=0.0)
        dstrength = tl.load(dst_row + tt * R + ranks, mask=keep_r, other=0.0)
        g = tl.load(dout_row + t * gts + values * gvs, mask=keep_v,
                    other=0.0)

        # 1. the lanes, last lane first (the exact reverse of the forward).
        hit = keep_r & (slot0 == ranks)
        hit2 = merge & keep_r & (second == ranks)
        if merge:
            ds = tl.where(hit2[:, None], 0.0, ds)
        dpay0 = tl.sum(tl.where(hit[:, None], ds, 0.0), 0)
        if erase0:
            ds = tl.where(hit[:, None], 0.0, ds)
        # 2. the DECAY adjoint: dlam = <ds, S_t> (a per-block partial), then
        #    ds <- lam * ds.
        tl.store(ddl_row + vblock * SEGT + tt,
                 tl.sum(tl.sum(_mul_rn(ds, acc), axis=1), axis=0))
        ds = _mul_rn(ds, lam)
        # 3. the MERGE-blend scatter against the pre-decay rows.
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
        #    path only -- the score path lives on the key channel now.
        ds = _add_rn(ds, _mul_rn(w[:, None], g[None, :]))
        dnr = _mul_rn(dstrength, tl.where(live, 1.0, 0.0))
        unit_r = tl.where(nr[:, None] > 0, _div_rn32(acc, nr[:, None]), 0.0)
        ds = _add_rn(ds, _mul_rn(dnr[:, None], unit_r))
        # 5. dpayload: the APPEND/CORRECT passthrough; the MERGE blend's
        #    gradient went to the state rows, POP/NONE carry nothing.
        dpay = tl.where(carries, dpay0, 0.0)
        tl.store(dp_row + t * H * V + values,
                 dpay.to(dpayload.dtype.element_ty), mask=keep_v)

    tl.store(dstate_seg + row.to(tl.int64) * R * V + ranks[:, None] * V
             + values[None, :], ds, mask=keep)


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k10_bwd_key_walk_kernel(
    ktraj, write_keys, query_keys, carry_in,
    dc_ch, dnk_ch, nk_ch, scal,
    slot, op,
    dwrite_keys, dquery_keys, dkeys_seg,
    SEGT, T, T0, R, KD,
    wbs, wts, whs, wks,
    qbs, qts, qhs, qks,
    H: tl.constexpr, KP: tl.constexpr, RP: tl.constexpr,
):
    """The key-table adjoint, one program per (batch*head) row: the
    ``rank x key_dim`` tile in registers, walked in reverse.  Per token: the
    LANE adjoint first (the forward's last action) -- a re-file emits the row's
    accumulated adjoint into ``dwrite_keys`` and terminates the pre-write
    key's path (POP writes a constant zero and emits nothing; a MERGE source
    row's adjoint is dropped) -- then the READ adjoint's c-term and key-norm
    term accumulate into the table, and the query-key adjoint's clamp-select
    identity produces ``dquery_keys``.  Keys are not decayed: no DECAY adjoint
    on this channel.  The cross-axis reduction ``duqk`` is a native rank sum
    (floor channel, the K6 ``duq`` precedent)."""
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
    channel_base = batch_index.to(tl.int64) * T * H * 2 + head_index * 2

    for i in tl.range(0, SEGT):
        tt = SEGT - 1 - i
        t = T0 + tt
        op0 = tl.load(op + channel_base + t * H * 2 + 0)
        slot0 = tl.load(slot + channel_base + t * H * 2 + 0).to(tl.int32)
        second = tl.load(slot + channel_base + t * H * 2 + 1).to(tl.int32)
        merge = tl.load(op + channel_base + t * H * 2 + 1) != _NONE_CODE
        wk = tl.load(wk_row + t * wts + kdims * wks, mask=keep_k,
                     other=0.0).to(tl.float32)
        qk = tl.load(qk_row + t * qts + kdims * qks, mask=keep_k,
                     other=0.0).to(tl.float32)
        # 1. the key-lane adjoint: APPEND/CORRECT/MERGE re-filed the target
        #    row from THIS token's write key (emit its gradient); POP wrote a
        #    constant zero (emit nothing); either way the pre-write key's
        #    gradient terminates.  A MERGE's source row was zeroed: dropped.
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
        # 2. the read adjoint's key terms: the c-term (dc_j * uqk) and the
        #    key-norm term (dnk_j * unit_key), against the PRE-lane table.
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
        # 3. dquery_keys: duqk = sum_j dc_j * K_j (a native rank sum, floor
        #    channel), then the clamp-select cross term (the K6 identity).
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


def _k10_backward_body(payload, write_keys, query_keys, decay, dreads,
                       schedule, live_pre, ckpt_state, ckpt_live, ckpt_keys,
                       rank, segment_tokens, scale):
    """The whole window's adjoint: segments in reverse, each recomputed from
    its entry checkpoints (the gate-C doctrine) -- K4's value replay, the key
    replay, the token-parallel prep, then the two reverse walks.  Memory: K6's
    arithmetic plus the key trajectory (``segt x rank x 16`` fp32, ~1 MB per
    segment at the registered shape) and three extra rank-vector channels.
    """
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
        # 2. the segment's key trajectory (the key-lane replay).
        ktraj = torch.empty((rows, segt, R, key_dim), dtype=torch.float32,
                            device=device)
        _k10_key_traj_kernel[(rows,)](
            ckpt_keys[seg], write_keys, schedule.slot, schedule.op, ktraj,
            segt, tokens, start, R, key_dim,
            *write_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        # 3. prep: the per-token weights and adjoint coefficients, both sides.
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
        # 4. the value walk (value-blocked reverse adjoint).
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
        # 5. the key walk (the key-table adjoint tile, whole key dim in one
        #    block at the registered key_dim 16).
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


@torch.library.custom_op("x45::k10_scan_fwd", mutates_args=())
def _k10_scan_fwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor, state: torch.Tensor,
    live: torch.Tensor, keys: torch.Tensor, rank: int, segment_tokens: int,
    tau_match: float, tau_merge: float, tau_pop: float, tie_break: float,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """The whole (layer, window) separated scan as ONE opaque op: the
    persistent walk's every channel as its outputs, so autograd itself keeps
    the backward's saves alive (the K4/K6 hosting pattern)."""
    _check_separated_shapes(payload, write_keys, query_keys, decay, state,
                            live, keys, int(rank), int(segment_tokens))
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _k10_forward_launch(payload, write_keys, query_keys, decay, state,
                              live, keys, pol32, int(rank), int(segment_tokens),
                              collect=False,
                              warps=_seq_warps(int(rank), payload.shape[-1]),
                              stages=3)
    return (out["reads"], out["state"], out["live"], out["keys"], out["slot"],
            out["op"], out["erase"], out["live_after"], out["payload"],
            out["live_pre"], out["ckpt_state"], out["ckpt_live"],
            out["ckpt_keys"])


@_k10_scan_fwd_op.register_fake
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


def _k10_scan_setup(ctx, inputs, output):
    payload, write_keys, query_keys, decay, state, live, keys, rank, \
        segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale = inputs
    # the backward needs the inputs' values and the walk's own recorded
    # channels; the reads, exit state/liveness/keys carry no gradient path.
    ctx.save_for_backward(payload, write_keys, query_keys, decay, state, live,
                          keys, *output[4:])        # slot..ckpt_keys
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)


@torch.library.custom_op("x45::k10_scan_bwd", mutates_args=())
def _k10_scan_bwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor, query_keys: torch.Tensor,
    decay: torch.Tensor, dreads: torch.Tensor, slot: torch.Tensor,
    op: torch.Tensor, erase: torch.Tensor, live_after: torch.Tensor,
    payload_field: torch.Tensor, live_pre: torch.Tensor,
    ckpt_state: torch.Tensor, ckpt_live: torch.Tensor,
    ckpt_keys: torch.Tensor, rank: int, segment_tokens: int, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """The whole window's adjoint as its own opaque op: the segment loop
    (value replay -> key replay -> prep -> both walks) runs inside, decisions
    as recorded constants (the standing doctrine)."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _k10_backward_body(payload, write_keys, query_keys, decay, dreads,
                              schedule, live_pre, ckpt_state, ckpt_live,
                              ckpt_keys, int(rank), int(segment_tokens),
                              float(scale))


@_k10_scan_bwd_op.register_fake
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


def _k10_scan_backward(ctx, dreads, dstate_out, dlive, dkeys_out, dslot, dop,
                        derase, dlive_after, dpayload_field, dlive_pre,
                        dckpt_state, dckpt_live, dckpt_keys):
    """The adjoint: dpayload, dwrite_keys, dquery_keys, ddecay, dstate, dkeys;
    the decision channel and the liveness carry no gradient by construction."""
    payload, write_keys, query_keys, decay, state, live, keys, slot, op, \
        erase, live_after, payload_field, live_pre, ckpt_state, ckpt_live, \
        ckpt_keys = ctx.saved_tensors
    batch, tokens, heads, value_dim = payload.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=payload.device)
    dpayload, dwrite_keys, dquery_keys, ddecay, dstate, dkeys = \
        _k10_scan_bwd_op(payload, write_keys, query_keys, decay, dreads, slot,
                         op, erase, live_after, payload_field, live_pre,
                         ckpt_state, ckpt_live, ckpt_keys, ctx.rank,
                         ctx.segment_tokens, ctx.scale)
    return (dpayload, dwrite_keys, dquery_keys, ddecay, dstate, None, dkeys,
            None, None, None, None, None, None, None)


torch.library.register_autograd("x45::k10_scan_fwd", _k10_scan_backward,
                                setup_context=_k10_scan_setup)


# --------------------------------------------------------------- the wrapper


def separated_scan(payload, write_keys, query_keys, decay, *, policy: RoutingPolicy,
                   rank=None, rank_max=None,
                   segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                   initial_state=None, initial_live=None, initial_keys=None,
                   collect: bool = False) -> RoutedResult:
    """One whole (layer, window) separated-key scan at PER-TOKEN granularity.

    The K6 wrapper's drop-in for the separated arm: the registered five-op
    policy evaluated against the CURRENT state and key table at every token,
    addressing on the key channel.  The kernel path is ONE opaque op pair per
    scan; every refusal takes the EAGER ORACLE -- the same semantics, the
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
        raise ValueError("the append-only control is refused by the separated "
                          "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the separated registration: "
                          "growth is a chunk-boundary concept")
    if rank != rank_max:
        raise ValueError("the separated registration requires rank == "
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
            out = _k10_forward_launch(
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
            _k10_scan_fwd_op(payload, write_keys, query_keys, decay, state,
                             live, keys, rank, int(segment_tokens),
                             policy.tau_match, policy.tau_merge,
                             policy.tau_pop, policy.tie_break,
                             policy.read_scale)
        return RoutedResult(reads, state_out, live_out, rank, ScanStats(),
                            None, None)
    # the eager oracle: the same semantics, the parity fixed point
    ref = separated_sequential_oracle_scan(payload, write_keys, query_keys,
                                           decay, policy=policy, rank=rank,
                                           rank_max=rank_max,
                                           initial_state=state,
                                           initial_live=live,
                                           initial_keys=keys, collect=False)
    return RoutedResult(ref["reads"], ref["state"], ref["live"], rank,
                        ScanStats(), None, None)


# ------------------------------------------------------- the mixer and the model


def mimetic_key_family(width: int, key_dim: int, heads: int,
                       seed: int = MIMETIC_SEED) -> torch.Tensor:
    """The fixed orthonormal family: one QR-orthonormal map ``Q_h`` per head
    (``width x key_dim``, orthonormal columns), seeded -- the mimetic birth's
    deterministic geometry.  Keys are born spread by construction: the emitted
    key at birth is ``Q_h^T x_current``, a fixed function of the token whose
    pairwise geometry is an invariant of the embeddings (the Wheeler
    mimetic-init receipt: spread is an invariant of the family, not a training
    outcome)."""
    families = []
    for head in range(heads):
        generator = torch.Generator().manual_seed(seed + head)
        A = torch.randn(width, key_dim, generator=generator)
        Q, _ = torch.linalg.qr(A)          # reduced: (width, key_dim)
        families.append(Q)
    return torch.stack(families)           # (heads, width, key_dim)


class SeparatedSequentialMixer(nn.Module):
    """The K10 mixer: the K6 sequential mixer's projections and DECAY gate with
    the content query REPLACED by the key channel -- the writer gains a small
    key projection of its causal write context, the reader's query moves to
    the key space, and the scan swaps to the separated one.  The mixer's input
    is ``norm1``'s per-position RMSNormed output (no second norm is added; the
    conversion is parameter-neutral at the registered shape)."""

    def __init__(self, width: int, heads: int, value_dim: int, rank: int,
                 rank_max: int, key_dim: int, policy: RoutingPolicy,
                 ctx: int = DEFAULT_CONTEXT,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
        super().__init__()
        self.heads, self.value_dim = heads, value_dim
        self.rank, self.rank_max = rank, rank_max
        self.policy = policy
        self.key_dim, self.ctx = int(key_dim), int(ctx)
        self.segment_tokens = int(segment_tokens)
        self.width = width
        self.value = nn.Linear(width, heads * value_dim, bias=False)
        self.key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.query_key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.out = nn.Linear(heads * value_dim, width, bias=False)
        self.decay = nn.Linear(width, heads, bias=True)
        with torch.no_grad():
            lam0 = math.exp(-1.0 / DECAY_HORIZON_TOKENS)
            self.decay.bias.fill_(math.log(lam0 / (1.0 - lam0)))
        self.apply_mimetic_init()

    def apply_mimetic_init(self, seed: int = MIMETIC_SEED):
        """Mimetic birth: both key projections are born AT the fixed
        orthonormal family -- only the CURRENT-token context position is
        non-zero, the family's map applied to the position, identical for the
        write-key and query-key projections (the birth machine mimics the
        registered fused content-addressing: the read score at birth is the
        ``key_dim``-sketch of the content cosine).  Free drift afterwards: no
        rotation constraint, no re-orthonormalisation."""
        with torch.no_grad():
            family = mimetic_key_family(self.width, self.key_dim, self.heads,
                                        seed)
            for projection in (self.key, self.query_key):
                projection.weight.zero_()
            for head in range(self.heads):
                block = family[head].t()          # (key_dim, width)
                rows = slice(head * self.key_dim, (head + 1) * self.key_dim)
                cols = slice((self.ctx - 1) * self.width, self.ctx * self.width)
                self.key.weight[rows, cols] = block
                self.query_key.weight[rows, cols] = block

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
        result = separated_scan(payload, write_keys, query_keys, decay,
                                policy=self.policy, rank=self.rank,
                                rank_max=self.rank_max,
                                segment_tokens=self.segment_tokens)
        return (self.out(result.reads.reshape(batch, tokens,
                                              self.heads * self.value_dim)),
                result)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mix(x)
        return out


def convert_model_to_separated_keys(model, key_dim: int = DEFAULT_KEY_DIM,
                                    ctx: int = DEFAULT_CONTEXT,
                                    segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                                    seed: int = MIMETIC_SEED):
    """Swap every ``MatrixMixer`` for the separated one, sharing parameters.

    The K6 conversion pattern (a shallow re-wrap: the ``nn.Linear`` modules
    and therefore the parameters the optimizer sees are the very same
    objects), with the addressing swap: the content-query projection is
    REMOVED (dead parameters would weight-decay under an optimizer that saw
    them, and the honest machine does not carry them) and the key/query-key
    projections are ADDED at the mimetic family.  At the registered shape
    (width 576, heads 2, value_dim 128, key_dim 16, ctx 4) the swap is
    parameter-NEUTRAL: ``heads*value_dim*width == 2 * ctx*width*heads*key_dim``
    = 147,456 per mixer -- asserted here, so the converted model is the audited
    149,981,808 parameters exactly.  ``model.py`` is untouched.
    """
    total_before = sum(p.numel() for p in model.parameters())
    for block in model.blocks:
        mixer = block.mixer
        width = mixer.value.weight.shape[1]
        sep = SeparatedSequentialMixer.__new__(SeparatedSequentialMixer)
        torch.nn.Module.__init__(sep)
        sep.__dict__.update(mixer.__dict__)
        sep._modules = dict(mixer._modules)     # own copy; the old mixer keeps its
        del sep.query                            # the dead content query, removed
        sep.key_dim, sep.ctx = int(key_dim), int(ctx)
        sep.segment_tokens = int(segment_tokens)
        sep.width = int(width)
        sep.key = nn.Linear(ctx * width, mixer.heads * int(key_dim), bias=False)
        sep.query_key = nn.Linear(ctx * width,
                                 mixer.heads * int(key_dim), bias=False)
        sep.apply_mimetic_init(seed)
        block.mixer = sep
    total_after = sum(p.numel() for p in model.parameters())
    mixer = model.blocks[0].mixer
    neutral_geometry = (mixer.heads * mixer.value_dim * mixer.width
                        == 2 * mixer.ctx * mixer.width * mixer.heads
                        * mixer.key_dim)
    if neutral_geometry:
        # the registered shape: the swap must be parameter-neutral to the
        # parameter (the probe asserts the audited count on top of this)
        if total_after != total_before:
            raise ValueError(f"the separated conversion must be parameter-"
                             f"neutral at this geometry: {total_before} "
                             f"before vs {total_after} after")
    else:
        model.conversion_parameter_delta = total_after - total_before
    return model


def span_key_std(write_keys: torch.Tensor) -> torch.Tensor:
    """The KEY-COLLAPSE watch (the Wheeler receipt's guard): the std over a
    window's tokens of the emitted write keys, per key dimension, then the
    mean over dimensions -- ``(b, heads)``.  The collapse signature is this
    quantity going to zero (one address absorbing every write)."""
    return write_keys.to(torch.float32).std(dim=1).mean(dim=-1)


__all__ = ["DEFAULT_KEY_DIM", "DEFAULT_CONTEXT", "DEFAULT_SEGMENT_TOKENS",
           "MIMETIC_SEED", "MAX_KEY_TILE", "RoutedResult", "ScanStats",
           "MatrixSchedule", "RoutingPolicy", "EPS",
           "SeparatedSequentialMixer", "convert_model_to_separated_keys",
           "mimetic_key_family", "separated_available",
           "separated_sequential_oracle_scan", "separated_scan",
           "span_key_std", "_k10_forward_launch", "_k10_backward_body"]
