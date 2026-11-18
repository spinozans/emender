"""K12 -- THE LEARNED PLACEMENT ARM: the APPEND's free-slot choice made a
learned policy on the dead-key residue, on the K10 separated-key lineage.

Why this file exists
--------------------
The op-set audit (``RESULT_OP_SET_AUDIT.md`` 1.1/3) named the gap: our APPEND's
WRITE half is instantiated (the rank-1 lane), but its ALLOCATION half -- WHICH
free slot the new row lands in -- is a fixed first-free-ordinal rule the model
cannot choose.  The operator named the lever: a learned placement policy,
natural form "write where the evicted content lived" -- score the free slots by
the similarity of the new write key to the DEAD row's last-key residue --
generalising to learned ADDRESS LOCALITY (related content clustering in slot
space).  This module is that policy, registered in
``PREREGISTRATION_PLACEMENT.md`` (a DRAFT until the operator signs it off).

The semantics (the registration's 1, pre-stated; the supervisor's rulings cited)
-------------------------------------------------------------------------------
1. **The residue channel (P1).**  A new RECORDED channel ``Res`` (per head,
   ``rank x key_dim``): at a row's DEATH token -- POP's target row, MERGE's
   source row -- ``Res[j] := K[j]``, the dying row's last key, recorded
   PRE-CLEAR (the key table's own value at that token; keys are not decayed,
   so it is the address the evicted content lived at).  Never-used slots keep
   a zero residue.  The residue is state dependence EXACTLY through a recorded
   channel: placement scores consume it and nothing else that is not a
   current-token input or the recorded liveness.
2. **The placement score and the birth tie-break (P5, supervisor-ruled).**  At
   an APPEND decision the placement score of a FREE slot ``j`` is
   ``s_j = w . cos(wk_t, Res_j) + (+0.0)`` in the K6 defined-order arithmetic
   (the same folds and EPS-clamped norms as the read scores -- FINITE at a
   zero residue), and the APPEND lands in
   ``argmax over free slots of s_j, lowest index wins ties`` -- the lane's
   established first-max semantics, the SAME construction in the oracle and
   the kernel.  ``w`` is one per-head scalar per mixer born at EXACT ``+0.0``:
   the ``+0.0`` add is part of the spec (IEEE ``-0.0 + +0.0 = +0.0``) so at
   birth EVERY free slot scores exactly ``+0.0`` and the argmax reproduces
   K10's first-free rule BIT-EXACTLY -- birth parity, and bit-equality
   FOREVER with ``w`` frozen at ``+0.0`` (the never-fires falsifier control).
3. **P2 -- decisions inviolable.**  Placement selects WHICH free slot an
   APPEND lands in; it never touches the op code, the thresholds, the payload,
   the key values, the liveness arithmetic or any live row (P3: the APPEND
   target is a free slot by definition).  The discrete op sequence at fixed
   inputs is bit-invariant to ``w``'s VALUE in the sense that only the APPEND
   slot channel moves; the op/erase/live_after channels cannot.
4. **The zero-gradient FINDING, receipted prominently (the supervisor's own
   ruling).**  Under P2+P3 the placement effect is purely DISCRETE, and the
   recorded-schedule doctrine makes the slot a constant in the backward -- so
   the STRUCTURAL gradient to ``w`` is exactly zero: the eager oracle's
   autograd returns ``dL/dw == 0`` on every fixture (asserted in the tests).
   "The model can learn where to write" therefore REQUIRES an admitted
   estimator; that requirement is a structural fact about this architecture
   (the same wall the zero-learned-management doctrine built Paper 1 on, now
   surfaced from the placement side).
5. **The admitted estimator (supervisor-approved, option (a)).**  The kernel
   path's backward adds ONE small reduction on top of K10's imported
   backward: at each APPEND token, with the write-content sensitivity
   ``g_t = <dpayload_t, p_t>`` (the EXISTING value walk's own output -- no new
   walk), the placement scores ``s_j = w . sim_j`` over free slots and
   ``p = softmax(s over free)``,
   ``dL/dw = sum_t g_t . p_{j*} . (sim_{j*} - sum_j p_j . sim_j)`` -- a
   straight-through softmax surrogate: forward HARD and bit-exact, gradient as
   if the APPEND's write had been soft-split across the free slots.  The score
   inputs are DETACHED (the write key is decision-side, the residue is a
   detached recording): ONLY the per-head ``w`` learns (+48 parameters at the
   registered shape, stated not hidden).  The estimator's bias is disclosed
   verbatim in the registration: it captures the write-content sensitivity of
   the chosen slot, NOT the full trajectory sensitivity of the slot choice.
   The oracle (refusal) path keeps the HONEST hard semantics -- its ``w``
   gradient is the structural zero of finding 4 -- and the probe asserts the
   kernel path is the path taken at the registered shape.

The arithmetic spec is K10's verbatim (defined-order folds, fp32
decision/read arithmetic, correctly-rounded IEEE ops, libdevice ``exp``,
``mul.rn``/``add.rn``, no FMA contraction), extended by the placement folds
(the ``key_dim``-axis residue norms/scores, the same ``_split_fold_axis2``
form) and the estimator's per-token scalar chain (``mul.rn``/``sub.rn``/
``add.rn``, the softmax with max-subtraction as the read's).

The kernels
------------
``_k12_seq_fwd_kernel``   K10's persistent walk with the residue tile
                          (``rank x key_dim`` fp32) beside the key tile: the
                          placement scores at APPEND tokens (inside a scalar
                          ``if appendable`` guard -- the ~2 % of tokens that
                          append), the residue recording at deaths
                          (pre-clear), and the collect-only placement
                          receipt channels (the chosen slot's residue
                          similarity and the free-slot mean).
``_k12_placement_grad_kernel``  the estimator: one program per (batch*head)
                          row replaying the key and residue tiles over the
                          whole window (copies only, bit-exact against the
                          forward's own tables), accumulating the per-head
                          ``dL/dw`` scalar.

The backward's SIX value/key gradients are K10's own
(``_k10_backward_body`` imported VERBATIM -- the recorded schedule fully
determines them; placement changed only WHICH free slot the schedule's APPEND
lane names).  The wiring is ONE opaque op pair per (layer, window) scan
(``x45::k12_scan_fwd``/``x45::k12_scan_bwd``, the K4/K6/K10 hosting pattern)
with the same refusals by name.

Doctrine, unchanged: no ``k10_*``/``k11_*`` source is edited; K10's spec
helpers, its backward body, its selection chain and its mixer/conversion are
IMPORTED.  The K11 lane (cards 0/1) is untouched by this file.
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
from .k1_schedule_scan import _add_rn, _mul_rn, _sub_rn
from .k10_separated_keys import (
    DEFAULT_CONTEXT, DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS, MIMETIC_SEED,
    SeparatedSequentialMixer, _check_separated_shapes, _k10_backward_body,
    _key_selection, _split_fold_axis2, convert_model_to_separated_keys,
    mimetic_key_family, separated_available)
from .k6_sequential_matrix import (_div_rn32, _fold_axis2, _fold_vec, _fp32,
                                   _log2_up, _norm_defined, _pow2, _round_f32,
                                   _seq_warps, _sqrt_rn32, tree_fold)
from .key_separation_probe import causal_context
from .model import DECAY_FLOOR, DECAY_HORIZON_TOKENS
from .scan import RoutedResult, ScanStats
from .schedule import EPS, RoutingPolicy

__all__ = ["DEFAULT_KEY_DIM", "DEFAULT_CONTEXT", "DEFAULT_SEGMENT_TOKENS",
           "MIMETIC_SEED", "RoutedResult", "ScanStats", "MatrixSchedule",
           "RoutingPolicy", "EPS",
           "PlacementSeparatedMixer", "convert_model_to_placement",
           "convert_model_to_separated_keys", "placement_available",
           "placement_oracle_scan", "placement_scan",
           "placement_parameter_delta", "separated_available",
           "_k12_forward_launch", "_k12_backward_body"]


def placement_available(rank: int, value_dim: int, key_dim: int,
                        value_limit: int = 16384,
                        key_limit: int = 8192) -> bool:
    """The persistent walk holds the whole store, its key table AND its
    residue table (the residue tile shares the key tile's register budget)."""
    return separated_available(rank, value_dim, key_dim, value_limit,
                                key_limit)


# ------------------------------------------------------------- the oracle


def placement_oracle_scan(payload, write_keys, query_keys, decay, placement_w,
                          *, policy: RoutingPolicy, rank=None, rank_max=None,
                          initial_state=None, initial_live=None,
                          initial_keys=None, initial_residue=None,
                          collect=True):
    """The placement arm's eager per-token oracle: K10's oracle with the
    residue channel and the learned free-slot choice.  The parity fixed
    point; differentiable through the value/key channels exactly as K10's,
    and HONEST about the placement weight: the slot choice is discrete, so
    the structural gradient to ``placement_w`` through this oracle is EXACTLY
    ZERO (the finding; the kernel path's backward carries the admitted
    straight-through estimator instead -- the registration's disclosure).
    """
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape[:-1] != payload.shape[:-1]:
        raise ValueError("keys must share the payload's (batch, tokens, heads)")
    if placement_w.shape != (heads,) or placement_w.dtype != torch.float32 \
            or not placement_w.is_contiguous():
        raise ValueError(f"placement_w must be a contiguous fp32 (heads,) "
                         f"tensor; got {tuple(placement_w.shape)} "
                         f"{placement_w.dtype}")
    if not policy.ops:
        raise ValueError("the append-only control is refused by the placement "
                         "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the placement registration: "
                         "it is a chunk-boundary concept")
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    if int(rank) != int(rank_max):
        raise ValueError("the placement registration requires rank == "
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
    res = (torch.zeros(batch, heads, rank, key_dim, dtype=compute_dtype,
                       device=device) if initial_residue is None
           else initial_residue)
    p32 = _fp32(payload)
    wk32 = _fp32(write_keys)
    qk32 = _fp32(query_keys)
    d32 = _fp32(decay)
    w32 = placement_w.to(compute_dtype)
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
        # ---- the read: K10 verbatim (certified weights on the key channel).
        nr = _norm_defined(state, -1, VP)                      # (b, h, rank)
        nk = _norm_defined(keys, -1, KP)                       # (b, h, rank)
        nqk = _norm_defined(qk_t, -1, KP)                       # (b, h)
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
        # ---- the decisions: K10's registered chain on the DETACHED key
        # channel, verbatim.
        keys_det = keys.detach()
        nkd = _norm_defined(keys_det, -1, KP)
        wk_det = wk_t.detach()
        uk = wk_det / _norm_defined(wk_det, -1, KP).clamp_min(EPS).unsqueeze(-1)
        sim = tree_fold(uk.unsqueeze(2) * keys_det, -1, KP) \
            / nkd.clamp_min(EPS)                               # (b, h, rank)
        sim = sim.masked_fill(~live, -2.0)
        sel = _key_selection(sim, live, tau_match, tau_merge, tau_pop, tie,
                             rank, RP)
        # ---- the PLACEMENT: the learned free-slot choice (the registration's
        # 1.2).  The score of a free slot is w * cos(wk_t, Res_j) + (+0.0) --
        # defined-order, EPS-clamped (finite at a zero residue), so at w = +0.0
        # EVERY free slot scores exactly +0.0 and the argmax's first-max
        # semantics reproduces K10's first-free rule bit-exactly.
        free_mask = ~live                                      # (b, h, rank)
        res_n = _norm_defined(res, -1, KP)                     # (b, h, rank)
        simres = tree_fold(uk.unsqueeze(2) * res, -1, KP) \
            / res_n.clamp_min(EPS)                            # (b, h, rank)
        s_place = w32.view(1, -1, 1) * simres + 0.0
        ps = torch.where(free_mask, s_place,
                         torch.full_like(s_place, float("-inf")))
        placed = ps.argmax(-1)                                # first max wins
        any_free = free_mask.any(-1)
        placed = torch.where(any_free, placed, torch.zeros_like(placed))
        # the op chain already ruled APPEND; placement only names the slot.
        sel["slot0"] = torch.where(sel["op0"].eq(APPEND), placed,
                                   sel["slot0"])
        # ---- the strength and the payload lane: K10 verbatim.
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
        lane0 = torch.where(sel["carries"].unsqueeze(-1), p_t,
                            torch.zeros_like(p_t))
        if blend is not None:
            lane0 = torch.where(sel["merge"].unsqueeze(-1), blend, lane0)
        # ---- the key lane: K10 verbatim; the RESIDUE recording (pre-clear)
        # rides beside it -- POP's target row and MERGE's source row leave
        # their last key in the residue channel as they die.
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
        # the residue recording: the dying row's last key, PRE-clear.
        res_at = res[b_index.expand(batch, heads),
                     h_index.expand(batch, heads), sel["slot0"]]
        res = res.index_put(
            (b_index.expand(batch, heads), h_index.expand(batch, heads),
             sel["slot0"]),
            torch.where(sel["op0"].eq(POP).unsqueeze(-1), keys_at, res_at))
        live = live.index_put((b_index.expand(batch, heads),
                               h_index.expand(batch, heads), sel["slot0"]),
                              torch.where(sel["op0"].ne(NONE), sel["live0"],
                                          live.gather(-1,
                                                      sel["slot0"].unsqueeze(-1)
                                                      ).squeeze(-1)))
        if bool(sel["merge"].any()):
            # masked per (b,h): only a MERGING lane's source row dies.  (The
            # unmasked form -- zeroing every lane's second row whenever ANY
            # lane merges -- is a latent multi-lane bug in the K10 oracle's
            # MERGE branch, unreachable by K10's own gate fixtures and inert
            # in production (training takes the kernel path, which guards per
            # lane); this copy fixes it and the gate receipt receipts BOTH.)
            keys_at2 = keys[b_index.expand(batch, heads),
                            h_index.expand(batch, heads), sel["second_index"]]
            rows2 = state[b_index.expand(batch, heads),
                          h_index.expand(batch, heads), sel["second_index"]]
            live2 = live.gather(-1, sel["second_index"].unsqueeze(-1)
                                ).squeeze(-1)
            state = state.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads), sel["second_index"]),
                torch.where(sel["merge"].unsqueeze(-1),
                            torch.zeros(batch, heads, value_dim,
                                        dtype=compute_dtype, device=device),
                            rows2))
            keys = keys.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.where(sel["merge"].unsqueeze(-1),
                            torch.zeros(batch, heads, key_dim,
                                        dtype=compute_dtype, device=device),
                            keys_at2))
            res_at2 = res[b_index.expand(batch, heads),
                          h_index.expand(batch, heads), sel["second_index"]]
            res = res.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.where(sel["merge"].unsqueeze(-1), keys_at2, res_at2))
            live = live.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads), sel["second_index"]),
                torch.where(sel["merge"],
                            torch.zeros(batch, heads, dtype=torch.bool,
                                        device=live.device), live2))
    schedule = MatrixSchedule(**fields)
    return dict(reads=reads, state=state, live=live, keys=keys, residue=res,
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
def _k12_seq_fwd_kernel(
    payload, write_keys, query_keys, decay, placement_w,
    state_in, live_in, keys_in, residue_in,
    reads, state_out, live_out, keys_out, residue_out,
    slot_out, op_out, erase_out, live_after_out, payload_out, live_pre,
    ckpt_state, ckpt_live, ckpt_keys, weights_out,
    place_sim, place_free_mean,
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
    """K10's whole-window walk with the residue tile and the placement choice:
    per token the K10 arithmetic verbatim, the placement scores at APPEND
    tokens (a scalar ``if appendable`` guard), the residue recording at deaths
    (pre-clear), and the collect-only placement receipt channels."""
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
    racc = tl.load(residue_in + row.to(tl.int64) * R * KD
                   + ranks[:, None] * KD + kdims[None, :], mask=keep_rk,
                   other=0.0)
    live = tl.load(live_in + row * R + ranks, mask=keep_r, other=0) != 0
    w_place = tl.load(placement_w + head_index)

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
    place_sim_base = place_sim + batch_index.to(tl.int64) * T * H \
        + head_index
    place_mean_base = place_free_mean + batch_index.to(tl.int64) * T * H \
        + head_index

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

        # ---- the decisions: the registered selection chain on the KEY channel
        # (K10 verbatim).
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
        # ---- the PLACEMENT (the registration's 1.2): the free slots scored by
        # the new write key's similarity to the dead row's last-key residue;
        # lowest index wins ties; at w = +0.0 every free slot scores exactly
        # +0.0 (the spec's +0.0 add normalises the -0.0 corner) and the
        # argmax reproduces K10's first-free rule bit-exactly.
        placed_slot = tl.zeros((), dtype=tl.int32)
        psim_chosen = tl.zeros((), dtype=tl.float32)
        psim_free_mean = tl.zeros((), dtype=tl.float32)
        if appendable:
            nres = tl.reshape(_sqrt_rn32(
                _split_fold_axis2(_mul_rn(racc, racc), KP, LOGK)), (RP,))
            simres = _div_rn32(
                tl.reshape(_split_fold_axis2(_mul_rn(uk[None, :], racc),
                                             KP, LOGK), (RP,)),
                tl.maximum(nres, _EPS))
            s_place = _add_rn(_mul_rn(w_place, simres), 0.0)
            s_free = tl.where(free, s_place, float('-inf'))
            s_top = tl.max(s_free, 0)
            placed_found = tl.min(tl.where(s_free == s_top, ranks,
                                           _BIG_INDEX), 0)
            placed_slot = tl.where(placed_found < _BIG_INDEX, placed_found, 0)
            psim_chosen = tl.sum(tl.where(ranks == placed_slot, simres, 0.0),
                                 0)
            psim_free_mean = _div_rn32(
                tl.sum(tl.where(free, simres, 0.0), 0),
                free_count.to(tl.float32))
        full = free_count <= 0
        pop = (~correct) & (~appendable) & full & (worst_value <= TAU_POP) \
            & live_any
        op0 = tl.where(correct, tl.where(merge, _MERGE_CODE, _CORRECT_CODE),
                       tl.where(appendable, _APPEND_CODE,
                                tl.where(pop, _POP_CODE, _NONE_CODE)))
        slot0 = tl.where(correct, best,
                         tl.where(appendable, placed_slot, worst))
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
        if COLLECT:
            tl.store(place_sim_base + t * H, psim_chosen)
            tl.store(place_mean_base + t * H, psim_free_mean)

        # ---- apply: DECAY (values only), then the value lanes in order (K6
        # verbatim), then the residue recording (pre-clear), then the KEY lane.
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
        # the residue channel: the dying rows' last keys, recorded PRE-clear
        # (kacc still holds this token's pre-lane table here).
        if op0 == _POP_CODE:
            hit_res = keep_r & (slot0 == ranks)
            racc = tl.where(hit_res[:, None], kacc, racc)
        if merge:
            hit_res2 = keep_r & (second == ranks)
            racc = tl.where(hit_res2[:, None], kacc, racc)
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
    tl.store(residue_out + row.to(tl.int64) * R * KD + ranks[:, None] * KD
             + kdims[None, :], racc, mask=keep_rk)


# ------------------------------------------------------------ the forward wrapper


def _check_placement_shapes(payload, write_keys, query_keys, decay,
                            placement_w, state, live, keys, residue, rank,
                            segment_tokens):
    """Refuse anything the placement path cannot express, before any launch."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    _ = tokens, value_dim
    if not (payload.is_cuda and write_keys.is_cuda and query_keys.is_cuda
            and decay.is_cuda):
        raise ValueError("the placement scan is a device kernel; inputs must "
                         "be on CUDA")
    if placement_w.shape != (heads,) or placement_w.dtype != torch.float32 \
            or not placement_w.is_contiguous():
        raise ValueError(f"placement_w must be a contiguous fp32 (heads,) "
                         f"tensor; got {tuple(placement_w.shape)} "
                         f"{placement_w.dtype}")
    if residue.shape != (batch, heads, rank, key_dim):
        raise ValueError(f"residue {tuple(residue.shape)} is not (batch, "
                         f"heads, rank, key_dim) = "
                         f"{(batch, heads, rank, key_dim)}")
    for name, tensor in (("residue", residue), ("keys", keys)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be fp32; the algebra is fp32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if tokens < 1 or int(segment_tokens) < 1:
        raise ValueError("tokens and segment_tokens must be positive")
    # K10's own checks carry the rest (state/keys/live/key shapes).
    _check_separated_shapes(payload, write_keys, query_keys, decay, state,
                            live, keys, rank, segment_tokens)


def _k12_forward_launch(payload, write_keys, query_keys, decay, placement_w,
                        state, live, keys, residue, policy_f32, rank,
                        segment_tokens, collect, warps, stages):
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
    out_residue = torch.empty_like(residue)
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
    place_sim = (torch.empty((batch, tokens, heads), dtype=torch.float32,
                             device=device) if collect
                else torch.empty((1,), dtype=torch.float32, device=device))
    place_free_mean = (torch.empty((batch, tokens, heads),
                                   dtype=torch.float32, device=device)
                       if collect
                       else torch.empty((1,), dtype=torch.float32,
                                        device=device))
    tau_match, tau_merge, tau_pop, tie, scale = policy_f32
    handle = _k12_seq_fwd_kernel[(batch * heads,)](
        payload, write_keys, query_keys, decay, placement_w,
        state, live.view(torch.int8), keys, residue,
        reads, out_state, out_live.view(torch.int8), out_keys, out_residue,
        slot, op, erase.view(torch.int8), live_after.view(torch.int8),
        payload_field, live_pre.view(torch.int8),
        ckpt_state, ckpt_live.view(torch.int8), ckpt_keys, weights,
        place_sim, place_free_mean,
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
                residue=out_residue, slot=slot, op=op, erase=erase,
                live_after=live_after, payload=payload_field,
                live_pre=live_pre, ckpt_state=ckpt_state,
                ckpt_live=ckpt_live, ckpt_keys=ckpt_keys, weights=weights,
                place_sim=place_sim, place_free_mean=place_free_mean,
                kernel=handle)


# ---------------------------------------------- the estimator's backward reduction


@triton.jit(do_not_specialize=['T'])
def _k12_placement_grad_kernel(
    write_keys, slot, op, live_pre, g_chan, keys_in, residue_in, placement_w,
    dw_out,
    T, R, KD,
    wbs, wts, whs, wks,
    gbs, gts, ghs,
    H: tl.constexpr, KP: tl.constexpr, RP: tl.constexpr,
    LOGR: tl.constexpr, LOGK: tl.constexpr,
):
    """The admitted straight-through estimator (the registration's 1.5): one
    program per (batch*head) row replaying the key and residue tables over the
    whole window (copies only -- bit-exact against the forward's own tiles),
    accumulating the per-head dL/dw:
    ``dL/dw = sum_t g_t * p_{j*} * (sim_{j*} - sum_j p_j * sim_j)`` with
    ``p = softmax(w*sim over free slots)`` at APPEND tokens and
    ``g_t = <dpayload_t, p_t>`` the value walk's own write-content
    sensitivity.  The score inputs are DETACHED by construction (the write
    keys enter as saved constants here); only the placement weight learns."""
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
    racc = tl.load(residue_in + row.to(tl.int64) * R * KD
                   + ranks[:, None] * KD + kdims[None, :], mask=keep_rk,
                   other=0.0)
    w_place = tl.load(placement_w + head_index)
    wk_row = write_keys + batch_index * wbs + head_index * whs
    channel_base = batch_index.to(tl.int64) * T * H * 2 + head_index * 2
    live_base = live_pre + batch_index.to(tl.int64) * T * H * R + head_index * R
    g_row = g_chan + batch_index * gbs + head_index * ghs
    acc = tl.zeros((), dtype=tl.float32)
    for t in tl.range(0, T):
        op0 = tl.load(op + channel_base + t * H * 2 + 0)
        slot0 = tl.load(slot + channel_base + t * H * 2 + 0).to(tl.int32)
        second = tl.load(slot + channel_base + t * H * 2 + 1).to(tl.int32)
        merge = tl.load(op + channel_base + t * H * 2 + 1) != _NONE_CODE
        live = tl.load(live_base + t * H * R + ranks, mask=keep_r,
                       other=0) != 0
        free = keep_r & (~live)
        wk = tl.load(wk_row + t * wts + kdims * wks, mask=keep_k,
                     other=0.0).to(tl.float32)
        if op0 == _APPEND_CODE:
            # the placement math, bit-identical to the forward's own scores.
            nwk = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(wk, wk), KP, LOGK)),
                             (1,))
            uk = _div_rn32(wk, tl.broadcast_to(tl.maximum(nwk, _EPS), (KP,)))
            nres = tl.reshape(_sqrt_rn32(
                _split_fold_axis2(_mul_rn(racc, racc), KP, LOGK)), (RP,))
            simres = _div_rn32(
                tl.reshape(_split_fold_axis2(_mul_rn(uk[None, :], racc),
                                             KP, LOGK), (RP,)),
                tl.maximum(nres, _EPS))
            s_place = _add_rn(_mul_rn(w_place, simres), 0.0)
            s_free = tl.where(free, s_place, float('-inf'))
            s_top = tl.max(s_free, 0)
            ex = libdevice.exp(s_free - s_top)
            p = _div_rn32(ex, tl.broadcast_to(
                tl.reshape(_fold_vec(ex, RP, LOGR), (1,)), (RP,)))
            sim_c = tl.sum(tl.where(ranks == slot0, simres, 0.0), 0)
            p_c = tl.sum(tl.where(ranks == slot0, p, 0.0), 0)
            p_mean_sim = tl.sum(_fold_vec(_mul_rn(p, simres), RP, LOGR), 0)
            g_t = tl.load(g_row + t * gts)
            term = _mul_rn(_mul_rn(g_t, p_c), _sub_rn(sim_c, p_mean_sim))
            acc = _add_rn(acc, term)
        # the residue recording (pre-clear), then the key lane -- the forward's
        # own arithmetic, replayed.
        if op0 == _POP_CODE:
            hit_res = keep_r & (slot0 == ranks)
            racc = tl.where(hit_res[:, None], kacc, racc)
        if merge:
            hit_res2 = keep_r & (second == ranks)
            racc = tl.where(hit_res2[:, None], kacc, racc)
        carries_key = op0 != _NONE_CODE
        pop_gate = tl.where(op0 == _POP_CODE, 0.0, 1.0)
        key_lane = _mul_rn(wk, pop_gate)
        hitk = carries_key & keep_r & (slot0 == ranks)
        kacc = tl.where(hitk[:, None], key_lane[None, :], kacc)
        if merge:
            hit2k = keep_r & (second == ranks)
            kacc = tl.where(hit2k[:, None], 0.0, kacc)
    tl.store(dw_out + row, acc)


def _k12_placement_grad_launch(write_keys, slot, op, live_pre, g, keys_in,
                               residue_in, placement_w, rank, key_dim):
    """The estimator over one whole window: returns ``dL/dw`` per (batch,
    head) -- the op's backward sums over batch to the (heads,) input."""
    batch, tokens, heads = g.shape
    R = int(rank)
    rows = batch * heads
    dw = torch.empty((rows,), dtype=torch.float32, device=g.device)
    KP, RP = _pow2(key_dim), _pow2(R)
    _k12_placement_grad_kernel[(rows,)](
        write_keys, slot, op, live_pre.view(torch.int8), g, keys_in,
        residue_in, placement_w, dw,
        tokens, R, key_dim,
        *write_keys.stride(), *g.stride(),
        H=heads, KP=KP, RP=RP, LOGR=_log2_up(RP), LOGK=_log2_up(KP),
        num_warps=4)
    return dw.view(batch, heads)


# ------------------------------------------------- the backward body


def _k12_backward_body(payload, write_keys, query_keys, decay, placement_w,
                       dreads, schedule, live_pre, ckpt_state, ckpt_live,
                       ckpt_keys, keys_in, residue_in, rank, segment_tokens,
                       scale):
    """K10's six gradients VERBATIM (the recorded schedule fully determines
    them; ``_k10_backward_body`` is imported, never edited) PLUS the admitted
    straight-through estimator's ``dL/dw`` (heads,) -- the registration's
    disclosed surrogate, the only channel whose gradient is not the true
    adjoint."""
    six = _k10_backward_body(payload, write_keys, query_keys, decay, dreads,
                             schedule, live_pre, ckpt_state, ckpt_live,
                             ckpt_keys, int(rank), int(segment_tokens),
                             float(scale))
    dpayload, dwrite_keys, dquery_keys, ddecay, dstate, dkeys = six
    batch, tokens, heads, _value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    g = (dpayload.to(torch.float32) * payload.to(torch.float32)).sum(-1)
    dw_bh = _k12_placement_grad_launch(
        write_keys, schedule.slot, schedule.op, live_pre, g, keys_in,
        residue_in, placement_w, int(rank), key_dim)
    dw = dw_bh.sum(0)                       # (heads,) -- batch accumulated
    return (dpayload, dwrite_keys, dquery_keys, ddecay, dw, dstate, dkeys)


# ------------------------------------------------------------ the op pair


@torch.library.custom_op("x45::k12_scan_fwd", mutates_args=())
def _k12_scan_fwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor,
    placement_w: torch.Tensor, state: torch.Tensor, live: torch.Tensor,
    keys: torch.Tensor, residue: torch.Tensor, rank: int,
    segment_tokens: int, tau_match: float, tau_merge: float, tau_pop: float,
    tie_break: float, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """The whole (layer, window) placement scan as ONE opaque op: the
    persistent walk's every K10 channel as its outputs (the residue table
    stays internal -- no gradient path, no replay need), so autograd itself
    keeps the backward's saves alive."""
    _check_placement_shapes(payload, write_keys, query_keys, decay,
                            placement_w, state, live, keys, residue,
                            int(rank), int(segment_tokens))
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _k12_forward_launch(payload, write_keys, query_keys, decay,
                              placement_w, state, live, keys, residue, pol32,
                              int(rank), int(segment_tokens), collect=False,
                              warps=_seq_warps(int(rank), payload.shape[-1]),
                              stages=3)
    return (out["reads"], out["state"], out["live"], out["keys"], out["slot"],
            out["op"], out["erase"], out["live_after"], out["payload"],
            out["live_pre"], out["ckpt_state"], out["ckpt_live"],
            out["ckpt_keys"])


@_k12_scan_fwd_op.register_fake
def _(payload, write_keys, query_keys, decay, placement_w, state, live, keys,
      residue, rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break,
      scale):
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


def _k12_scan_setup(ctx, inputs, output):
    payload, write_keys, query_keys, decay, placement_w, state, live, keys, \
        residue, rank, segment_tokens, tau_match, tau_merge, tau_pop, \
        tie_break, scale = inputs
    # the backward needs the inputs' values and the walk's own recorded
    # channels; the estimator additionally replays the key and residue tables
    # from their window-entry values (saved here), and needs placement_w.
    ctx.save_for_backward(payload, write_keys, query_keys, decay, placement_w,
                          state, live, keys, residue, *output[4:])
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)


@torch.library.custom_op("x45::k12_scan_bwd", mutates_args=())
def _k12_scan_bwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor, query_keys: torch.Tensor,
    decay: torch.Tensor, placement_w: torch.Tensor, keys: torch.Tensor,
    residue: torch.Tensor, dreads: torch.Tensor, slot: torch.Tensor,
    op: torch.Tensor, erase: torch.Tensor, live_after: torch.Tensor,
    payload_field: torch.Tensor, live_pre: torch.Tensor,
    ckpt_state: torch.Tensor, ckpt_live: torch.Tensor,
    ckpt_keys: torch.Tensor, rank: int, segment_tokens: int, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole window's adjoint as its own opaque op: K10's segment loop
    (imported verbatim) plus the estimator's reduction; decisions as recorded
    constants (the standing doctrine), the placement slot among them."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _k12_backward_body(payload, write_keys, query_keys, decay,
                              placement_w, dreads, schedule, live_pre,
                              ckpt_state, ckpt_live, ckpt_keys, keys, residue,
                              int(rank), int(segment_tokens), float(scale))


@_k12_scan_bwd_op.register_fake
def _(payload, write_keys, query_keys, decay, placement_w, keys, residue,
      dreads, slot, op, erase, live_after, payload_field, live_pre,
      ckpt_state, ckpt_live, ckpt_keys, rank, segment_tokens, scale):
    device = payload.device
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    return (torch.empty_like(payload), torch.empty_like(write_keys),
            torch.empty_like(query_keys),
            torch.empty((batch, tokens, heads), dtype=torch.float32,
                        device=device),
            torch.empty((placement_w.shape[0],), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, key_dim), dtype=torch.float32,
                        device=device))


def _k12_scan_backward(ctx, dreads, dstate_out, dlive, dkeys_out, dslot, dop,
                        derase, dlive_after, dpayload_field, dlive_pre,
                        dckpt_state, dckpt_live, dckpt_keys):
    """The adjoint: dpayload, dwrite_keys, dquery_keys, ddecay, dplacement_w
    (the admitted estimator), dstate, dkeys; the decision channel, the
    liveness and the residue carry no gradient by construction; the scalar
    policy arguments carry none by type."""
    payload, write_keys, query_keys, decay, placement_w, state, live, keys, \
        residue, slot, op, erase, live_after, payload_field, live_pre, \
        ckpt_state, ckpt_live, ckpt_keys = ctx.saved_tensors
    batch, tokens, heads, value_dim = payload.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=payload.device)
    dpayload, dwrite_keys, dquery_keys, ddecay, dplacement_w, dstate, dkeys \
        = _k12_scan_bwd_op(payload, write_keys, query_keys, decay, placement_w,
                           keys, residue, dreads, slot, op, erase, live_after,
                           payload_field, live_pre, ckpt_state, ckpt_live,
                           ckpt_keys, ctx.rank, ctx.segment_tokens, ctx.scale)
    return (dpayload, dwrite_keys, dquery_keys, ddecay, dplacement_w, dstate,
            None, dkeys, None,
            None, None, None, None, None, None, None)


torch.library.register_autograd("x45::k12_scan_fwd", _k12_scan_backward,
                                setup_context=_k12_scan_setup)


# --------------------------------------------------------------- the wrapper


def placement_scan(payload, write_keys, query_keys, decay, placement_w, *,
                   policy: RoutingPolicy, rank=None, rank_max=None,
                   segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                   initial_state=None, initial_live=None, initial_keys=None,
                   initial_residue=None, collect: bool = False) -> RoutedResult:
    """One whole (layer, window) separated-key scan at PER-TOKEN granularity
    with the learned free-slot placement at APPEND.

    The K10 wrapper's drop-in with the placement channel: the registered
    five-op policy evaluated against the CURRENT state and key table at every
    token, addressing on the key channel, the APPEND's free-slot choice by
    the placement policy.  The kernel path is ONE opaque op pair per scan;
    every refusal takes the EAGER ORACLE -- the same forward semantics, the
    parity fixed point, whose placement-weight gradient is the honest
    structural zero (the finding; the estimator lives on the kernel path,
    and the probe asserts that path is the one taken):

    * ``policy.ops=False`` (the append-only control) and ``policy.grow`` are
      refused outright (chunk-dial concepts); a rank prefix is refused by name;
    * a store the persistent walk cannot hold (``placement_available``), or
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
    if placement_w.shape != (heads,) or placement_w.dtype != torch.float32 \
            or not placement_w.is_contiguous():
        raise ValueError(f"placement_w must be a contiguous fp32 (heads,) "
                         f"tensor; got {tuple(placement_w.shape)} "
                         f"{placement_w.dtype}")
    if not policy.ops:
        raise ValueError("the append-only control is refused by the placement "
                         "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the placement registration: "
                         "growth is a chunk-boundary concept")
    if rank != rank_max:
        raise ValueError("the placement registration requires rank == "
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
    residue = (torch.zeros(batch, heads, rank_max, key_dim,
                           dtype=torch.float32, device=device)
               if initial_residue is None else initial_residue)
    if collect:
        pol32 = (_round_f32(policy.tau_match), _round_f32(policy.tau_merge),
                 _round_f32(policy.tau_pop), _round_f32(policy.tie_break),
                 _round_f32(policy.read_scale))
        with torch.no_grad():
            out = _k12_forward_launch(
                payload, write_keys, query_keys, decay, placement_w, state,
                live, keys, residue, pol32, rank, segment_tokens,
                collect=True, warps=_seq_warps(rank, value_dim), stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        return RoutedResult(out["reads"], out["state"], out["live"], rank,
                            ScanStats(), schedule, out["weights"])
    if (payload.is_cuda and placement_available(rank, value_dim, key_dim)):
        reads, state_out, live_out, _keys, _slot, _op, _erase, _live_after, \
            _payload, _live_pre, _ckpt_state, _ckpt_live, _ckpt_keys = \
            _k12_scan_fwd_op(payload, write_keys, query_keys, decay,
                             placement_w, state, live, keys, residue, rank,
                             int(segment_tokens), policy.tau_match,
                             policy.tau_merge, policy.tau_pop,
                             policy.tie_break, policy.read_scale)
        return RoutedResult(reads, state_out, live_out, rank, ScanStats(),
                            None, None)
    # the eager oracle: the same forward semantics, the parity fixed point
    # (its placement-weight gradient is the structural zero -- the finding)
    ref = placement_oracle_scan(payload, write_keys, query_keys, decay,
                                placement_w, policy=policy, rank=rank,
                                rank_max=rank_max, initial_state=state,
                                initial_live=live, initial_keys=keys,
                                initial_residue=residue, collect=False)
    return RoutedResult(ref["reads"], ref["state"], ref["live"], rank,
                        ScanStats(), None, None)


# ------------------------------------------------------- the mixer and the model


class PlacementSeparatedMixer(nn.Module):
    """The K12 mixer: the K10 separated mixer VERBATIM (its projections, its
    DECAY gate, its mimetic key birth) plus one per-head placement weight
    ``w`` born at EXACT ``+0.0`` (P5: at birth the mixer IS the banked K10
    winner bit for bit -- the zero-born argmax tie-break reproduces the
    first-free rule), trainable through the admitted straight-through
    estimator on the kernel path.  ``placement_trainable=False`` pins ``w``
    at ``placement_init``: the never-fires falsifier control at 0, the
    supervisor-ruled FIXED-w bracket arms at {0.25, 0.5, 1.0}."""

    def __init__(self, width: int, heads: int, value_dim: int, rank: int,
                 rank_max: int, key_dim: int, policy: RoutingPolicy,
                 ctx: int = DEFAULT_CONTEXT,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                 placement_init: float = 0.0,
                 placement_trainable: bool = True):
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
        self.placement_weight = nn.Parameter(
            torch.full((heads,), float(placement_init)),
            requires_grad=bool(placement_trainable))
        self.apply_mimetic_init()

    def apply_mimetic_init(self, seed: int = MIMETIC_SEED):
        """K10's mimetic key birth, verbatim: both key projections born AT
        the fixed orthonormal family -- only the CURRENT-token context
        position non-zero, identical for the write-key and query-key
        projections.  The placement weight is NOT part of the mimetic
        family: it is born at exact zero whatever its arm."""
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
        result = placement_scan(payload, write_keys, query_keys, decay,
                                self.placement_weight, policy=self.policy,
                                rank=self.rank, rank_max=self.rank_max,
                                segment_tokens=self.segment_tokens)
        return (self.out(result.reads.reshape(batch, tokens,
                                              self.heads * self.value_dim)),
                result)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mix(x)
        return out


def convert_model_to_placement(model, placement_init: float = 0.0,
                               placement_trainable: bool = True,
                               segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
    """Swap every ``SeparatedSequentialMixer`` for the placement one, sharing
    parameters (the K10/K11 conversion pattern: the ``nn.Linear`` modules the
    optimizer already saw are the very same objects), adding ONE per-head
    placement weight per mixer -- born at exact ``+0.0`` and trainable (the
    learned arm), or pinned at ``placement_init`` (the falsifier control and
    the fixed-w bracket).  Every value-side and key-side module is carried
    over UNTOUCHED, so at birth the converted model's discrete schedule is
    the banked K10 winner's bit for bit.  ``k10_separated_keys.py`` is not
    edited; this builds on its conversion's output.  The parameter DELTA is
    the registration's stated accounting (+48 at the registered shape),
    asserted by the probe, not hidden."""
    for block in model.blocks:
        mixer = block.mixer
        if not isinstance(mixer, SeparatedSequentialMixer):
            raise ValueError("convert_model_to_placement builds on "
                             "convert_model_to_separated_keys' output; got "
                             f"{type(mixer).__name__}")
        pm = PlacementSeparatedMixer.__new__(PlacementSeparatedMixer)
        torch.nn.Module.__init__(pm)
        pm.__dict__.update(mixer.__dict__)
        pm._modules = dict(mixer._modules)     # own copy; the old mixer keeps its
        pm.segment_tokens = int(segment_tokens)
        pm.placement_weight = nn.Parameter(
            torch.full((mixer.heads,), float(placement_init)),
            requires_grad=bool(placement_trainable))
        block.mixer = pm
    return model


def placement_parameter_delta(depth: int, heads: int) -> int:
    """The registration's parameter accounting, computed not hardcoded: one
    placement weight per head per mixer."""
    return depth * heads


__all__ = ["DEFAULT_KEY_DIM", "DEFAULT_CONTEXT", "DEFAULT_SEGMENT_TOKENS",
           "MIMETIC_SEED", "RoutedResult", "ScanStats", "MatrixSchedule",
           "RoutingPolicy", "EPS", "APPEND", "CORRECT", "MERGE", "NONE",
           "POP", "PlacementSeparatedMixer", "convert_model_to_placement",
           "convert_model_to_separated_keys", "placement_available",
           "placement_oracle_scan", "placement_scan",
           "placement_parameter_delta", "separated_available",
           "_k12_forward_launch", "_k12_backward_body",
           "_k12_placement_grad_launch"]
