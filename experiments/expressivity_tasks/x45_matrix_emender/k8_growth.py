"""K8 -- GROWTH WIRED onto the untangled machine: the registered growth rule on
the per-token sequential arm, with the operator's target (transformer-parity
addressable state) as the question the probe exists to answer.

Why this file exists
--------------------
No run in the program's history has ever had ``grow=True``.  The rule has been
registered from the beginning (``x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER
.md``: growth "at chunk boundaries only", ``r' = max(r + stride_floor, ceil(g*r))``
snapped to the precompiled set {64..2048}, trigger "a deterministic chunk
statistic, fixed before the run (write-residual mass and/or append-collision
rate), reported per chunk"), its legality is PROVEN in Lean (zero-injection:
``LiftedMemory`` / ``injection_propagation`` -- new rows enter as exact zeros and
the monoid and scan theorems hold; ``formal/lean/OPERATOR_ALGEBRA_STATUS.md``),
the registered implementation exists (``scan.repeat_write_rate`` +
``keys.growth_target`` inside ``scan.routed_chunk_scan``), and both K6 and K7
were built growth-ready (schedule-recorded shapes, per-snap constexpr kernels,
eager boundary re-dispatch).  What was missing is the WIRING on the per-token
arm -- whose own draft registration refused growth as a chunk-boundary concept
it did not carry -- and the trainable high-rank path growth needs (K7 is
forward-only; K6's backward kernels and K4's trajectory replay are
register-capped at rank 128 / 512).

The registered form, taken verbatim (operator-confirmed this session)
---------------------------------------------------------------------
* **Granularity**: the trigger is evaluated at fixed 64-token statistic
  boundaries MID-WINDOW (64 = the registered ``DEFAULT_CHUNK_TOKENS`` and the K6
  segment length).  A fire re-dispatches the scan to the grown rank AT THAT
  BOUNDARY -- the eager boundary re-dispatch K6/K7 were built ready for.
* **Signal**: the registered append-collision statistic, ``scan.repeat_write_rate``
  IMPORTED AND CALLED VERBATIM (the share of writes reusing a slot written
  within the last ``grow_window_tokens``=512 tokens), smoothed over the last
  ``grow_window``=4 boundaries, firing at mean >= ``grow_trigger``, window and
  ``last_write`` reset on fire -- ``routed_chunk_scan``'s exact bookkeeping --
  then ``r' = min(cap, growth_target(r, geometric, stride_floor))``.
* **Zero-injection**: at a fire the state tensor grows (new rows exact zeros,
  liveness off until first written) -- the legality-proven operation.  No
  physical shrink anywhere (POP/MERGE shrink content; rank never shrinks).

The wiring: two passes per (layer, window) scan
-----------------------------------------------
``growing_scan`` runs a PROBE pass then the REAL pass:

1. **The probe pass** (no grad): span-walks the window at the current rank --
   K6's ``_seq_forward_launch`` at rank <= 128, K7's ``_k7_walk_kernel`` above
   (launched directly; it is imported, never edited) -- evaluating the trigger
   at every 64-token boundary from the walk's own recorded schedule until it
   fires.  The boundary state comes from the walk's segment checkpoints
   (rank <= 128) or a schedule replay (``_k8_traj_kernel``, above).  The output
   is the SPAN TABLE -- the list of ``(t0, t1, rank)`` the real pass replays --
   plus the growth events and the per-boundary rates ("reported per chunk").
2. **The real pass**: the span table through autograd.  Spans at rank <= 128
   call K6's op pair ``x45::seq_scan_fwd`` VERBATIM on the slice (zero new
   kernels; ``grow``-never-fires is bit-equal to the banked K6 path by
   construction).  Spans above 128 call the new ``x45::k8_hi_scan_fwd/bwd``
   pair.  Spans chain through zero-injection pads (``F.pad``; the pad's own
   backward slices the old rows and drops the zero rows' gradient -- exactly
   the adjoint of the linear embedding).  The probe and real passes run the
   SAME kernels on the SAME inputs, so the real span's recorded schedule is
   bit-equal to the probe's prefix -- asserted in the tests, and the reason the
   re-dispatch needs no mid-walk host sync.

The high-rank backward (the K7-style re-grid of K6's adjoint)
--------------------------------------------------------------
K6's backward is the segment-recompute doctrine with two whole-rank register
kernels; at rank 2048 their tiles are 32 768+ floats per program.  K8 re-grids
them over SLOT BLOCKS -- the K7 discipline -- with every cross-block scalar
combined in ONE fixed-order torch tree (the Lean full-population law: a tile's
internal defined tree plus one tree over the tile results IS the whole-axis
tree, zeros padding exactly):

* ``_k8_traj_kernel`` -- the slot-blocked schedule replay (K1's walk at ANY
  rank): one program per (row, slot-tile) holds a 64x128 register tile and
  applies the recorded schedule (DECAY ``mul.rn``, lanes ``add.rn``, single-row
  writes).  Every op is column- and slot-independent given the schedule, so
  the blocking is bit-exact against the K6/K7 forward walks by construction.
  Three modes: exit-only (the probe's boundary extraction), per-token
  trajectory (the backward's 64-token segment recompute) and per-64-token
  checkpoints (the span's segment entries, recomputed in the backward instead
  of saved in the forward -- at rank 2048 forward-saved checkpoints would be
  ~554 MB per scan and ~13 GB per microbatch; one extra K1-class replay per
  span is the honest price, receipted in gate D).
* ``_k8_prep_scores/mass/weights/adj_kernel`` -- the per-token read weights and
  adjoint coefficients recomputed from the trajectory, in four phases around
  the torch combines (the softmax denominator, and the ``dsum``/``duu`` adjoint
  scalars, span the rank).  The per-row expressions are K6's own (value-axis
  trees unchanged), so the recomputed weights are BIT-EQUAL to the forward's
  own collected channel -- asserted in the gates.
* ``_k8_bwd_walk_kernel`` -- the reverse adjoint walk: one program per
  (row, value-block) sweeps the slot tiles per token in two sub-sweeps (the
  lane zeroing + the single-nonzero ``dpay0`` gather + the blend weights; then
  the DECAY adjoint, the MERGE scatter and the read adjoint), the adjoint
  carried in global memory, ``dlam``/``duq`` as register accumulators stored
  once per token, CTA barriers between the sweeps (the K7 cross-warp lesson),
  determinism asserted across runs.

Floor discipline, stated: spans at rank <= 128 keep K6's banked gate-B status
verbatim (this file does not touch their kernels).  Above 128 the adjoint's
rank-axis reductions (``duq``, ``dlam``) are tile sums sequenced across slot
blocks -- an order no torch expression reproduces exactly -- so the K8
gradients are held to the established FLOOR gates (kernel vs the eager oracle
that replays the SAME recorded span table; floor = the fp32 oracle against its
own fp64 twin; ``NOISE_FACTOR`` x), never bit-equality.  Decisions are
recorded constants throughout (the standing doctrine), and the growth
decisions themselves are host-side constants of the run -- deterministic
functions of the recorded schedule -- so nothing ever differentiates through a
trigger.

``oracle_growing_scan`` replays a recorded span table with the eager per-token
oracles (K6's below the register gate, K7's above), attached -- the gate-B
fixed point of the whole growing chain.

New file only: the ``k6_*``/``k7_*`` sources, both preregistrations and
``lattice.py`` are untouched (imported).
"""
from __future__ import annotations

import dataclasses
from typing import List, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .algebra import APPEND, CORRECT, NONE, MatrixSchedule
from .k1_schedule_scan import NONE_CODE, _add_rn, _mul_rn
from .k4_fused_backward import persistent_scan_trajectory
from .k6_sequential_matrix import (
    BT_BWD, BV_BWD, DEFAULT_SEGMENT_TOKENS, _check_seq_shapes, _k6_bwd_prep_kernel,
    _k6_bwd_walk_kernel, _log2_up, _pow2, _round_f32, _seq_forward_launch,
    _seq_warps, sequential_available, sequential_oracle_scan,
    sequential_scan, tree_fold)
from .k7_high_rank_store import (
    K7_WALK_TILE, MAX_WALK_TILE, MAX_WALK_VALUE_DIM, _div_rn32, _fold_axis2,
    _fold_vec, _k7_walk_kernel, _sqrt_rn32, sequential_oracle)
from .keys import MAX_RANK, ORTHONORMAL_RANKS, growth_target, snap_rank
from .model import DECAY_FLOOR, MatrixMixer
from .scan import repeat_write_rate
from .schedule import EPS, RoutingPolicy

# The trigger/statistic boundary: the registered chunk length (``scan.py``'s
# DEFAULT_CHUNK_TOKENS) and the K6 segment length.  Growth decisions live here;
# everything finer stays per-token (K6's semantics, untouched).
GROW_BOUNDARY_TOKENS = 64
# The slot tile every K8 kernel holds: BR x VP fp32 in registers, the lane's
# measured no-spill tile (K7's own).
K8_BR = 64
# The backward walk's value block (the K4/K6 partials discipline).
K8_BV = 16
# The prep kernels' token block.
K8_BT = 16
# The rank K6's register walk refuses beyond (its own gate, imported).
K6_RANK_LIMIT = 128

_EPS = tl.constexpr(EPS)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))


# ============================================================== the replay kernel


@triton.jit(do_not_specialize=['T', 'R', 'ROWS'])
def _k8_traj_kernel(
    state_in, live_in, decay, slot, op, erase, live_after, payload,
    state_out, live_out, traj, ckpt_state, ckpt_live,
    T, R, ROWS,
    dbs, dts, dhs,
    sbs, sts, shs, sls,
    pbs, pst, psh, psl,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BR: tl.constexpr, L: tl.constexpr,
    SEG: tl.constexpr, MODE: tl.constexpr, STAGES: tl.constexpr,
):
    """The slot-blocked schedule replay: K1's walk at ANY rank, in three modes.

    One program per (row, slot-tile) holds a ``BR x VP`` register tile and walks
    the recorded schedule: per token the pre-DECAY state is captured (mode 1),
    then DECAY (``mul.rn``), then every lane in order -- a single-row write:
    the payload verbatim for an erasing lane, ``add_rn`` for an additive one;
    ``live_after`` for the liveness (the K4 trajectory kernel's own guarded
    form).  All of it is column- and slot-independent given the schedule, so
    the blocking is bit-exact against the K6/K7 forward walks.

    ``MODE`` 0: the exit state only (the probe's boundary extraction).
    ``MODE`` 1: the per-token pre-DECAY trajectory into ``traj``
                ``(rows, T, R, V)`` (the backward's segment recompute -- the
                state each token's read saw, K4's own contract).
    ``MODE`` 2: the per-``SEG``-token entry checkpoints into
                ``ckpt_state``/``ckpt_live`` ``(nseg, rows, R, V)``.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    ranks = block * BR + tl.arange(0, BR)
    keep_r = ranks < R
    values = tl.arange(0, VP)
    keep_v = values < V
    keep = keep_r[:, None] & keep_v[None, :]

    acc = tl.load(state_in + row.to(tl.int64) * R * V
                  + ranks[:, None] * V + values[None, :], mask=keep, other=0.0)
    live = tl.load(live_in + row.to(tl.int64) * R + ranks, mask=keep_r,
                   other=0) != 0
    decay_row = decay + batch_index * dbs + head_index * dhs
    channel_base = batch_index.to(tl.int64) * sbs + head_index * shs
    payload_base = payload + batch_index.to(tl.int64) * pbs + head_index * psh
    traj_base = traj + row.to(tl.int64) * T * R * V
    ck_row = row.to(tl.int64)

    for t in tl.range(0, T, num_stages=STAGES):
        if MODE == 1:
            tl.store(traj_base + t.to(tl.int64) * R * V
                     + ranks[:, None] * V + values[None, :], acc, mask=keep)
        if MODE == 2:
            if t % SEG == 0:
                seg = t // SEG
                tl.store(ckpt_state + (seg * ROWS + ck_row) * R * V
                         + ranks[:, None] * V + values[None, :], acc,
                         mask=keep)
                tl.store(ckpt_live + (seg * ROWS + ck_row) * R + ranks,
                         live.to(tl.int8), mask=keep_r)
        lam = tl.load(decay_row + t * dts).to(tl.float32)
        acc = _mul_rn(acc, lam)
        for lane in tl.static_range(L):
            channel = t * sts + lane * sls
            if tl.load(op + channel_base + channel) != _NONE_CODE:
                target = tl.load(slot + channel_base + channel).to(tl.int32)
                hit = keep_r & (target == ranks)
                written = tl.load(payload_base + t * pst + lane * psl + values,
                                  mask=keep_v, other=0.0)
                if tl.load(erase + channel_base + channel) != 0:
                    acc = tl.where(hit[:, None], written[None, :], acc)
                else:
                    acc = tl.where(hit[:, None],
                                   _add_rn(acc, written[None, :]), acc)
                live = tl.where(hit,
                                tl.load(live_after + channel_base + channel) != 0,
                                live)
    tl.store(state_out + row.to(tl.int64) * R * V + ranks[:, None] * V
             + values[None, :], acc, mask=keep)
    tl.store(live_out + row.to(tl.int64) * R + ranks, live.to(tl.int8),
             mask=keep_r)


def _launch_traj(state, live, decay, schedule, state_out, live_out, traj,
                 ckpt_state, ckpt_live, tokens, rank, rows, *, heads, mode,
                 seg_tokens, warps=4, stages=2):
    """The one launcher for all three replay modes (strides passed, slices OK).

    ``heads`` is explicit because MODE 1 takes its entry as a ``(rows, R, V)``
    checkpoint slab, where the head count is not recoverable from the shape.
    The DECAY GATE is upcast to fp32 here (K6's body and K7's kernel both do
    the same): a bf16 ``lam`` fed to the f32 ``mul.rn`` intrinsic is not a
    value-correct promotion on this stack -- the x44 §6.2 class, found here
    again as a 1e6-scale ddecay blowup under the training regime's autocast.
    """
    value_dim = state.shape[-1]
    decay = decay.to(torch.float32)
    _k8_traj_kernel[(rows, triton.cdiv(_pow2(rank), K8_BR))](
        state, live.view(torch.int8), decay,
        schedule.slot, schedule.op, schedule.erase.view(torch.int8),
        schedule.live_after.view(torch.int8), schedule.payload,
        state_out, live_out.view(torch.int8), traj, ckpt_state,
        ckpt_live.view(torch.int8),
        int(tokens), int(rank), int(rows),
        *decay.stride(),
        *schedule.slot.stride(),
        *schedule.payload.stride()[:4],
        H=heads, V=value_dim, VP=_pow2(value_dim), BR=K8_BR,
        L=schedule.lanes, SEG=int(seg_tokens), MODE=mode, STAGES=stages,
        num_warps=warps)


def traj_exit_state(state, live, decay, schedule, rank):
    """MODE 0: the exit state/liveness of replaying ``schedule`` from a state."""
    tokens = schedule.slot.shape[1]
    rows = state.shape[0] * state.shape[1]
    device = state.device
    state_out = torch.empty_like(state)
    live_out = torch.empty_like(live)
    dummy = torch.empty(1, dtype=torch.float32, device=device)
    dummy_live = torch.empty(1, dtype=torch.int8, device=device)
    _launch_traj(state, live, decay, schedule, state_out, live_out, dummy,
                 dummy, dummy_live, tokens, rank, rows,
                 heads=state.shape[1], mode=0, seg_tokens=1)
    return state_out, live_out


def traj_checkpoints(state, live, decay, schedule, rank, seg_tokens):
    """MODE 2: the per-``seg_tokens`` entry checkpoints of a whole span replay.

    Returns ``(exit_state, exit_live, ckpt_state (nseg, rows, R, V),
    ckpt_live (nseg, rows, R))`` -- the backward's segment entries, recomputed
    rather than forward-saved (the memory arithmetic in the module docstring).
    """
    tokens = schedule.slot.shape[1]
    batch, heads = state.shape[0], state.shape[1]
    rows = batch * heads
    value_dim = state.shape[-1]
    nseg = -(-tokens // int(seg_tokens))
    device = state.device
    state_out = torch.empty_like(state)
    live_out = torch.empty_like(live)
    ckpt_state = torch.empty((nseg, rows, int(rank), value_dim),
                             dtype=torch.float32, device=device)
    ckpt_live = torch.empty((nseg, rows, int(rank)), dtype=torch.bool,
                            device=device)
    dummy = torch.empty(1, dtype=torch.float32, device=device)
    _launch_traj(state, live, decay, schedule, state_out, live_out, dummy,
                 ckpt_state, ckpt_live, tokens, rank, rows,
                 heads=heads, mode=2, seg_tokens=seg_tokens)
    return state_out, live_out, ckpt_state, ckpt_live


def traj_segment(entry_state, entry_live, decay_seg, schedule_seg, rank,
                 heads):
    """MODE 1: the per-token pre-DECAY trajectory of one segment's replay.

    ``entry_state``/``entry_live`` are ``(rows, R, V)``/``(rows, R)`` slabs (a
    checkpoint), ``decay_seg``/``schedule_seg`` the segment's slices.  Returns
    ``traj (rows, segt, R, V)`` plus the exit state/live.
    """
    tokens = schedule_seg.slot.shape[1]
    rows = entry_state.shape[0]
    value_dim = entry_state.shape[-1]
    device = entry_state.device
    traj = torch.empty((rows, tokens, int(rank), value_dim),
                       dtype=torch.float32, device=device)
    state_out = torch.empty_like(entry_state)
    live_out = torch.empty_like(entry_live)
    dummy = torch.empty(1, dtype=torch.float32, device=device)
    dummy_live = torch.empty(1, dtype=torch.int8, device=device)
    _launch_traj(entry_state, entry_live, decay_seg, schedule_seg, state_out,
                 live_out, traj, dummy, dummy_live, tokens, rank, rows,
                 heads=heads, mode=1, seg_tokens=tokens)
    return traj, state_out, live_out


# ============================================================== the prep kernels


@triton.jit(do_not_specialize=['SEGT', 'R', 'T0'])
def _k8_prep_scores_kernel(
    traj, query, dout, live_pre,
    nr_ch, scores_ch, dw_ch, nq_ch, maxes_ch,
    SEGT, R, T0,
    qbs, qts, qhs, qvs,
    gbs, gts, ghs, gvs,
    lbs, lts, lhs, lrs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BT: tl.constexpr, BR: tl.constexpr, SB: tl.constexpr,
    LOGV: tl.constexpr, SCALE: tl.constexpr,
):
    """Prep phase 1: the per-row norms, read scores, ``dw`` and the tile maxes.

    K6's prep expressions per row, unchanged (the value-axis defined trees);
    the rank axis is the grid's slot-block axis.  The score channel is RAW
    (finite at dead rows too -- phase 4's ``z`` needs it); the live mask is
    applied where the softmax consumes it.
    """
    row = tl.program_id(0)
    tok_block = tl.program_id(1)
    sb = tl.program_id(2)
    batch_index = row // H
    head_index = row % H
    ranks = sb * BR + tl.arange(0, BR)
    keep_r = ranks < R
    values = tl.arange(0, VP)
    keep_v = values < V
    traj_base = traj + row.to(tl.int64) * SEGT * R * V
    ch_base = row.to(tl.int64) * SEGT * R
    q_row = query + batch_index * qbs + head_index * qhs
    g_row = dout + batch_index * gbs + head_index * ghs
    live_row = live_pre + batch_index.to(tl.int64) * lbs + head_index * lhs
    for i in tl.range(0, BT):
        local = tok_block * BT + i
        if local < SEGT:
            t = T0 + local
            q = tl.load(q_row + t * qts + values * qvs, mask=keep_v,
                        other=0.0).to(tl.float32)
            g = tl.load(g_row + t * gts + values * gvs, mask=keep_v,
                        other=0.0)
            live_b = tl.load(live_row + t * lts + ranks * lrs, mask=keep_r,
                             other=0) != 0
            acc = tl.load(traj_base + local.to(tl.int64) * R * V
                          + ranks[:, None] * V + values[None, :],
                          mask=keep_r[:, None] & keep_v[None, :], other=0.0)
            nr = tl.reshape(_sqrt_rn32(_fold_axis2(_mul_rn(acc, acc), VP,
                                                   LOGV)), (BR,))
            nq = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(q, q), VP, LOGV)),
                            (1,))
            uq = _div_rn32(q, tl.broadcast_to(tl.maximum(nq, _EPS), (VP,)))
            score = _div_rn32(
                _mul_rn(tl.reshape(_fold_axis2(_mul_rn(uq[None, :], acc), VP,
                                                LOGV), (BR,)), SCALE),
                tl.maximum(nr, _EPS))
            dw = tl.reshape(_fold_axis2(_mul_rn(g[None, :], acc), VP, LOGV),
                            (BR,))
            ch = ch_base + local.to(tl.int64) * R + ranks
            tl.store(nr_ch + ch, nr, mask=keep_r)
            tl.store(scores_ch + ch, score, mask=keep_r)
            tl.store(dw_ch + ch, dw, mask=keep_r)
            if sb == 0:
                tl.store(nq_ch + row.to(tl.int64) * SEGT + local
                         + tl.arange(0, 1), tl.reshape(nq, (1,)))
            masked = tl.where(live_b, score, float('-inf'))
            tl.store(maxes_ch + (row.to(tl.int64) * SEGT + local) * SB + sb,
                     tl.max(masked, 0))


@triton.jit(do_not_specialize=['SEGT', 'R', 'T0'])
def _k8_prep_mass_kernel(
    scores_ch, live_pre, gmax_ch, masses_ch,
    SEGT, R, T0,
    lbs, lts, lhs, lrs,
    H: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr, SB: tl.constexpr,
    LOGBR: tl.constexpr,
):
    """Prep phase 2: the per-block softmax masses, THE explicit scalars.

    ``m_b = defined tree over the tile of exp(score - gmax)`` -- dead rows load
    as ``-inf`` after the live mask, so their ``exp`` is exactly 0 and the
    zero padding of the tile tree is exact (the Lean law's carrier).
    """
    row = tl.program_id(0)
    tok_block = tl.program_id(1)
    sb = tl.program_id(2)
    batch_index = row // H
    head_index = row % H
    ranks = sb * BR + tl.arange(0, BR)
    keep_r = ranks < R
    live_row = live_pre + batch_index.to(tl.int64) * lbs + head_index * lhs
    ch_base = row.to(tl.int64) * SEGT * R
    for i in tl.range(0, BT):
        local = tok_block * BT + i
        if local < SEGT:
            t = T0 + local
            live_b = tl.load(live_row + t * lts + ranks * lrs, mask=keep_r,
                             other=0) != 0
            score = tl.load(scores_ch + ch_base + local.to(tl.int64) * R
                            + ranks, mask=keep_r, other=0.0)
            masked = tl.where(live_b, score, float('-inf'))
            gmax = tl.load(gmax_ch + row.to(tl.int64) * SEGT + local)
            expo = libdevice.exp(masked - gmax)
            tl.store(masses_ch + (row.to(tl.int64) * SEGT + local) * SB + sb
                     + tl.arange(0, 1),
                     tl.reshape(_fold_vec(expo, BR, LOGBR), (1,)))


@triton.jit(do_not_specialize=['SEGT', 'R', 'T0'])
def _k8_prep_weights_kernel(
    scores_ch, nr_ch, dw_ch, live_pre, gmax_ch, denom_ch,
    probs_ch, w_ch, dprobs_ch, pdsum_ch,
    SEGT, R, T0,
    lbs, lts, lhs, lrs,
    H: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr, SB: tl.constexpr,
    LOGBR: tl.constexpr,
):
    """Prep phase 3: ONE normalisation by the full-population mass.

    ``probs = exp(score - gmax)/denom`` (exactly 0 where ``denom == 0`` -- the
    certified all-dead zero attention), ``w = probs * strength`` (the
    registered expression), and the ``dsum`` adjoint's per-block partials
    ``fold(dprobs * probs)``.
    """
    row = tl.program_id(0)
    tok_block = tl.program_id(1)
    sb = tl.program_id(2)
    batch_index = row // H
    head_index = row % H
    ranks = sb * BR + tl.arange(0, BR)
    keep_r = ranks < R
    live_row = live_pre + batch_index.to(tl.int64) * lbs + head_index * lhs
    ch_base = row.to(tl.int64) * SEGT * R
    for i in tl.range(0, BT):
        local = tok_block * BT + i
        if local < SEGT:
            t = T0 + local
            live_b = tl.load(live_row + t * lts + ranks * lrs, mask=keep_r,
                             other=0) != 0
            score = tl.load(scores_ch + ch_base + local.to(tl.int64) * R
                            + ranks, mask=keep_r, other=0.0)
            nr = tl.load(nr_ch + ch_base + local.to(tl.int64) * R + ranks,
                         mask=keep_r, other=0.0)
            dw = tl.load(dw_ch + ch_base + local.to(tl.int64) * R + ranks,
                         mask=keep_r, other=0.0)
            gmax = tl.load(gmax_ch + row.to(tl.int64) * SEGT + local)
            denom = tl.load(denom_ch + row.to(tl.int64) * SEGT + local)
            masked = tl.where(live_b, score, float('-inf'))
            expo = libdevice.exp(masked - gmax)
            probs = tl.where(denom > 0, _div_rn32(expo, denom), 0.0)
            strength = _mul_rn(nr, tl.where(live_b, 1.0, 0.0))
            w = _mul_rn(probs, strength)
            dprobs = _mul_rn(dw, strength)
            ch = ch_base + local.to(tl.int64) * R + ranks
            tl.store(probs_ch + ch, probs, mask=keep_r)
            tl.store(w_ch + ch, w, mask=keep_r)
            tl.store(dprobs_ch + ch, dprobs, mask=keep_r)
            tl.store(pdsum_ch + (row.to(tl.int64) * SEGT + local) * SB + sb
                     + tl.arange(0, 1),
                     tl.reshape(_fold_vec(_mul_rn(dprobs, probs), BR, LOGBR),
                                (1,)))


@triton.jit(do_not_specialize=['SEGT', 'R', 'T0'])
def _k8_prep_adj_kernel(
    probs_ch, dprobs_ch, scores_ch, dw_ch, dsum_ch,
    dscores_ch, dstrength_ch, z_ch, pduu_ch,
    SEGT, R, T0,
    H: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr, SB: tl.constexpr,
    LOGBR: tl.constexpr,
):
    """Prep phase 4: the adjoint coefficients from the combined scalars.

    ``dscores = probs * (dprobs - dsum)``, ``dstrength = dw * probs``,
    ``z = dscores * scores`` (K6's own expressions; ``scores`` raw) and the
    ``duu`` partials ``fold(z)``.
    """
    row = tl.program_id(0)
    tok_block = tl.program_id(1)
    sb = tl.program_id(2)
    ranks = sb * BR + tl.arange(0, BR)
    keep_r = ranks < R
    ch_base = row.to(tl.int64) * SEGT * R
    for i in tl.range(0, BT):
        local = tok_block * BT + i
        if local < SEGT:
            probs = tl.load(probs_ch + ch_base + local.to(tl.int64) * R
                            + ranks, mask=keep_r, other=0.0)
            dprobs = tl.load(dprobs_ch + ch_base + local.to(tl.int64) * R
                             + ranks, mask=keep_r, other=0.0)
            score = tl.load(scores_ch + ch_base + local.to(tl.int64) * R
                            + ranks, mask=keep_r, other=0.0)
            dw = tl.load(dw_ch + ch_base + local.to(tl.int64) * R + ranks,
                         mask=keep_r, other=0.0)
            dsum = tl.load(dsum_ch + row.to(tl.int64) * SEGT + local)
            dscores = _mul_rn(probs, dprobs - dsum)
            dstrength = _mul_rn(dw, probs)
            z = _mul_rn(dscores, score)
            ch = ch_base + local.to(tl.int64) * R + ranks
            tl.store(dscores_ch + ch, dscores, mask=keep_r)
            tl.store(dstrength_ch + ch, dstrength, mask=keep_r)
            tl.store(z_ch + ch, z, mask=keep_r)
            tl.store(pduu_ch + (row.to(tl.int64) * SEGT + local) * SB + sb
                     + tl.arange(0, 1),
                     tl.reshape(_fold_vec(z, BR, LOGBR), (1,)))


# ========================================================= the reverse walk kernel


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0', 'R'])
def _k8_bwd_walk_kernel(
    traj, query, dout, decay, carry, live_pre,
    nr_ch, w_ch, dscores_ch, dstrength_ch, z_ch, scal,
    slot, op, erase,
    dquery, dpayload, dlam_partial,
    SEGT, T, T0, R, NVB,
    qbs, qts, qhs, qvs,
    gbs, gts, ghs, gvs,
    dbs, dts, dhs,
    lbs, lts, lhs, lrs,
    sbs, sts, shs, sls,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BR: tl.constexpr, NT: tl.constexpr, BV: tl.constexpr, L: tl.constexpr,
    SCALE: tl.constexpr,
):
    """The value-blocked, slot-swept reverse adjoint walk of one segment.

    One program per (row, value-block) owns the FULL rank axis for its value
    columns and walks the segment's tokens in reverse; the adjoint ``ds`` lives
    in global memory (the carry) and is swept per token in ``NT`` slot tiles,
    in TWO sub-sweeps (K6's per-token step order preserved exactly):

    sweep 1 -- the lanes: the MERGE-source zeroing, the single-nonzero
    ``dpay0`` gather (exact whatever the tile order -- only the tile owning
    ``slot0`` contributes), the blend weights ``fw``/``sw`` (the same
    single-nonzero exactness), and the erasing zero at ``slot0``;
    sweep 2 -- the DECAY adjoint's ``dlam`` partial (accumulated across tiles,
    stored once per token), ``ds <- lam * ds``, the MERGE-blend scatter, then
    the read adjoint's four state terms, ``duq`` and ``dquery``.

    ``dpay0``/``fw``/``sw``/``duq``/``dlam`` are register accumulators across
    the tile sweep -- no cross-PROGRAM traffic at all within a token; the CTA
    barriers close the cross-WARP store->load hazards between the sweeps (the
    K7 determinism lesson).  The rank-axis reductions that remain (``duq``,
    ``dlam``) are tile sums sequenced across blocks -- an order no torch
    expression reproduces exactly, which is why the gradient gates are floors.
    """
    row = tl.program_id(0)
    vblock = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    values = vblock * BV + tl.arange(0, BV)
    keep_v = values < V
    carry_base = carry + row.to(tl.int64) * R * V
    traj_base = traj + row.to(tl.int64) * SEGT * R * V
    ch_base = row.to(tl.int64) * SEGT * R
    scal_base = row.to(tl.int64) * SEGT * 2
    q_row = query + batch_index * qbs + head_index * qhs
    g_row = dout + batch_index * gbs + head_index * ghs
    decay_row = decay + batch_index * dbs + head_index * dhs
    live_row = live_pre + batch_index.to(tl.int64) * lbs + head_index * lhs
    channel_base = batch_index.to(tl.int64) * sbs + head_index * shs
    dq_row = dquery + batch_index.to(tl.int64) * T * H * V + head_index * V
    dp_row = dpayload + batch_index.to(tl.int64) * T * H * V + head_index * V
    ddl_row = dlam_partial + row.to(tl.int64) * NVB * SEGT

    for i in tl.range(0, SEGT):
        tt = SEGT - 1 - i
        t = T0 + tt
        op0 = tl.load(op + channel_base + t * sts + 0 * sls)
        op1 = tl.load(op + channel_base + t * sts + 1 * sls)
        slot0 = tl.load(slot + channel_base + t * sts + 0 * sls).to(tl.int32)
        second = tl.load(slot + channel_base + t * sts + 1 * sls).to(tl.int32)
        erase0 = tl.load(erase + channel_base + t * sts + 0 * sls) != 0
        merge = op1 != _NONE_CODE
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        lam = tl.load(decay_row + t * dts).to(tl.float32)
        nq = tl.load(scal + scal_base + tt * 2 + 0)
        duu = tl.load(scal + scal_base + tt * 2 + 1)
        q = tl.load(q_row + t * qts + values * qvs, mask=keep_v,
                    other=0.0).to(tl.float32)
        g = tl.load(g_row + t * gts + values * gvs, mask=keep_v, other=0.0)
        cq = tl.maximum(nq, _EPS)
        uq_v = _div_rn32(q, tl.broadcast_to(cq, (BV,)))

        # ---- sweep 1: the lanes' zeroing, the dpay0 gather, the blend weights.
        dpay0 = tl.zeros((BV,), dtype=tl.float32)
        fw = tl.zeros((1,), dtype=tl.float32)
        sw = tl.zeros((1,), dtype=tl.float32)
        for b in tl.range(0, NT):
            ranks = b * BR + tl.arange(0, BR)
            keep_r = ranks < R
            keep = keep_r[:, None] & keep_v[None, :]
            ds = tl.load(carry_base + ranks[:, None] * V + values[None, :],
                         mask=keep, other=0.0)
            live_b = tl.load(live_row + t * lts + ranks * lrs, mask=keep_r,
                             other=0) != 0
            hit = keep_r & (slot0 == ranks)
            hit2 = keep_r & (second == ranks)
            if merge:
                ds = tl.where(hit2[:, None], 0.0, ds)
            dpay0 += tl.sum(tl.where(hit[:, None], ds, 0.0), 0)
            nr_b = tl.load(nr_ch + ch_base + tt.to(tl.int64) * R + ranks,
                           mask=keep_r, other=0.0)
            strength_b = _mul_rn(nr_b, tl.where(live_b, 1.0, 0.0))
            fw += tl.sum(tl.where(ranks == slot0, strength_b, 0.0), 0)
            sw += tl.sum(tl.where(ranks == second, strength_b, 0.0), 0)
            if erase0:
                ds = tl.where(hit[:, None], 0.0, ds)
            tl.store(carry_base + ranks[:, None] * V + values[None, :], ds,
                     mask=keep)
        tl.debug_barrier()
        fwm = tl.maximum(fw, _EPS)
        swm = tl.maximum(sw, _EPS)
        total = fwm + swm
        a_mix = _div_rn32(fwm, total)
        b_mix = _div_rn32(swm, total)

        # ---- sweep 2: the DECAY adjoint, the MERGE scatter, the read adjoint.
        duq = tl.zeros((BV,), dtype=tl.float32)
        dlam = tl.zeros((1,), dtype=tl.float32)
        for b in tl.range(0, NT):
            ranks = b * BR + tl.arange(0, BR)
            keep_r = ranks < R
            keep = keep_r[:, None] & keep_v[None, :]
            ds = tl.load(carry_base + ranks[:, None] * V + values[None, :],
                         mask=keep, other=0.0)
            acc = tl.load(traj_base + tt.to(tl.int64) * R * V
                          + ranks[:, None] * V + values[None, :], mask=keep,
                          other=0.0)
            live_b = tl.load(live_row + t * lts + ranks * lrs, mask=keep_r,
                             other=0) != 0
            nr_b = tl.load(nr_ch + ch_base + tt.to(tl.int64) * R + ranks,
                           mask=keep_r, other=0.0)
            w_b = tl.load(w_ch + ch_base + tt.to(tl.int64) * R + ranks,
                          mask=keep_r, other=0.0)
            dscores_b = tl.load(dscores_ch + ch_base + tt.to(tl.int64) * R
                                + ranks, mask=keep_r, other=0.0)
            dstrength_b = tl.load(dstrength_ch + ch_base + tt.to(tl.int64) * R
                                  + ranks, mask=keep_r, other=0.0)
            z_b = tl.load(z_ch + ch_base + tt.to(tl.int64) * R + ranks,
                          mask=keep_r, other=0.0)
            hit = keep_r & (slot0 == ranks)
            hit2 = keep_r & (second == ranks)
            dlam += tl.sum(tl.sum(_mul_rn(ds, acc), 1), 0)
            ds = _mul_rn(ds, lam)
            if merge:
                ds = tl.where(hit[:, None],
                              _add_rn(ds, _mul_rn(a_mix, dpay0)[None, :]), ds)
                ds = tl.where(hit2[:, None],
                              _add_rn(ds, _mul_rn(b_mix, dpay0)[None, :]), ds)
            ds = _add_rn(ds, _mul_rn(w_b[:, None], g[None, :]))
            cr = tl.maximum(nr_b, _EPS)
            dnr = _mul_rn(dstrength_b, tl.where(live_b, 1.0, 0.0))
            dnr = dnr + tl.where(nr_b >= _EPS,
                                 _div_rn32(_mul_rn(z_b, -1.0), cr), 0.0)
            unit_r = tl.where(nr_b[:, None] > 0,
                              _div_rn32(acc, nr_b[:, None]), 0.0)
            ds = _add_rn(ds, _mul_rn(dnr[:, None], unit_r))
            dt = _div_rn32(_mul_rn(dscores_b, SCALE), cr)
            ds = _add_rn(ds, _mul_rn(dt[:, None], uq_v[None, :]))
            duq += tl.sum(_mul_rn(dt[:, None], acc), 0)
            tl.store(carry_base + ranks[:, None] * V + values[None, :], ds,
                     mask=keep)
        tl.debug_barrier()
        tl.store(ddl_row + vblock * SEGT + tt + tl.arange(0, 1), dlam)
        dq = _div_rn32(duq, tl.broadcast_to(cq, (BV,))) \
            - _mul_rn(uq_v, tl.where(nq >= _EPS,
                                     tl.broadcast_to(_div_rn32(duu, nq),
                                                     (BV,)), 0.0))
        tl.store(dq_row + t * H * V + values, dq.to(dquery.dtype.element_ty),
                 mask=keep_v)
        dpay = tl.where(carries, dpay0, 0.0)
        tl.store(dp_row + t * H * V + values,
                 dpay.to(dpayload.dtype.element_ty), mask=keep_v)


# ================================================================= host wrappers


def _hi_walk_launch(query, payload, decay, state, live, pol32, rank, collect,
                    warps=8, stages=2):
    """The K7 walk kernel launched directly (imported, never edited): the hi
    path's forward at any snap, with the recorded channels."""
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    rows = batch * heads
    rp, vp = _pow2(rank), _pow2(value_dim)
    br = max(16, min(K7_WALK_TILE, MAX_WALK_TILE // vp))
    nt = triton.cdiv(rp, br)
    work = state.contiguous().clone()
    live_scr = live.contiguous().view(torch.int8).clone()
    reads = torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device)
    shape = (batch, tokens, heads, 2)
    slot = torch.empty(shape, dtype=torch.int64, device=device)
    op = torch.empty(shape, dtype=torch.int64, device=device)
    erase = torch.empty(shape, dtype=torch.bool, device=device)
    live_after = torch.empty(shape, dtype=torch.bool, device=device)
    payload_field = torch.empty((*shape, value_dim), dtype=torch.float32,
                                device=device)
    if collect:
        live_pre = torch.empty((batch, tokens, heads, int(rank)),
                               dtype=torch.bool, device=device)
        weights = torch.empty((batch, tokens, heads, int(rank)),
                              dtype=torch.float32, device=device)
    else:
        live_pre = torch.empty(1, dtype=torch.bool, device=device)
        weights = torch.empty(1, dtype=torch.float32, device=device)
    scratch = [torch.empty((rows, int(rank)), dtype=torch.float32,
                           device=device) for _ in range(4)]
    tau_match, tau_merge, tau_pop, tie, scale = pol32
    handle = _k7_walk_kernel[(rows,)](
        query, payload, decay, work, live_scr, reads, slot, op,
        erase.view(torch.int8), live_after.view(torch.int8), payload_field,
        live_pre.view(torch.int8), weights, *scratch,
        tokens, int(rank),
        *query.stride(), *payload.stride(), *decay.stride(),
        H=heads, V=value_dim, VP=vp, RP=rp, L=2, BR=br, NT=nt,
        LOGV=_log2_up(vp), LOGR=_log2_up(rp),
        TAU_MATCH=tau_match, TAU_MERGE=tau_merge, TAU_POP=tau_pop,
        TIE=tie, SCALE=scale, COLLECT=1 if collect else 0, STAGES=stages,
        num_warps=warps)
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return dict(reads=reads, state=work, live=live_scr.view(torch.bool),
                schedule=schedule, live_pre=live_pre, weights=weights,
                kernel=handle)


def _hi_prep(traj, query, dout, live_pre, t0, segt, rank, scale):
    """The four prep phases plus the ONE-tree torch combines, one segment.

    Returns the walk kernel's channels ``nr, w, dscores, dstrength, z`` and the
    scalars ``(nq, duu)`` per token.  The combines are the defined balanced
    tree over the slot-block axis -- bit-identical to the whole-axis tree
    (zeros pad exactly), the Lean full-population law.
    """
    rows, _, R = traj.shape[0], traj.shape[1], int(rank)
    value_dim = query.shape[-1]
    device = traj.device
    sb = triton.cdiv(_pow2(R), K8_BR)
    logbr = _log2_up(K8_BR)
    ch = lambda: torch.empty((rows, segt, R), dtype=torch.float32,
                             device=device)
    nr_ch, scores_ch, dw_ch, probs_ch, w_ch, dprobs_ch, dscores_ch, \
        dstrength_ch, z_ch = (ch() for _ in range(9))
    nq_ch = torch.empty((rows, segt), dtype=torch.float32, device=device)
    maxes = torch.empty((rows, segt, sb), dtype=torch.float32, device=device)
    grid = (rows, triton.cdiv(segt, K8_BT), sb)
    _k8_prep_scores_kernel[grid](
        traj, query, dout, live_pre, nr_ch, scores_ch, dw_ch, nq_ch, maxes,
        segt, R, int(t0),
        *query.stride(), *dout.stride(), *live_pre.stride(),
        H=query.shape[2], V=value_dim, VP=_pow2(value_dim), BT=K8_BT, BR=K8_BR,
        SB=sb, LOGV=_log2_up(_pow2(value_dim)),
        SCALE=_round_f32(scale), num_warps=4)
    # the max combine is exact (order-free); the all-dead guard is K6's own.
    live_any = live_pre[:, t0:t0 + segt].any(-1)          # (b, segt, h)
    live_any = live_any.permute(0, 2, 1).reshape(rows, segt)
    gmax = torch.where(live_any, maxes.amax(-1),
                      torch.zeros_like(maxes.amax(-1)))
    masses = torch.empty((rows, segt, sb), dtype=torch.float32,
                          device=device)
    _k8_prep_mass_kernel[grid](
        scores_ch, live_pre, gmax, masses,
        segt, R, int(t0),
        *live_pre.stride(),
        H=query.shape[2], BT=K8_BT, BR=K8_BR, SB=sb, LOGBR=logbr, num_warps=4)
    # THE ONE REDUCTION: the full-population exact mass.
    denom = tree_fold(masses, -1, sb)
    pdsum = torch.empty((rows, segt, sb), dtype=torch.float32, device=device)
    _k8_prep_weights_kernel[grid](
        scores_ch, nr_ch, dw_ch, live_pre, gmax, denom,
        probs_ch, w_ch, dprobs_ch, pdsum,
        segt, R, int(t0),
        *live_pre.stride(),
        H=query.shape[2], BT=K8_BT, BR=K8_BR, SB=sb, LOGBR=logbr, num_warps=4)
    dsum = tree_fold(pdsum, -1, sb)
    pduu = torch.empty((rows, segt, sb), dtype=torch.float32, device=device)
    _k8_prep_adj_kernel[grid](
        probs_ch, dprobs_ch, scores_ch, dw_ch, dsum,
        dscores_ch, dstrength_ch, z_ch, pduu,
        segt, R, int(t0),
        H=query.shape[2], BT=K8_BT, BR=K8_BR, SB=sb, LOGBR=logbr, num_warps=4)
    duu = tree_fold(pduu, -1, sb)
    scal = torch.stack((nq_ch, duu), dim=-1).contiguous()
    return dict(nr=nr_ch, w=w_ch, dscores=dscores_ch, dstrength=dstrength_ch,
                z=z_ch, scal=scal)


def _hi_backward_body(query, payload, decay, dreads, dstate_out, state, live,
                      schedule, live_pre, rank, segment_tokens, scale):
    """The whole hi-rank span's adjoint: checkpoints once, then segments in
    reverse -- recompute (traj) -> prep -> walk -- the gate-C doctrine with
    every shape read from the recorded span."""
    batch, tokens, heads, value_dim = query.shape
    rows = batch * heads
    R = int(rank)
    device = query.device
    nvb = triton.cdiv(value_dim, K8_BV)
    # the DECAY GATE upcast once, K6's own pattern: the kernels' ``mul.rn`` is
    # f32 and a bf16 ``lam`` is not a value-correct promotion on this stack.
    decay = decay.to(torch.float32)
    # 1. the span's segment-entry checkpoints, REPLAYED (not forward-saved):
    #    one extra K1-class walk per span, the memory arithmetic's price.
    _, _, ckpt_state, ckpt_live = traj_checkpoints(
        state, live, decay, schedule, R, segment_tokens)
    nseg = ckpt_state.shape[0]
    dquery = torch.empty(query.shape, dtype=query.dtype, device=device)
    dpayload = torch.empty(payload.shape, dtype=payload.dtype, device=device)
    ddecay = torch.empty((batch, heads, tokens), dtype=torch.float32,
                         device=device)
    # the span's exit-state gradient seeds the adjoint carry (the spans chain;
    # K6's own backward never needed this because its windows never chain).
    if dstate_out is None:
        carry = torch.zeros((batch, heads, R, value_dim), dtype=torch.float32,
                           device=device)
    else:
        # clone, not view: the bwd op must not return its own input
        carry = dstate_out.to(torch.float32).contiguous().clone()
    nt = triton.cdiv(_pow2(R), K8_BR)
    for seg in reversed(range(nseg)):
        start = seg * int(segment_tokens)
        stop = min(start + int(segment_tokens), tokens)
        segt = stop - start
        # 2. the segment's per-token trajectory from its entry checkpoint.
        traj, _, _ = traj_segment(ckpt_state[seg], ckpt_live[seg],
                                  decay[:, start:stop],
                                  schedule.chunk(start, stop), R, heads)
        # 3. prep: the weights and adjoint coefficients.
        prep = _hi_prep(traj, query, dreads, live_pre, start, segt, R, scale)
        # 4. the reverse walk.
        dlam_partial = torch.empty((rows, nvb, segt), dtype=torch.float32,
                                   device=device)
        _k8_bwd_walk_kernel[(rows, nvb)](
            traj, query, dreads, decay, carry, live_pre,
            prep["nr"], prep["w"], prep["dscores"], prep["dstrength"],
            prep["z"], prep["scal"],
            schedule.slot, schedule.op, schedule.erase.view(torch.int8),
            dquery, dpayload, dlam_partial,
            segt, tokens, start, R, nvb,
            *query.stride(), *dreads.stride(), *decay.stride(),
            *live_pre.stride(),
            *schedule.slot.stride(),
            H=heads, V=value_dim, VP=_pow2(value_dim), BR=K8_BR, NT=nt,
            BV=K8_BV, L=schedule.lanes, SCALE=_round_f32(scale), num_warps=4)
        ddecay[:, :, start:stop] = dlam_partial.sum(dim=1).view(batch, heads,
                                                                 segt)
    return dquery, dpayload, ddecay.permute(0, 2, 1), carry


# ------------------------------------------------------------ the hi op pair


@torch.library.custom_op("x45::k8_hi_scan_fwd", mutates_args=())
def _k8_hi_scan_fwd_op(
    query: torch.Tensor, payload: torch.Tensor, decay: torch.Tensor,
    state: torch.Tensor, live: torch.Tensor, rank: int,
    segment_tokens: int, tau_match: float, tau_merge: float, tau_pop: float,
    tie_break: float, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor]:
    """One hi-rank span's sequential scan as ONE opaque op: the K7 walk's every
    channel as outputs (the K4/K5A/K5B/K6 hosting pattern), so autograd itself
    keeps the backward's saves alive.  The entry state is an explicit input, so
    ``dstate`` is a real gradient output and the spans chain."""
    batch, tokens, heads, value_dim = query.shape
    if not (query.is_cuda and payload.is_cuda and decay.is_cuda
            and state.is_cuda):
        raise ValueError("the K8 hi scan is a device kernel; inputs must be "
                         "on CUDA")
    if payload.shape != query.shape:
        raise ValueError("query and payload disagree")
    if decay.shape != (batch, tokens, heads):
        raise ValueError(f"decay must be {(batch, tokens, heads)}, got "
                         f"{tuple(decay.shape)}")
    if state.shape != (batch, heads, int(rank), value_dim):
        raise ValueError(f"state {tuple(state.shape)} is not (batch, heads, "
                         f"rank, value_dim) = {(batch, heads, rank, value_dim)}")
    if live.shape != (batch, heads, int(rank)) or live.dtype != torch.bool:
        raise ValueError("live must be a bool (batch, heads, rank) tensor")
    if state.dtype != torch.float32 or not state.is_contiguous():
        raise ValueError("state must be contiguous fp32; the algebra is fp32")
    if not live.is_contiguous():
        raise ValueError("live must be contiguous")
    if value_dim > MAX_WALK_VALUE_DIM:
        raise ValueError(f"the walk's value ceiling is {MAX_WALK_VALUE_DIM}, "
                         f"got {value_dim}; refused by name")
    if tokens < 1 or int(segment_tokens) < 1:
        raise ValueError("tokens and segment_tokens must be positive")
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _hi_walk_launch(query, payload, decay, state, live, pol32,
                          int(rank), collect=True)
    return (out["reads"], out["state"], out["live"],
            out["schedule"].slot, out["schedule"].op,
            out["schedule"].erase, out["schedule"].live_after,
            out["schedule"].payload, out["live_pre"], out["weights"])


@_k8_hi_scan_fwd_op.register_fake
def _(query, payload, decay, state, live, rank, segment_tokens, tau_match,
      tau_merge, tau_pop, tie_break, scale):
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    shape = (batch, tokens, heads, 2)
    return (torch.empty((batch, tokens, heads, value_dim),
                        dtype=torch.float32, device=device),
            torch.empty_like(state),
            torch.empty_like(live),
            torch.empty(shape, dtype=torch.int64, device=device),
            torch.empty(shape, dtype=torch.int64, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty((*shape, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, tokens, heads, rank), dtype=torch.bool,
                        device=device),
            torch.empty((batch, tokens, heads, rank), dtype=torch.float32,
                        device=device))


def _k8_hi_scan_setup(ctx, inputs, output):
    query, payload, decay, state, live, rank, segment_tokens, tau_match, \
        tau_merge, tau_pop, tie_break, scale = inputs
    # the weights channel is NOT saved: the backward recomputes it (bit-equal,
    # the prep's own gate); saving it would pin a rank-wide fp32 channel per
    # span for nothing.
    ctx.save_for_backward(query, payload, decay, state, live, *output[3:9])
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)


@torch.library.custom_op("x45::k8_hi_scan_bwd", mutates_args=())
def _k8_hi_scan_bwd_op(
    query: torch.Tensor, payload: torch.Tensor, decay: torch.Tensor,
    dreads: torch.Tensor, dstate_out: torch.Tensor, state: torch.Tensor,
    live: torch.Tensor,
    slot: torch.Tensor, op: torch.Tensor, erase: torch.Tensor,
    live_after: torch.Tensor, payload_field: torch.Tensor,
    live_pre: torch.Tensor, rank: int, segment_tokens: int, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The hi-rank span's adjoint as its own opaque op: the checkpoint replay,
    the prep phases and the reverse walk run inside, decisions as recorded
    constants, the span's exit-state gradient seeding the carry."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _hi_backward_body(query, payload, decay, dreads, dstate_out,
                             state, live,
                             schedule, live_pre, int(rank), int(segment_tokens),
                             float(scale))


@_k8_hi_scan_bwd_op.register_fake
def _(query, payload, decay, dreads, dstate_out, state, live, slot, op, erase,
      live_after, payload_field, live_pre, rank, segment_tokens, scale):
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    return (torch.empty(query.shape, dtype=query.dtype, device=device),
            torch.empty(payload.shape, dtype=payload.dtype, device=device),
            torch.empty((batch, tokens, heads), dtype=torch.float32,
                        device=device),
            torch.empty_like(state))


def _k8_hi_scan_backward(ctx, dreads, dstate_out, dlive, dslot, dop, derase,
                         dlive_after, dpayload_field, dlive_pre, dweights):
    query, payload, decay, state, live, slot, op, erase, live_after, \
        payload_field, live_pre = ctx.saved_tensors
    batch, tokens, heads, value_dim = query.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=query.device)
    dquery, dpayload, ddecay, dstate = _k8_hi_scan_bwd_op(
        query, payload, decay, dreads,
        torch.zeros_like(state) if dstate_out is None else dstate_out,
        state, live, slot, op, erase,
        live_after, payload_field, live_pre, ctx.rank, ctx.segment_tokens,
        ctx.scale)
    # 12 inputs: query, payload, decay, state, live + 7 host constants.
    return (dquery, dpayload, ddecay, dstate, None, None, None, None, None,
            None, None, None)


torch.library.register_autograd("x45::k8_hi_scan_fwd", _k8_hi_scan_backward,
                                setup_context=_k8_hi_scan_setup)


# --------------------------------------------------------- the low-span op pair


def _k8_low_backward_body(query, payload, decay, dreads, dstate_out, schedule,
                          live_pre, ckpt_state, ckpt_live, rank,
                          segment_tokens, scale):
    """K6's ``_seq_backward_body`` VERBATIM (its own kernels, imported), with
    the one thing K6's windows never needed: the span's exit-state gradient
    seeds the segment-reverse adjoint carry, because K8's spans chain.  With
    ``dstate_out`` zero this is K6's own body exactly -- the same kernels, the
    same launch parameters, the same fixed-order combines."""
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    R = int(rank)
    SEG = int(segment_tokens)
    decay32 = decay.to(torch.float32).permute(0, 2, 1).contiguous()
    dreads_c = dreads.contiguous()
    dquery = torch.empty(query.shape, dtype=query.dtype, device=device)
    dpayload = torch.empty(payload.shape, dtype=payload.dtype, device=device)
    ddecay = torch.empty((batch, heads, tokens), dtype=torch.float32,
                         device=device)
    RP, VP = _pow2(R), _pow2(value_dim)
    LOGV, LOGR = _log2_up(VP), _log2_up(RP)
    nseg = -(-tokens // SEG)
    if dstate_out is None:
        carry = torch.zeros(batch, heads, R, value_dim, dtype=torch.float32,
                            device=device)
    else:
        # clone, not view: the bwd op must not return its own input
        carry = dstate_out.to(torch.float32).contiguous().clone()
    rows = batch * heads
    nblocks = triton.cdiv(value_dim, BV_BWD)
    for seg in reversed(range(nseg)):
        start = seg * SEG
        stop = min(start + SEG, tokens)
        segt = stop - start
        zeros_w = torch.zeros(batch, heads, segt, R, dtype=torch.float32,
                              device=device)
        _, _, _, traj = persistent_scan_trajectory(
            ckpt_state[seg], ckpt_live[seg], zeros_w,
            decay32[:, :, start:stop], schedule.chunk(start, stop))
        nr_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        w_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dscores_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                                 device=device)
        dstrength_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                                   device=device)
        z_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        scal = torch.empty((rows, segt, 2), dtype=torch.float32, device=device)
        grid = (rows, triton.cdiv(segt, BT_BWD))
        _k6_bwd_prep_kernel[grid](
            traj, query, dreads_c, live_pre.view(torch.int8),
            nr_ch, w_ch, dscores_ch, dstrength_ch, z_ch, scal,
            segt, tokens, start, R,
            *query.stride(), *dreads_c.stride(),
            H=heads, V=value_dim, VP=VP, RP=RP, LOGV=LOGV, LOGR=LOGR,
            BT=BT_BWD, SCALE=_round_f32(scale), num_warps=8)
        dstate_seg = torch.empty((batch, heads, R, value_dim),
                                 dtype=torch.float32, device=device)
        dlam_partial = torch.empty((rows, nblocks, segt), dtype=torch.float32,
                                   device=device)
        _k6_bwd_walk_kernel[(rows, nblocks)](
            traj, query, dreads_c, decay32, live_pre.view(torch.int8), carry,
            nr_ch, w_ch, dscores_ch, dstrength_ch, z_ch, scal,
            schedule.slot, schedule.op, schedule.erase.view(torch.int8),
            dquery, dpayload, dlam_partial, dstate_seg,
            segt, tokens, start, R, nblocks,
            *query.stride(), *dreads_c.stride(),
            H=heads, V=value_dim, VP=VP, RP=RP, LOGR=LOGR, BT=BT_BWD,
            BV=BV_BWD, SCALE=_round_f32(scale), num_warps=4)
        ddecay[:, :, start:stop] = dlam_partial.sum(dim=1).view(batch, heads,
                                                                segt)
        carry = dstate_seg
    return dquery, dpayload, ddecay.permute(0, 2, 1), carry


@torch.library.custom_op("x45::k8_span_scan_fwd", mutates_args=())
def _k8_span_scan_fwd_op(
    query: torch.Tensor, payload: torch.Tensor, decay: torch.Tensor,
    state: torch.Tensor, live: torch.Tensor, rank: int,
    segment_tokens: int, tau_match: float, tau_merge: float, tau_pop: float,
    tie_break: float, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor]:
    """One rank <= 128 span's sequential scan: K6's ``_seq_forward_launch``
    VERBATIM (bit-identical forward to K6's own op), re-hosted as a K8 op
    pair because K8's spans chain -- K6's own backward drops the exit-state
    gradient (its windows never chain), which is correct there and wrong
    here."""
    _check_seq_shapes(query, payload, decay, state, live, int(rank),
                      int(segment_tokens))
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _seq_forward_launch(query, payload, decay, state, live, pol32,
                              int(rank), int(segment_tokens), collect=False,
                              warps=_seq_warps(int(rank), query.shape[-1]),
                              stages=3)
    return (out["reads"], out["state"], out["live"], out["slot"], out["op"],
            out["erase"], out["live_after"], out["payload"], out["live_pre"],
            out["ckpt_state"], out["ckpt_live"])


@_k8_span_scan_fwd_op.register_fake
def _(query, payload, decay, state, live, rank, segment_tokens, tau_match,
      tau_merge, tau_pop, tie_break, scale):
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    nseg = -(-tokens // int(segment_tokens))
    shape = (batch, tokens, heads, 2)
    return (torch.empty((batch, tokens, heads, value_dim),
                        dtype=torch.float32, device=device),
            torch.empty_like(state),
            torch.empty_like(live),
            torch.empty(shape, dtype=torch.int64, device=device),
            torch.empty(shape, dtype=torch.int64, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty((*shape, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, tokens, heads, rank), dtype=torch.bool,
                        device=device),
            torch.empty((nseg, batch, heads, rank, value_dim),
                        dtype=torch.float32, device=device),
            torch.empty((nseg, batch, heads, rank), dtype=torch.bool,
                        device=device))


def _k8_span_scan_setup(ctx, inputs, output):
    query, payload, decay, state, live, rank, segment_tokens, tau_match, \
        tau_merge, tau_pop, tie_break, scale = inputs
    ctx.save_for_backward(query, payload, decay, state, live, *output[3:])
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)


@torch.library.custom_op("x45::k8_span_scan_bwd", mutates_args=())
def _k8_span_scan_bwd_op(
    query: torch.Tensor, payload: torch.Tensor, decay: torch.Tensor,
    dreads: torch.Tensor, dstate_out: torch.Tensor, slot: torch.Tensor,
    op: torch.Tensor, erase: torch.Tensor, live_after: torch.Tensor,
    payload_field: torch.Tensor, live_pre: torch.Tensor,
    ckpt_state: torch.Tensor, ckpt_live: torch.Tensor, rank: int,
    segment_tokens: int, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The rank <= 128 span's adjoint: K6's segment-recompute backward (its own
    prep and walk kernels, imported verbatim), the carry seeded from the
    span's exit-state gradient."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _k8_low_backward_body(query, payload, decay, dreads, dstate_out,
                                  schedule, live_pre, ckpt_state, ckpt_live,
                                  int(rank), int(segment_tokens), float(scale))


@_k8_span_scan_bwd_op.register_fake
def _(query, payload, decay, dreads, dstate_out, slot, op, erase, live_after,
      payload_field, live_pre, ckpt_state, ckpt_live, rank, segment_tokens,
      scale):
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    return (torch.empty(query.shape, dtype=query.dtype, device=device),
            torch.empty(payload.shape, dtype=payload.dtype, device=device),
            torch.empty((batch, tokens, heads), dtype=torch.float32,
                        device=device),
            torch.empty_like(ckpt_state[0]))


def _k8_span_scan_backward(ctx, dreads, dstate_out, dlive, dslot, dop,
                           derase, dlive_after, dpayload_field, dlive_pre,
                           dckpt_state, dckpt_live):
    query, payload, decay, state, live, slot, op, erase, live_after, \
        payload_field, live_pre, ckpt_state, ckpt_live = ctx.saved_tensors
    batch, tokens, heads, value_dim = query.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=query.device)
    if dstate_out is None:
        dstate_out = torch.zeros_like(state)
    dquery, dpayload, ddecay, dstate = _k8_span_scan_bwd_op(
        query, payload, decay, dreads, dstate_out, slot, op, erase,
        live_after, payload_field, live_pre, ckpt_state, ckpt_live, ctx.rank,
        ctx.segment_tokens, ctx.scale)
    # 12 inputs: query, payload, decay, state, live + 7 host constants.
    return (dquery, dpayload, ddecay, dstate, None, None, None, None, None,
            None, None, None)


torch.library.register_autograd("x45::k8_span_scan_fwd", _k8_span_scan_backward,
                                setup_context=_k8_span_scan_setup)


# ============================================================ the growth machinery


class GrowthStats:
    """The per-scan growth record: spans, events, per-boundary rates."""

    def __init__(self):
        self.spans: List[Tuple[int, int, int]] = []
        self.events: List[dict] = []
        self.rates: List[Tuple[int, float]] = []

    def summary(self) -> dict:
        return dict(spans=[list(s) for s in self.spans],
                    events=self.events, rates=self.rates)


def _pad_state(state, new_rank):
    """The zero-injection: new rows enter as exact zeros (the legality-proven
    linear embedding; ``F.pad``'s own backward is its exact adjoint)."""
    old_rank = state.shape[2]
    return torch.nn.functional.pad(state, (0, 0, 0, int(new_rank) - old_rank))


def _pad_live(live, new_rank):
    old_rank = live.shape[2]
    return torch.cat([live, torch.zeros(live.shape[0], live.shape[1],
                                        int(new_rank) - old_rank,
                                        dtype=torch.bool, device=live.device)],
                     dim=2)


def _evaluate_trigger(schedule, last_write, t0, span_tokens, policy, window,
                       rank, cap):
    """``scan.routed_chunk_scan``'s per-chunk growth bookkeeping, boundary by
    boundary: the registered statistic VERBATIM (``scan.repeat_write_rate``),
    the registered smoothing, fire at the mean, window and ``last_write`` reset
    on fire.  Returns ``(fire_boundary, k, smoothed_rate, appended_rates)``.
    """
    appended = []
    fire = None
    k_fire = None
    smoothed = None
    for k in range(1, span_tokens // GROW_BOUNDARY_TOKENS + 1):
        chunk = schedule.chunk((k - 1) * GROW_BOUNDARY_TOKENS,
                               k * GROW_BOUNDARY_TOKENS)
        rate = repeat_write_rate(chunk, last_write,
                                t0 + (k - 1) * GROW_BOUNDARY_TOKENS,
                                policy.grow_window_tokens)
        window.append(rate)
        appended.append((t0 + k * GROW_BOUNDARY_TOKENS, rate))
        if len(window) >= policy.grow_window and rank < cap:
            recent = window[-policy.grow_window:]
            mean = sum(recent) / policy.grow_window
            if mean >= policy.grow_trigger:
                fire = t0 + k * GROW_BOUNDARY_TOKENS
                k_fire = k
                smoothed = mean
                break
    return fire, k_fire, smoothed, appended


def _probe_span(query, payload, decay, state, live, rank, segment_tokens,
                pol32, value_dim):
    """One probe span-walk at a constant rank: the K6 register kernel at
    rank <= 128, the K7 walk above -- no grad, the recorded schedule out."""
    if rank <= K6_RANK_LIMIT and sequential_available(rank, value_dim):
        out = _seq_forward_launch(query, payload, decay, state, live, pol32,
                                  rank, int(segment_tokens), collect=False,
                                  warps=_seq_warps(rank, value_dim), stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        return dict(schedule=schedule, state=out["state"], live=out["live"],
                    ckpt_state=out["ckpt_state"], ckpt_live=out["ckpt_live"])
    out = _hi_walk_launch(query, payload, decay, state, live, pol32, rank,
                          collect=False)
    return dict(schedule=out["schedule"], state=out["state"],
                live=out["live"], ckpt_state=None, ckpt_live=None)


@torch.no_grad()
def _probe_spans(query, payload, decay, state0, live0, policy, start_rank,
                 rank_cap, segment_tokens, pol32):
    """The probe pass: discover the span table, the growth events and the
    per-boundary rates.  Mutates nothing the real pass owns (its state/live are
    probe-local clones; ``last_write`` is the statistic's own bookkeeping,
    reset at every fire exactly as the registered chunked implementation
    resets it)."""
    batch, tokens, heads, value_dim = payload.shape
    device = query.device
    stats = GrowthStats()
    rank = int(start_rank)
    state = state0.clone()
    live = live0.clone()
    last_write = torch.zeros(batch, heads, rank, dtype=torch.int64,
                            device=device)
    window: List[float] = []
    t0 = 0
    while t0 < tokens:
        if rank >= rank_cap:
            stats.spans.append((t0, tokens, rank))
            break
        span_tokens = tokens - t0
        out = _probe_span(query[:, t0:], payload[:, t0:], decay[:, t0:],
                          state, live,
                          rank, segment_tokens, pol32, value_dim)
        fire, k, smoothed, appended = _evaluate_trigger(
            out["schedule"], last_write, t0, span_tokens, policy, window,
            rank, rank_cap)
        stats.rates.extend(appended)
        if fire is None:
            stats.spans.append((t0, tokens, rank))
            break
        stats.spans.append((t0, fire, rank))
        # the boundary state: the probe walk's own segment checkpoint at
        # rank <= 128 (the fire boundary is a 64-token multiple, hence a
        # segment entry), or the schedule replay above.
        if out["ckpt_state"] is not None:
            if fire - t0 == span_tokens:
                # the fire sits at the probe span's very end: the boundary
                # state IS the walk's exit state (no further segment).
                state = out["state"].clone()
                live = out["live"].clone()
            else:
                state = out["ckpt_state"][k].clone()
                live = out["ckpt_live"][k].clone()
        else:
            state, live = traj_exit_state(state, live, decay[:, t0:fire],
                                          out["schedule"].chunk(0, fire - t0),
                                          rank)
        new_rank = min(rank_cap,
                       growth_target(rank, policy.grow_geometric,
                                     policy.grow_stride_floor))
        stats.events.append(dict(boundary=int(fire), rank_before=int(rank),
                                  rank_after=int(new_rank),
                                  smoothed_rate=float(smoothed),
                                  window_rates=[float(r) for r
                                                in window[-policy.grow_window:]]))
        # zero-injection: the new rows enter exact zeros, liveness off.
        state = _pad_state(state, new_rank)
        live = _pad_live(live, new_rank)
        last_write = torch.zeros(batch, heads, new_rank, dtype=torch.int64,
                                device=device)
        window = []
        rank = new_rank
        t0 = fire
    return stats


class GrowthResult(NamedTuple):
    reads: torch.Tensor
    state: torch.Tensor
    live: torch.Tensor
    rank: int
    stats: GrowthStats
    collected: Optional[List[dict]]


def _check_grow_inputs(query, payload, decay, policy, start_rank, rank_cap):
    batch, tokens, heads, value_dim = payload.shape
    if not policy.ops:
        raise ValueError("the append-only control is refused by the sequential "
                         "registration; it is a chunk-dial control")
    if query.shape != payload.shape:
        raise ValueError("query and payload disagree")
    if decay.shape != (batch, tokens, heads):
        raise ValueError(f"decay must be {(batch, tokens, heads)}, got "
                         f"{tuple(decay.shape)}")
    if not (query.is_cuda and payload.is_cuda and decay.is_cuda):
        raise ValueError("the growing scan is a device path; inputs must be "
                         "on CUDA")
    if int(start_rank) < 1 or int(start_rank) > MAX_RANK:
        raise ValueError(f"the start rank must be in [1, {MAX_RANK}]")
    if snap_rank(int(start_rank)) != int(start_rank):
        raise ValueError(f"the start rank must be a precompiled snap "
                         f"{ORTHONORMAL_RANKS}, got {start_rank}")
    if int(rank_cap) < int(start_rank) or int(rank_cap) > MAX_RANK:
        raise ValueError(f"the rank cap must be in [start_rank, {MAX_RANK}]")
    hi_reachable = start_rank > K6_RANK_LIMIT or (policy.grow
                                                   and rank_cap > K6_RANK_LIMIT)
    if hi_reachable and value_dim > MAX_WALK_VALUE_DIM:
        raise ValueError(f"the walk's value ceiling is {MAX_WALK_VALUE_DIM}, "
                         f"got {value_dim}; refused by name")
    if tokens < 1:
        raise ValueError("tokens must be positive")
    return batch, tokens, heads, value_dim


def growing_scan(query, payload, decay, *, policy: RoutingPolicy,
                  start_rank: int, rank_cap: int = MAX_RANK,
                  segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                  collect: bool = False,
                  initial_state: Optional[torch.Tensor] = None,
                  initial_live: Optional[torch.Tensor] = None) -> GrowthResult:
    """One whole (layer, window) scan with the REGISTERED growth rule wired.

    The drop-in at the mixer's boundary: K6's per-token semantics at every
    token, the registered growth decisions at 64-token boundaries from the
    model's own recorded schedule (the append-collision statistic), the
    zero-injection at every fire, eager re-dispatch to the next snap's
    kernels.  ``collect=True`` is the diagnostics instrument (no graph, every
    span's schedule/weights/liveness recorded); ``collect=False`` is the
    training path (the op pair per span, spans chained through the pads).
    """
    batch, tokens, heads, value_dim = _check_grow_inputs(
        query, payload, decay, policy, start_rank, rank_cap)
    device = query.device
    start_rank, rank_cap = int(start_rank), int(rank_cap)
    segment_tokens = int(segment_tokens)
    pol32 = (_round_f32(policy.tau_match), _round_f32(policy.tau_merge),
             _round_f32(policy.tau_pop), _round_f32(policy.tie_break),
             _round_f32(policy.read_scale))
    state0 = (torch.zeros(batch, heads, start_rank, value_dim,
                          dtype=torch.float32, device=device)
              if initial_state is None else initial_state)
    live0 = (torch.zeros(batch, heads, start_rank, dtype=torch.bool,
                         device=device) if initial_live is None
             else initial_live)
    if state0.shape != (batch, heads, start_rank, value_dim) \
            or state0.dtype != torch.float32 or not state0.is_contiguous():
        raise ValueError("initial_state must be a contiguous fp32 (batch, "
                         "heads, start_rank, value_dim) tensor")
    if live0.shape != (batch, heads, start_rank) or live0.dtype != torch.bool:
        raise ValueError("initial_live must be a bool (batch, heads, "
                         "start_rank) tensor")

    if policy.grow:
        stats = _probe_spans(query, payload, decay, state0, live0, policy,
                             start_rank, rank_cap, segment_tokens, pol32)
    else:
        stats = GrowthStats()
        stats.spans.append((0, tokens, start_rank))
    spans = stats.spans

    if collect:
        # the diagnostics instrument: the raw launches, no autograd boundary,
        # every span's schedule + read weights + pre-decision liveness, the
        # spans chained through the same zero-injection pads as the real pass.
        collected = []
        state, live = state0.clone(), live0.clone()
        reads = torch.empty((batch, tokens, heads, value_dim),
                            dtype=torch.float32, device=device)
        with torch.no_grad():
            for index, (t0, t1, rank) in enumerate(spans):
                qs, ps, ds = query[:, t0:t1], payload[:, t0:t1], decay[:, t0:t1]
                if rank <= K6_RANK_LIMIT and sequential_available(rank,
                                                                  value_dim):
                    out = _seq_forward_launch(qs, ps, ds, state, live, pol32,
                                              rank, segment_tokens,
                                              collect=True,
                                              warps=_seq_warps(rank, value_dim),
                                              stages=3)
                    schedule = MatrixSchedule(slot=out["slot"],
                                              erase=out["erase"],
                                              payload=out["payload"],
                                              op=out["op"],
                                              live_after=out["live_after"])
                    weights, live_pre = out["weights"], out["live_pre"]
                else:
                    out = _hi_walk_launch(qs, ps, ds, state, live, pol32, rank,
                                          collect=True)
                    schedule, weights, live_pre = (out["schedule"],
                                                   out["weights"],
                                                   out["live_pre"])
                reads[:, t0:t1] = out["reads"]
                state, live = out["state"], out["live"]
                collected.append(dict(t0=int(t0), t1=int(t1), rank=int(rank),
                                      schedule=schedule, weights=weights,
                                      live_pre=live_pre, state=out["state"],
                                      live=out["live"]))
                if index + 1 < len(spans) and spans[index + 1][2] > rank:
                    state = _pad_state(state, spans[index + 1][2])
                    live = _pad_live(live, spans[index + 1][2])
        return GrowthResult(reads, state, live, spans[-1][2], stats, collected)

    # ---- the real pass: the span table through autograd.
    state, live = state0, live0
    reads = torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device)
    for index, (t0, t1, rank) in enumerate(spans):
        qs, ps, ds = query[:, t0:t1], payload[:, t0:t1], decay[:, t0:t1]
        if rank <= K6_RANK_LIMIT and sequential_available(rank, value_dim):
            # K6's forward launch VERBATIM, re-hosted as the K8 span op pair
            # so the exit-state gradient chains (K6's own op drops it -- its
            # windows never chain).  The grow-never-fires forward is
            # bit-equal to the banked K6 path by construction.
            reads_s, state, live, _slot, _op, _erase, _live_after, _pay, \
                _live_pre, _ckpt_s, _ckpt_l = _k8_span_scan_fwd_op(
                    qs, ps, ds, state, live, rank, segment_tokens,
                    policy.tau_match, policy.tau_merge, policy.tau_pop,
                    policy.tie_break, policy.read_scale)
        else:
            reads_s, state, live, _slot, _op, _erase, _live_after, _pay, \
                _live_pre, _weights = _k8_hi_scan_fwd_op(
                    qs, ps, ds, state, live, rank, segment_tokens,
                    policy.tau_match, policy.tau_merge, policy.tau_pop,
                    policy.tie_break, policy.read_scale)
        reads[:, t0:t1] = reads_s
        if index + 1 < len(spans):
            nxt_rank = spans[index + 1][2]
            if nxt_rank > rank:
                state = _pad_state(state, nxt_rank)
                live = _pad_live(live, nxt_rank)
    return GrowthResult(reads, state, live, spans[-1][2], stats, None)


# --------------------------------------------------- the eager replay oracle


def oracle_growing_scan(query, payload, decay, *, policy: RoutingPolicy,
                        spans: List[Tuple[int, int, int]],
                        collect: bool = False,
                        initial_state: Optional[torch.Tensor] = None,
                        initial_live: Optional[torch.Tensor] = None):
    """Replay a RECORDED span table with the eager per-token oracles, ATTACHED.

    The gate-B fixed point of the whole growing chain: spans at rank <= 128 use
    K6's oracle (the K6 kernels' own fixed point), spans above use K7's oracle
    (the walk's), and the spans chain through the same zero-injection pads.
    Differentiable end to end.
    """
    quiet = dataclasses.replace(policy, grow=False)
    batch, tokens, heads, value_dim = payload.shape
    device = query.device
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    rank0 = int(spans[0][2])
    state = (torch.zeros(batch, heads, rank0, value_dim, dtype=dtype,
                         device=device) if initial_state is None
             else initial_state.to(dtype))
    live = (torch.zeros(batch, heads, rank0, dtype=torch.bool, device=device)
            if initial_live is None else initial_live.clone())
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=dtype,
                        device=device)
    per_span = []
    for index, (t0, t1, rank) in enumerate(spans):
        rank = int(rank)
        oracle = (sequential_oracle_scan
                  if rank <= K6_RANK_LIMIT else sequential_oracle)
        if oracle is sequential_oracle_scan:
            res = oracle(query[:, t0:t1], payload[:, t0:t1], decay[:, t0:t1],
                         policy=quiet, rank=rank, rank_max=rank,
                         initial_state=state, initial_live=live,
                         collect=collect)
        else:
            res = oracle(query[:, t0:t1], payload[:, t0:t1], decay[:, t0:t1],
                         policy=quiet, rank=rank, initial_state=state,
                         initial_live=live, collect=collect)
        reads[:, t0:t1] = res["reads"]
        state, live = res["state"], res["live"]
        per_span.append(dict(t0=t0, t1=t1, rank=rank,
                             schedule=res["schedule"],
                             weights=res.get("weights"),
                             live_pre=res.get("live_pre")))
        if index + 1 < len(spans):
            nxt = int(spans[index + 1][2])
            if nxt > rank:
                state = _pad_state(state, nxt)
                live = _pad_live(live, nxt)
    return dict(reads=reads, state=state, live=live, per_span=per_span)


# ------------------------------------------------------------- the mixer


class GrowingMatrixMixer(MatrixMixer):
    """The K8 mixer: the parent's projections and DECAY gate verbatim, the scan
    swapped for the growing sequential one -- per-token decisions, the
    registered growth rule at 64-token boundaries, the rank capped at
    ``rank_cap`` (the operator's 2048 target by default).

    Constructed by :func:`convert_model_to_growing` from an existing
    ``MatrixMixer`` (the same parameter modules, the same init -- the K6
    conversion pattern), so ``model.parameters()`` is unchanged: rank is
    capacity, never parameters.
    """

    def __init__(self, *args,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                 rank_cap: int = MAX_RANK, **kwargs):
        super().__init__(*args, **kwargs)
        self.segment_tokens = int(segment_tokens)
        self.rank_cap = int(rank_cap)
        self.growth_events: List[dict] = []
        self.last_rank = int(self.rank)

    def mix(self, x: torch.Tensor):
        batch, tokens, _ = x.shape
        query = self.query(x).view(batch, tokens, self.heads, self.value_dim)
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
        decay = torch.sigmoid(self.decay(x)).clamp(min=DECAY_FLOOR,
                                                    max=1.0 - 1e-6)
        result = growing_scan(query, payload, decay, policy=self.policy,
                              start_rank=self.rank, rank_cap=self.rank_cap,
                              segment_tokens=self.segment_tokens,
                              collect=self.collect_telemetry)
        self.growth_events = [dict(event) for event in result.stats.events]
        self.last_rank = int(result.rank)
        return (self.out(result.reads.reshape(
            batch, tokens, self.heads * self.value_dim)), result)


def convert_model_to_growing(model, segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                             rank_cap: int = MAX_RANK, grow: bool = True,
                             start_rank: Optional[int] = None):
    """Swap every ``MatrixMixer`` for the growing one, sharing parameters.

    The K6 conversion pattern verbatim: a shallow re-wrap whose ``__dict__``
    receives the old mixer's, so the ``nn.Linear`` modules (the parameters the
    optimizer sees, and the init the seed produced) are the very same objects.
    ``model.py`` and every registered file stay untouched.
    """
    for block in model.blocks:
        mixer = block.mixer
        seq = GrowingMatrixMixer.__new__(GrowingMatrixMixer)
        torch.nn.Module.__init__(seq)
        seq.__dict__.update(mixer.__dict__)
        seq.segment_tokens = int(segment_tokens)
        seq.rank_cap = int(rank_cap)
        seq.policy = dataclasses.replace(mixer.policy, grow=bool(grow))
        if start_rank is not None:
            seq.rank = int(start_rank)
        seq.growth_events = []
        seq.last_rank = int(seq.rank)
        block.mixer = seq
    return model


__all__ = ["GROW_BOUNDARY_TOKENS", "GrowthResult", "GrowthStats",
           "GrowingMatrixMixer", "K6_RANK_LIMIT", "K8_BR", "K8_BV",
           "convert_model_to_growing", "growing_scan",
           "oracle_growing_scan", "traj_checkpoints", "traj_exit_state",
           "traj_segment"]
