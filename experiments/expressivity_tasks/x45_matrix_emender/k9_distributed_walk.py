"""K9 -- THE DISTRIBUTED-PERSISTENT WALK: the high-rank per-token state kept
on-chip, combined per token by the Lean one-tree reduction.

Why this file exists
--------------------
The K7 walk (``k7_high_rank_store.k7_sequential_walk``) removed the rank
ceiling but left the price: at rank 2048 its one-program-per-row design
streams the whole 16 MB store from HBM twice per token and pays the dependent
defined-order tree folds serially -- fold-bound at 275 GB/s effective,
374.3 ms per (layer, window) scan (receipts/k7_high_rank.json), 414 ms with
the recorded channels (receipts/k8_gates.json, gate D).  The K9 brief: keep
the state ON-CHIP, distributed across CTAs, and combine per token by the
Lean full-population one-tree reduction -- exact, no sketch, no new semantics.

GATE 0 was measured BEFORE this build (``k9_barrier_economics.py`` ->
``receipts/k9_barrier_economics.json``; the design fork is argued in
``DESIGN_K9_DISTRIBUTED_WALK.md`` and the variant was approved by the
supervisor).  Two facts force the design here:

1. **The smem-resident form is impossible on this stack.**  Triton 3.5.1 has
   no grid-wide barrier (``tl.extra.cuda`` carries only the PDL utilities)
   and no cooperative launch; and the residency arithmetic refuses the probe
   shape outright: a 64 KB CTA fits once per SM (142 on this RTX 6000 Ada),
   so 9.3 MB of state is addressable on-chip against the 16 MB the scan
   needs.  The unlock for a future stack is named in the design doc; here the
   standing directive holds -- no new CUDA C++.
2. **The Triton-expressible per-token barrier is the kernel-launch boundary**,
   and its economics pass the pre-stated rule: 19.2 us eager (118 ms/scan --
   over budget) but 0.56 us as a CUDA-graph node (the lane's own K5B
   whole-scan discipline), so 2048 tokens x 3 syncs = 3.4 ms/scan, under the
   40 ms budget.  The store rides the L2 (16 MB < 96 MB, measured hot at
   ~1.2 TB/s for the 256-CTA sweep) -- wholly on-chip at the probe shape.

The design (``DESIGN_K9_DISTRIBUTED_WALK.md`` section 2), exact per the
registered semantics:

* grid ``(row, slot-block)``: one CTA per (batch*head row, NT slot blocks of
  BR=64 slots x 128 values -- the lane's proven 8 192-float walk tile);
* per token THREE kernels, the launch boundaries being the barrier:
  ``_k9_score_kernel``   every CTA folds its tile's per-row DEFINED value-axis
                         trees (norm, read score, decision cosine) into the
                         whole-rank scratch channels -- K7's serial sweep A
                         distributed; program (row, 0) additionally performs
                         the previous token's read-numerator ONE-TREE combine;
  ``_k9_decide_kernel``  ONE program per row runs K7's decision phase VERBATIM
                         on the whole-rank scratch (the softmax denominator is
                         the defined whole-axis tree IN ONE PROGRAM -- the
                         weights channel stays bit-exact against the oracle),
                         the registered selection chain, the MERGE blend, the
                         lane; it writes the decision channel and six scalars;
  ``_k9_apply_kernel``   every CTA computes its block's read-numerator partial
                         (native per-tile sum -- the floor channel), applies
                         DECAY in place, and the OWNING block alone executes
                         the op in its tile (the erasing write or the captured
                         decayed row + lane, the MERGE source zeroing, the
                         liveness update).  A NONE op writes nothing.
* the read numerator's per-block partials are combined in ONE defined
  balanced tree over the slot-block axis (a tile's internal tree plus one
  tree over the tile results IS the whole-axis tree, zeros padding exactly)
  -- the Lean full-population law, the K7 chunked path's own mechanism.
* the whole 3*T+1 launch sequence is captured into ONE whole-scan CUDA graph
  per (shape, snap, policy) slot -- static buffers, one replay per scan, zero
  host syncs; the eager body stays available and is the determinism
  cross-check.

The visibility analysis (the K7 nondeterminism class, closed -- design doc
section 4): cross-CTA writes are ordered by the launch boundary (an implicit
device fence between dependent launches, the stream order inside the graph);
no kernel reads back what it writes in the same launch; and the liveness
update deliberately MOVES from the decision kernel to the apply sweep's
owning block, because a decision program that both loads the whole liveness
vector and stores ``live[slot0]`` would race one warp's store against
another warp's load of the same byte inside one CTA.  The final bytes and
their order are exactly K7's.

Routing and composition
-----------------------
* ``rank x value_dim <= MAX_SEQ_TILE`` (16 384, K6's on-chip register budget)
  is REFUSED by name: the K6/K7 paths own it (bit-pinned, already fast).
  ``k9_auto_walk`` routes those ranks to ``k7_sequential_walk`` (imported,
  never edited) and everything above to the distributed walk.
* ``k9_span_walk`` composes with K8's growth machinery at the span-API level:
  a RECORDED span table replays through the auto route, the state chained
  through K8's own zero-injection pads (imported ``_pad_state``/``_pad_live``),
  every snap its own cached graph -- the growth re-dispatch with its state
  copy-in at the snap change.

Forward only, as every kernel in this lane: ``check_no_grad`` refuses a graph.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .algebra import MatrixSchedule
from .k1_schedule_scan import _add_rn, _mul_rn, check_no_grad
from .k6_sequential_matrix import MAX_SEQ_TILE, sequential_available
from .k7_high_rank_store import (
    _APPEND_CODE, _BIG_INDEX, _CORRECT_CODE, _EPS, _MERGE_CODE, _NONE_CODE,
    _POP_CODE, MAX_WALK_VALUE_DIM, _div_rn32, _fold_axis0, _fold_axis2,
    _fold_vec, _log2_up, _pow2, _round_f32, _sqrt_rn32, k7_sequential_walk,
)
from .schedule import RoutingPolicy

#: the slot-block width (slots) one distributed CTA owns: BR x 128 values is
#: the lane's proven 8 192-float walk tile (K7's own K7_WALK_TILE), now ONE CTA
#: per tile instead of one program's serial loop over all of them -- measured
#: the fastest width at the probe shape (56.1 ms vs 64.2 ms at BLOCK=128, the
#: brief's smem-design constant; receipted in receipts/k9_gates.json)
K9_BLOCK_SLOTS = 64
#: the whole-scan graphs are cached per (shape, snap, policy); a bounded
#: slot table (the K5B discipline) keeps the static buffers owned
MAX_K9_SLOTS = 16


# ============================================================== the three kernels


@triton.jit(do_not_specialize=['t', 'tp', 'T', 'R'])
def _k9_score_kernel(
    query, payload, state, live_scr,
    scr_score, scr_sim, scr_strength,
    partials, reads,
    t, tp, T, R,
    qbs, qts, qhs, qvs,
    pbs, pts, phs, pvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BR: tl.constexpr, NT: tl.constexpr, LOGNT: tl.constexpr, LOGV: tl.constexpr,
    SCALE: tl.constexpr, COMBINE: tl.constexpr,
):
    """K7's sweep A distributed: per (row, slot-block), the per-row DEFINED
    value-axis folds of this block's tile into the whole-rank scratch channels.

    The per-row folds are row-local (a tile never splits a row's value axis)
    and their expressions are K7's sweep A verbatim, so the scratch values are
    bit-identical to the ones K7's single program computes serially.  Program
    (row, 0) additionally combines the PREVIOUS token's read-numerator
    partials in ONE defined balanced tree over the slot-block axis -- the
    Lean full-population law (a tile's internal tree plus one tree over the
    tile results IS the whole-axis tree, zeros padding exactly).
    """
    row = tl.program_id(0)
    b_idx = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    ranks_b = b_idx * BR + tl.arange(0, BR)
    keep_b = ranks_b < R
    values = tl.arange(0, VP)
    keep_v = values < V

    # ---- the one-tree combine of the previous token's read numerator
    if COMBINE:
        if b_idx == 0:
            parts = tl.load(partials + row.to(tl.int64) * NT * VP
                            + tl.arange(0, NT)[:, None] * VP + values[None, :])
            read_o = tl.reshape(_fold_axis0(parts, NT, LOGNT), (VP,))
            reads_base = reads + batch_index.to(tl.int64) * T * H * V \
                + head_index * V
            tl.store(reads_base + tp * H * V + values, read_o, mask=keep_v)

    state_base = state + row.to(tl.int64) * R * V
    live_base = live_scr + row.to(tl.int64) * R
    scr_base = row.to(tl.int64) * R

    # the token's query/payload unit vectors (row-local folds, K7's own)
    query_row = query + batch_index * qbs + head_index * qhs
    payload_row = payload + batch_index * pbs + head_index * phs
    q = tl.load(query_row + t * qts + values * qvs, mask=keep_v,
                other=0.0).to(tl.float32)
    p = tl.load(payload_row + t * pts + values * pvs, mask=keep_v,
                other=0.0).to(tl.float32)
    nq = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(q, q), VP, LOGV)), (1,))
    uq = _div_rn32(q, tl.broadcast_to(tl.maximum(nq, _EPS), (VP,)))
    npv = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(p, p), VP, LOGV)), (1,))
    pv = _div_rn32(p, tl.broadcast_to(tl.maximum(npv, _EPS), (VP,)))

    tile = tl.load(state_base + ranks_b[:, None] * V + values[None, :],
                   mask=keep_b[:, None] & keep_v[None, :], other=0.0)
    live_b = tl.load(live_base + ranks_b, mask=keep_b, other=0) != 0
    nr_b = tl.reshape(_sqrt_rn32(_fold_axis2(_mul_rn(tile, tile), VP, LOGV)),
                      (BR,))
    strength_b = _mul_rn(nr_b, tl.where(live_b & keep_b, 1.0, 0.0))
    score_b = _div_rn32(
        _mul_rn(tl.reshape(_fold_axis2(_mul_rn(uq[None, :], tile), VP, LOGV),
                           (BR,)), SCALE),
        tl.maximum(nr_b, _EPS))
    sim_b = _div_rn32(
        tl.reshape(_fold_axis2(_mul_rn(pv[None, :], tile), VP, LOGV), (BR,)),
        tl.maximum(strength_b, _EPS))
    score_b = tl.where(live_b & keep_b, score_b, float('-inf'))
    sim_b = tl.where(live_b & keep_b, sim_b, -2.0)
    tl.store(scr_score + scr_base + ranks_b, score_b, mask=keep_b)
    tl.store(scr_sim + scr_base + ranks_b, sim_b, mask=keep_b)
    tl.store(scr_strength + scr_base + ranks_b, strength_b, mask=keep_b)


@triton.jit(do_not_specialize=['t', 'T', 'R'])
def _k9_decide_kernel(
    payload, state, live_scr,
    scr_score, scr_sim, scr_strength, scr_w, lane0_scr, scal,
    slot_out, op_out, erase_out, live_after_out, payload_out,
    live_pre, weights_out,
    t, T, R,
    pbs, pts, phs, pvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, RP: tl.constexpr,
    L: tl.constexpr, LOGR: tl.constexpr,
    TAU_MATCH: tl.constexpr, TAU_MERGE: tl.constexpr, TAU_POP: tl.constexpr,
    TIE: tl.constexpr, COLLECT: tl.constexpr,
):
    """K7's decision phase VERBATIM, one program per row, on the whole-rank
    scratch channels the score sweep just wrote (the launch boundary is the
    barrier that makes every warp's scratch stores visible).

    Writes the decision channel in the scan's ``(b, T, h, lanes)`` layout, the
    whole-rank weights scratch for the apply sweep, the lane vector, and six
    decision scalars (``slot0, second, op0, erase0, live0, merge``) for the
    owning block.  It does NOT touch the liveness: a program that both loads
    the whole liveness vector and stores ``live[slot0]`` would race one warp's
    store against another warp's load of the same byte inside one CTA (the K7
    nondeterminism class at this granularity), so the liveness update lives in
    the apply sweep's owning block, where no program reads liveness at all.
    """
    row = tl.program_id(0)
    batch_index = row // H
    head_index = row % H
    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = tl.arange(0, VP)
    keep_v = values < V

    scr_base = row.to(tl.int64) * R
    live_base = live_scr + row.to(tl.int64) * R
    state_base = state + row.to(tl.int64) * R * V
    payload_row = payload + batch_index * pbs + head_index * phs
    live_pre_base = live_pre + batch_index.to(tl.int64) * T * H * R \
        + head_index * R
    weights_base = weights_out + batch_index.to(tl.int64) * T * H * R \
        + head_index * R
    slot_base = slot_out + batch_index.to(tl.int64) * T * H * L \
        + head_index * L
    op_base = op_out + batch_index.to(tl.int64) * T * H * L + head_index * L
    erase_base = erase_out + batch_index.to(tl.int64) * T * H * L \
        + head_index * L
    live_after_base = live_after_out + batch_index.to(tl.int64) * T * H * L \
        + head_index * L
    payload_base = payload_out + batch_index.to(tl.int64) * T * H * L * V \
        + head_index * L * V

    live_vec = tl.load(live_base + ranks, mask=keep_r, other=0) != 0
    if COLLECT:
        tl.store(live_pre_base + t * H * R + ranks, live_vec.to(tl.int8),
                 mask=keep_r)

    # ---- the read weights: K7's decision phase, the defined whole-axis tree
    # in ONE program -- no split, so the weights are bit-exact by construction.
    score = tl.load(scr_score + scr_base + ranks, mask=keep_r,
                    other=float('-inf'))
    sim = tl.load(scr_sim + scr_base + ranks, mask=keep_r, other=-2.0)
    strength = tl.load(scr_strength + scr_base + ranks, mask=keep_r, other=0.0)
    live_r = live_vec & keep_r
    live_any = tl.sum(live_r.to(tl.int32), 0) > 0
    safe = tl.where(live_any, score, 0.0)
    pmax = tl.max(safe, 0)
    expo = libdevice.exp(safe - pmax)
    denom = tl.reshape(_fold_vec(tl.where(live_r, expo, 0.0), RP, LOGR), (1,))
    probs = tl.where(live_any,
                     _div_rn32(expo, tl.broadcast_to(denom, (RP,))), 0.0)
    w = _mul_rn(probs, strength)
    if COLLECT:
        tl.store(weights_base + t * H * R + ranks, w, mask=keep_r)
    tl.store(scr_w + scr_base + ranks, w, mask=keep_r)

    # ---- the registered selection chain (schedule.route_chunk at chunk 1)
    penalty = _mul_rn(ranks.to(tl.float32), TIE)
    sel_score = tl.where(keep_r, sim - penalty, -3.0)
    best_top = tl.max(sel_score, 0)
    best = tl.min(tl.where(sel_score == best_top, ranks, _BIG_INDEX), 0)
    best_value = tl.sum(tl.where(ranks == best, sim, 0.0), 0)
    second_score = tl.where(ranks == best, -3.0, sel_score)
    second_top = tl.max(second_score, 0)
    second = tl.min(tl.where(second_score == second_top, ranks, _BIG_INDEX),
                    0)
    second_value = tl.sum(tl.where(ranks == second, sim, 0.0), 0)
    worst_score = tl.where(keep_r, sel_score, float('inf'))
    worst_top = tl.min(worst_score, 0)
    worst = tl.min(tl.where(worst_score == worst_top, ranks, _BIG_INDEX), 0)
    worst_value = tl.sum(tl.where(ranks == worst, sim, 0.0), 0)

    correct = best_value >= TAU_MATCH
    merge = correct & (second_value >= TAU_MERGE)
    free = keep_r & (~live_vec)
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
    # the MERGE blend from the CURRENT rows: two gathered rows, one add --
    # the same two products K7's masked tile sum meets in exactly one add.
    fw = tl.maximum(tl.sum(tl.where(ranks == best, strength, 0.0), 0), _EPS)
    sw = tl.maximum(tl.sum(tl.where(ranks == second, strength, 0.0), 0), _EPS)
    total = fw + sw
    a_mix = _div_rn32(fw, total)
    b_mix = _div_rn32(sw, total)
    carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
    p = tl.load(payload_row + t * pts + values * pvs, mask=keep_v,
                other=0.0).to(tl.float32)
    if merge:
        row_best = tl.load(state_base + best * V + values, mask=keep_v,
                           other=0.0)
        row_second = tl.load(state_base + second * V + values, mask=keep_v,
                            other=0.0)
        lane0 = _add_rn(_mul_rn(a_mix, row_best),
                        _mul_rn(b_mix, row_second))
    else:
        lane0 = tl.where(carries, p, 0.0)

    # ---- record the decision channel (the scan's (b, T, h, lanes) layout)
    tl.store(slot_base + t * H * L + 0, slot0.to(tl.int64))
    tl.store(slot_base + t * H * L + 1, second.to(tl.int64))
    tl.store(op_base + t * H * L + 0, op0.to(tl.int64))
    tl.store(op_base + t * H * L + 1,
             tl.where(merge, _MERGE_CODE, _NONE_CODE).to(tl.int64))
    tl.store(erase_base + t * H * L + 0, erase0.to(tl.int8))
    tl.store(erase_base + t * H * L + 1, merge.to(tl.int8))
    tl.store(live_after_base + t * H * L + 0, live0.to(tl.int8))
    tl.store(live_after_base + t * H * L + 1, tl.zeros((), dtype=tl.int8))
    tl.store(payload_base + t * H * L * V + 0 * V + values, lane0,
             mask=keep_v)
    tl.store(payload_base + t * H * L * V + 1 * V + values,
             tl.zeros((VP,), dtype=tl.float32), mask=keep_v)

    # ---- the handoff to the apply sweep: the lane and six decision scalars
    tl.store(lane0_scr + row.to(tl.int64) * VP + values, lane0, mask=keep_v)
    scal_base = scal + row.to(tl.int64) * 8
    tl.store(scal_base + 0, slot0.to(tl.int32))
    tl.store(scal_base + 1, second.to(tl.int32))
    tl.store(scal_base + 2, op0.to(tl.int32))
    tl.store(scal_base + 3, erase0.to(tl.int32))
    tl.store(scal_base + 4, live0.to(tl.int32))
    tl.store(scal_base + 5, merge.to(tl.int32))


@triton.jit(do_not_specialize=['t', 'R'])
def _k9_apply_kernel(
    decay, state, live_scr,
    scr_w, lane0_scr, scal, partials,
    t, R,
    dbs, dts, dhs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BR: tl.constexpr, NT: tl.constexpr,
):
    """K7's sweep B + lanes distributed: every CTA owns its tile, computes its
    block's read-numerator partial (a native per-tile sum -- the floor
    channel, the one order no torch expression reproduces), applies DECAY in
    place (``mul.rn``), and the OWNING block alone executes the op in its
    tile: the erasing write or the captured decayed row + lane (the exact
    single-nonzero gather), the MERGE source zeroing, and the liveness update
    (moved here from the decision kernel -- no program reads liveness in
    this launch, which is what closes the intra-CTA store->load race).  A
    NONE op writes nothing, exactly as K7.
    """
    row = tl.program_id(0)
    b_idx = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    ranks_b = b_idx * BR + tl.arange(0, BR)
    keep_b = ranks_b < R
    values = tl.arange(0, VP)
    keep_v = values < V

    state_base = state + row.to(tl.int64) * R * V
    live_base = live_scr + row.to(tl.int64) * R
    lam = tl.load(decay + batch_index * dbs + t * dts + head_index * dhs
                  ).to(tl.float32)

    tile = tl.load(state_base + ranks_b[:, None] * V + values[None, :],
                   mask=keep_b[:, None] & keep_v[None, :], other=0.0)
    w_b = tl.load(scr_w + row.to(tl.int64) * R + ranks_b, mask=keep_b,
                  other=0.0)
    # the read-numerator partial for this block: a native per-tile sum, the
    # zero-padded columns exact -- combined across blocks in ONE tree by the
    # next token's score program (the Lean full-population law)
    partial = tl.sum(_mul_rn(w_b[:, None], tile), 0)
    tl.store(partials + row.to(tl.int64) * NT * VP + b_idx * VP + values,
             partial)

    # ---- the decision scalars (written by the decision kernel, one launch
    # boundary ago)
    scal_base = scal + row.to(tl.int64) * 8
    slot0 = tl.load(scal_base + 0)
    second = tl.load(scal_base + 1)
    op0 = tl.load(scal_base + 2)
    erase0 = tl.load(scal_base + 3) != 0
    live0 = tl.load(scal_base + 4) != 0
    merge = tl.load(scal_base + 5) != 0
    active0 = op0 != _NONE_CODE

    decayed = _mul_rn(tile, lam)
    hit = keep_b & (ranks_b == slot0)
    owns_slot0 = tl.sum(hit.to(tl.int32), 0) > 0
    if active0 & owns_slot0:
        lane0 = tl.load(lane0_scr + row.to(tl.int64) * VP + values,
                        mask=keep_v, other=0.0)
        if erase0:
            new_row = lane0
        else:
            # the decayed content this lane sees, captured in-tile (the exact
            # single-nonzero gather -- free of any cross-warp hazard)
            old_row = tl.sum(tl.where(hit[:, None], decayed, 0.0), 0)
            new_row = _add_rn(old_row, lane0)
        decayed = tl.where(hit[:, None], new_row[None, :], decayed)
        tl.store(live_base + slot0, live0.to(tl.int8))
    hit2 = merge & keep_b & (ranks_b == second)
    owns_second = tl.sum(hit2.to(tl.int32), 0) > 0
    if merge & owns_second:
        decayed = tl.where(hit2[:, None], 0.0, decayed)
        tl.store(live_base + second, tl.zeros((), dtype=tl.int8))

    tl.store(state_base + ranks_b[:, None] * V + values[None, :], decayed,
             mask=keep_b[:, None] & keep_v[None, :])


@triton.jit(do_not_specialize=['tp', 'T'])
def _k9_read_combine_kernel(
    partials, reads,
    tp, T,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    NT: tl.constexpr, LOGNT: tl.constexpr,
):
    """The final read-numerator combine (the last token's one-tree reduction;
    every earlier token's is folded into the next token's score sweep)."""
    row = tl.program_id(0)
    batch_index = row // H
    head_index = row % H
    values = tl.arange(0, VP)
    keep_v = values < V
    parts = tl.load(partials + row.to(tl.int64) * NT * VP
                    + tl.arange(0, NT)[:, None] * VP + values[None, :])
    read_o = tl.reshape(_fold_axis0(parts, NT, LOGNT), (VP,))
    reads_base = reads + batch_index.to(tl.int64) * T * H * V + head_index * V
    tl.store(reads_base + tp * H * V + values, read_o, mask=keep_v)


# ============================================================== the host wrappers


def _check_k9_inputs(query, payload, decay, state, live, rank, value_dim):
    """Refuse anything the distributed walk cannot express, before any launch."""
    batch, tokens, heads, _ = payload.shape
    if not (query.is_cuda and payload.is_cuda and decay.is_cuda
            and state.is_cuda):
        raise ValueError("the distributed walk is a device kernel; inputs "
                         "must be on CUDA")
    if query.shape != payload.shape:
        raise ValueError("query and payload disagree")
    if decay.shape != (batch, tokens, heads):
        raise ValueError(f"decay must be {(batch, tokens, heads)}, got "
                         f"{tuple(decay.shape)}")
    if state.shape != (batch, heads, rank, value_dim):
        raise ValueError(f"state {tuple(state.shape)} is not (batch, heads, "
                         f"rank, value_dim) = {(batch, heads, rank, value_dim)}")
    if live.shape != (batch, heads, rank) or live.dtype != torch.bool:
        raise ValueError("live must be a bool (batch, heads, rank) tensor")
    if state.dtype != torch.float32 or not state.is_contiguous():
        raise ValueError("state must be contiguous fp32; the algebra is fp32")
    if not live.is_contiguous():
        raise ValueError("live must be contiguous")
    if value_dim > MAX_WALK_VALUE_DIM:
        raise ValueError(f"the walk's probe ceiling is value_dim <= "
                          f"{MAX_WALK_VALUE_DIM}, got {value_dim}; refused by "
                          f"name, not silently spilled")
    # THE activation threshold: rank x value_dim within the on-chip register
    # budget is the K6 kernel's own shape -- the existing K6/K7 paths own it
    # (bit-pinned, already fast); the distributed walk is the route above it.
    if sequential_available(rank, value_dim, limit=MAX_SEQ_TILE):
        raise ValueError(f"rank {rank} x value_dim {value_dim} fits the "
                         f"on-chip register budget ({MAX_SEQ_TILE} floats): "
                         "the K6/K7 paths own it (k9_auto_walk routes there); "
                         "the distributed walk activates above the threshold")
    if tokens < 1:
        raise ValueError("tokens must be positive")


def _k9_pol32(policy: RoutingPolicy):
    return (_round_f32(policy.tau_match), _round_f32(policy.tau_merge),
            _round_f32(policy.tau_pop), _round_f32(policy.tie_break),
            _round_f32(policy.read_scale))


class _K9GraphSlot:
    """One (shape, snap, policy) graph instance: the static buffers' owner.

    The K5B discipline: every buffer the walk touches is allocated once here,
    the whole 3*T+1 launch sequence is captured into ONE whole-scan CUDA
    graph, and each scan is a copy-in, ONE replay, and a clone-out.  The
    decision data never touches the host.
    """

    def __init__(self, key, batch, tokens, heads, rank, value_dim, pol32,
                 collect, br, warps, device):
        self.key = key
        self.collect = bool(collect)
        self.handles = {}
        self.graph = None
        self.device = device
        rows = batch * heads
        rp, vp = _pow2(rank), _pow2(value_dim)
        if br & (br - 1) or br > rp:
            raise ValueError(f"the slot-block width must be a power of two "
                             f"not above the padded rank, got {br}")
        nt = rp // br
        self.nt = nt
        # the static inputs (contiguous; copy_ takes any source layout)
        self.query_s = torch.empty((batch, tokens, heads, value_dim),
                                   dtype=torch.float32, device=device)
        self.payload_s = torch.empty_like(self.query_s)
        self.decay_s = torch.empty((batch, tokens, heads),
                                   dtype=torch.float32, device=device)
        # the walk's working store (mutated by the graph; cloned out)
        self.work_state = torch.empty((batch, heads, rank, value_dim),
                                      dtype=torch.float32, device=device)
        self.work_live = torch.empty((batch, heads, rank), dtype=torch.int8,
                                     device=device)
        # the per-token scratch (reused across tokens; ordered by the launches)
        self.scr_score = torch.empty((rows, rank), dtype=torch.float32,
                                     device=device)
        self.scr_sim = torch.empty_like(self.scr_score)
        self.scr_strength = torch.empty_like(self.scr_score)
        self.scr_w = torch.empty_like(self.scr_score)
        self.lane0_scr = torch.empty((rows, vp), dtype=torch.float32,
                                     device=device)
        self.scal = torch.empty((rows, 8), dtype=torch.int32, device=device)
        self.partials = torch.empty((rows, nt, vp), dtype=torch.float32,
                                    device=device)
        # the outputs
        self.reads = torch.empty((batch, tokens, heads, value_dim),
                                 dtype=torch.float32, device=device)
        shape = (batch, tokens, heads, 2)
        self.slot = torch.empty(shape, dtype=torch.int64, device=device)
        self.op = torch.empty(shape, dtype=torch.int64, device=device)
        self.erase = torch.empty(shape, dtype=torch.bool, device=device)
        self.live_after = torch.empty(shape, dtype=torch.bool, device=device)
        self.payload_field = torch.empty((*shape, value_dim),
                                         dtype=torch.float32, device=device)
        if collect:
            self.live_pre = torch.empty((batch, tokens, heads, rank),
                                        dtype=torch.bool, device=device)
            self.weights = torch.empty((batch, tokens, heads, rank),
                                      dtype=torch.float32, device=device)
        else:
            # the COLLECT flag skips the diagnostic stores; one-element
            # stand-ins keep the launch signature (never written, never read)
            self.live_pre = torch.empty(1, dtype=torch.bool, device=device)
            self.weights = torch.empty(1, dtype=torch.float32, device=device)
        # the launch constants
        self.launch = dict(batch=batch, tokens=tokens, heads=heads, rank=rank,
                           value_dim=value_dim, rp=rp, vp=vp, br=br, nt=nt,
                           rows=rows, warps=warps,
                           logv=_log2_up(vp), logr=_log2_up(rp),
                           lognt=_log2_up(nt), pol32=pol32)

    # -- the 3*T+1 launch sequence (the eager path AND the captured body) --

    def body(self, keep_handles: bool = False):
        c = self.launch
        T, H, R, V = c["tokens"], c["heads"], c["rank"], c["value_dim"]
        rows, vp, br, nt = c["rows"], c["vp"], c["br"], c["nt"]
        collect = 1 if self.collect else 0
        tau_match, tau_merge, tau_pop, tie, scale = c["pol32"]
        qst, pst, dst = (self.query_s.stride(), self.payload_s.stride(),
                         self.decay_s.stride())
        for t in range(T):
            hs = _k9_score_kernel[(rows, nt)](
                self.query_s, self.payload_s, self.work_state, self.work_live,
                self.scr_score, self.scr_sim, self.scr_strength,
                self.partials, self.reads,
                t, t - 1, T, R,
                *qst, *pst,
                H=H, V=V, VP=vp, BR=br, NT=nt,
                LOGNT=c["lognt"], LOGV=c["logv"], SCALE=scale,
                COMBINE=0 if t == 0 else 1,
                num_warps=c["warps"])
            hd = _k9_decide_kernel[(rows,)](
                self.payload_s, self.work_state, self.work_live,
                self.scr_score, self.scr_sim, self.scr_strength, self.scr_w,
                self.lane0_scr, self.scal,
                self.slot, self.op, self.erase.view(torch.int8),
                self.live_after.view(torch.int8), self.payload_field,
                self.live_pre.view(torch.int8), self.weights,
                t, T, R,
                *pst,
                H=H, V=V, VP=vp, RP=c["rp"], L=2,
                LOGR=c["logr"],
                TAU_MATCH=tau_match, TAU_MERGE=tau_merge, TAU_POP=tau_pop,
                TIE=tie, COLLECT=collect,
                num_warps=c["warps"])
            ha = _k9_apply_kernel[(rows, nt)](
                self.decay_s, self.work_state, self.work_live,
                self.scr_w, self.lane0_scr, self.scal, self.partials,
                t, R,
                *dst,
                H=H, V=V, VP=vp, BR=br, NT=nt,
                num_warps=c["warps"])
        hc = _k9_read_combine_kernel[(rows,)](
            self.partials, self.reads,
            T - 1, T,
            H=H, V=V, VP=vp, NT=nt, LOGNT=c["lognt"], num_warps=c["warps"])
        if keep_handles:
            # the last eager run's compiled-kernel metadata (registers/spills)
            self.handles = dict(score=hs, decide=hd, apply=ha, combine=hc)

    # -- the K5B capture discipline: eager warmup, side-stream warmup, capture

    def capture(self, keep_handles: bool = False):
        self.body(keep_handles)
        torch.cuda.synchronize(self.device)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            self.body()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.body()
        return self

    # -- one scan: copy-in, one replay (or the eager body), clone-out

    def run(self, query, payload, decay, state, live, use_graph=True):
        self.query_s.copy_(query)
        self.payload_s.copy_(payload)
        self.decay_s.copy_(decay)
        if state is None:
            self.work_state.zero_()
        else:
            self.work_state.copy_(state)
        if live is None:
            self.work_live.zero_()
        else:
            self.work_live.copy_(live.view(torch.int8))
        if use_graph and self.graph is not None:
            self.graph.replay()
        else:
            self.body()
        schedule = MatrixSchedule(
            slot=self.slot.clone(), erase=self.erase.clone(),
            payload=self.payload_field.clone(), op=self.op.clone(),
            live_after=self.live_after.clone())
        return dict(reads=self.reads.clone(), state=self.work_state.clone(),
                    live=self.work_live.view(torch.bool).clone(),
                    schedule=schedule,
                    weights=(self.weights.clone() if self.collect
                             else self.weights),
                    live_pre=(self.live_pre.clone() if self.collect
                              else self.live_pre),
                    kernel=self.handles, graph=self.graph,
                    slot=self)


_K9_SLOTS: dict = {}


def reset_k9_graphs() -> int:
    """Drop every cached graph slot (tests and memory hygiene)."""
    n = len(_K9_SLOTS)
    _K9_SLOTS.clear()
    return n


def _k9_slot_for(batch, tokens, heads, rank, value_dim, pol32, collect, br,
                 warps, device):
    key = (batch, tokens, heads, int(rank), value_dim, pol32, bool(collect),
           int(br), int(warps))
    slot = _K9_SLOTS.get(key)
    if slot is None:
        if len(_K9_SLOTS) >= MAX_K9_SLOTS:
            _K9_SLOTS.pop(next(iter(_K9_SLOTS)))     # FIFO eviction
        slot = _K9GraphSlot(key, batch, tokens, heads, int(rank), value_dim,
                            pol32, bool(collect), int(br), int(warps), device)
        _K9_SLOTS[key] = slot
    return slot


def k9_sequential_walk(query: torch.Tensor, payload: torch.Tensor,
                       decay: torch.Tensor, *, policy: RoutingPolicy,
                       rank: int, initial_state: torch.Tensor = None,
                       initial_live: torch.Tensor = None, collect: bool = True,
                       graph: bool = True, br: int = K9_BLOCK_SLOTS,
                       warps: int = 8):
    """The K6/K7 per-token walk at HIGH rank: the state distributed across
    slot-block CTAs, L2-resident on-chip, combined per token by the Lean
    one-tree reduction -- one whole-scan CUDA-graph replay per scan.

    Same semantics as K7's walk (the registered five-op chain at chunk
    granularity 1, decisions and reads against the CURRENT state, the
    certified read, the registered update order); the parity fixed point is
    :func:`k7_high_rank_store.sequential_oracle` (decisions, weights, liveness
    and state bit-exact by construction; the read at the measured floor).
    Decisions/weights are also bit-exact against ``k7_sequential_walk``.  The
    by-name refusals: the on-chip register budget (rank x value_dim <=
    MAX_SEQ_TILE -- the K6/K7 paths own it), value_dim above the probe
    ceiling, and the append-only/growth controls exactly as K7 refuses them.
    """
    if not policy.ops:
        raise ValueError("the append-only control is refused by the sequential "
                         "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused at the walk boundary: it is a "
                          "chunk-boundary concept (k9_span_walk composes a "
                          "recorded span table with the growth re-dispatch)")
    batch, tokens, heads, value_dim = payload.shape
    rank = int(rank)
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                         device=query.device) if initial_state is None
             else initial_state)
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool,
                        device=query.device) if initial_live is None
            else initial_live)
    _check_k9_inputs(query, payload, decay, state, live, rank, value_dim)
    check_no_grad(query, payload, decay, state)
    slot = _k9_slot_for(batch, tokens, heads, rank, value_dim,
                        _k9_pol32(policy), collect, br, warps, query.device)
    if graph:
        if slot.graph is None:
            slot.capture(keep_handles=True)
    else:
        slot.body(keep_handles=True)
    return slot.run(query, payload, decay,
                    None if initial_state is None else state,
                    None if initial_live is None else live,
                    use_graph=graph)


def k9_auto_walk(query: torch.Tensor, payload: torch.Tensor,
                 decay: torch.Tensor, *, policy: RoutingPolicy, rank: int,
                 **kwargs):
    """The auto-path's high-rank route: the on-chip register budget routes to
    the EXISTING K7 walk (imported, never edited -- bit-pinned and already
    fast there), everything above to the K9 distributed walk."""
    batch, _, heads, value_dim = payload.shape
    if sequential_available(int(rank), value_dim, limit=MAX_SEQ_TILE):
        return k7_sequential_walk(query, payload, decay, policy=policy,
                                  rank=rank, **kwargs)
    return k9_sequential_walk(query, payload, decay, policy=policy, rank=rank,
                              **kwargs)


def k9_span_walk(query: torch.Tensor, payload: torch.Tensor,
                 decay: torch.Tensor, *, policy: RoutingPolicy,
                 spans, initial_state: torch.Tensor = None,
                 initial_live: torch.Tensor = None, collect: bool = True,
                 **kwargs):
    """Growth re-dispatch composed at the span-API level (K8's machinery).

    Replays a RECORDED span table ``[(t0, t1, rank), ...]`` (the growth
    events' own output, ``k8_growth.GrowthStats.spans``): each span runs the
    auto route -- the existing K7 path at or below the on-chip budget, the
    K9 distributed walk above -- and the state chains through K8's own
    zero-injection pads (``_pad_state``/``_pad_live``, imported), so a snap
    change re-dispatches to the next snap's cached graph with the state
    copy-in at the boundary (the new rows exact zeros, the prefix-reveal
    invariant).  Full integration into ``k8_growth.growing_scan``'s dispatch
    needs an edit to ``k8_growth.py`` (out of scope by the standing rule);
    this is the API-level composition.
    """
    from .k8_growth import _pad_live, _pad_state
    batch, tokens, heads, value_dim = payload.shape
    spans = [(int(a), int(b), int(r)) for a, b, r in spans]
    if not spans or spans[0][0] != 0 or spans[-1][1] != tokens:
        raise ValueError(f"the span table must cover [0, {tokens}) exactly, "
                         f"got {spans}")
    for (a0, b0, _), (a1, _, _) in zip(spans, spans[1:]):
        if b0 != a1:
            raise ValueError(f"the span table must be contiguous, got {spans}")
    state = (torch.zeros(batch, heads, spans[0][2], value_dim,
                         dtype=torch.float32, device=query.device)
              if initial_state is None else initial_state)
    live = (torch.zeros(batch, heads, spans[0][2], dtype=torch.bool,
                        device=query.device) if initial_live is None
            else initial_live)
    reads = torch.empty((batch, tokens, heads, value_dim),
                        dtype=torch.float32, device=query.device)
    collected = []
    for index, (t0, t1, rank) in enumerate(spans):
        out = k9_auto_walk(query[:, t0:t1], payload[:, t0:t1],
                           decay[:, t0:t1], policy=policy, rank=rank,
                           initial_state=state, initial_live=live,
                           collect=collect, **kwargs)
        reads[:, t0:t1] = out["reads"]
        state, live = out["state"], out["live"]
        if collect:
            collected.append(dict(t0=t0, t1=t1, rank=rank,
                                  schedule=out["schedule"],
                                  weights=out["weights"],
                                  live_pre=out["live_pre"]))
        if index + 1 < len(spans) and spans[index + 1][2] > rank:
            state = _pad_state(state, spans[index + 1][2])
            live = _pad_live(live, spans[index + 1][2])
    return dict(reads=reads, state=state, live=live, spans=spans,
                collected=collected)


__all__ = ["K9_BLOCK_SLOTS", "MAX_K9_SLOTS", "k9_auto_walk",
           "k9_sequential_walk", "k9_span_walk", "reset_k9_graphs"]
