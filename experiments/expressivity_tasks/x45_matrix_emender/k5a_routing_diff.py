"""K5A -- the routing region's DIFFERENTIABLE channels as one fused Triton pair.

Why this file exists
--------------------
K5 left the launch-count program with one named owner: the routing region's
*differentiable* channels.  Gate A fused the discrete decision chain and moved
~0 of the host gap, because the region's ~40 compiled kernels per chunk live in
the channels that carry gradient -- the read weights (softmax*strength, forward
+ backward, ~20/chunk), the payload lanes (~8/chunk), and the similarity
preprocessing (norms/divs/bmm, ~8/chunk) -- plus gate A's own decision kernel,
which costs 0.50 s of *device* time per step as a separate launch (one program
per (batch, head) row, a latency-bound serial token walk).

This stage fuses all of it into **one small set of launches per chunk**:

``_k5a_prep_kernel``   one program per ``(row, token_block)`` -- the read
                       weights (the unit-normalised ``scale*cos`` bmm as a
                       ``tl.dot``, the live-masked softmax, the strength
                       multiplier), the per-token selections from the
                       similarity (best/second/worst with the tie-break
                       penalty and torch's first-index rule), the MERGE blend,
                       and the per-block need counts.
``_k5a_emit_kernel``   one program per ``(row, token_block)`` -- the free-slot
                       ordinal resolution (the ONE cross-token dependency:
                       each block loads the earlier blocks' need counts, a
                       handful of scalars, and finishes the walk in
                       registers), the five-op code selection and the payload
                       lane select.  The decision walk is therefore
                       **token-block parallel** -- gate A's 0.50 s/step serial
                       kernel is folded in and its cost gone.
``_k5a_bwd_scores_kernel`` + ``_k5a_bwd_rows_kernel``   the adjoint: the
                       softmax/normalisation backward into ``dquery``, the
                       payload-lane mask into ``dpayload``, and the three
                       state-row contributions (the score path through
                       ``unit(rows)``, the strength-multiplier path through
                       ``row_strength``, and the MERGE blend scatter)
                       accumulated as per-(row, value-block) partials summed
                       in a fixed order -- no atomics anywhere.

What deliberately stays in torch, and why (the settled parity ruling)
---------------------------------------------------------------------
The decision inputs.  A decision is an argmax against thresholds, so a 1-ulp
difference in ``similarity`` can flip it, and the probe receipts show the
similarity chain is NOT reproducible from Triton on this stack: the bf16
payload norm (``x.norm`` accumulates in torch's own reduction order) and the
bmm (cuBLAS/inductor's MMA tiling) both differ from any ``tl.dot`` blocking by
~1 ulp on half the entries.  So the router keeps the eager router's own
preprocessing verbatim -- ``strength``, ``stored``, ``value``, the similarity
bmm, four compiled kernels -- and the fused kernels *consume* those tensors.
Decisions, slots, op codes and payloads are then **bit-identical to
``schedule.route_chunk`` by construction**, on every fixture, with no
noise-floor allowance and no flip-test gamble; the read WEIGHTS (values, not
decisions) are the one channel recomputed in-kernel and are held to the
measured-floor discipline of the K23 precedent (``NOISE_FACTOR`` x a floor
measured first on the same fixture).

The blend weights are the same story: the MERGE blend's coefficients are
gathered from the torch ``strength``, and the blend arithmetic uses the
explicit ``add.rn``/``mul.rn`` intrinsics (no FMA contraction), so the payload
lanes are bit-exact too.  The read-weights scores mirror the eager precision
chain exactly -- the dot is taken in the inputs' dtype and the scores are
rounded to bf16 under the model's autocast before the fp32 softmax -- so the
weights deviation is the reduction-order floor, not a precision change.

Gradient routing (the K23 lesson, honoured)
-------------------------------------------
In the registered path the routing gradient reaches exactly three tensors:
``query`` (through the read-weights scores), the state ``rows`` (through
``unit(rows)`` in the scores, through ``row_strength`` in the multiplier, and
through the MERGE blend content), and ``payload`` (the APPEND/CORRECT lane
passthrough).  The similarity branch is grad-dead on both arms (everything it
feeds ends in a comparison), so the custom op returns ``None`` for it and the
engine prunes the bmm backward exactly as eager does.  The adjoint is
first-order, like every kernel in this lane.

Hosting
-------
The K4 pattern: the forward is one opaque ``torch.library`` op
(``x45::routing_diff_fwd``) and the backward is its own opaque op
(``x45::routing_diff_bwd``), because ``torch.compile`` traces a registered
backward into the joint graph and the Triton launches must stay out of both
traces.  The bool schedule fields are allocated bool and passed to the kernel
as int8 views *inside* the op impl -- opaque, so the ``aten.view.dtype``
bitcast that inductor cannot lower never appears in the traced region (the
gate B lesson).

Refusals
--------
``rank > K5A_MAX_RANK`` (the selection and softmax tiles hold the whole rank
axis in registers) and the append-only control (``policy.ops=False``) take the
registered eager router verbatim -- refused by name, bit-exact trivially.
"""
from __future__ import annotations

import struct
from typing import Tuple

import torch
import triton
import triton.language as tl

from .algebra import (APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule)
from .k1_schedule_scan import _add_rn, _mul_rn
from .k5_routing_fusion import _round_threshold
from .read import EPS as READ_EPS
from .schedule import EPS, route_chunk

# The whole rank axis lives in one register tile (the softmax and the argmax
# both reduce over it); rank 128 is the registered shape and rank 512 is
# refused to the eager router by name.
K5A_MAX_RANK = 128
# The dtype-rounded thresholds are pure functions of frozen policy constants,
# so they are memoised: the op body would round five scalars per chunk, and
# under compile that is per-chunk Python the graph cannot hoist.
_ROUND_CACHE: dict = {}


def _cached_thresholds(tau_match, tau_merge, tau_pop, tie_break, dtype):
    key = (tau_match, tau_merge, tau_pop, tie_break, dtype)
    rounded = _ROUND_CACHE.get(key)
    if rounded is None:
        rounded = (_round_threshold(tau_match, dtype),
                   _round_threshold(tau_merge, dtype),
                   _round_threshold(tau_pop, dtype),
                   struct.unpack("<f", struct.pack("<f", tie_break))[0])
        _ROUND_CACHE[key] = rounded
    return rounded


# Triton constants for the op codes.
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))
_BIG_INDEX = tl.constexpr(2 ** 30)
_EPS = tl.constexpr(EPS if EPS == READ_EPS else 1e-6)

# Token/value block shapes.  BT is the token-block parallelism of all four
# kernels; BK is the value-axis chunk of the dots; BV is the value block of the
# row-adjoint partials.
BT_DEFAULT = 16
BK_DEFAULT = 32
BV_DEFAULT = 64


# --------------------------------------------------------------------- forward


# The MERGE blend's coefficients must divide exactly as eager does: Triton's
# fp32 ``/`` lowers to ``div.full.f32`` (a ~2-ulp approximation), while torch
# elementwise division is IEEE ``div.rn.f32`` -- and the blend weights are part
# of the bit-exact payload channel.
@triton.jit
def _div_rn(a, b):
    return tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit(do_not_specialize=['T', 'R'])
def _k5a_prep_kernel(
    query, rows, active, similarity, strength,
    weights, scratch_i32, scratch_f32, blend, counts,
    T, R,
    qbs, qhs, qts, qvs,
    rbs, rhs, rrs, rvs,
    abs_, ahs, ars,
    sbs, shs, sts, srs,
    tbs, ths, trs,
    wmax,
    NB,
    tau_match, tau_merge, tau_pop, tie_break, scale,
    H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    RP: tl.constexpr,
):
    """Per (row, token_block): read weights, selections, blend, need counts."""
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    ranks = tl.arange(0, RP)
    keep_r = ranks < R

    act = tl.load(active + batch_index * abs_ + head_index * ahs
                  + ranks * ars, mask=keep_r, other=0) != 0
    act = act & keep_r
    live_any = tl.sum(act.to(tl.int32), 0) > 0

    query_base = query + batch_index * qbs + head_index * qhs
    rows_base = rows + batch_index * rbs + head_index * rhs
    # the query and row norms (the read-weights score denominators)
    sq = tl.zeros((BT,), dtype=tl.float32)
    sr = tl.zeros((RP,), dtype=tl.float32)
    for k0 in range(0, V, BK):
        vals = k0 + tl.arange(0, BK)
        keep_v = vals < V
        q = tl.load(query_base + toks[:, None] * qts + vals[None, :] * qvs,
                    mask=keep_t[:, None] & keep_v[None, :], other=0.0) \
            .to(tl.float32)
        sq += tl.sum(q * q, axis=1)
        r = tl.load(rows_base + ranks[:, None] * rrs + vals[None, :] * rvs,
                    mask=keep_r[:, None] & keep_v[None, :], other=0.0) \
            .to(tl.float32)
        sr += tl.sum(r * r, axis=1)
    nq = tl.sqrt(sq)
    nr = tl.sqrt(sr)
    cq = tl.maximum(nq, _EPS)
    cr = tl.maximum(nr, _EPS)

    # the read-weights scores: scale * (unit(query) @ unit(rows)^T), taken in
    # the inputs' dtype and rounded exactly where eager rounds (bf16 under the
    # model's autocast), then the live-masked softmax in fp32.
    acc = tl.zeros((BT, RP), dtype=tl.float32)
    for k0 in range(0, V, BK):
        vals = k0 + tl.arange(0, BK)
        keep_v = vals < V
        q = tl.load(query_base + toks[:, None] * qts + vals[None, :] * qvs,
                    mask=keep_t[:, None] & keep_v[None, :], other=0.0)
        r = tl.load(rows_base + ranks[:, None] * rrs + vals[None, :] * rvs,
                    mask=keep_r[:, None] & keep_v[None, :], other=0.0)
        uq = (q.to(tl.float32) / cq[:, None]).to(query.dtype.element_ty)
        ur = (r.to(tl.float32) / cr[:, None]).to(query.dtype.element_ty)
        acc = tl.dot(uq, tl.trans(ur), acc, input_precision='ieee')
    su = acc * scale
    if query.dtype.element_ty == tl.bfloat16:
        scores = su.to(tl.bfloat16).to(tl.float32)
    else:
        scores = su
    mask2 = act[None, :] & keep_r[None, :]
    masked = tl.where(mask2, scores, float('-inf'))
    safe = tl.where(live_any, masked, 0.0)
    pmax = tl.max(safe, axis=1)
    expo = tl.exp(safe - pmax[:, None])
    probs = tl.where(live_any, expo / tl.sum(expo, axis=1)[:, None], 0.0)

    strength_row = tl.load(strength + batch_index * tbs + head_index * ths
                           + ranks * trs, mask=keep_r, other=0.0)
    w = probs * strength_row[None, :]
    # the full register width is stored (the columns above the active rank are
    # exact zeros: their probs are zero by the live-mask), so the weights
    # buffer needs no separate zero-fill kernel at the registered shape.  The
    # internal buffers are contiguous, so their strides are computed in-kernel
    # from the shapes rather than passed per launch.
    wrow = weights + row.to(tl.int64) * T * wmax
    tl.store(wrow + toks[:, None] * wmax + ranks[None, :], w,
             mask=keep_t[:, None] & (ranks[None, :] < wmax))

    # the selections, from the eager router's own similarity (never recomputed)
    sim = tl.load(similarity + batch_index * sbs + head_index * shs
                  + toks[:, None] * sts + ranks[None, :] * srs,
                  mask=keep_t[:, None] & keep_r[None, :], other=-2.0) \
        .to(tl.float32)
    sim = tl.where(act[None, :], sim, -2.0)
    penalty = ranks.to(tl.float32) * tie_break
    score = tl.where(keep_r[None, :], sim - penalty[None, :], -3.0)
    best_top = tl.max(score, axis=1)
    best = tl.min(tl.where(score == best_top[:, None], ranks, _BIG_INDEX), 1)
    best_value = tl.sum(tl.where(ranks[None, :] == best[:, None], sim, 0.0), 1)
    second_score = tl.where(ranks[None, :] == best[:, None], -3.0, score)
    second_top = tl.max(second_score, axis=1)
    second = tl.min(tl.where(second_score == second_top[:, None], ranks,
                             _BIG_INDEX), 1)
    second_value = tl.sum(tl.where(ranks[None, :] == second[:, None], sim, 0.0),
                          1)
    worst_score = tl.where(keep_r[None, :], score, float('inf'))
    worst_top = tl.min(worst_score, axis=1)
    worst = tl.min(tl.where(worst_score == worst_top[:, None], ranks,
                            _BIG_INDEX), 1)
    worst_value = tl.sum(tl.where(ranks[None, :] == worst[:, None], sim, 0.0),
                          1)

    correct = best_value >= tau_match
    merge = correct & (second_value >= tau_merge)
    need = keep_t & (~correct)
    tl.store(counts + row * NB + block, tl.sum(need.to(tl.int32), 0))

    flags = (correct.to(tl.int32) | (merge.to(tl.int32) << 1)
             | (need.to(tl.int32) << 2))
    scratch_base = row.to(tl.int64) * T * 4
    tl.store(scratch_i32 + scratch_base + toks * 4 + 0, best.to(tl.int32),
             mask=keep_t)
    tl.store(scratch_i32 + scratch_base + toks * 4 + 1, second.to(tl.int32),
             mask=keep_t)
    tl.store(scratch_i32 + scratch_base + toks * 4 + 2, worst.to(tl.int32),
             mask=keep_t)
    tl.store(scratch_i32 + scratch_base + toks * 4 + 3, flags, mask=keep_t)
    tl.store(scratch_f32 + row.to(tl.int64) * T + toks, worst_value,
             mask=keep_t)

    # the MERGE blend: gathered rows weighted by the torch strength, with the
    # eager expression's exact rounding (no FMA contraction).
    fw = tl.sum(tl.where(ranks[None, :] == best[:, None], strength_row[None, :],
                         0.0), 1)
    sw = tl.sum(tl.where(ranks[None, :] == second[:, None],
                         strength_row[None, :], 0.0), 1)
    fw = tl.maximum(fw, _EPS)
    sw = tl.maximum(sw, _EPS)
    total = fw + sw
    a = _div_rn(fw, total)
    b = _div_rn(sw, total)
    for k0 in range(0, V, BK):
        vals = k0 + tl.arange(0, BK)
        keep_v = vals < V
        first_row = tl.load(rows_base + best[:, None] * rrs + vals[None, :] * rvs,
                           mask=keep_t[:, None] & keep_v[None, :], other=0.0)
        second_row = tl.load(rows_base + second[:, None] * rrs
                             + vals[None, :] * rvs,
                             mask=keep_t[:, None] & keep_v[None, :], other=0.0)
        piece = _add_rn(_mul_rn(a[:, None], first_row),
                        _mul_rn(b[:, None], second_row))
        blend_row = blend + row.to(tl.int64) * T * V
        tl.store(blend_row + toks[:, None] * V + vals[None, :], piece,
                 mask=keep_t[:, None] & keep_v[None, :])


@triton.jit(do_not_specialize=['T', 'R'])
def _k5a_emit_kernel(
    active, payload, blend, scratch_i32, scratch_f32, counts,
    slot_out, op_out, erase_out, live_out, payload_out,
    T, R, NB,
    abs_, ahs, ars,
    pbs, phs, pts, pvs,
    tau_pop,
    H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    RP: tl.constexpr,
):
    """Per (row, token_block): the ordinal resolution and the schedule emission."""
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    ranks = tl.arange(0, RP)
    keep_r = ranks < R

    # the cross-block dependency is a handful of scalars: the earlier blocks'
    # need counts, prefix-summed in registers.
    prefix = tl.zeros((), dtype=tl.int32)
    for i in range(0, block):
        prefix += tl.load(counts + row * NB + i)

    scratch_base = row.to(tl.int64) * T * 4
    best = tl.load(scratch_i32 + scratch_base + toks * 4 + 0, mask=keep_t,
                   other=0)
    second = tl.load(scratch_i32 + scratch_base + toks * 4 + 1,
                     mask=keep_t, other=0)
    worst = tl.load(scratch_i32 + scratch_base + toks * 4 + 2,
                    mask=keep_t, other=0)
    flags = tl.load(scratch_i32 + scratch_base + toks * 4 + 3,
                    mask=keep_t, other=0)
    correct = (flags & 1) != 0
    merge = (flags & 2) != 0
    need = (flags & 4) != 0
    worst_value = tl.load(scratch_f32 + row.to(tl.int64) * T + toks,
                          mask=keep_t, other=0.0)

    act = tl.load(active + batch_index * abs_ + head_index * ahs
                  + ranks * ars, mask=keep_r, other=0) != 0
    act = act & keep_r
    free = keep_r & (~act)
    position = tl.cumsum(free.to(tl.int32), 0) - 1
    free_count = tl.sum(free.to(tl.int32), 0)
    any_active = tl.sum(act.to(tl.int32), 0) > 0

    local = tl.cumsum(need.to(tl.int32), 0) - 1
    ordinal = prefix + local
    appendable = need & (ordinal < free_count)
    match = free[None, :] & (position[None, :] == ordinal[:, None])
    found = tl.min(tl.where(match, ranks, _BIG_INDEX), 1)
    free_slot = tl.where(found < _BIG_INDEX, found, 0)
    full = free_count <= 0
    pop = (~correct) & (~appendable) & full & (worst_value <= tau_pop) \
        & any_active

    op0 = tl.where(correct, tl.where(merge, _MERGE_CODE, _CORRECT_CODE),
                   tl.where(appendable, _APPEND_CODE,
                            tl.where(pop, _POP_CODE, _NONE_CODE)))
    slot0 = tl.where(correct, best, tl.where(appendable, free_slot, worst))
    erase0 = (op0 == _CORRECT_CODE) | (op0 == _MERGE_CODE) | (op0 == _POP_CODE)
    live0 = op0 != _POP_CODE

    out_base = row.to(tl.int64) * T * 2
    tl.store(slot_out + out_base + toks * 2 + 0, slot0.to(tl.int64),
             mask=keep_t)
    tl.store(slot_out + out_base + toks * 2 + 1, second.to(tl.int64),
             mask=keep_t)
    tl.store(op_out + out_base + toks * 2 + 0, op0.to(tl.int64), mask=keep_t)
    tl.store(op_out + out_base + toks * 2 + 1,
             tl.where(merge, _MERGE_CODE, _NONE_CODE).to(tl.int64),
             mask=keep_t)
    tl.store(erase_out + out_base + toks * 2 + 0, erase0.to(tl.int8),
             mask=keep_t)
    tl.store(erase_out + out_base + toks * 2 + 1, merge.to(tl.int8),
             mask=keep_t)
    tl.store(live_out + out_base + toks * 2 + 0, live0.to(tl.int8),
             mask=keep_t)
    tl.store(live_out + out_base + toks * 2 + 1,
             tl.zeros((), dtype=tl.int8), mask=keep_t)

    # the payload lane: the incoming value for APPEND/CORRECT, the blend for
    # MERGE, zero otherwise (and lane 1 is dead by construction).  The carries
    # bit is written back into the scratch (bit 3) because the payload's
    # adjoint needs the POST-ordinal op, which only exists here.
    carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
    tl.store(scratch_i32 + scratch_base + toks * 4 + 3,
             flags | (carries.to(tl.int32) << 3), mask=keep_t)
    pay_base = row.to(tl.int64) * T * 2 * V
    for k0 in range(0, V, BK):
        vals = k0 + tl.arange(0, BK)
        keep_v = vals < V
        pay = tl.load(payload + batch_index * pbs + head_index * phs
                      + toks[:, None] * pts + vals[None, :] * pvs,
                      mask=keep_t[:, None] & keep_v[None, :], other=0.0) \
            .to(tl.float32)
        blend_row = blend + row.to(tl.int64) * T * V
        piece = tl.load(blend_row + toks[:, None] * V + vals[None, :],
                        mask=keep_t[:, None] & keep_v[None, :], other=0.0)
        lane0 = tl.where(carries[:, None], pay, 0.0)
        lane0 = tl.where(merge[:, None], piece, lane0)
        tl.store(payload_out + pay_base + toks[:, None] * (2 * V)
                 + vals[None, :], lane0,
                 mask=keep_t[:, None] & keep_v[None, :])
        tl.store(payload_out + pay_base + toks[:, None] * (2 * V) + V
                 + vals[None, :], tl.zeros((BT, BK), dtype=tl.float32),
                 mask=keep_t[:, None] & keep_v[None, :])


# -------------------------------------------------------------------- backward


@triton.jit(do_not_specialize=['T', 'R'])
def _k5a_bwd_scores_kernel(
    query, rows, active, strength, scratch_i32, dweights, dpay_field,
    dquery, dpayload, dscores_scratch, dstrength_p, ddot_p,
    T, R, NBT,
    qbs, qhs, qts, qvs,
    rbs, rhs, rrs, rvs,
    abs_, ahs, ars,
    tbs, ths, trs,
    kbs, khs, kts,
    gbs, ghs, gts, grs,
    dbs, dhs, dts, dls, dvs,
    scale,
    H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    RP: tl.constexpr,
):
    """The softmax/normalisation adjoint: dquery, dpayload, and the shared
    reduction partials (dscores, dstrength, dur.ur) the row adjoint consumes."""
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    ranks = tl.arange(0, RP)
    keep_r = ranks < R

    act = tl.load(active + batch_index * abs_ + head_index * ahs
                  + ranks * ars, mask=keep_r, other=0) != 0
    act = act & keep_r
    live_any = tl.sum(act.to(tl.int32), 0) > 0

    query_base = query + batch_index * qbs + head_index * qhs
    rows_base = rows + batch_index * rbs + head_index * rhs
    sq = tl.zeros((BT,), dtype=tl.float32)
    sr = tl.zeros((RP,), dtype=tl.float32)
    for k0 in range(0, V, BK):
        vals = k0 + tl.arange(0, BK)
        keep_v = vals < V
        q = tl.load(query_base + toks[:, None] * qts + vals[None, :] * qvs,
                    mask=keep_t[:, None] & keep_v[None, :], other=0.0) \
            .to(tl.float32)
        sq += tl.sum(q * q, axis=1)
        r = tl.load(rows_base + ranks[:, None] * rrs + vals[None, :] * rvs,
                    mask=keep_r[:, None] & keep_v[None, :], other=0.0) \
            .to(tl.float32)
        sr += tl.sum(r * r, axis=1)
    nq = tl.sqrt(sq)
    nr = tl.sqrt(sr)
    cq = tl.maximum(nq, _EPS)
    cr = tl.maximum(nr, _EPS)

    acc = tl.zeros((BT, RP), dtype=tl.float32)
    for k0 in range(0, V, BK):
        vals = k0 + tl.arange(0, BK)
        keep_v = vals < V
        q = tl.load(query_base + toks[:, None] * qts + vals[None, :] * qvs,
                    mask=keep_t[:, None] & keep_v[None, :], other=0.0)
        r = tl.load(rows_base + ranks[:, None] * rrs + vals[None, :] * rvs,
                    mask=keep_r[:, None] & keep_v[None, :], other=0.0)
        uq = (q.to(tl.float32) / cq[:, None]).to(query.dtype.element_ty)
        ur = (r.to(tl.float32) / cr[:, None]).to(query.dtype.element_ty)
        acc = tl.dot(uq, tl.trans(ur), acc, input_precision='ieee')
    su = acc * scale
    if query.dtype.element_ty == tl.bfloat16:
        scores = su.to(tl.bfloat16).to(tl.float32)
    else:
        scores = su
    mask2 = act[None, :] & keep_r[None, :]
    masked = tl.where(mask2, scores, float('-inf'))
    safe = tl.where(live_any, masked, 0.0)
    pmax = tl.max(safe, axis=1)
    expo = tl.exp(safe - pmax[:, None])
    probs = tl.where(live_any, expo / tl.sum(expo, axis=1)[:, None], 0.0)

    w_grad = tl.load(dweights + batch_index * gbs + head_index * ghs
                     + toks[:, None] * gts + ranks[None, :] * grs,
                     mask=keep_t[:, None] & keep_r[None, :], other=0.0)
    strength_row = tl.load(strength + batch_index * tbs + head_index * ths
                           + ranks * trs, mask=keep_r, other=0.0)
    dprobs = w_grad * strength_row[None, :]
    dsum = tl.sum(dprobs * probs, axis=1)
    dscores = probs * (dprobs - dsum[:, None])
    xrow = dscores_scratch + row.to(tl.int64) * T * R
    tl.store(xrow + toks[:, None] * R + ranks[None, :], dscores,
             mask=keep_t[:, None] & keep_r[None, :])
    # per-block partials of the two rank-axis reductions the row adjoint needs.
    dstrength = tl.sum(w_grad * probs, axis=0)
    tl.store(dstrength_p + (row * NBT + block) * RP + ranks, dstrength,
             mask=keep_r)
    # dur.ur (per rank) and duq.uq (per token) both reduce to dscores.su.
    ddot = tl.sum(dscores * su, axis=0)
    tl.store(ddot_p + (row * NBT + block) * RP + ranks, ddot, mask=keep_r)
    duu = tl.sum(dscores * su, axis=1)

    flags = tl.load(scratch_i32 + batch_index * kbs + head_index * khs
                    + toks * kts + 3, mask=keep_t, other=0)
    carries = (flags & 8) != 0
    for k0 in range(0, V, BK):
        vals = k0 + tl.arange(0, BK)
        keep_v = vals < V
        q = tl.load(query_base + toks[:, None] * qts + vals[None, :] * qvs,
                    mask=keep_t[:, None] & keep_v[None, :], other=0.0)
        r = tl.load(rows_base + ranks[:, None] * rrs + vals[None, :] * rvs,
                    mask=keep_r[:, None] & keep_v[None, :], other=0.0)
        uq = q.to(tl.float32) / cq[:, None]
        ur = (r.to(tl.float32) / cr[:, None]).to(query.dtype.element_ty)
        du = tl.dot(dscores.to(query.dtype.element_ty), ur,
                    input_precision='ieee') * scale
        # the clamp's backward is a select, not a mask-multiply: a zero-norm
        # query would turn 0/0 into NaN and 0*NaN is still NaN.
        dq = du / cq[:, None] - uq * tl.where(nq >= _EPS, duu / nq,
                                             0.0)[:, None]
        yrow = dquery + row.to(tl.int64) * T * V
        tl.store(yrow + toks[:, None] * V + vals[None, :],
                 dq.to(query.dtype.element_ty),
                 mask=keep_t[:, None] & keep_v[None, :])
        dpay = tl.load(dpay_field + batch_index * dbs + head_index * dhs
                       + toks[:, None] * dts + 0 * dls + vals[None, :] * dvs,
                       mask=keep_t[:, None] & keep_v[None, :], other=0.0) \
            .to(tl.float32)
        dpay = tl.where(carries[:, None], dpay, 0.0)
        zrow = dpayload + row.to(tl.int64) * T * V
        tl.store(zrow + toks[:, None] * V + vals[None, :],
                 dpay.to(dpayload.dtype.element_ty),
                 mask=keep_t[:, None] & keep_v[None, :])


@triton.jit(do_not_specialize=['T', 'R'])
def _k5a_bwd_rows_kernel(
    query, rows, active, strength, scratch_i32, dpay_field,
    dscores_scratch, dstrength_p, ddot_p,
    drows,
    T, R, NBT,
    qbs, qhs, qts, qvs,
    rbs, rhs, rrs, rvs,
    abs_, ahs, ars,
    tbs, ths, trs,
    kbs, khs, kts,
    dbs, dhs, dts, dls, dvs,
    scale,
    H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, RP: tl.constexpr,
):
    """The state-row adjoint, per (row, value_block): the score path, the
    strength-multiplier path and the MERGE blend scatter, token blocks walked
    in-kernel so the only cross-program axis is the value block -- each program
    writes its own columns of ``drows`` directly (no atomics, no partials)."""
    row = tl.program_id(0)
    vblock = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    vals = vblock * BV + tl.arange(0, BV)
    keep_v = vals < V
    ranks = tl.arange(0, RP)
    keep_r = ranks < R

    act = tl.load(active + batch_index * abs_ + head_index * ahs
                  + ranks * ars, mask=keep_r, other=0) != 0
    act = act & keep_r

    query_base = query + batch_index * qbs + head_index * qhs
    rows_base = rows + batch_index * rbs + head_index * rhs
    # the full-V norms (the unit-adjoint denominators)
    sr = tl.zeros((RP,), dtype=tl.float32)
    for k0 in range(0, V, BK):
        inner = k0 + tl.arange(0, BK)
        keepk = inner < V
        r = tl.load(rows_base + ranks[:, None] * rrs + inner[None, :] * rvs,
                    mask=keep_r[:, None] & keepk[None, :], other=0.0) \
            .to(tl.float32)
        sr += tl.sum(r * r, axis=1)
    nr = tl.sqrt(sr)
    cr = tl.maximum(nr, _EPS)

    # the two rank-axis reductions are full-token sums of the score kernel's
    # per-block partials.
    dstrength = tl.zeros((RP,), dtype=tl.float32)
    ddot = tl.zeros((RP,), dtype=tl.float32)
    for i in range(0, NBT):
        dstrength += tl.load(dstrength_p + (row * NBT + i) * RP + ranks,
                             mask=keep_r, other=0.0)
        ddot += tl.load(ddot_p + (row * NBT + i) * RP + ranks,
                        mask=keep_r, other=0.0)

    strength_row = tl.load(strength + batch_index * tbs + head_index * ths
                           + ranks * trs, mask=keep_r, other=0.0)
    r_chunk = tl.load(rows_base + ranks[:, None] * rrs + vals[None, :] * rvs,
                      mask=keep_r[:, None] & keep_v[None, :], other=0.0) \
        .to(tl.float32)
    ur_chunk = r_chunk / cr[:, None]
    # the strength-multiplier path: d(||r||) = dstrength * active, and the
    # norm adjoint direction r/||r|| (zero at a zero row, torch's rule).
    unit_r = tl.where(nr[:, None] > 0, r_chunk / nr[:, None], 0.0)
    dr = (dstrength * act.to(tl.float32))[:, None] * unit_r
    # the unit(rows) adjoint's cross-token term is a full-token reduction, so
    # it is applied exactly once (not per token block); the clamp's backward
    # is a select, so a zero-norm row cannot leak 0/0 as NaN.
    dr += tl.where(nr[:, None] >= _EPS, -ur_chunk * (ddot / nr)[:, None], 0.0)

    scratch_base = batch_index * kbs + head_index * khs
    for block in range(0, NBT):
        toks = block * BT + tl.arange(0, BT)
        keep_t = toks < T
        dscores = tl.load(dscores_scratch + row.to(tl.int64) * T * R
                          + toks[:, None] * R + ranks[None, :],
                          mask=keep_t[:, None] & keep_r[None, :], other=0.0)
        # the query norms, then uq for this block's value block
        sq = tl.zeros((BT,), dtype=tl.float32)
        for k0 in range(0, V, BK):
            inner = k0 + tl.arange(0, BK)
            keepk = inner < V
            q = tl.load(query_base + toks[:, None] * qts
                        + inner[None, :] * qvs,
                        mask=keep_t[:, None] & keepk[None, :], other=0.0) \
                .to(tl.float32)
            sq += tl.sum(q * q, axis=1)
        nq = tl.sqrt(sq)
        cq = tl.maximum(nq, _EPS)
        q_chunk = tl.load(query_base + toks[:, None] * qts + vals[None, :] * qvs,
                          mask=keep_t[:, None] & keep_v[None, :], other=0.0) \
            .to(tl.float32)
        uq_chunk = q_chunk / cq[:, None]
        # the score path: dur = scale * dscores^T @ uq, then unit(rows)'s
        # adjoint du/c (the cross-token ur term is applied once, above).
        dur = tl.dot(tl.trans(dscores).to(query.dtype.element_ty),
                    uq_chunk.to(query.dtype.element_ty),
                    input_precision='ieee') * scale
        dr += dur / cr[:, None]
        # the MERGE blend scatter: two detached-weight rows per merge token.
        best = tl.load(scratch_i32 + scratch_base + toks * kts + 0,
                       mask=keep_t, other=0)
        second = tl.load(scratch_i32 + scratch_base + toks * kts + 1,
                         mask=keep_t, other=0)
        flags = tl.load(scratch_i32 + scratch_base + toks * kts + 3,
                        mask=keep_t, other=0)
        merge = (flags & 2) != 0
        fw = tl.sum(tl.where(ranks[None, :] == best[:, None],
                            strength_row[None, :], 0.0), 1)
        sw = tl.sum(tl.where(ranks[None, :] == second[:, None],
                             strength_row[None, :], 0.0), 1)
        fw = tl.maximum(fw, _EPS)
        sw = tl.maximum(sw, _EPS)
        a = fw / (fw + sw)
        b = sw / (fw + sw)
        dpay_chunk = tl.load(dpay_field + batch_index * dbs + head_index * dhs
                             + toks[:, None] * dts + 0 * dls
                             + vals[None, :] * dvs,
                             mask=keep_t[:, None] & keep_v[None, :],
                             other=0.0).to(tl.float32)
        # the MERGE blend scatter as its true algebra: a per-token one-hot
        # against the addressed rank, so dr += M @ dpay is one deterministic
        # dot instead of a serial per-token masked-add walk.  A token's best
        # and second rows are distinct, so the two one-hots are disjoint and
        # their sum is a single (rank, tokens) matrix.
        merge_f = merge.to(tl.float32)
        onehot = tl.where(ranks[:, None] == best[None, :],
                          a[None, :] * merge_f[None, :], 0.0) \
            + tl.where(ranks[:, None] == second[None, :],
                       b[None, :] * merge_f[None, :], 0.0)
        dr += tl.dot(onehot, dpay_chunk, input_precision='ieee')

    tl.store(drows + row.to(tl.int64) * R * V + ranks[:, None] * V
             + vals[None, :], dr, mask=keep_r[:, None] & keep_v[None, :])


# ----------------------------------------------------------- the custom op pair


def _prep_shape_checks(query, payload, rows, active, similarity, strength,
                       rank):
    if not (query.is_cuda and rows.is_cuda):
        raise ValueError("K5A routing kernels are device kernels")
    batch, heads, tokens, value_dim = query.shape
    if payload.shape != query.shape:
        raise ValueError("query and payload chunks disagree")
    if rows.shape != (batch, heads, rank, value_dim):
        raise ValueError(f"rows {tuple(rows.shape)} do not match the chunk "
                         f"(batch {batch}, heads {heads}, rank {rank}, "
                         f"value_dim {value_dim})")
    if active.shape != (batch, heads, rank) or strength.shape != active.shape:
        raise ValueError("the active/strength channels do not match the rank")
    if similarity.shape != (batch, heads, tokens, rank):
        raise ValueError("the similarity does not belong to this chunk")
    if rank > K5A_MAX_RANK:
        raise ValueError(f"rank {rank} exceeds the fused router's register "
                         f"budget {K5A_MAX_RANK}")


@torch.library.custom_op("x45::routing_diff_fwd", mutates_args=())
def _routing_diff_fwd_op(
    query: torch.Tensor, payload: torch.Tensor, rows: torch.Tensor,
    active: torch.Tensor, similarity: torch.Tensor, strength: torch.Tensor,
    rank_max: int, tau_match: float, tau_merge: float, tau_pop: float,
    tie_break: float, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """The chunk's whole routing region as ONE opaque op.

    Consumes the eager preprocessing's own ``similarity`` and ``strength``
    tensors (the decision inputs -- their reduction orders are part of the
    decision semantics and are never recomputed here) and emits the complete
    schedule channel, the read weights and the selection scratch the backward
    needs.  ``query``/``payload``/``rows`` carry gradient; ``active``,
    ``similarity`` and ``strength`` do not (the similarity branch is grad-dead
    in the registered path too).
    """
    rank = int(rows.shape[2])
    _prep_shape_checks(query, payload, rows, active, similarity, strength, rank)
    batch, heads, tokens, value_dim = query.shape
    device = query.device
    tau_match, tau_merge, tau_pop, tie_break = _cached_thresholds(
        tau_match, tau_merge, tau_pop, tie_break, similarity.dtype)
    RP = max(triton.next_power_of_2(rank), 16)

    slot = torch.empty((batch, heads, tokens, 2), dtype=torch.int64,
                       device=device)
    op = torch.empty((batch, heads, tokens, 2), dtype=torch.int64,
                     device=device)
    erase = torch.empty((batch, heads, tokens, 2), dtype=torch.bool,
                        device=device)
    live = torch.empty((batch, heads, tokens, 2), dtype=torch.bool,
                       device=device)
    payload_field = torch.empty((batch, heads, tokens, 2, value_dim),
                                dtype=torch.float32, device=device)
    # the prep kernel stores the full register width with exact zeros above
    # the active rank, so the buffer needs its own zero-fill only when the
    # allocated rank is wider than one register tile.
    weights = (torch.empty((batch, heads, tokens, rank_max),
                           dtype=torch.float32, device=device)
               if rank_max <= RP else
               torch.zeros((batch, heads, tokens, rank_max),
                           dtype=torch.float32, device=device))
    scratch_i32 = torch.empty((batch, heads, tokens, 4), dtype=torch.int32,
                              device=device)
    scratch_f32 = torch.empty((batch, heads, tokens, 1), dtype=torch.float32,
                              device=device)
    blend = torch.empty((batch, heads, tokens, value_dim), dtype=torch.float32,
                        device=device)
    nblocks = triton.cdiv(tokens, BT_DEFAULT)
    counts = torch.empty((batch * heads, nblocks), dtype=torch.int32,
                         device=device)

    _k5a_prep_kernel[(batch * heads, nblocks)](
        query, rows, active, similarity, strength,
        weights, scratch_i32, scratch_f32, blend, counts,
        tokens, rank,
        *query.stride(), *rows.stride(), *active.stride(),
        *similarity.stride(), *strength.stride(),
        rank_max, nblocks,
        tau_match=tau_match, tau_merge=tau_merge, tau_pop=tau_pop,
        tie_break=tie_break, scale=scale,
        H=heads, V=value_dim, BT=BT_DEFAULT, BK=BK_DEFAULT, RP=RP,
        num_warps=8)
    _k5a_emit_kernel[(batch * heads, nblocks)](
        active, payload, blend, scratch_i32, scratch_f32, counts,
        slot, op, erase.view(torch.int8), live.view(torch.int8),
        payload_field,
        tokens, rank, nblocks,
        *active.stride(), *payload.stride(),
        tau_pop=tau_pop,
        H=heads, V=value_dim, BT=BT_DEFAULT, BK=BK_DEFAULT, RP=RP,
        num_warps=4)
    return slot, op, erase, live, payload_field, weights, scratch_i32


@_routing_diff_fwd_op.register_fake
def _(query, payload, rows, active, similarity, strength, rank_max, tau_match,
      tau_merge, tau_pop, tie_break, scale):
    batch, heads, tokens, value_dim = query.shape
    device = query.device
    return (torch.empty((batch, heads, tokens, 2), dtype=torch.int64,
                        device=device),
            torch.empty((batch, heads, tokens, 2), dtype=torch.int64,
                        device=device),
            torch.empty((batch, heads, tokens, 2), dtype=torch.bool,
                        device=device),
            torch.empty((batch, heads, tokens, 2), dtype=torch.bool,
                        device=device),
            torch.empty((batch, heads, tokens, 2, value_dim),
                        dtype=torch.float32, device=device),
            torch.empty((batch, heads, tokens, rank_max), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, tokens, 4), dtype=torch.int32,
                        device=device))


@torch.library.custom_op("x45::routing_diff_bwd", mutates_args=())
def _routing_diff_bwd_op(
    query: torch.Tensor, payload: torch.Tensor, rows: torch.Tensor,
    active: torch.Tensor, strength: torch.Tensor, scratch_i32: torch.Tensor,
    dweights: torch.Tensor, dpay_field: torch.Tensor, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The routing adjoint as its own opaque op (the compile-hosting pattern
    the K4 stage receipted): recompute the scores/softmax from the saved
    chunk-entry tensors (the recompute-over-store doctrine -- the saved
    scratch is 16 KB per chunk, not the 512 KB the probabilities would be),
    then the two adjoint kernels."""
    rank = int(rows.shape[2])
    batch, heads, tokens, value_dim = query.shape
    device = query.device
    nblocks = triton.cdiv(tokens, BT_DEFAULT)
    nvalue = triton.cdiv(value_dim, BV_DEFAULT)
    RP = max(triton.next_power_of_2(rank), 16)

    dquery = torch.empty(query.shape, dtype=query.dtype, device=device)
    dpayload = torch.empty(payload.shape, dtype=payload.dtype, device=device)
    dscores = torch.empty((batch, heads, tokens, rank), dtype=torch.float32,
                          device=device)
    dstrength_p = torch.empty((batch * heads, nblocks, RP), dtype=torch.float32,
                              device=device)
    ddot_p = torch.empty((batch * heads, nblocks, RP), dtype=torch.float32,
                         device=device)
    drows = torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                        device=device)
    if not dweights.is_contiguous():
        dweights = dweights.contiguous()
    if not dpay_field.is_contiguous():
        dpay_field = dpay_field.contiguous()

    _k5a_bwd_scores_kernel[(batch * heads, nblocks)](
        query, rows, active, strength, scratch_i32, dweights, dpay_field,
        dquery, dpayload, dscores, dstrength_p, ddot_p,
        tokens, rank, nblocks,
        *query.stride(), *rows.stride(), *active.stride(), *strength.stride(),
        *scratch_i32.stride()[:3],
        *dweights.stride(), *dpay_field.stride(),
        scale=scale,
        H=heads, V=value_dim, BT=BT_DEFAULT, BK=BK_DEFAULT, RP=RP, num_warps=8)
    _k5a_bwd_rows_kernel[(batch * heads, nvalue)](
        query, rows, active, strength, scratch_i32, dpay_field,
        dscores, dstrength_p, ddot_p,
        drows,
        tokens, rank, nblocks,
        *query.stride(), *rows.stride(), *active.stride(), *strength.stride(),
        *scratch_i32.stride()[:3], *dpay_field.stride(),
        scale=scale,
        H=heads, V=value_dim, BT=BT_DEFAULT, BK=BK_DEFAULT, BV=BV_DEFAULT,
        RP=RP, num_warps=8)
    return dquery, dpayload, drows


@_routing_diff_bwd_op.register_fake
def _(query, payload, rows, active, strength, scratch_i32, dweights,
      dpay_field, scale):
    batch, heads, tokens, value_dim = query.shape
    rank = rows.shape[2]
    device = query.device
    return (torch.empty((batch, heads, tokens, value_dim), dtype=query.dtype,
                        device=device),
            torch.empty((batch, heads, tokens, value_dim), dtype=payload.dtype,
                        device=device),
            torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                        device=device))


def _routing_diff_setup(ctx, inputs, output):
    query, payload, rows, active, similarity, strength, rank_max, \
        tau_match, tau_merge, tau_pop, tie_break, scale = inputs
    ctx.save_for_backward(query, payload, rows, active, similarity, strength,
                          output[-1])
    ctx.scaling = float(scale)


def _routing_diff_backward(ctx, dslot, dop, derase, dlive, dpay_field,
                           dweights, dscratch):
    """The adjoint: dquery, dpayload and drows; every other input is grad-dead.

    ``similarity`` gets ``None`` (everything it feeds ends in a comparison, on
    this arm exactly as in the registered eager path), so the engine prunes the
    bmm backward the same way eager prunes its own dead branch.
    """
    query, payload, rows, active, similarity, strength, scratch = \
        ctx.saved_tensors
    batch, heads, tokens, value_dim = query.shape
    device = query.device
    if dweights is None:
        dweights = torch.zeros((batch, heads, tokens, rows.shape[2]),
                               dtype=torch.float32, device=device)
    if dpay_field is None:
        dpay_field = torch.zeros((batch, heads, tokens, 2, value_dim),
                                 dtype=torch.float32, device=device)
    dquery, dpayload, drows = _routing_diff_bwd_op(
        query, payload, rows, active, strength, scratch, dweights, dpay_field,
        ctx.scaling)
    return dquery, dpayload, drows, None, None, None, None, None, None, None, \
        None, None


torch.library.register_autograd("x45::routing_diff_fwd", _routing_diff_backward,
                                setup_context=_routing_diff_setup)


# ------------------------------------------------------------------ the router


def routing_diff_route_chunk(index, token_offset, state, live, query, payload,
                             policy, rank):
    """A drop-in for ``schedule.route_chunk`` with the routing region fused.

    The preprocessing that feeds DECISIONS stays the eager router's own torch
    ops (four compiled kernels: the two norms, the stored division and the
    similarity bmm -- their reduction orders are the decision semantics), and
    everything downstream of ``similarity``/``strength`` -- the read weights,
    the selections, the five-op codes, the payload lanes and their whole
    adjoint -- is the fused pair above.  Decisions, slots, op codes, liveness
    and payloads are therefore **bit-identical to ``route_chunk`` by
    construction**; the read weights are the one recomputed channel, held to
    the measured-floor discipline (``K23_READ.md`` §4 precedent) by the gate
    tests.  ``rank > K5A_MAX_RANK`` and the append-only control take the
    eager router verbatim.
    """
    rank = int(rank)
    if not policy.ops or rank > K5A_MAX_RANK:
        return route_chunk(index, token_offset, state, live, query, payload,
                           policy, rank)
    batch, heads, tokens, value_dim = payload.shape
    device = payload.device
    active = live[:, :, :rank]
    rows = state[:, :, :rank]
    frozen_rows = rows.detach()
    # The eager router's own preprocessing, verbatim in VALUES -- and built
    # from detached inputs, because nothing differentiable ever consumed it:
    # similarity feeds only argmax/gather-into-comparison, a branch that is
    # grad-dead in the registered path too (the eager engine prunes it at
    # runtime).  Under torch.compile the dead branch is NOT pruned from the
    # joint graph, so leaving it attached would compile and run a dead
    # backward (measured: ~3 bf16 gemms plus their pointer-array setup per
    # chunk) for gradients that are provably zero.  Detaching removes it
    # without changing any gradient.
    payload_value = payload.detach()
    strength = frozen_rows.norm(dim=-1) * active.to(torch.float32)
    stored = frozen_rows / strength.clamp_min(EPS).unsqueeze(-1)
    value = payload_value / payload_value.norm(dim=-1,
                                              keepdim=True).clamp_min(EPS)
    similarity = value @ stored.transpose(-1, -2)

    slot, op, erase, live_after, payload_field, weights, _scratch = \
        _routing_diff_fwd_op(
            query, payload, rows, active, similarity, strength,
            int(state.shape[2]), policy.tau_match, policy.tau_merge,
            policy.tau_pop, policy.tie_break, policy.read_scale)
    schedule = MatrixSchedule(
        slot=slot.permute(0, 2, 1, 3),
        erase=erase.permute(0, 2, 1, 3),
        payload=payload_field.permute(0, 2, 1, 3, 4),
        op=op.permute(0, 2, 1, 3),
        live_after=live_after.permute(0, 2, 1, 3))
    return schedule, weights


__all__ = ['K5A_MAX_RANK', 'routing_diff_route_chunk']
