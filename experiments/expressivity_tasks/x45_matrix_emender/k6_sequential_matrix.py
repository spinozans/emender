"""K6 -- THE SEQUENTIAL MATRIX ARM: the registered five-op algebra at per-token
granularity, as one persistent whole-window kernel with a coarse-checkpoint
backward.

Why this file exists
--------------------
``RESULT_CHUNK_SEQUENTIALITY.md`` measured that the registered chunking is a
quality culprit: the chunk dial climbs monotonically as chunks shrink
(2.4173 -> 2.2186 nats/byte at chunk 128 -> 8, one LR, one seed, matched 20M
exposure), MERGE starves 27x as state freshens, and the dial cannot reach its
limit -- chunk 2 OOMs because whole-scan checkpoint memory scales with
chunks/window (1024 chunk-entry saves x 1 MB per scan at the registered shape).
This module is the limit itself, registered in
``PREREGISTRATION_SEQUENTIAL_MATRIX.md`` (a DRAFT until the operator signs it
off): **the registered five-op algebra evaluated at its natural PER-TOKEN
granularity** -- same operators, same thresholds, same values, decisions and
reads against the CURRENT state every token, with the parallel machinery kept
coarse exactly where it is safe (the backward's checkpoints).

The semantics are the chunk-1 limit of the registered dial
---------------------------------------------------------
At token ``t`` the policy sees the state as left by token ``t-1`` (post-write,
post-decay) -- the state the registered router sees at a chunk-1 boundary:

1. the registered selection chain (``schedule.route_chunk``'s logic at chunk
   granularity 1): unit payload vs unit live rows, dead rows masked to -2, the
   index tie-break penalty, the first-index rule, the five thresholds.  APPEND
   takes the FIRST free slot (the chunk-1 limit of the ordinal rule); MERGE
   blends the two rows with the registered detached strength weights; POP
   takes the worst row under the registered capacity rule;
2. the certified read (``read.py``'s formula, unchanged) taken against that
   same current state -- softmax over live rows of ``scale*cos`` times the row
   strength, no renormaliser, all-dead exactly zero attention;
3. the scan's registered order (``algebra.sequential_matrix_scan``): read,
   DECAY, then every lane in order.

The arithmetic spec (the K5A obstacle, resolved by definition)
--------------------------------------------------------------
The registered arm's decision similarity is a bf16 cuBLAS bmm whose reduction
order is not reproducible from a kernel (~48 % bit-equal, the banked K5A probe),
which is why K5A kept the decision inputs in torch.  Here the oracle is NEW, so
the registration DEFINES the arithmetic and both sides implement the same
definition:

* every reduction is a **balanced binary tree over the reduced axis** (the axis
  padded to its next power of two with exact zeros), a chain of
  ``view(..., n/2, 2)`` pair-sums -- bit-exact between torch and Triton;
* all decision/read arithmetic is **fp32** (inputs upcast once at entry);
* correctly-rounded IEEE ops only: ``sqrt.rn``/``div.rn`` (bit-identical to
  torch's), and the one transcendental -- the read softmax's ``exp`` -- is
  torch's own libdevice function (``libdevice.exp``, verified bit-exact where
  Triton's default ``tl.exp`` is not).

Decisions, payload lanes, read weights AND reads are therefore **bit-exact
against the eager per-token oracle by construction**; the parity gates are
``torch.equal``.

The kernels
------------
``_k6_seq_fwd_kernel``  ONE program per (batch*head) row holds the ENTIRE store
                        (``rank x d_v`` fp32) in registers and walks the window
                        serially: per token the defined-order norms, cosines,
                        five-op selection, MERGE blend, read softmax and read,
                        then DECAY and the lanes with ``mul.rn``/``add.rn`` (no
                        FMA contraction).  It records the decision channel, the
                        per-token pre-decision liveness and the segment-entry
                        checkpoints.  One launch per scan, no atomics,
                        bit-deterministic.
``persistent_scan_trajectory``  (reused verbatim from K4) replays a segment's
                        recorded schedule from its entry checkpoint to recompute
                        the per-token states in the backward -- the gate-C
                        recompute-over-store doctrine at a granularity
                        INDEPENDENT of the decision granularity (the knob the
                        dial does not have: 34 MB of checkpoints per scan at
                        the registered shape, against chunk-2's 1.05 GB).
``_k6_bwd_prep_kernel``  token-block PARALLEL per segment: recomputes the
                        per-token read weights in the defined order and their
                        adjoint coefficients, using the two algebraic identities
                        that keep the unit-adjoint cheap:
                        ``(dur*ur)_j = dscores_j * scores_j`` and
                        ``(duq*uq)_t = fold(dscores * scores)``.
``_k6_bwd_walk_kernel``  value-blocked reverse walk per segment: the adjoint
                        tile (``RP x BV``) in registers, the lanes last-first,
                        the DECAY adjoint, the MERGE-blend scatter against the
                        pre-decay rows, then the read adjoint's four state terms
                        and ``dquery``.  The value axis is the only blocked axis
                        (given the prep kernel's per-token rank vectors the
                        adjoint is column-independent); ``dlam`` crosses value
                        blocks and is summed over the block axis in a fixed
                        order in torch (the K4 partials pattern).

Doctrine, unchanged: decisions are recorded constants in the backward (no
decision logits exist); the read weights, payloads and DECAY gate carry
gradient to ``query``/``payload``/``decay`` and the state.  The adjoint's
cross-axis reductions (``dw``, the softmax's denominator adjoint, ``duq``) are
autograd-ordered on the oracle side, so the GRADIENT gates are floor-discipline
(re-measured floors, ``NOISE_FACTOR`` x), not bit-equality -- stated, not
hidden.  The backward is first-order.

The wiring: ONE opaque ``torch.library`` op pair per (layer, window) scan
(``x45::seq_scan_fwd``/``x45::seq_scan_bwd``, the K4/K5A/K5B hosting pattern).
The eager oracle is the fallback for every refusal, by name, with the SAME
semantics; ``policy.grow`` is refused outright (growth is a chunk-boundary
concept this registration does not carry).
"""
from __future__ import annotations

import struct
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .algebra import (APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule)
from .k1_schedule_scan import NONE_CODE, _add_rn, _mul_rn
from .k4_fused_backward import persistent_scan_trajectory
from .model import DECAY_FLOOR, MatrixMixer
from .scan import RoutedResult, ScanStats
from .schedule import EPS, RoutingPolicy

# The whole store must sit in ONE program's registers (decisions and reads need
# full rows), so the sequential kernel's size gate is the FULL tile, not the
# K4 lane's rank-major slice: rank x value_dim fp32 values.
MAX_SEQ_TILE = 16384          # 128 x 128 fp32 = 64 regs/thread at 8 warps
# The coarse backward checkpoint granularity (the registration's 64 tokens).
DEFAULT_SEGMENT_TOKENS = 64
# Token block of the backward's token-parallel prep kernel, and the value block
# of its reverse-walk kernel (the K4 partials discipline).
BT_BWD = 16
BV_BWD = 16


def sequential_available(rank: int, value_dim: int,
                         limit: int = MAX_SEQ_TILE) -> bool:
    """Can one program hold the whole ``rank x value_dim`` store?"""
    return _pow2(rank) * _pow2(value_dim) <= limit


# ------------------------------------------------------------- the spec's arithmetic


def tree_fold(x: torch.Tensor, dim: int, width: int) -> torch.Tensor:
    """The DEFINED reduction order: a balanced binary tree over ``dim``.

    ``dim`` is padded to ``width`` (a power of two) with exact zeros, then
    folded by repeatedly viewing the axis as ``(n/2, 2)`` and summing the
    pairs, ``log2(width)`` times.  Zeros contribute exactly 0 at every step
    and every step is one IEEE fp32 add per pair, so this is bit-reproducible
    in Triton (the ``_fold_*`` helpers below).  Non-contiguous inputs are made
    contiguous first.
    """
    width = int(width)
    if width < 1 or width & (width - 1):
        raise ValueError(f"the fold width must be a power of two, got {width}")
    if x.shape[dim] > width:
        raise ValueError(f"axis {dim} of {tuple(x.shape)} exceeds the fold "
                         f"width {width}")
    x = x.contiguous()
    dim = dim % x.dim()
    if x.shape[dim] < width:
        # F.pad's pad list is LAST-axis-first: (left_n, right_n, left_{n-1}, ...)
        pad = [0] * (2 * x.dim())
        pos = 2 * (x.dim() - 1 - dim)
        pad[pos + 1] = width - x.shape[dim]
        x = torch.nn.functional.pad(x, pad)
    n = width
    while n > 1:
        shape = list(x.shape)
        shape[dim:dim + 1] = [n // 2, 2]
        x = x.view(*shape).sum(dim + 1)
        n //= 2
    return x.squeeze(dim)


def _norm_defined(x: torch.Tensor, dim: int, width: int) -> torch.Tensor:
    """``sqrt(tree_fold(x*x))`` -- the DEFINED norm (torch sqrt is sqrt.rn).

    The zero-norm backward is a SELECT, not a division: a zero row's gradient
    is exactly 0 (torch's own norm rule, and the walk kernels' ``where``), so
    the sqrt never sees 0/0 in its chain.  The forward values are bit-identical
    to the plain ``sqrt(fold)``.
    """
    s = tree_fold(x * x, dim, width)
    mask = (s > 0).to(x.dtype)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    return torch.sqrt(safe) * mask


def _fp32(x: torch.Tensor) -> torch.Tensor:
    """Upcast to the spec's fp32 -- but never DOWN: the fp64 oracle twin (the
    floor reference and the gradcheck instrument) must stay fp64."""
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.to(torch.float32)
    return x


def _round_f32(value: float) -> float:
    """The threshold in the comparison's own dtype (fp32 here, as in K5A)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _pow2(n: int) -> int:
    return int(triton.next_power_of_2(int(n)))


def _log2_up(width: int) -> int:
    return max(1, (int(width) - 1).bit_length())


# ------------------------------------------------------------------ Triton helpers


@triton.jit
def _sqrt_rn32(x):
    return tl.inline_asm_elementwise('sqrt.rn.f32 $0, $1;', '=f,f', [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _div_rn32(a, b):
    return tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _fold_axis2(x, W: tl.constexpr, LOG: tl.constexpr):
    """``(R, W) -> (R, 1)``: the defined balanced tree over the LAST axis."""
    for i in tl.static_range(LOG):
        if W >= (2 << i):
            x = tl.sum(tl.reshape(x, (x.shape[0], W // (2 << i), 2)), 2)
    return x


@triton.jit
def _fold_axis0(x, W: tl.constexpr, LOG: tl.constexpr):
    """``(W, V) -> (1, V)``: the defined balanced tree over axis 0."""
    for i in tl.static_range(LOG):
        if W >= (2 << i):
            x = tl.sum(tl.reshape(x, (W // (2 << i), 2, x.shape[1])), 1)
    return x


@triton.jit
def _fold_vec(x, W: tl.constexpr, LOG: tl.constexpr):
    """``(W,) -> (1,)``: the defined balanced tree."""
    for i in tl.static_range(LOG):
        if W >= (2 << i):
            x = tl.sum(tl.reshape(x, (W // (2 << i), 2)), 1)
    return x


_EPS = tl.constexpr(EPS)
_BIG_INDEX = tl.constexpr(1 << 30)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))


# ------------------------------------------------------------------- the forward kernel


@triton.jit(do_not_specialize=['T', 'SEG'])
def _k6_seq_fwd_kernel(
    query, payload, decay, state_in, live_in,
    reads, state_out, live_out,
    slot_out, op_out, erase_out, live_after_out, payload_out, live_pre,
    ckpt_state, ckpt_live, weights_out,
    T, R, ROWS, SEG,
    qbs, qts, qhs, qvs,
    pbs, pts, phs, pvs,
    dbs, dts, dhs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    RP: tl.constexpr, L: tl.constexpr,
    LOGV: tl.constexpr, LOGR: tl.constexpr,
    TAU_MATCH: tl.constexpr, TAU_MERGE: tl.constexpr, TAU_POP: tl.constexpr,
    TIE: tl.constexpr, SCALE: tl.constexpr, COLLECT: tl.constexpr,
    STAGES: tl.constexpr,
):
    """One program per (batch*head) row; the whole window in registers.

    Per token: record the liveness the policy sees, then the read weights and
    the read against the current state, then the decisions against the same
    state, then DECAY and the lanes (``mul.rn``/``add.rn``, no contraction).
    The store tile ``acc`` (``RP x VP``) is register-resident for the whole
    window; padded rows and value columns are exact zeros throughout.
    """
    row = tl.program_id(0)
    batch_index = row // H
    head_index = row % H

    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = tl.arange(0, VP)
    keep_v = values < V
    keep = keep_r[:, None] & keep_v[None, :]

    acc = tl.load(state_in + row.to(tl.int64) * R * V
                  + ranks[:, None] * V + values[None, :], mask=keep, other=0.0)
    live = tl.load(live_in + row * R + ranks, mask=keep_r, other=0) != 0

    query_row = query + batch_index * qbs + head_index * qhs
    payload_row = payload + batch_index * pbs + head_index * phs
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
        # ---- the state as this token's read and decisions see it (pre-DECAY).
        tl.store(live_pre_base + t * H * R + ranks, live.to(tl.int8),
                 mask=keep_r)
        if t % SEG == 0:
            seg = t // SEG
            ck = (seg * ROWS + row).to(tl.int64)
            tl.store(ckpt_state + ck * R * V + ranks[:, None] * V
                     + values[None, :], acc, mask=keep)
            tl.store(ckpt_live + ck * R + ranks, live.to(tl.int8),
                     mask=keep_r)
        q = tl.load(query_row + t * qts + values * qvs, mask=keep_v,
                    other=0.0).to(tl.float32)
        p = tl.load(payload_row + t * pts + values * pvs, mask=keep_v,
                    other=0.0).to(tl.float32)
        lam = tl.load(decay_row + t * dts).to(tl.float32)

        # ---- the read weights and the read (the certified form, defined order).
        # NOTE: every product in this walk uses the mul.rn intrinsic -- Triton
        # contracts a plain `a * b` into a following add as an FMA (one rounding
        # instead of two), the x44 RESULT_CHUNK_SCAN §6.2 failure class; torch
        # eager rounds every op separately, so the intrinsic is what makes the
        # walk bit-equal to the oracle.  The score's division sits AFTER the
        # value fold (the spec's own defined order):
        # ``score_j = scale * fold(uq * S_j) / max(nr_j, EPS)`` -- the certified
        # ``scale * cos`` with its clamp, one division per row instead of a
        # full normalised tile (the register budget), bit-reproducible in torch
        # identically.  A dead row's content is exactly zero (POP and the MERGE
        # source write zeros; every other write makes the row live), so a dead
        # row's numerator is 0 and the mask takes it out anyway.
        nr = tl.reshape(_sqrt_rn32(_fold_axis2(_mul_rn(acc, acc), VP, LOGV)),
                        (RP,))
        strength = _mul_rn(nr, tl.where(live & keep_r, 1.0, 0.0))
        nq = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(q, q), VP, LOGV)), (1,))
        uq = _div_rn32(q, tl.broadcast_to(tl.maximum(nq, _EPS), (VP,)))
        scores = _div_rn32(
            _mul_rn(tl.reshape(_fold_axis2(_mul_rn(uq[None, :], acc),
                                           VP, LOGV), (RP,)), SCALE),
            tl.maximum(nr, _EPS))
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
        # The READ's rank-axis reduction is the one channel at the measured
        # floor: a native tl.sum (Triton's own reduction order, ~1 us/token
        # against the defined tree's ~11 -- the tree forces a layout
        # conversion per level at the whole-store width).  The gate-A precedent
        # is exactly this (K1/K23: state and decisions bit-exact, reads at the
        # fp32 floor); the read feeds no decision, so no flip can come of it.
        read_o = tl.sum(_mul_rn(w[:, None], acc), 0)
        tl.store(reads_base + t * H * V + values, read_o, mask=keep_v)
        if COLLECT:
            tl.store(weights_base + t * H * R + ranks, w, mask=keep_r)

        # ---- the decisions: the registered selection chain, fp32, defined order.
        pv = _div_rn32(p, tl.broadcast_to(
            tl.maximum(tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(p, p), VP,
                                                      LOGV)), (1,)), _EPS),
            (VP,)))
        sim = _div_rn32(
            tl.reshape(_fold_axis2(_mul_rn(pv[None, :], acc), VP, LOGV), (RP,)),
            tl.maximum(strength, _EPS))
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
        # ---- the MERGE blend from the CURRENT rows (its weights are functions
        # of the recorded channel, so they are constants in the backward).
        # The blend is ONE native rank-axis sum of a coefficient-masked tile:
        # exactly two nonzero products (a*row_best, b*row_second) meet in
        # exactly one add whatever the reduction order, and zeros add exactly
        # 0 -- so the value is bit-exact against the oracle's mul-mul-add
        # with no order dependence at all.
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

        # ---- apply: DECAY, then the lanes in order (the registered order).
        # The old row's gather is order-independent-exact (a single nonzero
        # contributes itself whatever the tree), and only APPEND needs it --
        # an erasing write discards the row outright.
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

    tl.store(state_out + row.to(tl.int64) * R * V + ranks[:, None] * V
             + values[None, :], acc, mask=keep)
    tl.store(live_out + row * R + ranks, live.to(tl.int8), mask=keep_r)


# ------------------------------------------------------------ the forward wrapper


def _check_seq_shapes(query, payload, decay, state, live, rank, segment_tokens):
    """Refuse anything the sequential path cannot express, before any launch."""
    if not (query.is_cuda and payload.is_cuda and decay.is_cuda):
        raise ValueError("the sequential scan is a device kernel; inputs must "
                         "be on CUDA")
    batch, tokens, heads, value_dim = query.shape
    if payload.shape != query.shape:
        raise ValueError("query and payload disagree")
    if decay.shape != (batch, tokens, heads):
        raise ValueError(f"decay must be {(batch, tokens, heads)}, got "
                         f"{tuple(decay.shape)}")
    if state.shape != (batch, heads, rank, value_dim):
        raise ValueError(f"state {tuple(state.shape)} is not (batch, heads, "
                         f"rank, value_dim) = {(batch, heads, rank, value_dim)}")
    if live.shape != (batch, heads, rank) or live.dtype != torch.bool:
        raise ValueError("live must be a bool (batch, heads, rank) tensor")
    for name, tensor in (("state", state),):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be fp32; the algebra is fp32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if tokens < 1 or segment_tokens < 1:
        raise ValueError("tokens and segment_tokens must be positive")


def _seq_forward_launch(query, payload, decay, state, live, policy_f32,
                        rank, segment_tokens, collect, warps, stages):
    """The persistent walk: one launch, every output allocated contiguous."""
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    lanes = 2
    reads = torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device)
    out_state = torch.empty_like(state)
    out_live = torch.empty_like(live)
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
    weights = (torch.empty((batch, tokens, heads, rank), dtype=torch.float32,
                           device=device) if collect
              else torch.empty((1,), dtype=torch.float32, device=device))
    tau_match, tau_merge, tau_pop, tie, scale = policy_f32
    handle = _k6_seq_fwd_kernel[(batch * heads,)](
        query, payload, decay, state, live.view(torch.int8),
        reads, out_state, out_live.view(torch.int8),
        slot, op, erase.view(torch.int8), live_after.view(torch.int8),
        payload_field, live_pre.view(torch.int8),
        ckpt_state, ckpt_live.view(torch.int8), weights,
        tokens, rank, batch * heads, int(segment_tokens),
        *query.stride(), *payload.stride(), *decay.stride(),
        H=heads, V=value_dim, VP=_pow2(value_dim),
        RP=_pow2(rank), L=lanes,
        LOGV=_log2_up(_pow2(value_dim)), LOGR=_log2_up(_pow2(rank)),
        TAU_MATCH=tau_match, TAU_MERGE=tau_merge, TAU_POP=tau_pop,
        TIE=tie, SCALE=scale, COLLECT=1 if collect else 0,
        STAGES=stages, num_warps=warps)
    return dict(reads=reads, state=out_state, live=out_live, slot=slot, op=op,
                erase=erase, live_after=live_after, payload=payload_field,
                live_pre=live_pre, ckpt_state=ckpt_state, ckpt_live=ckpt_live,
                weights=weights, kernel=handle)


# ------------------------------------------------------- the eager per-token oracle


def _oracle_selection(frozen, live, p_det, tau_match, tau_merge, tau_pop, tie,
                      rank, RP, VP):
    """``schedule.route_chunk``'s decision chain for ONE token, defined order.

    ``frozen`` is the detached current state ``(b, h, rank, V)`` fp32,
    ``p_det`` the detached payload ``(b, h, V)``.  Returns the per-(b, h)
    decision fields as a dict -- everything downstream of the similarity is a
    comparison, so nothing here is differentiable, exactly as in the registered
    path.
    """
    strength = _norm_defined(frozen, -1, VP) * live.to(torch.float32)
    pv = p_det / _norm_defined(p_det, -1, VP).clamp_min(EPS).unsqueeze(-1)
    # the division after the fold, matching the kernel's defined order
    sim = tree_fold(pv.unsqueeze(2) * frozen, -1, VP) \
        / strength.clamp_min(EPS)              # (b, h, rank)
    sim = sim.masked_fill(~live, -2.0)
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
                erase0=erase0, live0=live0, strength=strength,
                carries=(op0 == APPEND) | (op0 == CORRECT))


def sequential_oracle_scan(query, payload, decay, *, policy: RoutingPolicy,
                           rank=None, rank_max=None, initial_state=None,
                           initial_live=None, collect=True):
    """The eager per-token oracle: the parity fixed point, differentiable.

    Walks the window token by token in torch, vectorised over (batch, head):
    per token the decisions (from the detached current state, the registered
    chain at chunk granularity 1) and the read (the certified weights from the
    ATTACHED current state -- the registered split: the weights are the read,
    the decisions are architecture), then DECAY and the lanes.  All
    reductions are the defined tree folds, so the kernel path is bit-equal to
    this by construction -- the fixed point the gates hold it to.
    """
    batch, tokens, heads, value_dim = payload.shape
    if not policy.ops:
        raise ValueError("the append-only control is refused by the sequential "
                         "registration; it is a chunk-dial control and this "
                         "arm runs the five-op policy")
    if policy.grow:
        raise ValueError("growth is refused by the sequential registration: "
                         "it is a chunk-boundary concept this arm does not "
                         "carry")
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    if int(rank) != int(rank_max):
        raise ValueError("the sequential registration requires rank == "
                         "rank_max (the registered shape); a rank prefix is "
                         "refused by name")
    device = query.device
    compute_dtype = torch.promote_types(_fp32(query).dtype, torch.float32)
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=compute_dtype,
                         device=device) if initial_state is None
             else initial_state)
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
            if initial_live is None else initial_live.clone())
    q32 = _fp32(query)
    p32 = _fp32(payload)
    d32 = _fp32(decay)
    RP, VP = _pow2(rank), _pow2(value_dim)
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
        q_t, p_t, lam_t = q32[:, t], p32[:, t], d32[:, t]     # (b, h, V) / (b, h)
        # ---- the read: certified weights from the ATTACHED current state.
        # The score's division sits after the fold (the spec's defined order,
        # matching the kernel exactly): score_j = scale*fold(uq*S_j)/max(nr_j,EPS).
        nr = _norm_defined(state, -1, VP)                        # (b, h, rank)
        nq = _norm_defined(q_t, -1, VP)                          # (b, h)
        uq = q_t / nq.clamp_min(EPS).unsqueeze(-1)
        scores = (scale * tree_fold(uq.unsqueeze(2) * state, -1, VP)) \
            / nr.clamp_min(EPS)
        live_f = live.to(torch.float32)
        any_live = live.any(-1, keepdim=True)
        masked = scores.masked_fill(~live, float("-inf"))
        safe = torch.where(any_live, masked, torch.zeros_like(masked))
        pmax = safe.max(-1).values
        expo = torch.exp(safe - pmax.unsqueeze(-1))
        denom = tree_fold(expo * live_f, -1, RP)   # (b, h)
        # the all-dead guard: the division's BACKWARD would otherwise compute
        # 0 * inf where the denominator is an exact zero (the forward's `where`
        # selects the prob away, but the chain still flows through it).  The
        # safe denominator is bit-identical wherever it is used.
        any_live_flat = live.any(-1)
        denom_safe = torch.where(any_live_flat, denom, torch.ones_like(denom))
        probs = torch.where(any_live, expo / denom_safe.unsqueeze(-1),
                             torch.zeros_like(expo))
        w = probs * nr * live_f
        reads[:, t] = tree_fold(w.unsqueeze(-1) * state, -2, RP)  # (b, h, V)
        if collect:
            weights_out[:, t] = w
        # ---- the decisions: the registered chain on the DETACHED state.
        sel = _oracle_selection(state.detach(), live, p_t.detach(), tau_match,
                                tau_merge, tau_pop, tie, rank, RP, VP)
        # ---- the payload lane: incoming value (attached), or the MERGE blend
        # (content attached, weights detached -- the registered construction).
        blend = None
        if bool(sel["merge"].any()):
            rows = state[b_index.expand(batch, heads),
                         h_index.expand(batch, heads), sel["best_index"]]
            rows2 = state[b_index.expand(batch, heads),
                          h_index.expand(batch, heads), sel["second_index"]]
            fw = sel["strength"].gather(-1, sel["best_index"].unsqueeze(-1)
                                         ).clamp_min(EPS).squeeze(-1)
            sw = sel["strength"].gather(-1, sel["second_index"].unsqueeze(-1)
                                         ).clamp_min(EPS).squeeze(-1)
            total = fw + sw
            blend = ((fw / total).detach().unsqueeze(-1) * rows
                     + (sw / total).detach().unsqueeze(-1) * rows2)
        lane0 = torch.where(sel["carries"].unsqueeze(-1), p_t,
                            torch.zeros_like(p_t))
        if blend is not None:
            lane0 = torch.where(sel["merge"].unsqueeze(-1), blend, lane0)
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
        # ---- apply: DECAY, then the lanes in order (the registered order).
        state = state * lam_t.unsqueeze(-1).unsqueeze(-1)
        rows_at = state[b_index.expand(batch, heads),
                       h_index.expand(batch, heads), sel["slot0"]]
        written = torch.where(sel["erase0"].unsqueeze(-1), lane0,
                              rows_at + lane0)
        state = state.index_put((b_index.expand(batch, heads),
                                 h_index.expand(batch, heads),
                                 sel["slot0"]), written)
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
                torch.zeros(batch, heads, value_dim, dtype=torch.float32,
                            device=device))
            live = live.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.zeros(batch, heads, dtype=torch.bool,
                            device=live.device))
    schedule = MatrixSchedule(**fields)
    return dict(reads=reads, state=state, live=live, schedule=schedule,
                weights=weights_out, live_pre=live_pre_out, rank=int(rank))


# ------------------------------------------------- the backward: prep + walk


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k6_bwd_prep_kernel(
    traj, query, dout, live_pre,
    nr_ch, w_ch, dscores_ch, dstrength_ch, z_ch, scal,
    SEGT, T, T0, R,
    qbs, qts, qhs, qvs,
    gbs, gts, ghs, gvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, RP: tl.constexpr,
    LOGV: tl.constexpr, LOGR: tl.constexpr, BT: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Token-block parallel: recompute the per-token read weights (bit-identical
    to the forward's -- the same defined-order expressions) and their adjoint
    coefficients from the segment's recomputed trajectory.

    Per token ``t0 + tt`` (``tt`` local to the segment): ``nr``, ``scores``, the
    softmax, ``w``; then ``dw = fold(dout * S_t)`` (the read adjoint's value
    fold), ``dscores``, ``dstrength``, ``z = dscores * scores`` and
    ``duu = fold(dscores * scores)`` (the unit-adjoint identities).  Stored per
    token as rank vectors plus two scalars (``nq``, ``duu``).
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    ranks = tl.arange(0, RP)
    keep_r = ranks < R

    traj_base = traj + row.to(tl.int64) * SEGT * R * V
    q_row = query + batch_index * qbs + head_index * qhs
    dout_row = dout + batch_index * gbs + head_index * ghs
    live_base = (live_pre + batch_index.to(tl.int64) * T * H * R
                 + head_index * R)
    nr_row = nr_ch + row.to(tl.int64) * SEGT * R
    w_row = w_ch + row.to(tl.int64) * SEGT * R
    ds_row = dscores_ch + row.to(tl.int64) * SEGT * R
    dst_row = dstrength_ch + row.to(tl.int64) * SEGT * R
    z_row = z_ch + row.to(tl.int64) * SEGT * R
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
            q = tl.load(q_row + t * qts + values * qvs, mask=keep_v,
                        other=0.0).to(tl.float32)
            g = tl.load(dout_row + t * gts + values * gvs, mask=keep_v,
                        other=0.0)
            acc = tl.load(traj_base + local * R * V
                          + ranks[:, None] * V + values[None, :],
                          mask=keep_r[:, None] & keep_v[None, :], other=0.0)
            nr = tl.reshape(_sqrt_rn32(_fold_axis2(_mul_rn(acc, acc), VP,
                                                   LOGV)), (RP,))
            nq = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(q, q), VP, LOGV)),
                            (1,))
            uq = _div_rn32(q, tl.broadcast_to(tl.maximum(nq, _EPS), (VP,)))
            scores = _div_rn32(
                _mul_rn(tl.reshape(_fold_axis2(_mul_rn(uq[None, :], acc),
                                               VP, LOGV), (RP,)), SCALE),
                tl.maximum(nr, _EPS))
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
            tl.store(nr_row + local * R + ranks, nr, mask=keep_r)
            tl.store(w_row + local * R + ranks, w, mask=keep_r)
            tl.store(ds_row + local * R + ranks, dscores, mask=keep_r)
            tl.store(dst_row + local * R + ranks, dstrength, mask=keep_r)
            tl.store(z_row + local * R + ranks, z, mask=keep_r)
            tl.store(scal_row + local * 2 + tl.arange(0, 1),
                     tl.reshape(nq, (1,)))
            tl.store(scal_row + local * 2 + 1 + tl.arange(0, 1),
                     tl.reshape(duu, (1,)))


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k6_bwd_walk_kernel(
    traj, query, dout, decay32, live_pre, carry_in,
    nr_ch, w_ch, dscores_ch, dstrength_ch, z_ch, scal,
    slot, op, erase,
    dquery, dpayload, dlam_partial, dstate_seg,
    SEGT, T, T0, R, NB,
    qbs, qts, qhs, qvs,
    gbs, gts, ghs, gvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, RP: tl.constexpr,
    LOGR: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Value-blocked reverse walk of one segment: the adjoint tile in registers.

    Per token (reverse), given the prep kernel's per-token rank vectors: the
    lanes last-first (the payload gradient and the erasing zero), the DECAY
    adjoint (a per-block partial of ``dlam``, summed over blocks in torch),
    the MERGE-blend scatter against the pre-decay rows, then the read
    adjoint's three state terms and ``dquery``.  The value axis is the only
    blocked axis; every rank-axis reduction left is a small-tile native sum.
    """
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
    q_row = query + batch_index * qbs + head_index * qhs
    dout_row = dout + batch_index * gbs + head_index * ghs
    decay_row = decay32 + row * T
    live_base = (live_pre + batch_index.to(tl.int64) * T * H * R
                 + head_index * R)
    nr_row = nr_ch + row.to(tl.int64) * SEGT * R
    w_row = w_ch + row.to(tl.int64) * SEGT * R
    ds_row = dscores_ch + row.to(tl.int64) * SEGT * R
    dst_row = dstrength_ch + row.to(tl.int64) * SEGT * R
    z_row = z_ch + row.to(tl.int64) * SEGT * R
    scal_row = scal + row.to(tl.int64) * SEGT * 2
    dq_row = dquery + batch_index.to(tl.int64) * T * H * V + head_index * V
    dp_row = dpayload + batch_index.to(tl.int64) * T * H * V + head_index * V
    ddl_row = dlam_partial + row.to(tl.int64) * NB * SEGT
    channel_base = batch_index.to(tl.int64) * T * H * 2 + head_index * 2

    for i in tl.range(0, SEGT):
        tt = SEGT - 1 - i
        t = T0 + tt
        op0 = tl.load(op + channel_base + t * H * 2 + 0)
        op1 = tl.load(op + channel_base + t * H * 2 + 1)
        slot0 = tl.load(slot + channel_base + t * H * 2 + 0).to(tl.int32)
        second = tl.load(slot + channel_base + t * H * 2 + 1).to(tl.int32)
        erase0 = tl.load(erase + channel_base + t * H * 2 + 0) != 0
        merge = op1 != _NONE_CODE
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        lam = tl.load(decay_row + t)
        live = tl.load(live_base + t * H * R + ranks, mask=keep_r,
                       other=0) != 0
        acc = tl.load(traj_base + tt * R * V + ranks[:, None] * V
                      + values[None, :], mask=keep, other=0.0)
        nr = tl.load(nr_row + tt * R + ranks, mask=keep_r, other=0.0)
        w = tl.load(w_row + tt * R + ranks, mask=keep_r, other=0.0)
        dscores = tl.load(ds_row + tt * R + ranks, mask=keep_r, other=0.0)
        dstrength = tl.load(dst_row + tt * R + ranks, mask=keep_r, other=0.0)
        z = tl.load(z_row + tt * R + ranks, mask=keep_r, other=0.0)
        nq = tl.load(scal_row + tt * 2 + 0)
        duu = tl.load(scal_row + tt * 2 + 1)
        q = tl.load(q_row + t * qts + values * qvs, mask=keep_v,
                    other=0.0).to(tl.float32)
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
        #    ds <- lam * ds (now the gradient w.r.t. the pre-decay state).
        tl.store(ddl_row + vblock * SEGT + tt,
                 tl.sum(tl.sum(_mul_rn(ds, acc), axis=1), axis=0))
        ds = _mul_rn(ds, lam)
        # 3. the MERGE-blend scatter: the blend's content is the pre-decay
        #    rows, and its weights are recorded constants.
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
        # 4. the read adjoint: the direct term, the strength path and the two
        #    score-path terms -- all elementwise given the prep coefficients.
        cr = tl.maximum(nr, _EPS)
        ds = _add_rn(ds, _mul_rn(w[:, None], g[None, :]))
        dnr = _mul_rn(dstrength, tl.where(live, 1.0, 0.0))
        dnr = dnr + tl.where(nr >= _EPS, _div_rn32(_mul_rn(z, -1.0), cr),
                             0.0)
        unit_r = tl.where(nr[:, None] > 0, _div_rn32(acc, nr[:, None]), 0.0)
        ds = _add_rn(ds, _mul_rn(dnr[:, None], unit_r))
        dt = _div_rn32(_mul_rn(dscores, SCALE), cr)
        ds = _add_rn(ds, _mul_rn(dt[:, None],
                                _div_rn32(q, tl.broadcast_to(
                                    tl.maximum(nq, _EPS), (BV,)))[None, :]))
        # 5. dquery: duq = sum_j dt_j * S_j (a small-tile rank sum), then the
        #    clamp-select cross term (the duu identity).
        duq = tl.sum(_mul_rn(dt[:, None], acc), 0)
        cq = tl.maximum(nq, _EPS)
        uq = _div_rn32(q, tl.broadcast_to(cq, (BV,)))
        dq = _div_rn32(duq, tl.broadcast_to(cq, (BV,))) \
            - _mul_rn(uq, tl.where(nq >= _EPS,
                                   tl.broadcast_to(_div_rn32(duu, nq), (BV,)),
                                   0.0))
        tl.store(dq_row + t * H * V + values, dq.to(dquery.dtype.element_ty),
                 mask=keep_v)
        # 6. dpayload: the APPEND/CORRECT passthrough; the MERGE blend's
        #    gradient went to the state rows, POP/NONE carry nothing.
        dpay = tl.where(carries, dpay0, 0.0)
        tl.store(dp_row + t * H * V + values,
                 dpay.to(dpayload.dtype.element_ty), mask=keep_v)

    tl.store(dstate_seg + row.to(tl.int64) * R * V + ranks[:, None] * V
             + values[None, :], ds, mask=keep)


def _seq_backward_body(query, payload, decay, dreads, schedule, live_pre,
                       ckpt_state, ckpt_live, rank, segment_tokens, scale):
    """The whole window's adjoint: segments in reverse, each recomputed from
    its entry checkpoint (the gate-C doctrine), prepped, and walked.

    Memory, stated: the checkpoints are ``ceil(T/SEG) + 1`` states (34 MB at
    the registered shape), the live per-segment scratch is one trajectory
    (67 MB) plus five rank-vector channels and two scalars per token (~2.6
    MB), and the segments run strictly in sequence -- the allocator reuses
    each segment's scratch for the next.  ``dlam`` crosses value blocks and
    its partials are summed in a fixed order in torch (the K4 pattern).
    """
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    R = int(rank)
    SEG = int(segment_tokens)
    decay32 = decay.to(torch.float32).permute(0, 2, 1).contiguous()  # (b,h,T)
    dreads_c = dreads.contiguous()
    dquery = torch.empty_like(query)
    dpayload = torch.empty_like(payload)
    ddecay = torch.empty(batch, heads, tokens, dtype=torch.float32,
                         device=device)
    RP, VP = _pow2(R), _pow2(value_dim)
    LOGV, LOGR = _log2_up(VP), _log2_up(RP)
    nseg = -(-tokens // SEG)
    carry = torch.zeros(batch, heads, R, value_dim, dtype=torch.float32,
                        device=device)
    rows = batch * heads
    nblocks = triton.cdiv(value_dim, BV_BWD)
    prep_pol = (_round_f32(scale),)
    for seg in reversed(range(nseg)):
        start = seg * SEG
        stop = min(start + SEG, tokens)
        segt = stop - start
        # 1. recompute the segment's trajectory from its entry checkpoint by
        #    replaying the recorded schedule (K4's own kernel -- the walk is
        #    the forward's walk, so the states are bit-identical to the ones
        #    a storing forward would have saved).
        zeros_w = torch.zeros(batch, heads, segt, R, dtype=torch.float32,
                              device=device)
        _, _, _, traj = persistent_scan_trajectory(
            ckpt_state[seg], ckpt_live[seg], zeros_w,
            decay32[:, :, start:stop], schedule.chunk(start, stop))
        # 2. prep: the per-token weights and adjoint coefficients.
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
        # 3. walk: the value-blocked reverse adjoint.
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
        ddecay[:, :, start:stop] = dlam_partial.sum(dim=1).view(
            batch, heads, segt)
        carry = dstate_seg
    dstate = carry
    return dquery, dpayload, ddecay.permute(0, 2, 1), dstate


# ------------------------------------------------------------ the op pair


@torch.library.custom_op("x45::seq_scan_fwd", mutates_args=())
def _seq_scan_fwd_op(
    query: torch.Tensor, payload: torch.Tensor, decay: torch.Tensor,
    state: torch.Tensor, live: torch.Tensor, rank: int, segment_tokens: int,
    tau_match: float, tau_merge: float, tau_pop: float, tie_break: float,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole (layer, window) sequential scan as ONE opaque op (the K4/K5A
    hosting pattern): the persistent walk's every channel as its outputs, so
    autograd itself keeps the backward's saves alive.  ``state``/``live`` are
    explicit inputs -- the window-entry state -- so ``dstate`` is a real
    gradient output and the parity fixtures can compare it.
    """
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


@_seq_scan_fwd_op.register_fake
def _(query, payload, decay, state, live, rank, segment_tokens, tau_match,
      tau_merge, tau_pop, tie_break, scale):
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    nseg = -(-tokens // int(segment_tokens))
    shape = (batch, tokens, heads, 2)
    return (torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty_like(state),
            torch.empty_like(live),
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
                        device=device))


def _seq_scan_setup(ctx, inputs, output):
    query, payload, decay, state, live, rank, segment_tokens, tau_match, \
        tau_merge, tau_pop, tie_break, scale = inputs
    # the backward needs the inputs' values (query, decay) and the walk's own
    # recorded channels; the reads and the exit state carry no gradient path.
    ctx.save_for_backward(query, payload, decay, state, live,
                           *output[3:])        # slot..ckpt_live
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)


@torch.library.custom_op("x45::seq_scan_bwd", mutates_args=())
def _seq_scan_bwd_op(
    query: torch.Tensor, payload: torch.Tensor, decay: torch.Tensor,
    dreads: torch.Tensor, slot: torch.Tensor, op: torch.Tensor,
    erase: torch.Tensor, live_after: torch.Tensor, payload_field: torch.Tensor,
    live_pre: torch.Tensor, ckpt_state: torch.Tensor, ckpt_live: torch.Tensor,
    rank: int, segment_tokens: int, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole window's adjoint as its own opaque op: the segment loop
    (recompute -> prep -> walk) runs inside, decisions as recorded constants."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _seq_backward_body(query, payload, decay, dreads, schedule,
                              live_pre, ckpt_state, ckpt_live, int(rank),
                              int(segment_tokens), float(scale))


@_seq_scan_bwd_op.register_fake
def _(query, payload, decay, dreads, slot, op, erase, live_after,
      payload_field, live_pre, ckpt_state, ckpt_live, rank, segment_tokens,
      scale):
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    return (torch.empty_like(query), torch.empty_like(payload),
            torch.empty((batch, tokens, heads), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                        device=device))


def _seq_scan_backward(ctx, dreads, dstate_out, dlive, dslot, dop, derase,
                        dlive_after, dpayload_field, dlive_pre, dckpt_state,
                        dckpt_live):
    """The adjoint: dquery, dpayload, ddecay, dstate; the decision channel and
    the liveness carry no gradient by construction (the standing doctrine)."""
    query, payload, decay, state, live, slot, op, erase, live_after, \
        payload_field, live_pre, ckpt_state, ckpt_live = ctx.saved_tensors
    batch, tokens, heads, value_dim = query.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=query.device)
    dquery, dpayload, ddecay, dstate = _seq_scan_bwd_op(
        query, payload, decay, dreads, slot, op, erase, live_after,
        payload_field, live_pre, ckpt_state, ckpt_live, ctx.rank,
        ctx.segment_tokens, ctx.scale)
    return dquery, dpayload, ddecay, dstate, None, None, None, None, None, \
        None, None, None


torch.library.register_autograd("x45::seq_scan_fwd", _seq_scan_backward,
                                setup_context=_seq_scan_setup)


def _seq_warps(rank: int, value_dim: int) -> int:
    """Warps for the whole-store tile: 16 floats per thread at 8 warps, the
    measured no-spill point at the registered shape (receipt, gate D)."""
    return 8 if _pow2(rank) * _pow2(value_dim) > 2048 else 4


# --------------------------------------------------------------- the wrapper


def sequential_scan(query, payload, decay, *, policy: RoutingPolicy, rank=None,
                    rank_max=None, segment_tokens=DEFAULT_SEGMENT_TOKENS,
                    initial_state=None, initial_live=None,
                    collect=False) -> RoutedResult:
    """One whole (layer, window) scan at PER-TOKEN granularity.

    The drop-in for ``scan.routed_chunk_scan`` at the mixer's boundary: the
    registered five-op policy evaluated against the CURRENT state at every
    token (the chunk-1 limit of the registered dial), reads included.  The
    kernel path is ONE opaque op pair per scan with a coarse-checkpoint
    backward; every refusal takes the EAGER ORACLE -- the same semantics, the
    parity fixed point -- never a silent granularity change:

    * ``policy.ops=False`` (the append-only control) and ``policy.grow`` are
      refused outright: both are chunk-dial concepts this registration does
      not carry;
    * a rank prefix (``rank != rank_max``) is refused by name (the registration
      runs the full allocated rank);
    * a store the persistent walk cannot hold (``sequential_available``), or
      non-CUDA inputs, take the oracle verbatim.

    ``collect=True`` (the diagnostics instrument) runs the walk without a
    graph and returns the recorded schedule and read weights.
    """
    batch, tokens, heads, value_dim = payload.shape
    device = query.device
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    rank, rank_max = int(rank), int(rank_max)
    if not policy.ops:
        raise ValueError("the append-only control is refused by the sequential "
                         "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the sequential registration: "
                         "growth is a chunk-boundary concept")
    if rank != rank_max:
        raise ValueError("the sequential registration requires rank == "
                         "rank_max (the registered shape); a rank prefix is "
                         "refused by name")
    state = (torch.zeros(batch, heads, rank_max, value_dim,
                         dtype=torch.float32, device=device)
             if initial_state is None else initial_state)
    live = (torch.zeros(batch, heads, rank_max, dtype=torch.bool, device=device)
            if initial_live is None else initial_live)
    if collect:
        # the record instrument: the raw launch, no autograd boundary
        pol32 = (_round_f32(policy.tau_match), _round_f32(policy.tau_merge),
                 _round_f32(policy.tau_pop), _round_f32(policy.tie_break),
                 _round_f32(policy.read_scale))
        with torch.no_grad():
            out = _seq_forward_launch(
                query, payload, decay, state, live, pol32, rank, segment_tokens,
                collect=True, warps=_seq_warps(rank, value_dim), stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                   payload=out["payload"], op=out["op"],
                                   live_after=out["live_after"])
        return RoutedResult(out["reads"], out["state"], out["live"], rank,
                           ScanStats(), schedule, out["weights"])
    if query.is_cuda and sequential_available(rank, value_dim):
        reads, state_out, live_out, _slot, _op, _erase, _live_after, \
            _payload, _live_pre, _ckpt_state, _ckpt_live = _seq_scan_fwd_op(
                query, payload, decay, state, live, rank, int(segment_tokens),
                policy.tau_match, policy.tau_merge, policy.tau_pop,
                policy.tie_break, policy.read_scale)
        return RoutedResult(reads, state_out, live_out, rank, ScanStats(),
                            None, None)
    # the eager oracle: the same semantics, the parity fixed point
    ref = sequential_oracle_scan(query, payload, decay, policy=policy,
                                  rank=rank, rank_max=rank_max,
                                  initial_state=state, initial_live=live,
                                  collect=False)
    return RoutedResult(ref["reads"], ref["state"], ref["live"], rank,
                        ScanStats(), None, None)


# ------------------------------------------------------- the mixer and the model


class SequentialMatrixMixer(MatrixMixer):
    """The K6 mixer: the parent's projections and DECAY gate verbatim, the
    scan swapped for the per-token sequential one.

    Constructed by :func:`convert_model_to_sequential` from an existing
    ``MatrixMixer`` (the same parameter modules, the same init), so a model
    built by the matched-arms build order and converted shares the sweep
    arms' init bit for bit and ``model.parameters()`` is unchanged.
    """

    def __init__(self, *args, segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.segment_tokens = int(segment_tokens)

    def mix(self, x: torch.Tensor):
        batch, tokens, _ = x.shape
        query = self.query(x).view(batch, tokens, self.heads, self.value_dim)
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
        decay = torch.sigmoid(self.decay(x)).clamp(min=DECAY_FLOOR,
                                                    max=1.0 - 1e-6)
        result = sequential_scan(query, payload, decay, policy=self.policy,
                                 rank=self.rank, rank_max=self.rank_max,
                                 segment_tokens=self.segment_tokens)
        return (self.out(result.reads.reshape(batch, tokens,
                                              self.heads * self.value_dim)),
                result)


def convert_model_to_sequential(model, segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
    """Swap every ``MatrixMixer`` for the sequential one, sharing parameters.

    The conversion is a shallow re-wrap: each new mixer's ``__dict__`` receives
    the old one's, so the ``nn.Linear`` modules (and therefore the parameters
    the optimizer sees, and the init the seed produced) are the very same
    objects.  ``model.py`` is untouched.
    """
    for block in model.blocks:
        mixer = block.mixer
        seq = SequentialMatrixMixer.__new__(SequentialMatrixMixer)
        torch.nn.Module.__init__(seq)
        seq.__dict__.update(mixer.__dict__)
        seq.segment_tokens = int(segment_tokens)
        block.mixer = seq
    return model


__all__ = ["DEFAULT_SEGMENT_TOKENS", "MAX_SEQ_TILE", "RoutedResult",
           "ScanStats", "MatrixMixer", "MatrixSchedule", "RoutingPolicy",
           "SequentialMatrixMixer", "EPS", "convert_model_to_sequential",
           "persistent_scan_trajectory", "sequential_available",
           "sequential_oracle_scan", "sequential_scan", "tree_fold"]
