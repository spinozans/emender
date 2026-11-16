"""K7 -- THE HIGH-RANK STORE: the rank ceiling removed from both sides of the scan.

Why this file exists
--------------------
Every one-program-owns-the-store kernel in this lane refuses the growth snap set
{1024, 2048} by name: ``hostloop_free.persistent_available`` (8 192 register
floats) refuses the persistent chunk-granular block scan beyond rank 512, K4's
fused backward inherits the same gate, and K6's per-token kernel refuses beyond
``rank x d_v`` = 16 384 floats (rank 128 at ``d_v`` 128) and falls back to the
eager oracle.  The refusal is honest -- a register tile is a real budget -- but
it caps the emender at ~4x transformer-order slot capacity.  This module removes
the ceiling on both sides, WITHOUT touching a single registered semantic:

Part A -- the slot-blocked chunked value scan (``k7_read_weights`` / ``k7_read``)
    The inter/intra read kernels of K2+K3 are re-gridded over slot blocks: the
    grid is ``(row, token-block, slot-block)`` and each program holds a RANK TILE
    of ``BR`` slots, so the read's parallelism scales with the rank instead of
    starving inside one program's serial rank loop.  The read's softmax moves
    into the kernels with it, and its denominator -- which no single program can
    see -- is combined by THE LAW the Lean truncation result fixes
    (``formal/lean/OPERATOR_ALGEBRA_STATUS.md``, EMENDER_ATTENTION_ALGEBRA):
    per-operator normalisation does NOT make truncation harmless
    (``Obstructions.truncation_sharpens_survivor``: a per-block renormalisation
    doubles a survivor's coefficient), and the exact resolution is to normalise
    by the FULL-POPULATION exact mass, carrying each block's exact mass as an
    explicit scalar and combining them in ONE reduction.  Implemented here:

        m_b = tree over the tile of exp(score - gmax)      the per-block masses,
                                                            explicit scalars
        denom = ONE balanced tree over the slot-block axis  the full-population mass
        w_j = mul.rn(div.rn(exp(score_j - gmax), denom), strength_j)
                                                            ONE normalisation, the
                                                            registered expression

    Never ``w_j = exp(score_j)/m_b`` -- that is the obstruction, and the gate
    that would catch it is the weights channel's parity against ``read.read_weights``
    (the single-sourced definition, unchanged).  The tile-internal and
    cross-block trees decompose the K6-defined whole-axis balanced tree
    bit-exactly (zeros pad exactly), so the denominator's order is reproducible.

    The read terms themselves keep K2+K3's structure: per slot-block partials
    (``tl.dot`` fp32 ``ieee``, the causal mask as a predicate, the intra term
    additionally masked to the steps whose addressed slot falls in the tile --
    each step belongs to exactly one slot block), combined in ONE fixed-order
    reduction (the K4 partials discipline: no atomics, one writer per element,
    bit-deterministic).  The state walk is K1, reused verbatim -- it is already
    slot-blocked at any rank.

Part B -- the high-rank sequential decision walk (``k7_sequential_walk``)
    The K6 per-token walk's store does not fit one program at rank 1024/2048
    (the receipted register ceiling), so the walk's state moves to GLOBAL memory
    and the program sweeps it in slot tiles.  The design fork -- global-tiled
    state vs compressed decision state -- is resolved in
    ``DESIGN_HIGH_RANK_WALK.md`` (written first): the compressed option is
    REFUSED because its scalar recurrences are not exact in the defined fp32
    order, APPEND breaks the strength recurrence outright, and the
    query-alignment sketch is a new registered semantic.  The walk keeps K6's
    arithmetic exactly (defined value-axis trees, ``mul.rn``/``div.rn``/
    ``sqrt.rn``, libdevice ``exp``, the first-index tie rules), so DECISIONS and
    READ WEIGHTS are bit-exact against the K7 oracle
    (``sequential_oracle``, written to the kernel's reduction order -- the K6
    precedent); the read OUTPUT is a native per-tile sum plus sequential
    cross-tile adds, the one order no torch expression reproduces, and is held
    to the measured floor (the K1/K23/K6 corner: the read feeds no decision).

Growth readiness (design requirement, not wired)
------------------------------------------------
Nothing here precludes mid-sequence rank change: every shape is read from the
tensors at call time, ``RP``/``SB``/``NT`` are the only rank-derived constexprs
(each snap in ``keys.ORTHONORMAL_RANKS`` is its own cached kernel binary), the
channels are allocated per call at the snap's width, and a growth event at a
window boundary re-dispatches to the next snap's kernels or its pre-captured
graph with the new rows exact zeros -- the prefix-reveal invariant the chunked
arm's growth already relies on.  Wiring a growth wave is out of scope.

Forward only, as every kernel in this lane: ``check_no_grad`` refuses a graph.
"""
from __future__ import annotations

import struct

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .algebra import APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule
from .k1_schedule_scan import _add_rn, _mul_rn, check_no_grad, k1_state_scan
from .k23_read import check_k23_shapes
from .schedule import EPS, RoutingPolicy

# The rank tile one Part-A program holds: BR x VP fp32 in registers, the same
# 8 192-float budget the lane's persistent kernels measured as the no-spill tile
# (NO_HOST_LOOP.md section 4).  The token and step tiles match K2+K3's scale.
K7_RANK_TILE = 64
K7_TOKEN_TILE = 16
K7_STEP_TILE = 64
# ``tl.dot`` wants every operand axis a multiple of 16; the value axis of the
# Part-A kernels is padded up to 16 and masked down (k23's MIN_TILE rule).
MIN_TILE = 16
# A value dim wider than this is a different register budget, not a different
# kernel: refused rather than silently spilling (k23's ceiling, kept).
MAX_VALUE_DIM = 256
# The walk's probe ceiling: its tile is BR x VP with BR x VP <= MAX_WALK_TILE.
MAX_WALK_TILE = 8192
MAX_WALK_VALUE_DIM = 128
K7_WALK_TILE = 64


# ----------------------------------------------------------- the defined order, shared

def _pow2(n: int) -> int:
    return int(triton.next_power_of_2(int(n)))


def _log2_up(width: int) -> int:
    return max(1, (int(width) - 1).bit_length())


def _round_f32(value: float) -> float:
    """The threshold in the comparison's own dtype (fp32 here, as in K5A/K6)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _fp32(x: torch.Tensor) -> torch.Tensor:
    """Upcast to the spec's fp32 -- but never DOWN: the fp64 oracle twin (the
    floor reference) must stay fp64 (the K6 lesson, section 6.4)."""
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.to(torch.float32)
    return x


def tree_fold(x: torch.Tensor, dim: int, width: int) -> torch.Tensor:
    """The DEFINED reduction order (the K6 registration's), self-contained here.

    A balanced binary tree over ``dim`` padded to ``width`` (a power of two) with
    exact zeros, a chain of ``view(..., n/2, 2)`` pair-sums.  Zero-padding a
    defined tree to a LARGER power of two is bit-identical, which is what makes
    the slot-block decomposition of a whole-axis tree exact: a tile's internal
    tree plus one tree over the per-tile results IS the whole-axis tree.
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
    """``sqrt(tree_fold(x*x))`` -- the DEFINED norm (torch sqrt is sqrt.rn)."""
    s = tree_fold(x * x, dim, width)
    mask = (s > 0).to(x.dtype)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    return torch.sqrt(safe) * mask


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


def _slot_blocks(rank: int, value_dim: int, tile: int = K7_RANK_TILE) -> int:
    """The slot-block count: a power of two by construction (RP and BR are)."""
    return triton.cdiv(_pow2(rank), tile)


# ============================================================== Part A: the read kernels


@triton.jit(do_not_specialize=['T', 'R'])
def _k7_scores_kernel(
    query, entry_state, live,
    scores_ch, strength_ch, maxes_ch,
    T, R,
    qbs, qhs, qts, qvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BT: tl.constexpr, BR: tl.constexpr, SB: tl.constexpr,
    LOGV: tl.constexpr, SCALE: tl.constexpr,
):
    """The chunk-entry scores per ``(row, token-block, slot-block)``.

    Each program holds a rank tile as ``(VP, BR)`` (transposed for the one
    ``tl.dot`` the cosine needs -- at chunk granularity the scores feed only the
    READ, the decisions being the recorded schedule, so the dot's fp32-ieee
    order is a floors question, exactly K2+K3's).  Writes the masked score
    channel (``-inf`` on dead rows, the registered mask), the strength channel
    ``nr * live`` and the tile's max -- the softmax's max is exact, so the
    cross-block max combine needs no order at all.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    sb = tl.program_id(2)
    batch_index = row // H
    head_index = row % H
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    ranks = sb * BR + tl.arange(0, BR)
    keep_r = ranks < R
    values = tl.arange(0, VP)
    keep_v = values < V

    query_row = query + batch_index * qbs + head_index * qhs
    q = tl.load(query_row + toks[:, None] * qts + values[None, :] * qvs,
                mask=keep_t[:, None] & keep_v[None, :], other=0.0)
    nq = tl.reshape(_sqrt_rn32(_fold_axis2(_mul_rn(q, q), VP, LOGV)), (BT,))
    uq = _div_rn32(q, tl.broadcast_to(tl.maximum(nq, _EPS)[:, None], (BT, VP)))

    tile = tl.load(entry_state + row.to(tl.int64) * R * V
                   + ranks[None, :] * V + values[:, None],
                   mask=keep_r[None, :] & keep_v[:, None], other=0.0)
    nr = tl.reshape(_sqrt_rn32(_fold_axis0(_mul_rn(tile, tile), VP, LOGV)), (BR,))
    unit_rows = _div_rn32(tile,
                          tl.broadcast_to(tl.maximum(nr, _EPS)[None, :], (VP, BR)))
    scores = _mul_rn(tl.dot(uq, unit_rows, input_precision='ieee'), SCALE)

    live_tile = tl.load(live + row * R + ranks, mask=keep_r, other=0) != 0
    masked = tl.where(live_tile[None, :] & keep_r[None, :], scores, float('-inf'))
    strength = _mul_rn(nr, tl.where(live_tile & keep_r, 1.0, 0.0))

    ch_base = row.to(tl.int64) * T * R
    offsets = toks[:, None] * R + ranks[None, :]
    channel_mask = keep_t[:, None] & keep_r[None, :]
    tl.store(scores_ch + ch_base + offsets, masked, mask=channel_mask)
    # the strength channel is per slot (the chunk-entry store), shared by every
    # token of the chunk -- broadcast over the token tile on the way out.
    tl.store(strength_ch + ch_base + offsets,
             tl.broadcast_to(strength[None, :], (BT, BR)), mask=channel_mask)
    tile_max = tl.max(masked, 1)
    tl.store(maxes_ch + row.to(tl.int64) * T * SB + toks * SB + sb, tile_max,
             mask=keep_t)


@triton.jit(do_not_specialize=['T', 'R'])
def _k7_mass_kernel(
    scores_ch, gmax_ch, masses_ch,
    T, R,
    H: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr, SB: tl.constexpr,
    LOGBR: tl.constexpr,
):
    """The per-block masses: THE explicit scalars the Lean law carries.

    ``m_b = tree over the tile of exp(score - gmax)`` -- dead rows arrive as
    ``-inf`` in the score channel, so their ``exp`` is exactly 0 and the tile
    tree's zero-padding is exact.  One scalar per (row, token, slot-block).
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    sb = tl.program_id(2)
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    ranks = sb * BR + tl.arange(0, BR)
    keep_r = ranks < R

    scores = tl.load(scores_ch + row.to(tl.int64) * T * R
                      + toks[:, None] * R + ranks[None, :],
                      mask=keep_t[:, None] & keep_r[None, :], other=float('-inf'))
    gmax = tl.load(gmax_ch + row.to(tl.int64) * T + toks, mask=keep_t, other=0.0)
    expo = libdevice.exp(scores - gmax[:, None])
    mass = tl.reshape(_fold_axis2(expo, BR, LOGBR), (BT,))
    tl.store(masses_ch + row.to(tl.int64) * T * SB + toks * SB + sb, mass,
             mask=keep_t)


@triton.jit(do_not_specialize=['T', 'R'])
def _k7_weights_kernel(
    scores_ch, strength_ch, gmax_ch, denom_ch, weights_out,
    T, R,
    H: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr,
):
    """ONE normalisation by the full-population mass: the registered expression.

    ``w_j = mul.rn(div.rn(exp(score_j - gmax), denom), strength_j)`` with ``denom``
    the combined full mass; an all-dead token has ``denom == 0`` and its weights
    are exactly zero (the certified all-dead zero attention).  Never a per-block
    renormalisation -- that is ``truncation_sharpens_survivor``.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    sb = tl.program_id(2)
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    ranks = sb * BR + tl.arange(0, BR)
    keep_r = ranks < R

    offsets = toks[:, None] * R + ranks[None, :]
    channel_mask = keep_t[:, None] & keep_r[None, :]
    ch_base = row.to(tl.int64) * T * R
    scores = tl.load(scores_ch + ch_base + offsets, mask=channel_mask,
                     other=float('-inf'))
    strength = tl.load(strength_ch + ch_base + offsets, mask=channel_mask,
                       other=0.0)
    gmax = tl.load(gmax_ch + row.to(tl.int64) * T + toks, mask=keep_t, other=0.0)
    denom = tl.load(denom_ch + row.to(tl.int64) * T + toks, mask=keep_t,
                    other=0.0)
    expo = libdevice.exp(scores - gmax[:, None])
    probs = tl.where(denom[:, None] > 0,
                     _div_rn32(expo, tl.broadcast_to(denom[:, None],
                                                      (BT, BR))), 0.0)
    w = _mul_rn(probs, strength)
    tl.store(weights_out + ch_base + offsets, w, mask=channel_mask)


@triton.jit(do_not_specialize=['T', 'N', 'R'])
def _k7_inter_kernel(
    weights, entry_state, log_gate, partials,
    T, N, R,
    wbs, whs, wt, wr,
    gbs, ghs, gs,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BT: tl.constexpr, BR: tl.constexpr, SB: tl.constexpr,
):
    """K3 re-gridded: one rank tile per program, a per-block partial out.

    The per-tile expression is k23's own (``dot(w * exp(gc_read), H_tile)``,
    fp32 ``ieee``); the rank axis that k23 walked in a serial in-program loop is
    the grid axis here, so the read's parallelism scales with the rank.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    sb = tl.program_id(2)
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    ranks = sb * BR + tl.arange(0, BR)
    keep_r = ranks < R
    values = tl.arange(0, VP)
    keep_v = values < V

    read_step = toks * L - 1
    first = read_step < 0
    gc_read = tl.load(log_gate + (row // H) * gbs + (row % H) * ghs
                      + tl.where(first, 0, read_step) * gs,
                      mask=keep_t & ~first, other=0.0)
    decay_weight = tl.exp(gc_read)

    w = tl.load(weights + (row // H) * wbs + (row % H) * whs
                + toks[:, None] * wt + ranks[None, :] * wr,
                mask=keep_t[:, None] & keep_r[None, :], other=0.0)
    rows_tile = tl.load(entry_state + row.to(tl.int64) * R * V
                        + ranks[:, None] * V + values[None, :],
                        mask=keep_r[:, None] & keep_v[None, :], other=0.0)
    partial = tl.dot(w * decay_weight[:, None], rows_tile, input_precision='ieee')

    pbase = (row.to(tl.int64) * SB + sb) * T * VP
    tl.store(partials + pbase + toks[:, None] * VP + values[None, :], partial,
             mask=keep_t[:, None] & keep_v[None, :])


@triton.jit(do_not_specialize=['T', 'N', 'R'])
def _k7_intra_kernel(
    weights, delta, log_gate, slot, intra_partials,
    T, N, R,
    wbs, whs, wt, wr,
    dbs, dhs, ds, dv,
    gbs, ghs, gs,
    kbs, khs, kt, kl,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BT: tl.constexpr, BU: tl.constexpr, NB: tl.constexpr,
):
    """K2 re-gridded over STEP blocks: the intra term's natural decomposition.

    Every step ``u`` contributes ``exp(gc_read_t - gc_u) * w_t[j_u] * d_u`` to
    every later token ``t`` -- a sum over the STEP axis, whose blocking is
    free (one fixed-order reduction over the per-block partials).  Blocking
    it by the steps' ADDRESSED SLOT instead multiplies the dot work by the
    slot-block count (each step belongs to one tile, so all but one program
    would zero the same coefficient matrix -- measured 2x the k23 block wall
    at rank 2048), which is why the step axis is the grid axis here and the
    causal predicate is k23's own, unchanged.  One writer per partial element,
    no atomics.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    n = tl.program_id(2)
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    values = tl.arange(0, VP)
    keep_v = values < V

    read_step = toks * L - 1
    first = read_step < 0
    gc_read = tl.load(log_gate + (row // H) * gbs + (row % H) * ghs
                      + tl.where(first, 0, read_step) * gs,
                      mask=keep_t & ~first, other=0.0)

    steps = n * BU + tl.arange(0, BU)
    keep_u = steps < N
    bound = tl.minimum(N, (block * BT + BT) * L)
    if n * BU < bound:
        gc_step = tl.load(log_gate + (row // H) * gbs + (row % H) * ghs
                          + steps * gs, mask=keep_u, other=0.0)
        target = tl.load(slot + (row // H) * kbs + (row % H) * khs
                        + (steps // L) * kt + (steps % L) * kl,
                        mask=keep_u, other=0).to(tl.int32)
        addressed = tl.load(weights + (row // H) * wbs + (row % H) * whs
                            + toks[:, None] * wt + target[None, :] * wr,
                            mask=keep_t[:, None] & keep_u[None, :], other=0.0)
        causal = (steps[None, :] < toks[:, None] * L) & keep_t[:, None] \
            & keep_u[None, :]
        exponent = tl.where(causal, gc_read[:, None] - gc_step[None, :], 0.0)
        coefficients = tl.where(causal, addressed * tl.exp(exponent), 0.0)
        payload = tl.load(delta + (row // H) * dbs + (row % H) * dhs
                          + steps[:, None] * ds + values[None, :] * dv,
                          mask=keep_u[:, None] & keep_v[None, :], other=0.0)
        partial = tl.dot(coefficients, payload, input_precision='ieee')
    else:
        partial = tl.zeros((BT, VP), dtype=tl.float32)

    pbase = (row.to(tl.int64) * NB + n) * T * VP
    tl.store(intra_partials + pbase + toks[:, None] * VP + values[None, :],
             partial, mask=keep_t[:, None] & keep_v[None, :])


def _check_weights_inputs(query: torch.Tensor, entry_state: torch.Tensor,
                          live: torch.Tensor) -> None:
    """The Part-A weights audit, before any launch."""
    if not (query.is_cuda and entry_state.is_cuda and live.is_cuda):
        raise ValueError("K7 is a device path; inputs must be on CUDA")
    if query.dim() != 4 or entry_state.dim() != 4:
        raise ValueError("query must be (batch, heads, tokens, value_dim) and the "
                         "entry state (batch, heads, rank, value_dim)")
    batch, heads, tokens, value_dim = query.shape
    rank = entry_state.shape[2]
    if entry_state.shape != (batch, heads, rank, value_dim):
        raise ValueError(f"entry state {tuple(entry_state.shape)} does not match "
                         f"the query {tuple(query.shape)}")
    if live.shape != (batch, heads, rank) or live.dtype != torch.bool:
        raise ValueError("live must be a bool (batch, heads, rank) tensor")
    if value_dim > MAX_VALUE_DIM:
        raise ValueError(f"K7 compiles for value dims up to {MAX_VALUE_DIM}, "
                         f"got {value_dim}")
    if entry_state.dtype != torch.float32 or not entry_state.is_contiguous():
        raise ValueError("the entry state must be contiguous fp32; the algebra "
                         "is fp32")
    if tokens < 1:
        raise ValueError("tokens must be positive")


def k7_read_weights(query: torch.Tensor, entry_state: torch.Tensor,
                     live: torch.Tensor, scale: float = 2.0,
                     bt: int = K7_TOKEN_TILE, br: int = K7_RANK_TILE):
    """The chunk's read-weight channel at ANY rank, as three slot-blocked launches.

    The registered formula (``read.read_weights``, the single-sourced
    definition): ``softmax over live entry rows of scale*cos(q_t, S_entry[j])``
    times the row strength, all-dead exactly zero, no renormaliser.  What is new
    is only the grid: the softmax's max (exact) and its denominator (the
    per-block masses carried as explicit scalars, combined in ONE defined tree
    reduction -- the Lean law) span the slot blocks.  ``read.py`` keeps the
    definition; the parity gate holds this to it at the measured floor, because
    torch's own softmax order is not reproducible from a kernel (the K5A
    receipt).

    ``query`` ``(batch, heads, tokens, value_dim)`` is the router's layout;
    returns ``(batch, heads, tokens, rank)`` fp32.
    """
    _check_weights_inputs(query, entry_state, live)
    check_no_grad(query)
    batch, heads, tokens, value_dim = query.shape
    rank = entry_state.shape[2]
    rows = batch * heads
    vp = max(MIN_TILE, _pow2(value_dim))
    if br * vp > 8192:
        br = max(16, 8192 // vp)
    sb = _slot_blocks(rank, value_dim, br)
    device = query.device
    scores = torch.empty((batch, heads, tokens, rank), dtype=torch.float32,
                         device=device)
    strength = torch.empty_like(scores)
    maxes = torch.empty((batch, heads, tokens, sb), dtype=torch.float32,
                         device=device)
    grid = (rows, triton.cdiv(tokens, bt), sb)
    _k7_scores_kernel[grid](
        query, entry_state, live.view(torch.int8), scores, strength, maxes,
        tokens, rank,
        *query.stride(),
        H=heads, V=value_dim, VP=vp, BT=bt, BR=br, SB=sb,
        LOGV=_log2_up(vp), SCALE=_round_f32(scale), num_warps=4)
    # the max combine: exact, so no order needs stating
    gmax = maxes.amax(dim=-1)
    gmax = torch.where(gmax.isfinite(), gmax, torch.zeros_like(gmax))
    masses = torch.empty((batch, heads, tokens, sb), dtype=torch.float32,
                         device=device)
    _k7_mass_kernel[grid](
        scores, gmax, masses,
        tokens, rank,
        H=heads, BT=bt, BR=br, SB=sb, LOGBR=_log2_up(br), num_warps=4)
    # THE ONE REDUCTION: the full-population exact mass as the defined balanced
    # tree over the slot-block axis (bit-identical to the whole-axis tree).
    denom = tree_fold(masses, -1, sb)
    weights = torch.empty_like(scores)
    _k7_weights_kernel[grid](
        scores, strength, gmax, denom, weights,
        tokens, rank,
        H=heads, BT=bt, BR=br, num_warps=4)
    return weights


def k7_read(weights: torch.Tensor, entry_state: torch.Tensor, chunk,
            schedule: MatrixSchedule, bt: int = K7_TOKEN_TILE,
            br: int = K7_RANK_TILE, bu: int = K7_STEP_TILE) -> torch.Tensor:
    """The chunk's read as two slot-blocked launches plus one fixed-order combine.

    ``weights`` is an input channel exactly as in K2+K3 (``read.read_weights``
    at the chunk-entry state, from ``schedule.route_chunk`` -- or
    :func:`k7_read_weights` at any rank).  Per slot block the inter and intra
    partials are ``tl.dot`` fp32-ieee tiles; the partials are summed over the
    slot-block axis in ONE defined tree reduction.  No atomics; one writer per
    element; bit-deterministic.
    """
    check_k23_shapes(weights, entry_state, chunk, schedule)
    check_no_grad(weights)
    batch, heads, tokens, rank = weights.shape
    value_dim = entry_state.shape[-1]
    rows = batch * heads
    vp = max(MIN_TILE, _pow2(value_dim))
    if br * vp > 8192:
        br = max(16, 8192 // vp)
    sb = _slot_blocks(rank, value_dim, br)
    nb = triton.cdiv(chunk.steps, bu)
    partials = torch.empty((batch, heads, sb, tokens, vp), dtype=torch.float32,
                           device=weights.device)
    intra_partials = torch.empty((batch, heads, nb, tokens, vp),
                                 dtype=torch.float32, device=weights.device)
    grid = (rows, triton.cdiv(tokens, bt), sb)
    _k7_inter_kernel[grid](
        weights, entry_state, chunk.log_gate, partials,
        tokens, chunk.steps, rank,
        *weights.stride(),
        chunk.log_gate.stride(0), chunk.log_gate.stride(1), chunk.log_gate.stride(2),
        H=heads, L=schedule.lanes, V=value_dim, VP=vp, BT=bt, BR=br, SB=sb,
        num_warps=4)
    _k7_intra_kernel[(rows, triton.cdiv(tokens, bt), nb)](
        weights, chunk.delta, chunk.log_gate, schedule.slot, intra_partials,
        tokens, chunk.steps, rank,
        *weights.stride(),
        *chunk.delta.stride(),
        chunk.log_gate.stride(0), chunk.log_gate.stride(1), chunk.log_gate.stride(2),
        schedule.slot.stride(0), schedule.slot.stride(2),
        schedule.slot.stride(1), schedule.slot.stride(3),
        H=heads, L=schedule.lanes, V=value_dim, VP=vp, BT=bt, BU=bu, NB=nb,
        num_warps=4)
    # the ONE join: two fixed-order tree reductions over the block axes, one
    # fp32 add -- the eager path's ``entry_read + injected``.
    return (tree_fold(partials, 2, sb)[..., :value_dim]
            + tree_fold(intra_partials, 2, nb)[..., :value_dim])


def k7_chunk(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
             decay: torch.Tensor, schedule: MatrixSchedule):
    """Drop-in for ``k23_chunk`` at ANY rank: K1 (verbatim) + the slot-blocked read.

    ``decay`` is head-major ``(batch, heads, tokens)`` as the block paths take
    it.  Three-plus launches per chunk become seven (the softmax's scores, mass
    and normalise passes are separate because the denominator spans slot
    blocks), which is the honest price of a grid whose rank axis is a grid axis;
    at ranks where K2+K3's single-program rank loop is not starved the existing
    path stays the faster one, and nothing here removes it.
    """
    check_no_grad(weights)
    chunk = k1_state_scan(state, live, decay.permute(0, 2, 1), schedule)
    reads = k7_read(weights, state, chunk, schedule)
    return reads, chunk.state, chunk.live


def k7_chunk_routed(query: torch.Tensor, state: torch.Tensor, live: torch.Tensor,
                    decay: torch.Tensor, schedule: MatrixSchedule,
                    scale: float = 2.0):
    """One chunk at any rank with the weights computed in-kernel (the split
    softmax) -- the path that removes the rank ceiling end to end."""
    weights = k7_read_weights(query, state, live, scale)
    return k7_chunk(state, live, weights, decay, schedule)


def _check_block(state, live, weights, decay, schedule):
    from .hostloop_free import check_block_shapes
    check_block_shapes(state, live, weights, decay, schedule)


def _check_block_routed(query, state, live, decay, schedule):
    """The block audit for the routed form: same boundary, the weights channel is
    produced by the kernels instead of being an input."""
    if not (query.is_cuda and state.is_cuda and decay.is_cuda
            and schedule.payload.is_cuda):
        raise ValueError("all block paths are device kernels; inputs must be on CUDA")
    batch, heads, rank, value_dim = state.shape
    tokens = schedule.tokens
    if query.shape != (batch, heads, tokens, value_dim):
        raise ValueError(f"query {tuple(query.shape)} does not match the state "
                         f"{tuple(state.shape)} and {tokens} tokens")
    if decay.shape != (batch, heads, tokens):
        raise ValueError(f"decay must be {(batch, heads, tokens)}, got "
                         f"{tuple(decay.shape)}")
    if schedule.payload.shape != (batch, tokens, heads, schedule.lanes,
                                  value_dim):
        raise ValueError("the schedule's payload does not belong to this block")
    for name, tensor in (("state", state), ("query", query), ("decay", decay),
                         ("payload", schedule.payload)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be fp32; the algebra is fp32")
    if not state.is_contiguous() or not live.is_contiguous():
        raise ValueError("state and live must be contiguous")
    if live.dtype != torch.bool or live.shape != (batch, heads, rank):
        raise ValueError("live must be a bool (batch, heads, rank) tensor")


def k7_block_scan(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
                  decay: torch.Tensor, schedule: MatrixSchedule,
                  chunk_tokens: int = 128):
    """A whole block at ANY rank with a given (recorded) weights channel."""
    _check_block(state, live, weights, decay, schedule)
    check_no_grad(state, weights, decay, schedule.payload)
    batch, heads, _, value_dim = state.shape
    tokens = schedule.tokens
    reads = torch.empty((batch, heads, tokens, value_dim), dtype=torch.float32,
                        device=state.device)
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        chunk_reads, state, live = k7_chunk(
            state, live, weights[:, :, start:stop], decay[:, :, start:stop],
            schedule.chunk(start, stop))
        reads[:, :, start:stop] = chunk_reads
    return reads, state, live


def k7_block_scan_routed(query: torch.Tensor, state: torch.Tensor,
                          live: torch.Tensor, decay: torch.Tensor,
                          schedule: MatrixSchedule, scale: float = 2.0,
                          chunk_tokens: int = 128):
    """A whole block at ANY rank, weights included: the refusal removed."""
    _check_block_routed(query, state, live, decay, schedule)
    check_no_grad(query, state, decay, schedule.payload)
    batch, heads, _, value_dim = state.shape
    tokens = schedule.tokens
    reads = torch.empty((batch, heads, tokens, value_dim), dtype=torch.float32,
                        device=state.device)
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        chunk_reads, state, live = k7_chunk_routed(
            query[:, :, start:stop], state, live, decay[:, :, start:stop],
            schedule.chunk(start, stop), scale)
        reads[:, :, start:stop] = chunk_reads
    return reads, state, live


# ======================================================== Part B: the sequential walk


@triton.jit(do_not_specialize=['T', 'R'])
def _k7_walk_kernel(
    query, payload, decay,
    state, live_scr,
    reads, slot_out, op_out, erase_out, live_after_out, payload_out,
    live_pre, weights_out,
    scr_score, scr_sim, scr_strength, scr_w,
    T, R,
    qbs, qts, qhs, qvs,
    pbs, pts, phs, pvs,
    dbs, dts, dhs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, RP: tl.constexpr,
    L: tl.constexpr, BR: tl.constexpr, NT: tl.constexpr,
    LOGV: tl.constexpr, LOGR: tl.constexpr,
    TAU_MATCH: tl.constexpr, TAU_MERGE: tl.constexpr, TAU_POP: tl.constexpr,
    TIE: tl.constexpr, SCALE: tl.constexpr, COLLECT: tl.constexpr,
    STAGES: tl.constexpr,
):
    """The high-rank sequential decision walk: the store in GLOBAL memory.

    One program per (batch*head) row walks the window token by token; the store
    is swept per token in ``NT`` slot tiles of ``BR x VP`` floats.  Sweep A
    (read-only) computes the per-row defined value-axis folds -- norms, read
    scores, decision cosines -- into scratch channels and nothing else; the
    decision phase then runs K6's own code on whole-rank register vectors (the
    softmax denominator is the defined whole-axis tree IN ONE PROGRAM -- no
    split, so the weights are bit-exact against the oracle); sweep B computes
    the read numerator (native per-tile sums, sequential across tiles -- the
    floor-disciplined channel) and applies DECAY in place; the lanes then write
    the addressed rows.  The register peak is the tile (8 192 floats) plus the
    whole-rank vectors, NOT the store -- that is the whole point.
    """
    row = tl.program_id(0)
    batch_index = row // H
    head_index = row % H
    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = tl.arange(0, VP)
    keep_v = values < V

    state_base = state + row.to(tl.int64) * R * V
    live_base = live_scr + row.to(tl.int64) * R
    scr_base = row.to(tl.int64) * R
    query_row = query + batch_index * qbs + head_index * qhs
    payload_row = payload + batch_index * pbs + head_index * phs
    decay_row = decay + batch_index * dbs + head_index * dhs
    reads_base = reads + batch_index.to(tl.int64) * T * H * V + head_index * V
    live_pre_base = live_pre + batch_index.to(tl.int64) * T * H * R \
        + head_index * R
    slot_base = slot_out + batch_index.to(tl.int64) * T * H * L + head_index * L
    op_base = op_out + batch_index.to(tl.int64) * T * H * L + head_index * L
    erase_base = erase_out + batch_index.to(tl.int64) * T * H * L + head_index * L
    live_after_base = live_after_out + batch_index.to(tl.int64) * T * H * L \
        + head_index * L
    payload_base = payload_out + batch_index.to(tl.int64) * T * H * L * V \
        + head_index * L * V
    weights_base = weights_out + batch_index.to(tl.int64) * T * H * R \
        + head_index * R

    for t in range(0, T):
        # ---- the state as this token's policy sees it (pre-DECAY, pre-write).
        live_vec = tl.load(live_base + ranks, mask=keep_r, other=0) != 0
        if COLLECT:
            tl.store(live_pre_base + t * H * R + ranks, live_vec.to(tl.int8),
                     mask=keep_r)
        q = tl.load(query_row + t * qts + values * qvs, mask=keep_v,
                    other=0.0).to(tl.float32)
        p = tl.load(payload_row + t * pts + values * pvs, mask=keep_v,
                    other=0.0).to(tl.float32)
        lam = tl.load(decay_row + t * dts).to(tl.float32)
        nq = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(q, q), VP, LOGV)), (1,))
        uq = _div_rn32(q, tl.broadcast_to(tl.maximum(nq, _EPS), (VP,)))
        npv = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(p, p), VP, LOGV)), (1,))
        pv = _div_rn32(p, tl.broadcast_to(tl.maximum(npv, _EPS), (VP,)))

        # ---- sweep A: the per-tile defined folds, into whole-rank scratch.
        for b in tl.range(0, NT, num_stages=STAGES):
            ranks_b = b * BR + tl.arange(0, BR)
            keep_b = ranks_b < R
            tile = tl.load(state_base + ranks_b[:, None] * V + values[None, :],
                           mask=keep_b[:, None] & keep_v[None, :], other=0.0)
            nr_b = tl.reshape(_sqrt_rn32(_fold_axis2(_mul_rn(tile, tile), VP,
                                                     LOGV)), (BR,))
            live_b = tl.load(live_base + ranks_b, mask=keep_b, other=0) != 0
            strength_b = _mul_rn(nr_b, tl.where(live_b & keep_b, 1.0, 0.0))
            score_b = _div_rn32(
                _mul_rn(tl.reshape(_fold_axis2(_mul_rn(uq[None, :], tile), VP,
                                                LOGV), (BR,)), SCALE),
                tl.maximum(nr_b, _EPS))
            sim_b = _div_rn32(
                tl.reshape(_fold_axis2(_mul_rn(pv[None, :], tile), VP, LOGV),
                           (BR,)),
                tl.maximum(strength_b, _EPS))
            score_b = tl.where(live_b & keep_b, score_b, float('-inf'))
            sim_b = tl.where(live_b & keep_b, sim_b, -2.0)
            tl.store(scr_score + scr_base + ranks_b, score_b, mask=keep_b)
            tl.store(scr_sim + scr_base + ranks_b, sim_b, mask=keep_b)
            tl.store(scr_strength + scr_base + ranks_b, strength_b,
                     mask=keep_b)
        # The scratch was just written by every warp of the program and the
        # decision phase reads it back in a different thread layout -- a CTA
        # barrier is what makes one warp's global store visible to another
        # warp's load (the CUDA __syncthreads guarantee; Triton inserts no
        # barrier for global store->load on its own).
        tl.debug_barrier()

        # ---- the decision phase: whole-rank register vectors, K6's own code.
        # The stored scores are already live-masked to -inf, so `safe` is the
        # K6 expression; the softmax denominator is the defined whole-axis tree
        # in ONE program -- no slot-block split, so the weights channel is
        # bit-exact against the oracle.
        score = tl.load(scr_score + scr_base + ranks, mask=keep_r,
                       other=float('-inf'))
        sim = tl.load(scr_sim + scr_base + ranks, mask=keep_r, other=-2.0)
        strength = tl.load(scr_strength + scr_base + ranks, mask=keep_r,
                           other=0.0)
        live_r = live_vec & keep_r
        live_any = tl.sum(live_r.to(tl.int32), 0) > 0
        safe = tl.where(live_any, score, 0.0)
        pmax = tl.max(safe, 0)
        expo = libdevice.exp(safe - pmax)
        denom = tl.reshape(_fold_vec(tl.where(live_r, expo, 0.0), RP, LOGR),
                           (1,))
        probs = tl.where(live_any,
                         _div_rn32(expo, tl.broadcast_to(denom, (RP,))), 0.0)
        w = _mul_rn(probs, strength)
        if COLLECT:
            tl.store(weights_base + t * H * R + ranks, w, mask=keep_r)
        tl.store(scr_w + scr_base + ranks, w, mask=keep_r)
        tl.debug_barrier()     # scr_w is re-read per tile in sweep B

        # ---- the registered selection chain (schedule.route_chunk at chunk 1).
        penalty = _mul_rn(ranks.to(tl.float32), TIE)
        sel_score = tl.where(keep_r, sim - penalty, -3.0)
        best_top = tl.max(sel_score, 0)
        best = tl.min(tl.where(sel_score == best_top, ranks, _BIG_INDEX), 0)
        best_value = tl.sum(tl.where(ranks == best, sim, 0.0), 0)
        second_score = tl.where(ranks == best, -3.0, sel_score)
        second_top = tl.max(second_score, 0)
        second = tl.min(tl.where(second_score == second_top, ranks,
                                 _BIG_INDEX), 0)
        second_value = tl.sum(tl.where(ranks == second, sim, 0.0), 0)
        worst_score = tl.where(keep_r, sel_score, float('inf'))
        worst_top = tl.min(worst_score, 0)
        worst = tl.min(tl.where(worst_score == worst_top, ranks, _BIG_INDEX),
                       0)
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
        # the same two products K6's masked tile sum meets in exactly one add.
        fw = tl.maximum(tl.sum(tl.where(ranks == best, strength, 0.0), 0),
                        _EPS)
        sw = tl.maximum(tl.sum(tl.where(ranks == second, strength, 0.0), 0),
                        _EPS)
        total = fw + sw
        a_mix = _div_rn32(fw, total)
        b_mix = _div_rn32(sw, total)
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        if merge:
            row_best = tl.load(state_base + best * V + values, mask=keep_v,
                               other=0.0)
            row_second = tl.load(state_base + second * V + values, mask=keep_v,
                                 other=0.0)
            lane0 = _add_rn(_mul_rn(a_mix, row_best),
                            _mul_rn(b_mix, row_second))
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
        tl.store(live_after_base + t * H * L + 1, tl.zeros((), dtype=tl.int8))
        tl.store(payload_base + t * H * L * V + 0 * V + values, lane0,
                 mask=keep_v)
        tl.store(payload_base + t * H * L * V + 1 * V + values,
                 tl.zeros((VP,), dtype=tl.float32), mask=keep_v)

        # ---- sweep B: the read numerator, the DECAY in place, and the capture
        # of the lane's target row (the single-nonzero gather -- exact whatever
        # the reduction order, and it avoids re-LOADING a row this program just
        # stored, which would be the same cross-warp hazard the barriers below
        # close).  Deliberately NOT software-pipelined: the loop stores back to
        # the very buffer it loads, and the pipeliner's load/store alias
        # analysis is not a thing to bet the walk's correctness on.
        read_o = tl.zeros((VP,), dtype=tl.float32)
        old_row = tl.zeros((VP,), dtype=tl.float32)
        for b in range(0, NT):
            ranks_b = b * BR + tl.arange(0, BR)
            keep_b = ranks_b < R
            tile = tl.load(state_base + ranks_b[:, None] * V + values[None, :],
                           mask=keep_b[:, None] & keep_v[None, :], other=0.0)
            w_b = tl.load(scr_w + scr_base + ranks_b, mask=keep_b, other=0.0)
            read_o += tl.sum(_mul_rn(w_b[:, None], tile), 0)
            decayed = _mul_rn(tile, lam)
            hit_b = (ranks_b == slot0) & keep_b
            old_row += tl.sum(tl.where(hit_b[:, None], decayed, 0.0), 0)
            tl.store(state_base + ranks_b[:, None] * V + values[None, :],
                     decayed, mask=keep_b[:, None] & keep_v[None, :])
        tl.store(reads_base + t * H * V + values, read_o, mask=keep_v)

        # ---- the lanes, in the registered order (after the DECAY).  A NONE op
        # writes nothing: its payload lane is exactly zero and its target is
        # provably live (a full store with no dead row -- else the worst row
        # would be dead and POP would have fired), so K6's unconditional write
        # is the identity and skipping it is bit-identical without leaning on
        # the invariant.  The live update carries the op != NONE guard the
        # oracle has.
        active0 = op0 != _NONE_CODE
        if active0:
            if erase0:
                tl.store(state_base + slot0 * V + values, lane0, mask=keep_v)
            else:
                # `old_row` was captured in sweep B (the decayed content this
                # lane sees, bit-identical to a re-load and free of the
                # cross-warp hazard one would be).
                tl.store(state_base + slot0 * V + values,
                         _add_rn(old_row, lane0), mask=keep_v)
            tl.store(live_base + slot0, live0.to(tl.int8))
        if merge:
            tl.store(state_base + second * V + values,
                     tl.zeros((VP,), dtype=tl.float32), mask=keep_v)
            tl.store(live_base + second, tl.zeros((), dtype=tl.int8))
        # The next token's sweeps read the state and the live channel this
        # token just wrote; close the token on a barrier so every warp sees
        # every other warp's writes (same guarantee, same reason).
        tl.debug_barrier()


def _check_walk_inputs(query, payload, decay, state, live, rank):
    batch, tokens, heads, value_dim = payload.shape
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
                         f"{MAX_WALK_VALUE_DIM} (a {MAX_WALK_TILE}-float tile), "
                         f"got {value_dim}; refused by name, not silently "
                         f"spilled")
    if not (query.is_cuda and payload.is_cuda and decay.is_cuda
            and state.is_cuda):
        raise ValueError("the sequential walk is a device kernel; inputs must "
                         "be on CUDA")
    if tokens < 1:
        raise ValueError("tokens must be positive")


def k7_sequential_walk(query: torch.Tensor, payload: torch.Tensor,
                        decay: torch.Tensor, *, policy: RoutingPolicy,
                        rank: int, initial_state: torch.Tensor = None,
                        initial_live: torch.Tensor = None,
                        collect: bool = True, br: int = K7_WALK_TILE,
                        warps: int = 8, stages: int = 2):
    """The K6 per-token walk at ANY rank: the global-tiled state, one launch.

    Same semantics as K6's kernel (the registered five-op chain at chunk
    granularity 1, decisions and reads against the CURRENT state, the certified
    read, the registered update order); the store lives in global memory and is
    swept per token in slot tiles, so no register ceiling applies.  The
    decision channel, the read weights, the pre-decision liveness, the exit
    state and the reads are returned; the parity fixed point is
    :func:`sequential_oracle` (decisions and weights bit-exact by construction,
    the read at the measured floor).  ``policy.grow`` and the append-only
    control are refused exactly as K6 refuses them.
    """
    batch, tokens, heads, value_dim = payload.shape
    rank = int(rank)
    if not policy.ops:
        raise ValueError("the append-only control is refused by the sequential "
                         "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the sequential registration: "
                         "it is a chunk-boundary concept (K7 keeps the shapes "
                         "growth-ready; wiring a growth wave is out of scope)")
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                         device=query.device) if initial_state is None
             else initial_state)
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool,
                        device=query.device) if initial_live is None
            else initial_live)
    _check_walk_inputs(query, payload, decay, state, live, rank)
    check_no_grad(query, payload, decay, state)
    device = query.device
    rows = batch * heads
    rp, vp = _pow2(rank), _pow2(value_dim)
    br = max(16, min(int(br), MAX_WALK_TILE // vp))
    nt = triton.cdiv(rp, br)
    work = state.contiguous().clone()          # the walk mutates the store
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
        live_pre = torch.empty((batch, tokens, heads, rank), dtype=torch.bool,
                               device=device)
        weights = torch.empty((batch, tokens, heads, rank),
                               dtype=torch.float32, device=device)
    else:
        # the COLLECT flag skips the diagnostic stores entirely; a one-element
        # stand-in keeps the launch signature (never written, never read).
        live_pre = torch.empty(1, dtype=torch.bool, device=device)
        weights = torch.empty(1, dtype=torch.float32, device=device)
    scratch = [torch.empty((rows, rank), dtype=torch.float32, device=device)
               for _ in range(4)]
    handle = _k7_walk_kernel[(rows,)](
        query, payload, decay,
        work, live_scr,
        reads, slot, op, erase.view(torch.int8),
        live_after.view(torch.int8), payload_field,
        live_pre.view(torch.int8), weights,
        *scratch,
        tokens, rank,
        *query.stride(), *payload.stride(), *decay.stride(),
        H=heads, V=value_dim, VP=vp, RP=rp, L=2, BR=br, NT=nt,
        LOGV=_log2_up(vp), LOGR=_log2_up(rp),
        TAU_MATCH=_round_f32(policy.tau_match),
        TAU_MERGE=_round_f32(policy.tau_merge),
        TAU_POP=_round_f32(policy.tau_pop),
        TIE=_round_f32(policy.tie_break), SCALE=_round_f32(policy.read_scale),
        COLLECT=1 if collect else 0, STAGES=stages, num_warps=warps)
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return dict(reads=reads, state=work, live=live_scr.view(torch.bool),
                schedule=schedule, weights=weights, live_pre=live_pre,
                kernel=handle)


# ------------------------------------------------------------- the K7 eager oracle


def _oracle_selection(frozen, live, p_det, tau_match, tau_merge, tau_pop, tie,
                      rank, RP, VP):
    """``schedule.route_chunk``'s decision chain for ONE token, defined order.

    Identical arithmetic to K6's oracle (the parity fixed point both K7
    implementations share); ``frozen`` is the detached current state.
    """
    strength = _norm_defined(frozen, -1, VP) * live.to(torch.float32)
    pv = p_det / _norm_defined(p_det, -1, VP).clamp_min(EPS).unsqueeze(-1)
    sim = tree_fold(pv.unsqueeze(2) * frozen, -1, VP) \
        / strength.clamp_min(EPS)
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


def sequential_oracle(query, payload, decay, *, policy: RoutingPolicy, rank: int,
                      initial_state=None, initial_live=None, collect=True):
    """The K7 eager per-token oracle: the parity fixed point of the high-rank walk.

    The decision and read-WEIGHT arithmetic is K6's defined order verbatim
    (balanced trees, fp32, correctly-rounded ops, torch's own exp), so the
    kernel's decisions and weights are bit-exact against this by construction.
    TWO stated differences from K6's oracle, both about the lane application and
    the read OUTPUT:

    * the read is reduced by torch's einsum -- the kernel's read is a native
      per-tile sum, an order no torch expression reproduces, so the read gate is
      the measured floor, never bit-equality (the K1/K23/K6 corner);
    * a NONE op writes NOTHING (K6's oracle writes ``row + 0.0``, which flips a
      ``-0.0`` element to ``+0.0``; skipping the write is what the kernel does,
      so the oracle matches the kernel without leaning on the live-store
      invariant that makes K6's unconditional write the identity).
    """
    batch, tokens, heads, value_dim = payload.shape
    if not policy.ops:
        raise ValueError("the append-only control is refused by the sequential "
                         "registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the sequential registration: "
                         "it is a chunk-boundary concept")
    rank = int(rank)
    device = query.device
    compute_dtype = torch.promote_types(_fp32(query).dtype, torch.float32)
    q32, p32 = _fp32(query).to(compute_dtype), _fp32(payload).to(compute_dtype)
    d32 = _fp32(decay).to(compute_dtype)
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=compute_dtype,
                         device=device) if initial_state is None
             else initial_state.clone())
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
            if initial_live is None else initial_live.clone())
    RP, VP = _pow2(rank), _pow2(value_dim)
    tau_match, tau_merge, tau_pop = (_round_f32(policy.tau_match),
                                     _round_f32(policy.tau_merge),
                                     _round_f32(policy.tau_pop))
    tie, scale = _round_f32(policy.tie_break), _round_f32(policy.read_scale)
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=compute_dtype,
                        device=device)
    fields = dict(
        slot=torch.zeros(batch, tokens, heads, 2, dtype=torch.int64,
                         device=device),
        erase=torch.zeros(batch, tokens, heads, 2, dtype=torch.bool,
                          device=device),
        payload=torch.zeros(batch, tokens, heads, 2, value_dim,
                            dtype=compute_dtype, device=device),
        op=torch.zeros(batch, tokens, heads, 2, dtype=torch.int64,
                       device=device),
        live_after=torch.zeros(batch, tokens, heads, 2, dtype=torch.bool,
                               device=device))
    weights_out = (torch.zeros(batch, tokens, heads, rank,
                               dtype=compute_dtype, device=device)
                   if collect else None)
    live_pre_out = (torch.zeros(batch, tokens, heads, rank, dtype=torch.bool,
                                device=device) if collect else None)
    b_index = torch.arange(batch, device=device)[:, None]
    h_index = torch.arange(heads, device=device)[None, :]
    bhe = b_index.expand(batch, heads)
    hhe = h_index.expand(batch, heads)
    for t in range(tokens):
        if collect:
            live_pre_out[:, t] = live
        q_t, p_t, lam_t = q32[:, t], p32[:, t], d32[:, t]
        # ---- the read weights: the certified formula, K6's defined order.
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
        any_live_flat = live.any(-1)
        denom_safe = torch.where(any_live_flat, denom, torch.ones_like(denom))
        probs = torch.where(any_live, expo / denom_safe.unsqueeze(-1),
                            torch.zeros_like(expo))
        w = probs * nr * live_f
        reads[:, t] = torch.einsum("bhr,bhrv->bhv", w, state)
        if collect:
            weights_out[:, t] = w
        # ---- the decisions: the registered chain on the DETACHED state.
        sel = _oracle_selection(state.detach(), live, p_t.detach(), tau_match,
                                tau_merge, tau_pop, tie, rank, RP, VP)
        blend = None
        if bool(sel["merge"].any()):
            rows = state[bhe, hhe, sel["best_index"]]
            rows2 = state[bhe, hhe, sel["second_index"]]
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
        active = sel["op0"].ne(NONE)
        written = torch.where(sel["erase0"].unsqueeze(-1), lane0,
                              state[bhe, hhe, sel["slot0"]] + lane0)
        state = state.index_put((bhe, hhe, sel["slot0"]),
                                torch.where(active.unsqueeze(-1), written,
                                            state[bhe, hhe, sel["slot0"]]))
        live = live.index_put((bhe, hhe, sel["slot0"]),
                              torch.where(active, sel["live0"],
                                          live.gather(-1,
                                                      sel["slot0"].unsqueeze(-1)
                                                      ).squeeze(-1)))
        if bool(sel["merge"].any()):
            state = state.index_put(
                (bhe, hhe, sel["second_index"]),
                torch.zeros(batch, heads, value_dim, dtype=state.dtype,
                            device=device))
            live = live.index_put(
                (bhe, hhe, sel["second_index"]),
                torch.zeros(batch, heads, dtype=torch.bool, device=live.device))
    schedule = MatrixSchedule(**fields)
    return dict(reads=reads, state=state, live=live, schedule=schedule,
                weights=weights_out, live_pre=live_pre_out, rank=rank)


__all__ = ["DESIGN_CHOICE_GLOBAL_TILED", "K7_RANK_TILE", "K7_TOKEN_TILE",
           "K7_STEP_TILE", "K7_WALK_TILE", "MAX_WALK_TILE",
           "MAX_WALK_VALUE_DIM", "MAX_VALUE_DIM", "k7_block_scan",
           "k7_block_scan_routed", "k7_chunk", "k7_chunk_routed", "k7_read",
           "k7_read_weights", "k7_sequential_walk", "sequential_oracle",
           "tree_fold"]

#: the resolved design fork of Part B -- see ``DESIGN_HIGH_RANK_WALK.md``
DESIGN_CHOICE_GLOBAL_TILED = "global-tiled state walk (option i)"
