"""K2 + K3 — the matrix store's read as two fused Triton kernels (forward only).

Why this file exists
--------------------
Stage K1 (``k1_schedule_scan``) removed 86 of the 87 kernels an eager chunk launched,
and left exactly one thing in torch: ``_reads_from_injections``, the read reduction over
the emitted injection list.  It is 23 kernels of *small* ops per chunk and, measured on
one 2048-token block call, more than half of the drop-in path's time (``K1_SCHEDULE_SCAN.md``
§4–5).  This file is that reduction as two ``tl.dot`` kernels:

``K3`` ``k23_read_inter``   the inter-chunk state term
    ``o_t += exp(gc_read_t) * (w_t @ H)``
    one program per ``(row, token_block)``, rank walked in blocks, ``tl.dot`` accumulating
    into ``(BT, V)`` registers.
``K2`` ``k23_read_intra``   the intra-chunk interaction
    ``o_t += sum_{u < t*lanes} exp(gc_read_t - gc_u) * w_t[j_u] * d_u``
    one program per ``(row, token_block)``, step blocks walked with the causal predicate
    ``u < t*lanes`` as a constexpr-shaped mask, ``tl.dot`` accumulating ``(BT, BU) @ (BU, V)``.

Both are joined by a single fp32 add inside the second kernel (it loads the inter term and
adds its own accumulator), which is exactly the eager path's ``entry_read + injected``, so
the whole read is **two launches per chunk** instead of 23.

The normalisation, stated exactly (and why it is not a "union" softmax)
----------------------------------------------------------------------
The certified read (``x30_tape_cell/naive_reference.py`` line 19) is
``sum_i softmax_i(score over live slots) * strength_i * value_i`` with no division by the
strength mass.  In this arm the softmax is taken **once per chunk against the chunk-entry
state**, by :func:`read.read_weights`, and handed to the scan as the ``weights`` channel:
``w_t[j] = softmax_j(scale * cos(q_t, S_entry[j])) * ||S_entry[j]||`` for live entry rows,
zero for dead ones.  The chunked reduction then adds the in-chunk injections *linearly with
those same weights*:

    o_t = exp(gc_read_t) * w_t @ H  +  sum_{u <= read_t} exp(gc_read_t - gc_u) w_t[j_u] d_u

so the denominator is over the entry-live slots, it is already inside ``w``, and it
multiplies the intra block too.  There is no per-token union denominator in this algebra:
adding ``sum_intra exp(score)`` to the denominator would be a *new* read, not this one, and
would break gate A against the eager reference by construction.  ``read.py`` stays the
single definition of the weights; this file only reduces them, as K1 does.  The consequence
is stated where it can be seen: a slot that is dead at chunk entry has ``w = 0``, so an
in-chunk write into it contributes nothing to that chunk's read (``K23_READ.md`` §7).

Two things the gate receipts turn into invariants rather than assertions:

* a slot that **dies mid-chunk** leaves the read by *exact cancellation* — the decayed entry
  term ``exp(gc) w H[j]`` and the POP's injection ``-exp(gc) w v`` are the same numbers with
  opposite signs, so the slot's contribution is zero from that token on;
* a token whose only support is an in-chunk injection (a live entry row with zero content)
  therefore gets ``w * d`` and nothing else — the sharpest available test of the coefficient.

Determinism and precision
-------------------------
Every ``tl.dot`` is ``input_precision='ieee'``: Triton's default for fp32 inputs is TF32, which
would put the read three orders of magnitude off the fp32 reference.  The reduction order
still differs from cuBLAS's (rank and step axes are blocked, accumulation in fp32 registers),
so the fused read is held to the **measured** fp32 sequential floor times ``NOISE_FACTOR``,
never to a fitted constant — ``K23_READ.md`` §4 has the numbers.  No atomics, no
read-modify-write races: ``K2`` accumulates into the inter term in program order, and its
``restore_value`` autotune hook keeps the tuning runs from adding themselves twice.

Forward only, and default-off until the gates passed
---------------------------------------------------
The fused read has no backward (same reason as K1: a graph through it would keep the read's
gradient and silently drop the state's), so it refuses one.  ``scan.chunk_scan(...,
fused=True)`` is the drop-in; the eager path is the oracle.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .algebra import MatrixSchedule
from .k1_schedule_scan import K1Chunk, check_no_grad

# ``tl.dot`` wants every operand axis a multiple of 16; the value axis is therefore
# padded up to a 16-multiple and masked down to ``V`` (all-true at the arm's V = 128).
MIN_TILE = 16
# A value dim wider than this would be a different register budget, not a different
# kernel: refused rather than silently spilling.
MAX_VALUE_DIM = 256

# Block shapes are autotuned on (rank, value_dim) — the two axes that decide how much
# of a tile fits in registers — while ``T`` (tokens) and ``N`` (steps) are
# ``do_not_specialize``d so a new sequence length does not trigger a recompile.
INTER_CONFIGS = [triton.Config({'BT': bt, 'BK': bk}, num_warps=4)
                 for bt in (16, 32, 64) for bk in (32, 64)]
INTRA_CONFIGS = [triton.Config({'BT': bt, 'BU': bu}, num_warps=4)
                 for bt in (16, 64) for bu in (32, 64)]


def check_k23_shapes(weights: torch.Tensor, entry_state: torch.Tensor,
                     chunk: K1Chunk, schedule: MatrixSchedule) -> None:
    """Refuse a shape the two kernels cannot express, before any launch."""
    if not (weights.is_cuda and entry_state.is_cuda and chunk.delta.is_cuda):
        raise ValueError("K2/K3 are device kernels; inputs must be on CUDA")
    if weights.dim() != 4 or entry_state.dim() != 4:
        raise ValueError("weights must be (batch, heads, tokens, rank) and the entry "
                         "state (batch, heads, rank, value_dim)")
    batch, heads, tokens, rank = weights.shape
    value_dim = entry_state.shape[-1]
    if entry_state.shape != (batch, heads, rank, value_dim):
        raise ValueError(f"entry state {tuple(entry_state.shape)} does not match the "
                         f"weights channel {tuple(weights.shape)}")
    if chunk.delta.shape[:3] != (batch, heads, chunk.steps):
        raise ValueError("the injection list does not belong to this chunk")
    if chunk.delta.shape[-1] != value_dim:
        raise ValueError("the injection list's value dim disagrees with the entry state")
    if schedule.lanes * tokens != chunk.steps:
        raise ValueError(f"{chunk.steps} steps for {tokens} tokens at "
                         f"{schedule.lanes} lanes")
    if value_dim > MAX_VALUE_DIM:
        raise ValueError(f"K2/K3 compile for value dims up to {MAX_VALUE_DIM}, "
                         f"got {value_dim}")
    for name, tensor in (("weights", weights), ("entry_state", entry_state),
                         ("delta", chunk.delta), ("log_gate", chunk.log_gate)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be fp32; the algebra is fp32")


@triton.autotune(configs=INTER_CONFIGS, key=['R', 'V'])
@triton.jit(do_not_specialize=['T', 'N'])
def _k23_inter_kernel(
    weights, entry_state, log_gate, reads,
    T, N, R,
    wbs, whs, wt, wr,
    sbs, shs, sr, sv,
    gbs, ghs, gs,
    obs, ohs, ot, ov,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
):
    """K3: ``reads_t = exp(gc_read_t) * (w_t @ H)``, one program per ``(row, tokens)``.

    ``read_step(t) = t*lanes - 1`` is the step whose cumulative log decay the read sees
    (the last lane of the previous token); token 0 has no such step, and the reference
    reads it as ``gc_read = 0``, i.e. the undecayed entry state — the same statement as
    ``exp(0) = 1`` without a branch.  Rank is walked in ``BK`` blocks and accumulated in
    fp32 registers, so the whole term is one ``tl.dot`` chain and no
    ``[rows, tokens, rank]`` tensor is ever materialised.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    vals = tl.arange(0, VP)
    keep_v = vals < V

    read_step = toks * L - 1
    first = read_step < 0
    log_gate_row = log_gate + (row // H) * gbs + (row % H) * ghs
    gc_read = tl.load(log_gate_row + tl.where(first, 0, read_step) * gs,
                      mask=keep_t & ~first, other=0.0)
    decay_weight = tl.exp(gc_read)

    weights_row = weights + (row // H) * wbs + (row % H) * whs
    state_row = entry_state + (row // H) * sbs + (row % H) * shs
    acc = tl.zeros((BT, VP), dtype=tl.float32)
    for start in range(0, R, BK):
        ranks = start + tl.arange(0, BK)
        keep_r = ranks < R
        w = tl.load(weights_row + toks[:, None] * wt + ranks[None, :] * wr,
                    mask=keep_t[:, None] & keep_r[None, :], other=0.0)
        rows = tl.load(state_row + ranks[:, None] * sr + vals[None, :] * sv,
                       mask=keep_r[:, None] & keep_v[None, :], other=0.0)
        acc = tl.dot(w * decay_weight[:, None], rows, acc, input_precision='ieee')

    out_row = reads + (row // H) * obs + (row % H) * ohs
    tl.store(out_row + toks[:, None] * ot + vals[None, :] * ov, acc,
             mask=keep_t[:, None] & keep_v[None, :])


@triton.autotune(configs=INTRA_CONFIGS, key=['R', 'V'], restore_value=['reads'])
@triton.jit(do_not_specialize=['T', 'N'])
def _k23_intra_kernel(
    weights, delta, log_gate, slot, reads,
    T, N, R,
    wbs, whs, wt, wr,
    dbs, dhs, ds, dv,
    gbs, ghs, gs,
    kbs, khs, kt, kl,
    obs, ohs, ot, ov,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BT: tl.constexpr, BU: tl.constexpr,
):
    """K2: ``reads_t += sum_{u < t*lanes} exp(gc_read_t - gc_u) * w_t[j_u] * d_u``.

    The causal predicate is ``u < t*lanes`` — a token's read is pre-write, so its own
    lanes never contribute, and the last lane of the *previous* token does with weight
    ``exp(0) = 1``.  ``w_t[j_u]`` is the gathered weight of the addressed entry slot, so
    the injection is weighted by the same normalised channel as the entry term (no second
    softmax, by design).  The exponent is evaluated only under the predicate, so an
    excluded pair cannot produce ``inf * 0``.

    The result is added to the inter term on the way out (one fp32 add, the eager
    expression's ``entry_read + injected``).  ``restore_value`` makes the autotune
    benchmarking runs restore ``reads`` afterwards, so tuning cannot accumulate the
    intra term several times.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    toks = block * BT + tl.arange(0, BT)
    keep_t = toks < T
    vals = tl.arange(0, VP)
    keep_v = vals < V

    read_step = toks * L - 1
    first = read_step < 0
    log_gate_row = log_gate + (row // H) * gbs + (row % H) * ghs
    gc_read = tl.load(log_gate_row + tl.where(first, 0, read_step) * gs,
                      mask=keep_t & ~first, other=0.0)

    weights_row = weights + (row // H) * wbs + (row % H) * whs
    delta_row = delta + (row // H) * dbs + (row % H) * dhs
    slot_row = slot + (row // H) * kbs + (row % H) * khs
    # Only the steps that precede some token of this block can contribute, so the walk
    # stops at the block's last read step instead of at the chunk's end.
    bound = tl.minimum(N, (block * BT + BT) * L)
    acc = tl.zeros((BT, VP), dtype=tl.float32)
    for start in range(0, bound, BU):
        steps = start + tl.arange(0, BU)
        keep_u = steps < N
        gc_step = tl.load(log_gate_row + steps * gs, mask=keep_u, other=0.0)
        target = tl.load(slot_row + (steps // L) * kt + (steps % L) * kl,
                         mask=keep_u, other=0).to(tl.int32)
        addressed = tl.load(weights_row + toks[:, None] * wt + target[None, :] * wr,
                            mask=keep_t[:, None] & keep_u[None, :], other=0.0)
        causal = (steps[None, :] < toks[:, None] * L) & keep_t[:, None] & keep_u[None, :]
        exponent = tl.where(causal, gc_read[:, None] - gc_step[None, :], 0.0)
        coefficients = tl.where(causal, addressed * tl.exp(exponent), 0.0)
        payload = tl.load(delta_row + steps[:, None] * ds + vals[None, :] * dv,
                          mask=keep_u[:, None] & keep_v[None, :], other=0.0)
        acc = tl.dot(coefficients, payload, acc, input_precision='ieee')

    out_row = reads + (row // H) * obs + (row % H) * ohs
    offsets = toks[:, None] * ot + vals[None, :] * ov
    store_mask = keep_t[:, None] & keep_v[None, :]
    entry_term = tl.load(out_row + offsets, mask=store_mask, other=0.0)
    tl.store(out_row + offsets, entry_term + acc, mask=store_mask)


def k23_read(weights: torch.Tensor, entry_state: torch.Tensor, chunk: K1Chunk,
             schedule: MatrixSchedule) -> torch.Tensor:
    """The chunk's read as two kernel launches: ``(batch, heads, tokens, value_dim)``.

    ``weights`` ``(batch, heads, tokens, rank)`` is the read-weight channel of *this*
    chunk (``read.read_weights`` at the chunk-entry state, from ``schedule.route_chunk``),
    ``entry_state`` ``(batch, heads, rank, value_dim)`` the state the chunk started from,
    ``chunk`` K1's emission for the same chunk.  Both kernels take strides, so a
    ``weights`` view sliced out of a sequence tensor needs no copy and the schedule's
    decision channel is read in its own layout (the step axis is decomposed into
    ``(token, lane)`` in-kernel, not on the host).
    """
    check_k23_shapes(weights, entry_state, chunk, schedule)
    check_no_grad(weights, entry_state)
    batch, heads, tokens, rank = weights.shape
    value_dim = entry_state.shape[-1]
    reads = torch.empty(batch, heads, tokens, value_dim, dtype=torch.float32,
                        device=weights.device)
    padded = max(MIN_TILE, triton.next_power_of_2(value_dim))
    grid = lambda meta: (batch * heads, triton.cdiv(tokens, meta['BT']))
    _k23_inter_kernel[grid](
        weights=weights, entry_state=entry_state, log_gate=chunk.log_gate, reads=reads,
        T=tokens, N=chunk.steps, R=rank,
        wbs=weights.stride(0), whs=weights.stride(1), wt=weights.stride(2),
        wr=weights.stride(3),
        sbs=entry_state.stride(0), shs=entry_state.stride(1), sr=entry_state.stride(2),
        sv=entry_state.stride(3),
        gbs=chunk.log_gate.stride(0), ghs=chunk.log_gate.stride(1),
        gs=chunk.log_gate.stride(2),
        obs=reads.stride(0), ohs=reads.stride(1), ot=reads.stride(2), ov=reads.stride(3),
        H=heads, L=schedule.lanes, V=value_dim, VP=padded)
    _k23_intra_kernel[grid](
        weights=weights, delta=chunk.delta, log_gate=chunk.log_gate,
        slot=schedule.slot, reads=reads,
        T=tokens, N=chunk.steps, R=rank,
        wbs=weights.stride(0), whs=weights.stride(1), wt=weights.stride(2),
        wr=weights.stride(3),
        dbs=chunk.delta.stride(0), dhs=chunk.delta.stride(1), ds=chunk.delta.stride(2),
        dv=chunk.delta.stride(3),
        gbs=chunk.log_gate.stride(0), ghs=chunk.log_gate.stride(1),
        gs=chunk.log_gate.stride(2),
        kbs=schedule.slot.stride(0), khs=schedule.slot.stride(2),
        kt=schedule.slot.stride(1), kl=schedule.slot.stride(3),
        obs=reads.stride(0), ohs=reads.stride(1), ot=reads.stride(2), ov=reads.stride(3),
        H=heads, L=schedule.lanes, V=value_dim, VP=padded)
    return reads


def k23_chunk(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
              decay: torch.Tensor, schedule: MatrixSchedule):
    """Drop-in for ``scan.chunk_scan``: ``(reads, state, live)`` from K1 + K2 + K3.

    Three launches per chunk: the state walk (K1), the inter term (K3) and the intra
    interaction (K2).  ``chunk_scan`` takes the DECAY gate head-major
    ``(batch, heads, tokens)``; the kernel indexes the schedule's head-minor layout, and
    the permutation is a view.
    """
    from .k1_schedule_scan import k1_state_scan

    check_no_grad(weights)
    chunk = k1_state_scan(state, live, decay.permute(0, 2, 1), schedule)
    reads = k23_read(weights, state, chunk, schedule)
    return reads, chunk.state, chunk.live


# ------------------------------------------------------------------------- diagnostic

def union_normalised_reads(query: torch.Tensor, decay: torch.Tensor,
                           schedule: MatrixSchedule, entry_state: torch.Tensor,
                           entry_live: torch.Tensor, scale: float) -> torch.Tensor:
    """DIAGNOSTIC ONLY — the read this arm does *not* implement, for one number.

    Recomputes :func:`read.read_weights` at every token against the state *as it is at
    that token* (entry rows decayed plus in-chunk writes), i.e. a softmax over the slots
    live at that token rather than over the chunk-entry set.  That is the tape store's
    per-token granularity; the registered matrix arm weights a whole chunk from its entry
    state, which is what makes the scan affine and chunk-parallel.

    Nothing on any runtime path calls this.  It exists so "does chunk-constant weighting
    cost anything measurable?" is a number in the receipt instead of an argument, and the
    test suite holds it to the registered read on a write-free fixture, where the two are
    the same read by construction.
    """
    from .algebra import NONE, apply_lane, update_liveness
    from .read import read_weights

    batch, tokens, heads, rank = query.shape
    value_dim = schedule.payload.shape[-1]
    state = entry_state.clone()
    live = entry_live.clone()
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=torch.float32,
                        device=query.device)
    for t in range(tokens):
        weights = read_weights(query[:, t:t + 1], state, live, scale)[:, 0]
        reads[:, t] = torch.einsum("bhr,bhrv->bhv", weights, state)
        state = state * decay[:, t, :, None, None]
        for lane in range(schedule.lanes):
            op = schedule.op[:, t, :, lane]
            if not bool(op.ne(NONE).any()):
                continue
            apply_lane(state, schedule.slot[:, t, :, lane], schedule.erase[:, t, :, lane],
                       schedule.payload[:, t, :, lane])
            update_liveness(live, schedule.slot[:, t, :, lane], op,
                            schedule.live_after[:, t, :, lane])
    return reads


__all__ = ['INTRA_CONFIGS', 'INTER_CONFIGS', 'MAX_VALUE_DIM', 'MIN_TILE',
           'check_k23_shapes', 'k23_chunk', 'k23_read', 'union_normalised_reads']
