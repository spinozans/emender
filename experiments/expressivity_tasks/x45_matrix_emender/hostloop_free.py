"""No host loop: the matrix store's 2048-token forward in one or three launches.

Why this file exists
--------------------
K2+K3 (``k23_read``) took a chunk from 87 kernels to 3, and a 2048-token block call
from 1 392 kernels to 48 -- and moved the wall/occupancy ratio the wrong way (5.16x ->
5.69x), because the remaining cost is 16 Python-dispatch steps of 3 launches each
(``K23_READ.md`` §5.3: 83 % of the wall is host dispatch).  GDN-2's forward is 4
launches for the same sequence because its chunk axis is a *grid* axis: its write
``S <- S(I - beta k k^T) + beta v k^T`` needs no selection, so every per-token quantity
is a function of the inputs and chunks are independent.

**This arm's five ops do need selection**, and the routing that supplies it is computed
from the chunk-entry state, so chunk ``c``'s decisions depend on chunk ``c-1``'s exit
state.  That is a genuine sequential chain (the runtime shadow of
``state_dependent_death_not_affine``, ``chunked_fold_equiv`` /
``combined_chunked_scan`` in ``formal/lean/EMENDER_OPERATOR_ALGEBRA.pdf``: the chunked
fold is equal only along the *state* axis, never along a parallel axis), so it cannot be
turned into a per-chunk-parallel grid without changing the semantics.  The host loop is
therefore removed in a different place: the sequential chain is moved *onto the device*
(one persistent program owning the whole store, Option A) or the host dispatches it once
(one captured graph over the same kernel pipeline, Option B), while Option C simply pays
the dispatch ``T/C`` times fewer.

Three paths, one interface
--------------------------
All of them take a whole 2048-token block at the boundary the receipts use -- state and
liveness in, the *recorded* decision channel and read weights as inputs, ``reads`` and
the exit state out -- so they differ only in how the sequence is sequenced:

``chunked_block_scan``   the K2+K3 baseline: ``T/C`` chunks x 3 launches (48 at C = 128).
                         Option C is this function at a larger ``C``.
``persistent_scan``      **Option A.** One kernel for the whole call.  Grid is
                         ``(batch*heads, value_dim/BV)`` -- no chunk axis, no host loop,
                         no cross-program synchronisation, because one program owns every
                         slot of its row (the state tile lives in its registers) and walks
                         the sequence token by token: read -> DECAY -> lanes.
``BlockGraph``           **Option B.** The same per-chunk pipeline, captured once into a
                         CUDA graph with static buffers; one replay per block call.  This
                         is legitimate *because* the pipeline is sequential -- the graph is
                         the device-side form of the host loop, not a parallelisation.

Semantics: unchanged, and as literal as possible
------------------------------------------------
``persistent_scan`` is the per-token oracle's order, verbatim (``algebra
.sequential_matrix_scan``): read the state as it stands, then decay, then every lane in
order, with ``mul.rn.f32``/``add.rn.f32`` inline asm so no FMA contraction can move the
trajectory by one ulp (the failure recorded in ``x44/RESULT_CHUNK_SCAN.md`` §6.2).  The
read weights are the *input* channel (``schedule.route_chunk`` -> ``read.read_weights``)
exactly as in K1/K2/K3, so nothing about the registered chunk-granular read changes; the
token loop does not re-route, it only applies the recorded schedule.  State and liveness
are therefore bit-exact against the oracle, and the read differs from it only by
reduction order (``NO_HOST_LOOP.md`` §2).

Forward only, like every path in this lane: no backward, so a graph through any of these
is refused rather than silently dropping the state's gradient.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .algebra import MatrixSchedule
from .k1_schedule_scan import NONE_CODE, _add_rn, _mul_rn, check_no_grad
from .k23_read import k23_chunk

# The state tile one persistent program holds in its registers: ``next_pow2(rank) *
# value_block`` fp32 values.  8 192 floats = 32 KB is 64 registers per thread at 4 warps,
# which keeps the whole store of a (row, value-block) resident with room for the read
# accumulator; at 16 384 floats the same code spills (measured, ``NO_HOST_LOOP.md`` §4:
# rank 512 at a block of 32 is 82 ms instead of 2.3).  A rank whose *narrowest* block
# exceeds this does not fit, and ``persistent_available`` says so instead of spilling.
MAX_PERSISTENT_TILE = 8192
# The value block every shape is measured best at (``NO_HOST_LOOP.md`` §4): narrower is
# more programs at the same per-token latency, and the read reduction is unchanged.
MIN_VALUE_BLOCK = 16


def persistent_available(rank: int, value_dim: int, limit: int = MAX_PERSISTENT_TILE) -> bool:
    """Can one program hold ``next_pow2(rank) x MIN_VALUE_BLOCK`` of the store?"""
    return triton.next_power_of_2(int(rank)) * MIN_VALUE_BLOCK <= limit


def value_block(rank: int, value_dim: int, limit: int = MAX_PERSISTENT_TILE) -> int:
    """The value columns one program owns: ``MIN_VALUE_BLOCK``, i.e. as narrow as it goes.

    The value axis is the only axis that may be blocked without cross-program traffic:
    every quantity in the walk (the read ``w_t @ S[:, v]``, the decay and the row writes)
    is column-independent, and the liveness channel each block keeps is the same function
    of the schedule, so the blocks agree without communicating.  The *rank* axis may not
    be blocked -- that is what would need a cross-program reduction per token -- so a rank
    whose whole tile does not fit is refused rather than split.

    Narrow beats wide at every shape measured (rank 64/128/512 at a block of 16/32/64/128):
    the per-token latency of the serial walk is set by the rank axis and the global loads,
    not by the value width, so a narrow block only buys more programs.  Nothing narrower
    than ``MIN_VALUE_BLOCK`` is compiled: a program would then own fewer value columns than
    one store transaction carries, so the per-token read store would dominate.  That
    boundary is a design choice, not a measurement -- the sweep starts at 16.
    """
    if not persistent_available(rank, value_dim, limit):
        raise ValueError(
            f"the persistent kernel needs next_pow2({rank}) * {MIN_VALUE_BLOCK} = "
            f"{triton.next_power_of_2(int(rank)) * MIN_VALUE_BLOCK} register-resident "
            f"floats and its budget is {limit}; rank {rank} at value dim {value_dim} "
            f"does not fit, so it must take the chunked path")
    return MIN_VALUE_BLOCK


def check_block_shapes(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
                       decay: torch.Tensor, schedule: MatrixSchedule) -> None:
    """Refuse anything the three paths cannot express, before any launch.

    One whole-sequence block: ``state`` ``(batch, heads, rank, d_v)``, ``live``
    ``(batch, heads, rank)``, ``weights`` ``(batch, heads, tokens, rank)``, ``decay``
    ``(batch, heads, tokens)`` and a whole-sequence ``schedule``.
    """
    if not (state.is_cuda and weights.is_cuda and schedule.payload.is_cuda):
        raise ValueError("all block paths are device kernels; inputs must be on CUDA")
    if state.dim() != 4 or weights.dim() != 4:
        raise ValueError("state must be (batch, heads, rank, d_v) and weights "
                         "(batch, heads, tokens, rank)")
    batch, heads, rank, value_dim = state.shape
    if weights.shape != (batch, heads, schedule.tokens, rank):
        raise ValueError(f"weights {tuple(weights.shape)} do not match the state "
                         f"{tuple(state.shape)} and {schedule.tokens} tokens")
    if decay.shape != (batch, heads, schedule.tokens):
        raise ValueError(f"decay must be {(batch, heads, schedule.tokens)}, "
                         f"got {tuple(decay.shape)}")
    if schedule.payload.shape != (batch, schedule.tokens, heads, schedule.lanes,
                                  value_dim):
        raise ValueError("the schedule's payload does not belong to this block")
    for name, tensor in (("state", state), ("weights", weights), ("decay", decay),
                         ("payload", schedule.payload)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be fp32; the algebra is fp32")
    if not state.is_contiguous() or not live.is_contiguous():
        raise ValueError("state and live must be contiguous")
    if live.dtype != torch.bool:
        raise ValueError("live must be a bool tensor")


# --------------------------------------------------------------------- Option A: one kernel

@triton.jit(do_not_specialize=['T', 'R'])
def _persistent_scan_kernel(
    state_in, live_in, weights, decay,
    slot, op, erase, live_after, payload,
    reads, state_out, live_out,
    T, R,
    wbs, whs, wt, wr,
    dbs, dhs, dt,
    sbs, st, sh, sl,
    pbs, pst, psh, psl,
    obs, ohs, ot, ov,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr,
    RP: tl.constexpr, BV: tl.constexpr, STAGES: tl.constexpr,
):
    """One program per ``(row, value_block)``; the whole sequence in registers.

    ``row = program_id(0)`` indexes ``(batch, head)`` and owns **every** slot of that row
    for a block of ``BV`` value columns, so the chunk chain needs no chunk axis: the walk
    is ``for t in range(T)`` with the store resident.  ``rank`` is masked from ``RP``
    (its next power of two) and the padding rows are exact zeros throughout -- they load
    as zero, are never addressed (a slot index is ``< R``) and are excluded from the
    store -- so they contribute nothing to the read reduction.

    Order per token, the oracle's (``algebra.sequential_matrix_scan``): read against the
    state as it stands, then the DECAY gate, then every lane in order.  ``mul.rn`` /
    ``add.rn`` inline asm keeps the trajectory bit-exact.

    ``STAGES`` pipelines the loop's global loads (the weight row, the DECAY gate and the
    payload) ahead of the iteration that consumes them: the walk is a serial chain, so
    latency, not arithmetic, is what each token costs, and without the pipeline the
    dependent loads make it ~2x slower (measured, ``NO_HOST_LOOP.md`` §4).
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H

    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = block * BV + tl.arange(0, BV)
    keep_v = values < V
    keep = keep_r[:, None] & keep_v[None, :]

    state_base = row * R * V
    acc = tl.load(state_in + state_base + ranks[:, None] * V + values[None, :],
                  mask=keep, other=0.0)
    live = tl.load(live_in + row * R + ranks, mask=keep_r, other=0) != 0

    weights_row = weights + batch_index * wbs + head_index * whs
    decay_row = decay + batch_index * dbs + head_index * dhs
    reads_row = reads + batch_index * obs + head_index * ohs
    payload_row = payload + batch_index * pbs + head_index * psh

    for t in tl.range(0, T, num_stages=STAGES):
        # The read is pre-write: the state left by token t-1's decay and ops.
        w = tl.load(weights_row + t * wt + ranks * wr, mask=keep_r, other=0.0)
        read = tl.sum(w[:, None] * acc, axis=0)
        tl.store(reads_row + t * ot + values * ov, read, mask=keep_v)
        lam = tl.load(decay_row + t * dt)
        acc = _mul_rn(acc, lam)
        for lane in tl.static_range(L):
            channel = batch_index * sbs + t * st + head_index * sh + lane * sl
            if tl.load(op + channel) != NONE_CODE:
                target = tl.load(slot + channel).to(tl.int32)
                hit = keep_r & (target == ranks)
                written = tl.load(payload_row + t * pst + lane * psl + values,
                                  mask=keep_v, other=0.0)
                tile = tl.broadcast_to(written[None, :], (RP, BV))
                # Erase (CORRECT / POP / MERGE) replaces the addressed row; APPEND adds
                # to it.  Every other row keeps its content bit-for-bit either way.
                if tl.load(erase + channel) != 0:
                    acc = tl.where(hit[:, None], tile, acc)
                else:
                    acc = tl.where(hit[:, None], _add_rn(acc, tile), acc)
                live = tl.where(hit, tl.load(live_after + channel) != 0, live)

    tl.store(state_out + state_base + ranks[:, None] * V + values[None, :], acc,
             mask=keep)
    # Every value block applies the same liveness rule to its own copy, so one of them
    # writes it -- a second writer would race identical bytes for no benefit.
    if block == 0:
        tl.store(live_out + row * R + ranks, live.to(tl.int8), mask=keep_r)


def warp_count(tile: int) -> int:
    """Warps for a register tile of ``tile`` floats: 16 floats per thread, capped at 4.

    Not autotuned, deliberately: ``@triton.autotune``'s cache key cannot see the value
    block (a constexpr), so it would hand every block shape the config it measured for the
    first one -- and its tuning runs launch the kernel several extra times on the real
    buffers, which is the opposite of what a zero-host-loop path is for.  The rule is one
    line and ``NO_HOST_LOOP.md`` §4 reports the sweep it came from: 4 warps is the
    measured optimum at and above a 2 048-float tile, 2 below it, and 8 is worse at every
    shape (that is also why the cap is 4, not 8).
    """
    return max(1, min(4, int(tile) // 512))


def persistent_scan(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
                    decay: torch.Tensor, schedule: MatrixSchedule,
                    block: int | None = None, warps: int | None = None,
                    stages: int = 3):
    """**Option A** -- the whole block call as one kernel launch.

    Returns ``(reads, state, live)`` with ``reads`` ``(batch, heads, tokens, d_v)``.
    ``block``, ``warps`` and ``stages`` are the kernel's tuning knobs and each has a
    measured default (:func:`value_block`, :func:`warp_count`); they are keyword arguments
    so the sweep in ``NO_HOST_LOOP.md`` §4 is reproducible, not so a caller has to pick.
    """
    check_block_shapes(state, live, weights, decay, schedule)
    check_no_grad(state, weights, decay, schedule.payload)
    batch, heads, rank, value_dim = state.shape
    tokens = schedule.tokens
    block = value_block(rank, value_dim) if block is None else int(block)
    if block < MIN_VALUE_BLOCK or block & (block - 1):
        raise ValueError(f"the value block must be a power of two >= {MIN_VALUE_BLOCK}")
    reads = torch.empty((batch, heads, tokens, value_dim), dtype=torch.float32,
                        device=state.device)
    out_state = torch.empty_like(state)
    out_live = torch.empty_like(live)
    grid = (batch * heads, triton.cdiv(value_dim, block))
    tile = triton.next_power_of_2(rank) * block
    warps = warp_count(tile) if warps is None else int(warps)
    _persistent_scan_kernel[grid](
        state, live.view(torch.int8), weights, decay,
        schedule.slot, schedule.op, schedule.erase.view(torch.int8),
        schedule.live_after.view(torch.int8), schedule.payload,
        reads, out_state, out_live.view(torch.int8),
        tokens, rank,
        *weights.stride(), *decay.stride(),
        *schedule.slot.stride(), *schedule.payload.stride()[:4],
        *reads.stride(),
        H=heads, L=schedule.lanes, V=value_dim,
        RP=triton.next_power_of_2(rank), BV=block, STAGES=stages, num_warps=warps)
    return reads, out_state, out_live


# ------------------------------------------------------- the baseline loop, and Option C

def chunked_block_scan(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
                       decay: torch.Tensor, schedule: MatrixSchedule,
                       chunk_tokens: int = 128):
    """The K2+K3 baseline over a whole block: ``ceil(T/C)`` chunks x 3 launches.

    Identical to ``scan.chunked_scan(..., fused=True)`` in kernels and in numbers; it
    exists so every path in this file has one signature.  **Option C** is this call at a
    larger ``chunk_tokens``: the dispatch cost is ``T/C`` launches, so C = 256 or 512 cuts
    the host loop 2-4x, and the intra-chunk interaction (K2) is the term that grows with
    C (``O(tokens * steps)`` per chunk).
    """
    check_block_shapes(state, live, weights, decay, schedule)
    check_no_grad(state, weights, decay, schedule.payload)
    batch, heads, tokens, _ = weights.shape
    value_dim = state.shape[-1]
    chunk_tokens = int(chunk_tokens)
    reads = torch.empty((batch, heads, tokens, value_dim), dtype=torch.float32,
                        device=state.device)
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        chunk_reads, state, live = k23_chunk(
            state, live, weights[:, :, start:stop], decay[:, :, start:stop],
            schedule.chunk(start, stop))
        reads[:, :, start:stop] = chunk_reads
    return reads, state, live


# --------------------------------------------------------------- Option B: one graph

class BlockGraph:
    """**Option B** -- the whole block call captured into one CUDA graph.

    The pipeline is sequential, so the graph does not parallelise anything: it *is* the
    host loop, moved into a replay.  That is what makes it legitimate here -- and why it
    needs the whole block to be static, which at this boundary it is: the schedule and the
    read weights are recorded inputs, and every shape is fixed by the block.

    Contract, stated because it is the trap this lane already hit at step granularity:

    * every input is copied into a static buffer at :meth:`run` (a handful of device
      copies, no allocation), so the caller's tensors need not persist;
    * the returned ``reads``/``state``/``live`` **are the static output buffers** and the
      next replay overwrites them.  ``run(..., clone=True)`` copies them out (and pays
      for the copy, which ``NO_HOST_LOOP.md`` §3 measures) -- the honest default is
      ``clone=False`` with the lifetime stated, exactly as ``torch.cuda.graph``'s own
      users have to state it.

    Capture happens once per (shape, chunk): a warm, uncompiled-graph call first, so the
    Triton kernels are compiled and their autotune cache is filled *outside* the capture
    (autotune benchmarks by launching the kernel, which must not happen inside a graph).
    """

    def __init__(self, batch: int, heads: int, tokens: int, rank: int, value_dim: int,
                 lanes: int, chunk_tokens: int = 128, device: str = "cuda"):
        self.batch, self.heads, self.tokens = int(batch), int(heads), int(tokens)
        self.rank, self.value_dim = int(rank), int(value_dim)
        self.lanes, self.chunk_tokens = int(lanes), int(chunk_tokens)
        self.device = device
        shape = (batch, tokens, heads, lanes)
        self.state_in = torch.zeros(batch, heads, rank, value_dim, device=device)
        self.live_in = torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
        self.weights_in = torch.zeros(batch, heads, tokens, rank, device=device)
        self.decay_in = torch.zeros(batch, heads, tokens, device=device)
        self.schedule_in = MatrixSchedule(
            slot=torch.zeros(shape, dtype=torch.int64, device=device),
            erase=torch.zeros(shape, dtype=torch.bool, device=device),
            payload=torch.zeros(*shape, value_dim, device=device),
            op=torch.zeros(shape, dtype=torch.int64, device=device),
            live_after=torch.zeros(shape, dtype=torch.bool, device=device))
        # The state carrying across chunks lives in a static buffer: that is the one
        # thing a graph cannot carry in Python.
        self.state_carry = torch.zeros(batch, heads, rank, value_dim, device=device)
        self.live_carry = torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
        self.reads_out = torch.zeros(batch, heads, tokens, value_dim, device=device)
        self.state_out = torch.zeros(batch, heads, rank, value_dim, device=device)
        self.live_out = torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None

    # -- capture ------------------------------------------------------------------
    def _body(self):
        """The captured region: copy the entry state in, run the chunks, copy out."""
        self.state_carry.copy_(self.state_in)
        self.live_carry.copy_(self.live_in)
        reads, state, live = chunked_block_scan(
            self.state_carry, self.live_carry, self.weights_in, self.decay_in,
            self.schedule_in, self.chunk_tokens)
        self.reads_out.copy_(reads)
        self.state_out.copy_(state)
        self.live_out.copy_(live)

    def capture(self) -> "BlockGraph":
        """Warm on a side stream, then capture. One replay per block call afterwards."""
        for _ in range(2):
            self._body()
        torch.cuda.synchronize(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        return self

    # -- use ----------------------------------------------------------------------
    def run(self, state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
            decay: torch.Tensor, schedule: MatrixSchedule, clone: bool = False):
        """One replay of the whole block call: ``(reads, state, live)``."""
        if self.graph is None:
            raise RuntimeError("BlockGraph.run before capture()")
        self.state_in.copy_(state)
        self.live_in.copy_(live)
        self.weights_in.copy_(weights)
        self.decay_in.copy_(decay)
        self.schedule_in.slot.copy_(schedule.slot)
        self.schedule_in.erase.copy_(schedule.erase)
        self.schedule_in.payload.copy_(schedule.payload)
        self.schedule_in.op.copy_(schedule.op)
        self.schedule_in.live_after.copy_(schedule.live_after)
        self.graph.replay()
        if clone:
            return (self.reads_out.clone(), self.state_out.clone(),
                    self.live_out.clone())
        return self.reads_out, self.state_out, self.live_out


# ---------------------------------------------------------------------------- dispatch

BLOCK_PATHS = ("chunked", "persistent", "auto")


def block_scan(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
               decay: torch.Tensor, schedule: MatrixSchedule, path: str = "auto",
               chunk_tokens: int = 128):
    """One whole block, by name.  ``auto`` is the persistent kernel where it fits.

    ``persistent`` raises when the store does not fit one program's registers
    (:func:`persistent_available`), which is the honest fallback condition: at rank 2048
    with a value dim of 128 the tile is 1 MB and rank must be blocked, and a blocked rank
    would need a cross-program reduction per token.
    """
    if path == "chunked":
        return chunked_block_scan(state, live, weights, decay, schedule, chunk_tokens)
    if path == "persistent":
        return persistent_scan(state, live, weights, decay, schedule)
    if path != "auto":
        raise ValueError(f"unknown block path {path!r}; expected one of {BLOCK_PATHS}")
    if persistent_available(state.shape[2], state.shape[3]):
        return persistent_scan(state, live, weights, decay, schedule)
    return chunked_block_scan(state, live, weights, decay, schedule, chunk_tokens)


__all__ = ['BLOCK_PATHS', 'BlockGraph', 'MAX_PERSISTENT_TILE', 'MIN_VALUE_BLOCK',
           'block_scan', 'check_block_shapes', 'chunked_block_scan',
           'persistent_available', 'persistent_scan', 'value_block', 'warp_count']
