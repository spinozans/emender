"""Chunked scan of the matrix-state Emender.  No checkpoint replay anywhere.

Formulation
-----------
Per token the state update is the delta rule

    S_t = (I - c_t k_{j_t} k_{j_t}^T) diag(lam_t) S_{t-1} + k_{j_t} p_t^T

with ``c_t`` the erase flag of the token's lane (CORRECT/POP erase, APPEND does
not) and ``p_t`` the payload.  Writing ``gc_s = sum_{u<=s} log lam_u`` over the
chunk's *step* axis (a token with ``L`` lanes occupies ``L`` steps; the DECAY gate
sits on lane 0 and the other lanes are neutral), the chunk decomposes as

    S_s = exp(gc_s) H + sum_{u<=s} exp(gc_s - gc_u) k_{j_u} d_u^T
    d_s = p_s - c_s ( exp(gc_s) H[j_s] + sum_{u<s, j_u=j_s} exp(gc_s-gc_u) d_u )

so ``d`` solves the unit lower triangular system ``(I + A) d = rhs`` with
``A[s,u] = c_s exp(gc_s - gc_u) [j_s = j_u] [u < s]``.  ``A`` is block-diagonal
*by slot*, so an erase in one direction cannot leak into another: the cross-slot
entries are structurally zero, not small.

Reads are ``o_t = w_t^T S_{t-1}`` (``w`` from ``read.read_weights``), i.e. the
chunk-entry state plus the in-chunk injection:

    o_t = exp(gc_{s0-1}) w_t^T H + sum_{u<s0} exp(gc_{s0-1}-gc_u) w_t[j_u] d_u .

Addressing is device-side and out-of-place: the rows a chunk addresses are
``state[.., slot]`` (a gather), the injection is accumulated into a fresh zeros
tensor with ``scatter_add_``, and the read addresses the weight channel with a
gather.  Measured reason (see ``RESULT_MATRIX_EMENDER.md``): expressing the same
three operations as one-hot *matmuls* -- ``onehot @ state`` and
``onehot^T @ delta`` -- costs ``rank`` times the arithmetic (128x at rank 128),
and took the arm from 21k to 9k tokens/s with no change in semantics.  None of the
ops here need a host synchronisation or a dynamic shape, so the block stays
capturable and autograd-friendly; the only index write is onto freshly allocated
temporaries.

No division of an accumulated quantity appears: the decay channel is a *sum* scan
in log space (the gate is floored strictly positive) and the erase/overwrite zeros
live in the triangular solve, which is a forward substitution.  The inter-chunk
carry is the exact state -- one masked matmul per chunk.  That is the honest form:
the entry state of chunk ``c`` depends on the deaths inside chunks ``< c`` and
those depend on the entry state (the runtime shadow of
``state_dependent_death_not_affine``), so the composed prefix-product form lives in
the tests rather than on the runtime path.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from .algebra import MAX_LANES, NONE, empty_schedule, op_histogram

DEFAULT_CHUNK_TOKENS = 64


class RoutedResult(NamedTuple):
    """Everything a routed scan produces, including its recorded decision channel."""

    reads: torch.Tensor
    state: torch.Tensor
    live: torch.Tensor
    rank: int
    stats: "ScanStats"
    schedule: torch.Tensor | None      # recorded decision channel (collect=True)
    weights: torch.Tensor | None       # recorded read-weight channel (collect=True)


class ScanStats:
    """Per-chunk telemetry: op counts, capacity pressure, rank trajectory.

    Filled only when the caller asks (``collect=True``) or the arm grows, because
    reporting a chunk costs a device synchronisation; the fixed-rank training path
    never pays it.
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self.rank_trajectory: list[dict] = []
        self.growth_events: list[dict] = []

    def add_chunk(self, index, rank, live_fraction, writes, repeat_rate, pressure, ops):
        self.chunks.append(dict(chunk=int(index), rank=int(rank), writes=int(writes),
                                live_fraction=float(live_fraction),
                                repeat_rate=float(repeat_rate),
                                pressure=float(pressure), ops=ops))
        if not self.rank_trajectory or self.rank_trajectory[-1]["rank"] != int(rank):
            self.rank_trajectory.append(dict(chunk=int(index), rank=int(rank)))

    def grow(self, index, before, after):
        self.growth_events.append(dict(chunk=int(index), rank_before=int(before),
                                       rank_after=int(after)))

    def summary(self) -> dict:
        return dict(chunks=self.chunks, rank_trajectory=self.rank_trajectory,
                    growth_events=self.growth_events)


def _steps(schedule, tokens):
    """Flatten ``(batch, tokens, heads, lanes, ...)`` onto the chunk's step axis."""
    batch, _, heads, lanes = schedule.slot.shape
    steps = tokens * lanes
    slot = schedule.slot.permute(0, 2, 1, 3).reshape(batch, heads, steps)
    erase = schedule.erase.permute(0, 2, 1, 3).reshape(batch, heads, steps)
    op = schedule.op.permute(0, 2, 1, 3).reshape(batch, heads, steps)
    live_after = schedule.live_after.permute(0, 2, 1, 3).reshape(batch, heads, steps)
    payload = schedule.payload.permute(0, 2, 1, 3, 4).reshape(
        batch, heads, steps, schedule.payload.shape[-1])
    return slot, erase, op, live_after, payload, lanes, steps


def chunk_scan(state, live, weights, decay, schedule, fused: bool = False):
    """One chunk.

    ``state`` ``(batch, heads, rank, d_v)``, ``live`` ``(batch, heads, rank)``,
    ``weights`` ``(batch, heads, tokens, rank)`` the fixed read weights of this
    chunk, ``decay`` ``(batch, heads, tokens)``, ``schedule`` the chunk's decision
    channel (the only carrier of the write payloads).  Returns ``(reads, state,
    live)`` with ``reads`` ``(batch, heads, tokens, d_v)``.

    ``fused=True`` runs stage K1 instead: one Triton kernel per chunk that walks
    the chunk's steps per slot in registers and emits the injection list, then the
    same read reduction over it (``k1_schedule_scan.k1_chunk``).  It is default-off
    -- the eager path below *is* the reference the fused one is held to.
    """
    if fused:
        from .k1_schedule_scan import k1_chunk
        return k1_chunk(state, live, weights, decay, schedule)
    dtype = torch.float32
    batch, heads, rank, value_dim = state.shape
    tokens = weights.shape[2]
    slot, erase, op, live_after, pay, lanes, steps = _steps(schedule, tokens)
    # An inactive lane is a no-op whatever payload it carries: mask it explicitly.
    pay = pay.to(dtype) * op.ne(NONE).unsqueeze(-1)
    weights = weights.to(dtype)

    # DECAY sits on lane 0 of every token; the other lanes are neutral steps.
    lam = torch.ones(batch, heads, steps, dtype=dtype, device=state.device)
    lam[:, :, ::lanes] = decay.to(dtype)
    gc = torch.cumsum(lam.log(), dim=-1)                        # (b, h, steps)
    zero = torch.zeros((), dtype=dtype, device=state.device)

    # The rows this chunk addresses (a gather: the store's rows are the keys).
    b_index = torch.arange(batch, device=state.device)[:, None, None]
    h_index = torch.arange(heads, device=state.device)[None, :, None]
    entry_rows = state[b_index, h_index, slot]                  # (b, h, steps, V)

    # Triangular coupling (block-diagonal by slot) and its solve.
    same = slot.unsqueeze(-1) == slot.unsqueeze(-2)             # (b, h, steps, steps)
    order = torch.arange(steps, device=state.device)
    lower = order.unsqueeze(-1) > order.unsqueeze(0)
    coefficients = torch.exp(gc.unsqueeze(-1) - gc.unsqueeze(-2))
    matrix = torch.where(same & lower, coefficients, zero) * erase.to(dtype).unsqueeze(-1)
    erase_scale = erase.to(dtype) * torch.exp(gc)
    rhs = pay - erase_scale.unsqueeze(-1) * entry_rows
    delta = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)

    # Reads at lane 0 of each token: entry state plus the in-chunk injection.
    read_index = torch.arange(tokens, device=state.device) * lanes - 1
    inside = (read_index >= 0).unsqueeze(0).unsqueeze(0)
    gc_read = torch.where(inside, gc[:, :, read_index.clamp(min=0)], zero)
    addressed = weights.gather(
        3, slot.unsqueeze(2).expand(batch, heads, tokens, steps))   # w_t[j_u]
    before = (torch.arange(tokens, device=state.device).unsqueeze(-1) * lanes
              > order.unsqueeze(0)).unsqueeze(0).unsqueeze(0)   # (1, 1, tokens, steps)
    read_matrix = torch.where(before, torch.exp(gc_read.unsqueeze(-1) - gc.unsqueeze(-2)),
                              zero) * addressed
    reads = (torch.exp(gc_read).unsqueeze(-1) * weights) @ state + read_matrix @ delta

    # Exit state: exact carry (decayed entry state plus the masked injection).
    injection = torch.exp(gc[:, :, -1:] - gc).unsqueeze(-1) * delta
    scattered = torch.zeros_like(state).scatter_add_(
        2, slot.unsqueeze(-1).expand(batch, heads, steps, value_dim), injection)
    new_state = torch.exp(gc[:, :, -1])[:, :, None, None] * state + scattered

    # Liveness at the chunk exit: the last lane touching a slot decides it.
    # (Integer bookkeeping only -- an amax, so it stays deterministic.)
    stamp = torch.arange(1, steps + 1, device=state.device, dtype=torch.int64)
    stamp = stamp.unsqueeze(0).unsqueeze(0).expand(batch, heads, steps)
    key = torch.where(op.ne(NONE), stamp * 2 + live_after.to(torch.int64), 0)
    # The chunk-entry liveness is the stamp-0 state, so a slot this chunk does not
    # touch keeps what it had (its key 0/1 loses the amax to any real touch >= 2).
    last = live.to(torch.int64).clone()
    last.scatter_reduce_(2, slot, key, reduce="amax", include_self=True)
    return reads, new_state, (last % 2).to(torch.bool)


def repeat_write_rate(schedule, last_write, step_offset, window_tokens):
    """Share of this chunk's writes that reuse a slot written inside the window.

    This is the registered "append-collision rate" statistic, made a function of
    the schedule alone (no state read) so growth is deterministic and reportable
    per chunk.  Updates ``last_write`` ``(batch, heads, rank)`` in place.
    """
    slot = schedule.slot[..., 0].permute(0, 2, 1)        # (batch, heads, tokens)
    written = schedule.op[..., 0].permute(0, 2, 1).ne(NONE)
    batch, heads, tokens = slot.shape
    step = step_offset + torch.arange(tokens, device=slot.device)
    step = step.unsqueeze(0).unsqueeze(0).expand(batch, heads, tokens)
    b_index = torch.arange(batch, device=slot.device)[:, None, None]
    h_index = torch.arange(heads, device=slot.device)[None, :, None]
    previous = last_write[b_index, h_index, slot]
    repeat = written & (previous >= (step - window_tokens)) & (previous >= 0)
    rate = repeat.sum().float() / written.sum().clamp_min(1).float()
    key = torch.where(written, step, torch.full_like(step, -1))
    last_write.scatter_reduce_(2, slot, key, reduce="amax", include_self=True)
    return float(rate)


def routed_chunk_scan(query, payload, decay, *, router, policy, rank=None,
                      rank_max=None, initial_state=None, initial_live=None,
                      chunk_tokens=None, collect=False, fused: bool = False):
    """Chunked scan whose fixed schedule (and read weights) come from ``router``.

    ``router(chunk_index, token_offset, state, live, query_chunk, payload_chunk,
    policy, rank)`` returns ``(schedule, read_weights)`` -- see
    ``schedule.route_chunk``.  The router reads the *detached* chunk-entry state, so
    from the scan's point of view the schedule is an input: nothing inside the scan
    differentiates a decision.  The read weights are *not* detached (they are the
    read), which is exactly the tape arm's split.

    With ``policy.grow`` the rank may widen at a chunk boundary.  Rows above the
    current rank are exact zeros (growth is a zero-injection), so widening is a
    bookkeeping change recorded in ``stats.rank_trajectory``.
    """
    chunk_tokens = DEFAULT_CHUNK_TOKENS if chunk_tokens is None else int(chunk_tokens)
    batch, tokens, heads, value_dim = payload.shape
    if rank_max is None:
        rank_max = int(value_dim if rank is None else rank)
    rank_max = int(rank_max)
    device = query.device
    state = (torch.zeros(batch, heads, rank_max, value_dim, dtype=torch.float32,
                         device=device) if initial_state is None else initial_state)
    live = (torch.zeros(batch, heads, rank_max, dtype=torch.bool, device=device)
            if initial_live is None else initial_live)
    rank_now = rank_max if rank is None else int(rank)
    stats = ScanStats()
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=torch.float32, device=device)
    schedule_out = None
    weights_out = None
    growth = getattr(policy, "grow", False)
    last_write = (torch.zeros(batch, heads, rank_now, dtype=torch.int64, device=device)
                  if (growth or collect) else None)
    window: list[float] = []
    for index, start in enumerate(range(0, tokens, chunk_tokens)):
        stop = min(start + chunk_tokens, tokens)
        qc = query[:, start:stop].permute(0, 2, 1, 3)
        pc = payload[:, start:stop].permute(0, 2, 1, 3)
        dc = decay[:, start:stop].permute(0, 2, 1)
        schedule, weights = router(index, start, state, live, qc, pc, policy, rank_now)
        if collect:
            if schedule_out is None:
                schedule_out = empty_schedule(batch, tokens, heads, value_dim,
                                              schedule.lanes, device=device)
                weights_out = torch.zeros(batch, tokens, heads, rank_max,
                                          dtype=torch.float32, device=device)
            _write_chunk(schedule_out, start, stop, schedule)
            weights_out[:, start:stop] = weights.permute(0, 2, 1, 3)
            stats.add_chunk(
                index, rank_now, live[:, :, :rank_now].float().mean(),
                int(schedule.op.ne(NONE).sum()),
                repeat_write_rate(schedule, last_write, start, policy.grow_window_tokens),
                float(schedule.op[..., 0].eq(NONE).float().mean()), op_histogram(schedule))
            window.append(stats.chunks[-1]["repeat_rate"])
        elif growth:
            window.append(repeat_write_rate(schedule, last_write, start,
                                            policy.grow_window_tokens))
        chunk_reads, state, live = chunk_scan(state, live, weights, dc, schedule, fused)
        reads[:, start:stop] = chunk_reads.permute(0, 2, 1, 3)
        if growth and len(window) >= policy.grow_window and rank_now < rank_max:
            if sum(window[-policy.grow_window:]) / policy.grow_window >= policy.grow_trigger:
                from .keys import growth_target
                grown = min(rank_max, growth_target(
                    rank_now, policy.grow_geometric, policy.grow_stride_floor))
                stats.grow(index, rank_now, grown)
                rank_now = grown
                last_write = torch.zeros(batch, heads, rank_now, dtype=torch.int64,
                                         device=device)
                window = []
        if collect:
            window = window[-policy.grow_window:]
    return RoutedResult(reads, state, live, rank_now, stats, schedule_out, weights_out)


def _write_chunk(schedule, start, stop, chunk):
    schedule.slot[:, start:stop] = chunk.slot
    schedule.erase[:, start:stop] = chunk.erase
    schedule.payload[:, start:stop] = chunk.payload
    schedule.op[:, start:stop] = chunk.op
    schedule.live_after[:, start:stop] = chunk.live_after


def chunked_scan(weights, decay, schedule, initial_state=None,
                 initial_live=None, chunk_tokens=None, fused: bool = False):
    """Chunked scan of a *given* fixed schedule (the parity twin of the oracle)."""
    chunk_tokens = DEFAULT_CHUNK_TOKENS if chunk_tokens is None else int(chunk_tokens)
    batch, tokens, heads, rank_max = weights.shape
    value_dim = schedule.payload.shape[-1]
    device = weights.device
    state = (torch.zeros(batch, heads, rank_max, value_dim, dtype=torch.float32,
                         device=device) if initial_state is None else initial_state)
    live = (torch.zeros(batch, heads, rank_max, dtype=torch.bool, device=device)
            if initial_live is None else initial_live)
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=torch.float32, device=device)
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        chunk_reads, state, live = chunk_scan(
            state, live, weights[:, start:stop].permute(0, 2, 1, 3),
            decay[:, start:stop].permute(0, 2, 1), schedule.chunk(start, stop), fused)
        reads[:, start:stop] = chunk_reads.permute(0, 2, 1, 3)
    return reads, state, live
