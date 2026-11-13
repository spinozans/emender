"""The decision schedule and the read weights: detached routing, per chunk.

Decisions are *architectural*, exactly as in the registered tape arm (argmax
match, thresholds, case codes): nothing here is differentiated, and nothing here
runs inside the scan.  What is new relative to the tape arm is only the granularity
-- a chunk's decisions are taken against the chunk-*entry* state, which is what
makes routing chunk-parallel and lets the scan stay affine.  The price is that two
tokens of one chunk may target the same slot; the scan resolves that with the
delta-rule order (later lane wins) and the repeat-rate statistic records it.

Cases, in order (thresholds are frozen policy constants recorded in the receipt):

``CORRECT``  the incoming value's best cosine to a live row is ``>= tau_match``:
             overwrite that row with the value (exact, orthonormal keys).
``MERGE``    as CORRECT *and* a second live row is ``>= tau_merge`` similar: CORRECT
             the target row with a norm-weighted blend of the two stored contents
             and POP the source -- the registration's "MERGE = two rank-1 ops".
``APPEND``   otherwise, if the active rank has a free slot: write the value there.
``POP``      otherwise (store full, nothing to overwrite, worst match
             ``<= tau_pop``): free the worst-matching live row.
``NONE``     otherwise: the write is dropped (the capacity-pressure case).

The read weights are produced here too, because they are computed from the same
chunk-entry state, but they are **not** a decision: the softmax and the strengths
stay attached to the graph (``read.read_weights`` is the single-sourced read), while
the liveness mask that says which rows exist is detached.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .algebra import APPEND, CORRECT, MAX_LANES, MERGE, NONE, MatrixSchedule, POP
from .read import read_weights

EPS = 1e-6


@dataclass(frozen=True)
class RoutingPolicy:
    """Frozen machinery constants.  ``ops=False`` is the append-only control."""

    ops: bool = True
    tau_match: float = 0.45
    tau_merge: float = 0.85
    tau_pop: float = 0.10
    tie_break: float = 1e-6
    read_scale: float = 2.0
    grow: bool = False
    grow_trigger: float = 0.5
    grow_window: int = 4
    grow_window_tokens: int = 512
    grow_geometric: float = 2.0
    grow_stride_floor: int = 64

    def __post_init__(self):
        if not -1.0 <= self.tau_pop <= self.tau_match <= 1.0:
            raise ValueError("thresholds must satisfy -1 <= tau_pop <= tau_match <= 1")
        if not -1.0 <= self.tau_merge <= 1.0:
            raise ValueError("tau_merge must be a cosine in [-1, 1]")
        if self.tie_break <= 0:
            raise ValueError("tie_break must be positive (deterministic lowest-index wins)")
        if self.read_scale <= 0:
            raise ValueError("read_scale must be positive")
        if not 0.0 <= self.grow_trigger <= 1.0:
            raise ValueError("grow_trigger must be a rate in [0, 1]")
        if self.grow_window < 1 or self.grow_window_tokens < 1:
            raise ValueError("growth windows must be positive")
        if self.grow_geometric <= 1.0 or self.grow_stride_floor < 1:
            raise ValueError("growth must be monotone: geometric > 1, stride floor >= 1")

    def record(self) -> dict:
        return dict(ops=self.ops, tau_match=self.tau_match, tau_merge=self.tau_merge,
                    tau_pop=self.tau_pop, tie_break=self.tie_break,
                    read_scale=self.read_scale, grow=self.grow,
                    grow_trigger=self.grow_trigger, grow_window=self.grow_window,
                    grow_window_tokens=self.grow_window_tokens,
                    grow_geometric=self.grow_geometric,
                    grow_stride_floor=self.grow_stride_floor)


def _blank(batch, tokens, heads, lanes, value_dim, device, dtype):
    shape = (batch, heads, tokens, lanes)
    return dict(
        slot=torch.zeros(shape, dtype=torch.int64, device=device),
        erase=torch.zeros(shape, dtype=torch.bool, device=device),
        payload=torch.zeros(*shape, value_dim, dtype=dtype, device=device),
        op=torch.zeros(shape, dtype=torch.int64, device=device),
        live_after=torch.zeros(shape, dtype=torch.bool, device=device))


def _schedule(fields) -> MatrixSchedule:
    # Routing works head-major; the scan consumes (batch, tokens, heads, lanes).
    return MatrixSchedule(
        slot=fields["slot"].permute(0, 2, 1, 3),
        erase=fields["erase"].permute(0, 2, 1, 3),
        payload=fields["payload"].permute(0, 2, 1, 3, 4),
        op=fields["op"].permute(0, 2, 1, 3),
        live_after=fields["live_after"].permute(0, 2, 1, 3))


def route_chunk(index, token_offset, state, live, query, payload, policy, rank):
    """Route one chunk and produce its read weights.

    ``state`` ``(batch, heads, rank_max, d_v)``, ``live`` ``(batch, heads,
    rank_max)``, ``query``/``payload`` ``(batch, heads, tokens, d_v)``.  Returns
    ``(schedule, weights)``: the schedule in the scan's
    ``(batch, tokens, heads, lanes)`` layout and the read weights
    ``(batch, heads, tokens, rank)``.
    """
    batch, heads, tokens, value_dim = payload.shape
    device, dtype = payload.device, torch.float32
    rank = int(rank)
    active = live[:, :, :rank]
    rows = state[:, :, :rank]
    # The read-weight channel is always as wide as the state, zero above the active
    # rank: the schedule is a prefix of the family, and the scan's matmuls line up
    # even when the rank grows at a chunk boundary inside this scan.
    weights = torch.zeros(batch, heads, tokens, state.shape[2], dtype=dtype, device=device)
    weights[:, :, :, :rank] = read_weights(query, rows, active, policy.read_scale)

    if not policy.ops:
        steps = token_offset + torch.arange(tokens, device=device)
        slot = (steps % rank).view(1, 1, tokens, 1).expand(batch, heads, tokens, 1)
        fields = _blank(batch, tokens, heads, 1, value_dim, device, dtype)
        fields["slot"][...] = slot
        fields["payload"][..., 0, :] = payload.to(dtype)
        fields["op"][...] = APPEND
        fields["live_after"][...] = True
        return _schedule(fields), weights

    frozen_rows = rows.detach()
    strength = frozen_rows.norm(dim=-1) * active.to(dtype)            # (b, h, rank)
    stored = frozen_rows / strength.clamp_min(EPS).unsqueeze(-1)
    value = payload / payload.norm(dim=-1, keepdim=True).clamp_min(EPS)
    similarity = value @ stored.transpose(-1, -2)                      # (b, h, tokens, rank)
    similarity = similarity.masked_fill(~active.unsqueeze(-2), -2.0)
    penalty = torch.arange(rank, dtype=dtype, device=device) * policy.tie_break
    score = similarity - penalty
    best_index = score.argmax(-1)
    best_value = similarity.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    second_score = score.scatter(-1, best_index.unsqueeze(-1), -3.0)
    second_index = second_score.argmax(-1)
    second_value = similarity.gather(-1, second_index.unsqueeze(-1)).squeeze(-1)

    correct = best_value >= policy.tau_match
    merge = correct & (second_value >= policy.tau_merge)

    free = ~active
    position = free.cumsum(-1) - 1
    free_count = free.sum(-1)
    need = ~correct
    ordinal = need.cumsum(-1) - 1
    appendable = need & (ordinal < free_count.unsqueeze(-1))
    free_slot = ((position.unsqueeze(-2) == ordinal.unsqueeze(-1)) & free.unsqueeze(-2))
    free_slot = free_slot.to(torch.int64).argmax(-1)

    worst_index = score.argmin(-1)
    worst_value = similarity.gather(-1, worst_index.unsqueeze(-1)).squeeze(-1)
    full = free_count.unsqueeze(-1) <= 0
    pop = (~correct) & (~appendable) & full & (worst_value <= policy.tau_pop) \
        & active.any(-1, keepdim=True)

    fields = _blank(batch, tokens, heads, MAX_LANES, value_dim, device, dtype)
    op = torch.where(correct,
                     torch.where(merge, torch.full_like(best_index, MERGE),
                                 torch.full_like(best_index, CORRECT)),
                     torch.where(appendable, torch.full_like(best_index, APPEND),
                                 torch.where(pop, torch.full_like(best_index, POP),
                                             torch.full_like(best_index, NONE))))
    slot = torch.where(correct, best_index,
                       torch.where(appendable, free_slot, worst_index))
    fields["slot"][..., 0] = slot
    fields["op"][..., 0] = op
    fields["erase"][..., 0] = (op == CORRECT) | (op == MERGE) | (op == POP)
    fields["live_after"][..., 0] = op != POP

    # Payloads: the incoming value for APPEND/CORRECT, zero for POP and for an
    # inactive lane, and for MERGE a norm-weighted blend of the two stored rows
    # (weights detached, content attached so gradients reach what was stored).
    # The row accesses are dense one-hot matmuls, never a gather.
    addresses = torch.arange(rank, device=device)
    first_mask = (best_index.unsqueeze(-1) == addresses).to(dtype)
    second_mask = (second_index.unsqueeze(-1) == addresses).to(dtype)
    first_row = first_mask @ rows
    second_row = second_mask @ rows
    first_weight = strength.gather(-1, best_index).clamp_min(EPS)
    second_weight = strength.gather(-1, second_index).clamp_min(EPS)
    total = first_weight + second_weight
    blend = ((first_weight / total).detach().unsqueeze(-1) * first_row
             + (second_weight / total).detach().unsqueeze(-1) * second_row)
    carries_value = (op == APPEND) | (op == CORRECT)
    payload_lane0 = torch.where(carries_value.unsqueeze(-1), payload,
                                torch.zeros_like(payload))
    payload_lane0 = torch.where((op == MERGE).unsqueeze(-1), blend, payload_lane0)
    fields["payload"][..., 0, :] = payload_lane0

    fields["slot"][..., 1] = second_index
    fields["op"][..., 1] = torch.where(merge, torch.full_like(second_index, MERGE),
                                       torch.full_like(second_index, NONE))
    fields["erase"][..., 1] = merge
    fields["live_after"][..., 1] = False
    return _schedule(fields), weights
