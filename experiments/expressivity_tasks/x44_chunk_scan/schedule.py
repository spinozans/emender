"""Per-token decision schedule consumed by the chunked scan.

The proved statement (``formal/lean/OPERATOR_ALGEBRA_STATUS.md`` section 2-4)
is conditional on a *fixed decision schedule*: gates, selected write slot,
pop target and magnitude, decay/death masks, and ages.  This module is that
schedule as data.  It is exactly the discrete channel the sequential X30 path
records (``x30_tape_cell.routing_trace.RoutingTrace``) plus the two magnitudes
the affine recurrence needs (append gate, pop factor).

``DecisionSchedule.from_naive_trace`` is the only builder used by the parity
tests: it re-reads a sequential ``naive_reference.tape_scan`` trace so the
chunked path and the sequential path are driven by the same decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

NO_SLOT = -1


@dataclass(frozen=True)
class DecisionSchedule:
    """Discrete decisions for one token range, flattened to ``[rows, tokens]``.

    ``rows`` indexes (batch element, head) pairs; the memory state of two rows
    never interacts.
    """

    alloc: torch.Tensor       # [rows, tokens] int64  append slot, -1 = none
    gate: torch.Tensor        # [rows, tokens] float  append gate magnitude
    correct: torch.Tensor     # [rows, tokens] int64  correction/merge slot, -1 = none
    beta: torch.Tensor        # [rows, tokens] float  delta-rule interpolation
    pop_slot: torch.Tensor    # [rows, tokens] int64  popped slot, -1 = none
    pop_factor: torch.Tensor  # [rows, tokens] float  strength multiplier 1-pop

    @property
    def rows(self) -> int:
        return self.alloc.shape[0]

    @property
    def tokens(self) -> int:
        return self.alloc.shape[1]

    @property
    def slots(self) -> int:
        raise AttributeError('slot count is a state property, not a schedule property')

    def check(self, rows: int, tokens: int, slots: int) -> 'DecisionSchedule':
        for name in ('alloc', 'gate', 'correct', 'beta', 'pop_slot', 'pop_factor'):
            value = getattr(self, name)
            if value.shape != (rows, tokens):
                raise ValueError(f'{name} must be [{rows}, {tokens}]')
        for name in ('alloc', 'correct', 'pop_slot'):
            value = getattr(self, name)
            if value.dtype != torch.int64:
                raise ValueError(f'{name} must be int64')
            if value.numel() and (int(value.min()) < -1 or int(value.max()) >= slots):
                raise ValueError(f'{name} outside [-1, {slots})')
        return self

    def to(self, device) -> 'DecisionSchedule':
        return DecisionSchedule(*(getattr(self, name).to(device) for name in
                                  ('alloc', 'gate', 'correct', 'beta',
                                   'pop_slot', 'pop_factor')))

    def fail_on_collision(self) -> None:
        """Reject schedules whose append and correction target the same row-token.

        The sequential reference gathers the correction's before-image from the
        pre-append store while writing the correction into the post-append
        store, so a collision has a defined but surprising meaning.  The
        arithmetic routing never produces one (an append target is free, a
        correction target is live); the chunked scan reports instead of
        guessing.
        """
        both = (self.alloc >= 0) & (self.correct >= 0)
        if bool(both.any()):
            raise ValueError('schedule appends and corrects at the same row-token')


def from_naive_trace(trace, pop_strength: torch.Tensor, write_strength: torch.Tensor,
                     append_logit: torch.Tensor, cfg) -> DecisionSchedule:
    """Build a schedule from a sequential ``naive_reference`` routing trace.

    ``pop_strength``/``write_strength``/``append_logit`` are the ``[B, T, NH]``
    inputs of that same sequential run.  The append gate is
    ``sigmoid(append_logit)``: the residual-dependent term
    ``append_err_coef * error`` is zero in production (``error_coef`` is a
    zero buffer in the X42 arms) and is rejected rather than silently dropped
    when non-zero -- see ``RESULT_CHUNK_SCAN.md``.
    """
    b, t, h = pop_strength.shape
    rows = b * h

    def flat(x):
        # rows index (batch, head) in that order, matching the [rows, T, dim]
        # layout the chunked scan expects for its token inputs.
        return x.reshape(b, t, h).transpose(1, 2).reshape(rows, t)

    alloc = flat(trace.alloc_target).contiguous()
    correct = flat(trace.matched).contiguous()
    gate = flat(torch.sigmoid(append_logit.float()))
    beta = flat(write_strength.float())
    fired = flat(trace.pop_fired)
    target = flat(trace.pop_target)
    pop_slot = torch.where(fired, target, torch.full_like(target, NO_SLOT))
    pop_factor = torch.where(fired, 1.0 - flat(pop_strength.float()),
                             torch.ones(rows, t, dtype=torch.float32,
                                        device=fired.device))
    return DecisionSchedule(alloc, gate, correct, beta, pop_slot, pop_factor)


__all__ = ['DecisionSchedule', 'from_naive_trace', 'NO_SLOT']
