"""The five Emender ops as rank-1 / diagonal updates on the matrix state.

The state is ``S in R^{r x d_v}`` per head.  Row ``j`` is the stored content
along key direction ``k_j`` of the active orthonormal prefix, so for orthonormal
keys ``S k_j == S[j]`` and the registration's five ops are, exactly,

===========  ==========================================  ==========================
op           registration form                           coefficient form
===========  ==========================================  ==========================
APPEND       ``S += v k_j^T``                            ``S[j] += v``
CORRECT      ``S += (v' - S k_j) k_j^T``                 ``S[j]  = v'``
POP          ``S -= (S k_j) k_j^T``                      ``S[j]  = 0``
DECAY        ``S *= lam``                                ``S     = lam * S``
MERGE        two rank-1 ops (CORRECT target + POP src)   both of the above
===========  ==========================================  ==========================

The three rank-1 writes are one update carrying an *erase* flag,
``S[j] = erase ? v' : S[j] + v'``, which is the delta rule
``S <- (I - e_j k_j^T) lam S + k_j v'^T``.  Everything the scan consumes is the
:class:`MatrixSchedule`: a fixed, detached decision channel of
``(slot, erase, payload, op, live_after)`` lanes -- exactly what
``MemoryAlgebra.ScanCompatibleRelativeTo`` requires, and nothing is recomputed
from rounded state inside the scan.

``sequential_matrix_scan`` is the per-token semantic oracle; ``scan.chunked_scan``
is held to it.  ``key_form_step`` writes the same op through the key table itself
(``S k_j`` as a real contraction) so the exactness tests exercise the
orthonormality premise rather than the coefficient-space shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

NONE, APPEND, CORRECT, POP, MERGE = 0, 1, 2, 3, 4
OP_NAMES = ("none", "append", "correct", "pop", "merge")
# Every lane count the arms use: `matrix_ops` needs two lanes (MERGE), the
# append-only control needs one.
MAX_LANES = 2


@dataclass
class MatrixSchedule:
    """The fixed decision channel for one sequence batch.

    All fields are ``(batch, tokens, heads, lanes)`` shaped (``payload`` adds a
    trailing ``d_v`` axis).  Lane ``l`` of token ``t`` is applied after lane
    ``l-1``; an inactive lane is ``op=NONE``.  Only lanes with ``op != NONE``
    touch the state and the liveness channel, so ``live_after`` is the death /
    birth bookkeeping that the scan must reproduce bit-exactly (``pop`` is the
    only op with ``live_after=False`` on a single-lane write; the source lane of
    a MERGE is the other one).
    """

    slot: torch.Tensor
    erase: torch.Tensor
    payload: torch.Tensor
    op: torch.Tensor
    live_after: torch.Tensor

    def __post_init__(self):
        if self.slot.shape != self.erase.shape or self.slot.shape != self.op.shape:
            raise ValueError("schedule slot/erase/op shapes disagree")
        if self.slot.shape != self.live_after.shape:
            raise ValueError("schedule live_after shape disagrees with slot")
        if self.payload.shape[:-1] != self.slot.shape:
            raise ValueError("schedule payload shape disagrees with slot")
        if self.slot.dtype != torch.int64 or self.op.dtype != torch.int64:
            raise ValueError("schedule slots and op codes must be int64")
        if self.erase.dtype != torch.bool or self.live_after.dtype != torch.bool:
            raise ValueError("schedule erase/live_after must be bool")

    @property
    def lanes(self) -> int:
        return int(self.slot.shape[-1])

    @property
    def tokens(self) -> int:
        return int(self.slot.shape[1])

    def chunk(self, start: int, stop: int) -> "MatrixSchedule":
        return MatrixSchedule(self.slot[:, start:stop], self.erase[:, start:stop],
                              self.payload[:, start:stop], self.op[:, start:stop],
                              self.live_after[:, start:stop])

    def to_record(self) -> dict:
        """Everything needed to reproduce this schedule (determinism receipt)."""
        return dict(slot=self.slot.to("cpu").tolist(), erase=self.erase.to("cpu").tolist(),
                    payload=self.payload.to("cpu").tolist(), op=self.op.to("cpu").tolist(),
                    live_after=self.live_after.to("cpu").tolist())


def empty_schedule(batch: int, tokens: int, heads: int, value_dim: int, lanes: int,
                   device=None, dtype=torch.float32) -> MatrixSchedule:
    shape = (batch, tokens, heads, lanes)
    return MatrixSchedule(
        slot=torch.zeros(shape, dtype=torch.int64, device=device),
        erase=torch.zeros(shape, dtype=torch.bool, device=device),
        payload=torch.zeros(*shape, value_dim, dtype=dtype, device=device),
        op=torch.zeros(shape, dtype=torch.int64, device=device),
        live_after=torch.zeros(shape, dtype=torch.bool, device=device))


def op_histogram(schedule: MatrixSchedule) -> dict:
    """Counts per op name over every lane (telemetry, never a decision)."""
    counts = torch.bincount(schedule.op.reshape(-1).to("cpu"), minlength=len(OP_NAMES))
    return {name: int(counts[i]) for i, name in enumerate(OP_NAMES)}


def apply_lane(state: torch.Tensor, slot: torch.Tensor, erase: torch.Tensor,
               payload: torch.Tensor) -> torch.Tensor:
    """One lane-step on ``state`` ``(batch, heads, rank, d_v)``, in place.

    ``slot``/``erase`` are ``(batch, heads)``, ``payload`` is
    ``(batch, heads, d_v)``.  Use ``torch.where`` rather than a masked in-place
    add so the non-erasing branch stays ``row + payload`` exactly.
    """
    batch, heads = slot.shape
    b = torch.arange(batch, device=state.device)[:, None]
    h = torch.arange(heads, device=state.device)[None, :]
    rows = state[b, h, slot]
    state[b, h, slot] = torch.where(erase[..., None], payload, rows + payload)
    return state


def update_liveness(live: torch.Tensor, slot: torch.Tensor, op: torch.Tensor,
                    live_after: torch.Tensor) -> torch.Tensor:
    """Apply one lane's ``live_after`` to the ``(batch, heads, rank)`` liveness."""
    batch, heads = slot.shape
    b = torch.arange(batch, device=live.device)[:, None]
    h = torch.arange(heads, device=live.device)[None, :]
    current = live[b, h, slot]
    live[b, h, slot] = torch.where(op.ne(NONE), live_after, current)
    return live


def sequential_matrix_scan(weights, decay, schedule, initial_state=None,
                           initial_live=None, return_liveness=False):
    """Per-token semantic oracle: read, then DECAY, then every lane in order.

    ``weights`` ``(batch, tokens, heads, rank)`` the read weights (``read.py``);
    ``decay``   ``(batch, tokens, heads)`` the multiplicative DECAY gate;
    ``schedule`` the fixed decision channel (see :class:`MatrixSchedule`), which
    is also the only carrier of the write payloads.

    The read at token ``t`` uses the state left by token ``t-1``: token ``t``'s
    own DECAY is part of token ``t``'s update, the same ordering the chunked scan
    implements.  Returns ``(reads, final_state)``, plus ``(final_live,
    liveness)`` when ``return_liveness=True`` (``liveness`` is the per-token
    channel, integer bookkeeping that must match the chunked scan bit-exactly).
    """
    batch, tokens, heads, rank = weights.shape
    value_dim = schedule.payload.shape[-1]
    device = weights.device
    dtype = torch.promote_types(weights.dtype, schedule.payload.dtype)
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=dtype, device=device)
             if initial_state is None else initial_state)
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
            if initial_live is None else initial_live.clone())
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=dtype, device=device)
    liveness = []
    for t in range(tokens):
        reads[:, t] = torch.einsum("bhr,bhrv->bhv", weights[:, t], state)
        state = state * decay[:, t, :, None, None]
        for lane in range(schedule.lanes):
            op = schedule.op[:, t, :, lane]
            if not bool(op.ne(NONE).any()):
                continue
            apply_lane(state, schedule.slot[:, t, :, lane], schedule.erase[:, t, :, lane],
                       schedule.payload[:, t, :, lane])
            update_liveness(live, schedule.slot[:, t, :, lane], op,
                            schedule.live_after[:, t, :, lane])
        if return_liveness:
            liveness.append(live.clone())
    if return_liveness:
        return reads, state, live, liveness
    return reads, state


def key_form_step(state, keys, slot, erase, payload):
    """Apply one op in the registration's *key* form, for the exactness tests.

    ``state`` is the ambient ``(dim, d_v)`` matrix, ``keys`` the ``(rank, dim)``
    orthonormal table, ``slot`` a python int and ``payload`` a ``(d_v,)`` vector.
    This is literally ``S += (v' - erase * S k_j) k_j^T`` with ``S k_j`` written
    as a contraction against the key table -- no coefficient-space shortcut, so a
    disagreement with :func:`apply_lane` can only come from the family failing to
    be orthonormal.
    """
    direction = keys[slot]
    stored = state.transpose(0, 1) @ direction           # (d_v,)
    residual = payload - stored if bool(erase) else payload
    return state + torch.outer(direction, residual)
