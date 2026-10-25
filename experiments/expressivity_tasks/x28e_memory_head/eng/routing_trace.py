"""Per-token per-head discrete routing trace for the X28E slot-store tape head.

Binding spec: ``experiments/expressivity_tasks/x28e_memory_head/REGISTRATION.md``
(replay-decision consistency gate, P1-6 engineering-freeze obligation):

    "the forward pass saves a discrete ROUTING TRACE (per token, per head:
    which of R-EMPTY/R-NOMATCH/R-ZERO-RESIDUAL/R-CONFLICT/R-CAP fired and the
    matched slot index, or NO-WRITE, plus the POP decision and its target) in
    the checkpoint payload; replay follows the trace — routing decisions are
    replay INPUTS, never recomputed from rounded state."

and (routing rule section): "The trace records POP and WRITE target slot
indices SEPARATELY when both occur (their targets need not coincide)."

All trace tensors are INTEGRAL (bool / int64) and never gradient-bearing:
routing is architectural, always (REGISTRATION: "routing is architectural,
always"). The zero-gradient-through-routing test asserts integrality and
``requires_grad=False`` for every field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

# ---------------------------------------------------------------------------
# Routing case codes (architectural; integral by construction).
# ---------------------------------------------------------------------------
NO_WRITE = 0          # W-OFF token: no write at all (traced as NO-WRITE)
R_EMPTY = 1           # no live slots                      -> APPEND
R_NOMATCH = 2         # best match score < match_threshold -> APPEND
R_ZERO_RESIDUAL = 3   # match && residual < eps_residual   -> APPEND
R_CONFLICT = 4        # match && residual >= eps_residual  -> DELTA-RULE CORRECTION
R_CAP = 5             # APPEND requested, budget full      -> MERGE (final override)

# Free-by-strength-death trace codes (G10: free-by-decay decisions join the
# routing trace; the replay-consistency gate covers them). A slot freed at the
# post-POP death check carries code 1; a slot freed at the end-of-token decay
# death check carries code 2. Codes are disjoint (a slot freed at the pop
# check is dead before the decay check). Replay applies the TRACED frees
# instead of recomputing the strength comparison from (rounded) state.
FREED_NONE = 0
FREED_POP = 1
FREED_DECAY = 2

CASE_NAMES: Dict[int, str] = {
    NO_WRITE: "NO-WRITE",
    R_EMPTY: "R-EMPTY",
    R_NOMATCH: "R-NOMATCH",
    R_ZERO_RESIDUAL: "R-ZERO-RESIDUAL",
    R_CONFLICT: "R-CONFLICT",
    R_CAP: "R-CAP",
}

ALL_CASES = (NO_WRITE, R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL, R_CONFLICT, R_CAP)

# Sentinel slot index: "no target" (no live slot / decision did not fire).
NO_SLOT = -1


@dataclass
class RoutingTrace:
    """Per-token per-head architectural routing decisions.

    All tensor fields are shaped ``[B, T, NH]``. ``no_write`` (== ``~w_on``)
    is provided as a property for receipt readability.

    Fields:
        case:        which routing case fired (codes above; NO-WRITE for W-OFF).
        matched:     write-path best-match slot index at decision time
                     (NO_SLOT when no live slot exists). This is the correction
                     target for R-CONFLICT / R-CAP and the informational match
                     for the append cases; replay uses it only for corrections.
        pop_fired:   whether the read-path POP fired (pop strength exceeded
                     ``pop_threshold`` and a live slot existed).
        pop_target:  the POP (read-path, query-argmax) slot index, recorded
                     SEPARATELY from the write target (NO_SLOT when no pop).
        alloc_target: slot index allocated by an APPEND (NO_SLOT when the
                     append degraded to R-CAP merge or no append fired).
        w_on:        W-ON / W-OFF write-enable decision (W-OFF == NO-WRITE).
        freed:       int8 [B,T,NH,S] strength-death trace (G10): per token,
                     which slots were freed — FREED_POP (1) at the post-pop
                     death check, FREED_DECAY (2) at the end-of-token decay
                     death check, 0 otherwise. Replay follows the trace:
                     liveness decisions are replay INPUTS, never recomputed
                     from (bf16-rounded) state. None = not collected (the
                     pre-G10 trace shape, kept for provenance only).
    """

    case: torch.Tensor          # int64 [B,T,NH]
    matched: torch.Tensor       # int64 [B,T,NH]
    pop_fired: torch.Tensor     # bool  [B,T,NH]
    pop_target: torch.Tensor   # int64 [B,T,NH]
    alloc_target: torch.Tensor  # int64 [B,T,NH]
    w_on: torch.Tensor         # bool  [B,T,NH]
    freed: Optional[torch.Tensor] = None  # int8 [B,T,NH,S] (G10; codes above)

    @property
    def no_write(self) -> torch.Tensor:
        return ~self.w_on

    @property
    def batch_size(self) -> int:
        return self.case.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.case.shape[1]

    @property
    def num_heads(self) -> int:
        return self.case.shape[2]

    def slice_tokens(self, t0: int, t1: int) -> "RoutingTrace":
        """View of tokens ``[t0, t1)`` (segment-aligned replay input)."""
        return RoutingTrace(
            case=self.case[:, t0:t1, :],
            matched=self.matched[:, t0:t1, :],
            pop_fired=self.pop_fired[:, t0:t1, :],
            pop_target=self.pop_target[:, t0:t1, :],
            alloc_target=self.alloc_target[:, t0:t1, :],
            w_on=self.w_on[:, t0:t1, :],
            freed=(self.freed[:, t0:t1, :] if self.freed is not None else None),
        )

    @staticmethod
    def cat(traces: List["RoutingTrace"]) -> "RoutingTrace":
        """Concatenate segment traces along the token axis."""
        assert len(traces) > 0
        have_freed = [t.freed is not None for t in traces]
        if any(have_freed):
            assert all(have_freed), "RoutingTrace.cat: mixed freed presence"
        return RoutingTrace(
            case=torch.cat([t.case for t in traces], dim=1),
            matched=torch.cat([t.matched for t in traces], dim=1),
            pop_fired=torch.cat([t.pop_fired for t in traces], dim=1),
            pop_target=torch.cat([t.pop_target for t in traces], dim=1),
            alloc_target=torch.cat([t.alloc_target for t in traces], dim=1),
            w_on=torch.cat([t.w_on for t in traces], dim=1),
            freed=(torch.cat([t.freed for t in traces], dim=1)
                   if all(have_freed) else None),
        )

    def assert_integral(self) -> None:
        """Routing is architectural: every field is integral and grad-free.

        Raises AssertionError otherwise. This is the structural half of the
        zero-gradient-through-routing gate.
        """
        for name, tensor in self._fields():
            assert not tensor.requires_grad, f"trace field {name} requires grad"
            if tensor.dtype == torch.bool:
                continue
            assert tensor.dtype in (torch.int64, torch.int32, torch.int16,
                                    torch.int8, torch.uint8), (
                f"trace field {name} has non-integral dtype {tensor.dtype}")
        if self.freed is not None:
            assert not self.freed.requires_grad, "trace field freed requires grad"
            assert self.freed.dtype in (torch.int8, torch.uint8), (
                f"trace field freed has non-integral dtype {self.freed.dtype}")

    def _fields(self):
        return [
            ("case", self.case),
            ("matched", self.matched),
            ("pop_fired", self.pop_fired),
            ("pop_target", self.pop_target),
            ("alloc_target", self.alloc_target),
            ("w_on", self.w_on),
        ]

    def counts(self) -> Dict[str, int]:
        """Per-case firing counts across all (batch, token, head) — receipts."""
        out = {CASE_NAMES[c]: 0 for c in ALL_CASES}
        for c in ALL_CASES:
            out[CASE_NAMES[c]] = int((self.case == c).sum())
        return out

    def summarize(self) -> str:
        c = self.counts()
        total = self.case.numel()
        lines = [f"RoutingTrace: B={self.batch_size} T={self.num_tokens} "
                 f"NH={self.num_heads} ({total} decisions)"]
        lines += [f"  {name:<17s} {count}" for name, count in c.items()]
        lines.append(f"  POP fired        {int(self.pop_fired.sum())}")
        lines.append(f"  W-ON tokens       {int(self.w_on.sum())}")
        return "\n".join(lines)


def assert_traces_equal(a: RoutingTrace, b: RoutingTrace, what: str = "") -> None:
    """Assert two traces are decision-identical (the P1-6 consistency gate,
    extended to free-by-decay per G10: the ``freed`` strength-death codes are
    compared whenever either trace carries them)."""
    fields = [(name, ta, tb) for name, ta, tb in zip(
        [f[0] for f in a._fields()], [f[1] for f in a._fields()],
        [f[1] for f in b._fields()])]
    if (a.freed is None) != (b.freed is None):
        raise AssertionError(
            f"{what}: trace field freed presence mismatch "
            f"({a.freed is None} vs {b.freed is None})")
    if a.freed is not None:
        fields.append(("freed", a.freed, b.freed))
    for name, ta, tb in fields:
        assert ta.shape == tb.shape, f"{what}: trace field {name} shape mismatch"
        assert torch.equal(ta, tb), (
            f"{what}: trace field {name} differs between forward and replay "
            f"at {(ta != tb).nonzero()[:10].tolist()}")
