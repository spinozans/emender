"""Optional occurrence-specific posterior inspection for one fixed-weight read.

Mass is inherited from the ORIGINAL jointly normalized endpoint distribution. A
continuation narrows that mass; it isn't a new row or an observation in the LM.
Only the requested fork is inspected, one archive token per surviving occurrence.
"""

from dataclasses import dataclass
import math

import torch

from .index import Trail


@dataclass(frozen=True)
class Continuation:
    index: object
    trail: Trail
    log_weights: torch.Tensor  # original per-endpoint log probabilities, in trail order

    @property
    def log_mass(self):
        return self.log_weights.logsumexp(0).item()

    @property
    def mass(self):
        return math.exp(self.log_mass)

    def branch(self):
        paths, ends = self.index.branch(self.trail)
        where = {r: i for i, r in enumerate(self.trail.rows)}

        def weights(rows):
            return self.log_weights[[where[r] for r in rows]]

        kids = {
            t: Continuation(self.index, path, weights(path.rows))
            for t, path in paths.items()
        }
        end_log = weights(ends).logsumexp(0).item()
        total = self.log_mass
        # Unconditional masses use the full corpus/local/null partition; conditional
        # branches include END. Empty parent returns no branches and zero END mass.
        conditional = {t: math.exp(child.log_mass - total) for t, child in kids.items()}
        end_cond = math.exp(end_log - total) if ends else 0.0
        return Fork(kids, conditional, math.exp(end_log), end_cond)


@dataclass(frozen=True)
class Fork:
    children: dict
    conditional: dict
    end_mass: float
    end_conditional: float


def refined_cursor(index, plan, q, bank, answer):
    if (
        bank.layout.version != index.version
        or plan.layout != bank.layout
        or plan.batch != 1
    ):
        raise ValueError("foreign or batched cursor plan")
    spans = [range(a, b) for _, a, b in plan.exact]
    rows = spans[0] if len(spans) == 1 else tuple(i for sr in spans for i in sr)
    trail = Trail(index.version, rows)
    # Only the actual exact-scored endpoints are enumerated. Unrefined background
    # stays in the partition even if no explicit occurrence handle is requested.
    chunks = [
        bank.keys[s.start : s.stop] @ q * (q.numel() ** -0.5)
        + bank.bias[s.start : s.stop]
        for s in spans
    ]
    s = torch.cat(chunks) if chunks else q.new_empty(0)
    return Continuation(
        index, trail, (s.double() - answer.log_partition[0].double()).cpu()
    )
