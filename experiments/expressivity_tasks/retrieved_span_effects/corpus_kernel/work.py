"""Device-ready integer routes. Reusable across optimizer steps; contains no learned state."""

from dataclasses import dataclass

import torch

from .plan import Plan, tiles


@dataclass(frozen=True)
class Work:
    plan: Plan
    rank: int
    local_lengths: tuple
    exact: torch.Tensor
    edge: torch.Tensor
    nodes: torch.Tensor
    local: torch.Tensor
    tile: int

    def check(self, plan, rank, local, device, tile):
        if (
            self.plan != plan
            or self.rank != rank
            or self.local_lengths != local
            or self.exact.device != device
            or tile != self.tile
        ):
            raise ValueError("stale descriptor layout")


def bind(plan, rank, local_lengths, device, *, tile=16):
    if (
        rank <= 0
        or tile <= 0
        or (
            local_lengths
            and (len(local_lengths) != plan.batch or min(local_lengths) < 0)
        )
    ):
        raise ValueError("invalid dispatch geometry")

    def make(rs):
        return torch.tensor(tiles(rs, tile), dtype=torch.int64, device=device).reshape(
            -1, 3
        )

    local = []
    start = 0
    for i, n in enumerate(local_lengths):
        local.append((i, start, start + n))
        start += n
    nodes = ((i, j * rank, (j + 1) * rank) for i, j in plan.nodes)
    return Work(
        plan,
        rank,
        tuple(local_lengths),
        make(plan.exact),
        make(plan.edge),
        make(nodes),
        make(local),
        tile,
    )
