"""Exact rank allocation subproblem, not a complete parallel tape cell."""
import torch


def allocate(free, append):
    """1-D bool masks -> int64 destinations; reject capacity overflow.

    No frees may occur between tokens. Slot order is logical ascending order.
    This diagnostic implementation synchronizes for the overflow check.
    """
    if free.ndim != 1 or append.ndim != 1 or free.dtype != torch.bool or append.dtype != torch.bool:
        raise ValueError('expected one-dimensional boolean masks')
    if free.device != append.device:
        raise ValueError('masks must share a device')
    ranks = append.to(torch.int64).cumsum(0) - 1
    slots = free.nonzero().flatten()
    if int(append.sum()) > slots.numel():
        raise ValueError('insufficient free slots')
    out = torch.full(append.shape, -1, dtype=torch.int64, device=append.device)
    out[append] = slots[ranks[append]]
    return out
