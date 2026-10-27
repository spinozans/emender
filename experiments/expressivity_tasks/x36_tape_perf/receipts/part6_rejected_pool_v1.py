"""Stable logical ordering for imported replay row adjoints.

The allocation is the measured maximum union cardinality across stores, never
the padded Triton lane count. Owners and journal addresses retain native rows.
"""
import torch


def build(entry_live, allocations):
    stores, slots = entry_live.shape
    union = entry_live.clone()
    allocation = allocations.reshape(stores, -1)
    # A sentinel column absorbs no-allocation events without clearing row zero.
    extended = torch.cat((union, union.new_zeros((stores, 1))), dim=1)
    extended.scatter_(1, torch.where(allocation >= 0, allocation, slots), True)
    union = extended[:, :slots]
    counts = union.sum(1)
    rows = max(1, int(counts.max().item()))
    inverse = union.cumsum(1, dtype=torch.int32) - 1
    inverse.masked_fill_(~union, -1)
    logical = torch.full((stores, rows), -1, device=union.device, dtype=torch.int32)
    store, slot = union.nonzero(as_tuple=True)
    logical[store, inverse[store, slot].long()] = slot.to(torch.int32)
    return logical, inverse, counts
