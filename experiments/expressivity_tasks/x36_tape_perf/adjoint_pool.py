"""Stable logical ordering for imported replay row adjoints.

The allocation is the measured maximum union cardinality across stores, never
the padded Triton lane count. Owners and journal addresses retain native rows.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def index_rows(Union, Inverse, Logical, S:tl.constexpr, PS:tl.constexpr, BS:tl.constexpr):
    n=tl.program_id(0)
    row=tl.arange(0,BS)
    active=tl.load(Union+n*S+row,row<S,False)
    rank=tl.cumsum(active.to(tl.int32),0)-1
    tl.device_assert((~active)|(rank<PS),'adjoint union exceeds reservation')
    tl.store(Inverse+n*S+row,tl.where(active,rank,-1),row<S)
    tl.store(Logical+n*PS+rank,row,active)


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
    union = union.contiguous()
    inverse = torch.empty_like(union,dtype=torch.int32)
    logical = torch.full((stores, rows), -1, device=union.device, dtype=torch.int32)
    index_rows[(stores,)](union,inverse,logical,slots,rows,triton.next_power_of_2(slots),
                         num_warps=16 if slots>=4096 else 4,debug=True)
    return logical, inverse, counts
