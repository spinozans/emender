"""Probe-only launch-axis repair for X31 journals exceeding CUDA grid.y.

Identical address mapping operations; store axis moves to y, log blocks to x.
No production source or kernel is changed.
"""
import triton
import triton.language as tl


@triton.jit
def rebase_transposed(Addresses, Counts, Mapping, Fields:tl.constexpr,
           Capacity, Block:tl.constexpr):
    n=tl.program_id(1).to(tl.int64)
    x=tl.program_id(0)*Block+tl.arange(0,Block)
    active=x<tl.load(Counts+n)
    address=tl.load(Addresses+n*Capacity+x,active,0).to(tl.int64)
    lo=tl.full((Block,),0,tl.int32)
    hi=tl.where(active,Fields,0)
    while tl.sum((lo<hi).to(tl.int32),0)>0:
        searching=lo<hi
        mid=(lo+hi)//2
        base=tl.load(Mapping+3*mid,searching,0)
        right=address>=base
        lo=tl.where(searching & right,mid+1,lo)
        hi=tl.where(searching & ~right,mid,hi)
    slot=lo-1
    valid=active & (slot>=0)
    old=tl.load(Mapping+3*slot,valid,0)
    new=tl.load(Mapping+3*slot+1,valid,0)
    size=tl.load(Mapping+3*slot+2,valid,0)
    matched=valid & (address>=old) & (address<old+size)
    tl.device_assert((~active)|matched,'undo address has no owning field')
    tl.store(Addresses+n*Capacity+x,address-old+new,matched)



class TransposedLaunch:
    def __getitem__(self,grid):
        return rebase_transposed[(grid[1],grid[0])]


def validate():
    import torch
    from ..x31_tape_cell_fused.backward import rebase
    old=torch.zeros(2048,device='cuda');new=torch.zeros_like(old)
    mapping=torch.tensor([[old.data_ptr(),new.data_ptr(),old.numel()*old.element_size()]],device='cuda',dtype=torch.int64)
    count=torch.tensor([73,199],device='cuda',dtype=torch.int32)
    addresses=old.data_ptr()+torch.arange(512,device='cuda',dtype=torch.int64).reshape(2,256)*4
    a=addresses.clone();b=addresses.clone()
    rebase[(2,1)](a,count,mapping,1,256,256,debug=True)
    rebase_transposed[(1,2)](b,count,mapping,1,256,256,debug=True)
    assert torch.equal(a,b),'transposed mapping differs from original'
    # Exercise the formerly invalid grid.y size with masked inactive blocks.
    large_capacity=65536*256+1
    c=addresses[0,:73].clone();counts=count[:1]
    rebase_transposed[(triton.cdiv(large_capacity,256),1)](c,counts,mapping,1,large_capacity,256,debug=True)
    assert torch.equal(c,a[0,:73])
    return dict(status='passed',small_mapping_bitwise=True,large_log_blocks=triton.cdiv(large_capacity,256))
