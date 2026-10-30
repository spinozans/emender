"""Copy counted journal prefixes without reading undefined capacity tails."""
import torch
import triton
import triton.language as tl

@triton.jit
def copy_counted(Source, Destination, Counts, Capacity: tl.constexpr, Block: tl.constexpr):
    n=tl.program_id(1).to(tl.int64)
    x=tl.program_id(0)*Block+tl.arange(0,Block)
    active=(x<Capacity) & (x<tl.load(Counts+n))
    value=tl.load(Source+n*Capacity+x,active,0)
    tl.store(Destination+n*Capacity+x,value,x<Capacity)


def counted_clone(source, counts):
    destination=torch.empty_like(source)
    copy_counted[(triton.cdiv(source.shape[1],256),len(counts))](
        source,destination,counts,source.shape[1],256)
    return destination

