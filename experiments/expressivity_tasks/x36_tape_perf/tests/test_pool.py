import gc

import pytest
import torch

from experiments.expressivity_tasks.x36_tape_perf.adjoint_pool import build
from experiments.expressivity_tasks.x36_tape_perf.cell_fused import tape_scan, TapeHeadConfig
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import init_state


@pytest.mark.parametrize('slots,live',[(37,0),(11776,1024),(11776,11744)])
def test_pool_reservation_is_union_cardinality(slots,live):
    entry=torch.zeros((3,slots),device='cuda',dtype=torch.bool)
    entry[0,-live:]=live>0
    entry[1,::7]=True
    allocations=torch.tensor([[0,0,-1],[slots-1,-1,slots-1],[-1,-1,-1]],device='cuda')
    logical,inverse,counts=build(entry,allocations)
    expected=entry.cpu()
    for n,events in enumerate(allocations.cpu().tolist()):
        for row in events:
            if row>=0:
                expected[n,row]=True
    assert logical.shape[-1]==max(1,int(expected.sum(1).max()))
    assert torch.equal(counts.cpu(),expected.sum(1))
    for n in range(3):
        rows=expected[n].nonzero().flatten()
        assert torch.equal(logical[n,:len(rows)].cpu().long(),rows)
        assert (logical[n,len(rows):]==-1).all()
        assert (inverse[n,~expected[n].to(entry.device)]==-1).all()
        assert torch.equal(inverse[n,rows.to('cuda')].cpu().long(),torch.arange(len(rows)))


def test_discarded_replay_releases_allocator_storage():
    cfg=TapeHeadConfig(n_heads=2,budget=2944,decay_rate=1.)
    state=init_state(2,cfg,'cuda')
    state.live[...,::101]=True
    state.strength[...,::101]=1.
    xs=[torch.randn((2,9,2,d),device='cuda',requires_grad=True) for d in (64,64,128)]
    xs += [torch.zeros((2,9,2),device='cuda') for _ in range(3)]
    def sample():
        out,final,_,_=tape_scan(*xs,state,cfg)
        out.sum().backward()
        # No access to the private autograd node and no explicit owner release.
    allocated=[]
    for _ in range(7):
        for x in xs:
            x.grad=None
        sample()
        gc.collect()
        torch.cuda.synchronize()
        allocated.append(torch.cuda.memory_allocated())
    print('lifetime_allocated_bytes:',allocated)
    assert max(allocated[2:])-min(allocated[2:])<1024*1024,allocated
