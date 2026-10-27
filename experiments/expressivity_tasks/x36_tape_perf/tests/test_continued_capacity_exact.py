"""Byte gates for the fixed-live shapes used by the capacity protocol."""
from contextlib import nullcontext
import gc
from unittest.mock import patch

import pytest
import torch

from experiments.expressivity_tasks.x30_tape_cell.naive_reference import init_state
from experiments.expressivity_tasks.x36_tape_perf.cell_fused import (
    TapeHeadConfig, original_scan, tape_scan,
)


CAPACITIES=(2944,5888,11776,8192,8224)
ENTRY_SHAPES=[(s,live) for s in CAPACITIES for live in (17,256,1024,2800)]
ENTRY_SHAPES += [(s,s-32) for s in CAPACITIES]


@pytest.mark.parametrize('slots,live',ENTRY_SHAPES)
def test_fixed_live_capacity_bytes(slots,live):
    torch.manual_seed(3606)
    cfg=TapeHeadConfig(n_heads=2,budget=slots,decay_rate=1.)
    seed=init_state(2,cfg,'cuda')
    seed.keys.normal_();seed.values.normal_()
    rows=torch.linspace(0,slots-1,live,device='cuda').long()
    seed.live[...,rows]=True
    seed.strength[...,rows]=1.
    seed.age[...,rows]=1
    data=[torch.randn((2,512,2,d),device='cuda') for d in (64,64,128)]
    data += [torch.zeros((2,512,2),device='cuda') for _ in range(3)]
    for mode in ('imported','continued'):
        results=[]
        for scan in (original_scan,tape_scan):
            xs=[x.detach().clone().requires_grad_(i in (1,2)) for i,x in enumerate(data)]
            from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeState
            state=TapeState(*(getattr(seed,n).clone() for n in
                             ('keys','values','strength','live','age','usage')))
            context=(patch('experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused.tape_scan',original_scan)
                     if scan is original_scan else nullcontext())
            with context:
                if mode=='continued':
                    with torch.no_grad():
                        _,state,_,_=scan(*(x[:,:1] for x in xs),state,cfg)
                out,final,_,_=scan(*xs,state,cfg)
            from experiments.expressivity_tasks.x36_tape_perf.tests.test_exact import oracle_rebase_grid
            with oracle_rebase_grid() if scan is original_scan else nullcontext():
                gradients=torch.autograd.grad(out.square().sum(),xs[1:3])
            if scan is tape_scan:
                pool=out.grad_fn.pool_stats
                assert pool['allocated_rows']==live
                assert (pool['union_counts']==live).all()
            results.append(tuple(x.detach().clone() for x in
                                 (out,*gradients,final.native.keys,final.native.values,
                                  final.native.strength,final.native.live)))
            if scan is original_scan:
                # The historical reference has the same raw-output owner
                # cycle. This discarded oracle node is never replayed again;
                # release it after taking independent byte snapshots so the
                # byte harness does not retain tens of GiB across test cases.
                # Production nodes receive no private reference modification.
                node=out.grad_fn
                node.storage=None
                node.address_owners=()
                node.replay=()
                final.replay_storage=None
                del node
            del out,final
            gc.collect()
        for index,(expected,actual) in enumerate(zip(*results)):
            assert torch.equal(expected.detach().contiguous().view(torch.uint8),
                               actual.detach().contiguous().view(torch.uint8)),(mode,slots,live,index)
