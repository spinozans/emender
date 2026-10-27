from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from experiments.expressivity_tasks.x30_tape_cell.naive_reference import init_state
from experiments.expressivity_tasks.x36_tape_perf.cell_fused import TapeHeadConfig, original_scan, tape_scan


@pytest.mark.parametrize('continued',[False,True])
@pytest.mark.parametrize('zero_read',[True,False])
def test_imported_negative_zero_cold_strength_bytes(continued,zero_read):
    cfg=TapeHeadConfig(n_heads=1,budget=8192,decay_rate=1.)
    results=[]
    for scan in (original_scan,tape_scan):
        torch.manual_seed(36060)
        state=init_state(1,cfg,'cuda')
        state.keys.fill_(-1.)
        if not zero_read:
            state.keys.normal_();state.values.normal_()
        state.strength.fill_(-0.)
        rows=torch.linspace(0,8191,17,device='cuda').long()
        state.live[...,rows]=True
        state.strength[...,rows]=1.
        xs=[torch.ones((1,9,1,d),device='cuda',requires_grad=i in (1,2))
            for i,d in enumerate((64,64,128))]
        xs += [torch.zeros((1,9,1),device='cuda') for _ in range(3)]
        context=(patch('experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused.tape_scan',original_scan)
                 if scan is original_scan else nullcontext())
        with context:
            if continued:
                with torch.no_grad():
                    _,state,_,_=scan(*(x[:,:1] for x in xs),state,cfg)
            out,final,_,_=scan(*xs,state,cfg)
        gradient=torch.autograd.grad(out.sum(),xs[1])[0]
        results.append(gradient.detach().clone())
        if scan is original_scan:
            node=out.grad_fn
            node.storage=None;node.address_owners=();node.replay=()
            final.replay_storage=None
    assert torch.equal(results[0].view(torch.uint8),results[1].view(torch.uint8))
