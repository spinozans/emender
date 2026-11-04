"""Conditional control laws, native/checkpoint backward, finite cap evidence."""
import os
import pytest
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_ablation_model import StoreIntegrated
from experiments.expressivity_tasks.x43_fork_resolved.c6 import count

pytestmark=pytest.mark.skipif(os.environ.get('GPU_LEASE_HELD') not in ('2','4'),reason='real lease required')


@pytest.mark.parametrize('control',['append_only','unbudgeted'])
def test_conditional_control_backward_and_capacity(control):
    torch.set_num_threads(1);torch.manual_seed(3401)
    model=StoreIntegrated(control).cuda().train()
    assert count(model)==2133700
    ids=torch.full((2,128),97,device='cuda',dtype=torch.long)
    answers=torch.zeros(2,128,260,device='cuda');answers[...,98]=1;answers[...,256]=1
    lengths=torch.ones(2,128,device='cuda',dtype=torch.long)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        lp,_=model(ids,answers,lengths)
        loss=-lp[...,99].mean()
    loss.backward()
    assert model.device_input.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for layer in model.telemetry:
        for row in layer:
            if control=='append_only':assert row['stale_retirements']==0
            else:
                assert row['budget']==257 and row['peak_occupancy']<257
                assert row['capacity_merges']==0
