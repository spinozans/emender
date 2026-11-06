"""Raw empty-read POP counters and masked physical tail remain byte-identical."""
import os
from types import SimpleNamespace
import pytest
import torch
from experiments.expressivity_tasks.x43_fork_resolved import c14_b_control_native as native
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_ablation_native import native_forward as reference
from experiments.expressivity_tasks.x43_fork_resolved.c14_launch_shape import install
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_step_composition import bits
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Policy

pytestmark=pytest.mark.skipif(os.environ.get('GPU_LEASE_HELD') not in ('3','5','6'),reason='requires approved real GPU lease')

@pytest.mark.parametrize('t',[12,128])
@pytest.mark.parametrize('append_only',[False,True])
def test_empty_pop_raw_trace_and_unwritten_tail(t,append_only):
    torch.set_num_threads(1);torch.manual_seed(314)
    policy=Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.)
    fields=dict(vars(policy.configs()[0]));fields.update(budget=257,slots=257,array_append_only=append_only)
    cfg=SimpleNamespace(**fields);b=2
    xs=[torch.randn(b,t,d,device='cuda') for d in (64,64,128,128)]
    xs += [torch.ones(b,t,device='cuda'),torch.zeros(b,t,device='cuda'),torch.full((b,t),-20.,device='cuda')]
    xs += [torch.zeros((),device='cuda') for _ in range(3)]
    metadata=([[None]*t for _ in range(b)],[b'a'*t]*b,[(j,0) for j in range(b)],[t,t-2],[[False]*t for _ in range(b)],[[97]*t for _ in range(b)])
    old=native.native_forward
    try:
        if old.__module__ == native.__name__:
            install()
        expected=reference(xs,cfg,policy.gate_policy(0),metadata)
        actual=native.native_forward(xs,cfg,policy.gate_policy(0),metadata)
        assert bits(actual)==bits(expected)
        assert int(actual[1][1][:,:,7].sum())==2*t-2
    finally:
        native.native_forward=old
