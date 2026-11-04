"""Conditional controls: unchanged scalar/native laws and batched gradients."""
from dataclasses import replace
from types import SimpleNamespace
import os
import pytest
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_ablation_native import eager
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_control_native import NativeReplay,native_reports
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_control_model import StoreIntegrated
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_suffix import queried_suffix
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Policy
from experiments.expressivity_tasks.x43_fork_resolved.c6 import count
pytestmark=pytest.mark.skipif(os.environ.get('GPU_LEASE_HELD') not in ('2','4'),reason='requires real lease')

@pytest.mark.parametrize('append_only',[False,True])
@pytest.mark.parametrize('budget',[1,32,257])
def test_control_replay_lifecycle_and_gradients(append_only,budget):
    torch.set_num_threads(1);torch.manual_seed(3401)
    p=Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.)
    cfg=replace(p.configs()[0],budget=budget);cfg=SimpleNamespace(**vars(cfg),slots=cfg.slots,array_append_only=append_only)
    policy=p.gate_policy(0);b,t=2,12;raw=[b'abababababab',b'aaaaaaaaaaaa'];windows=[(1,0),(2,0)]
    owners=[[queried_suffix(windows[j],raw[j][:i+1],min(i+1,1+i%3)) for i in range(t)] for j in range(b)]
    metadata=(owners,raw,windows,[t,t-2],[[i%3!=0 for i in range(t)] for _ in range(b)],[[98]*t,[97]*t])
    inputs=[torch.randn(b,t,d,device='cuda') for d in (64,64,128,128)]
    inputs += [torch.full((b,t),.8,device='cuda'),torch.randn(b,t,device='cuda'),torch.full((b,t),20.,device='cuda')]
    inputs += [torch.tensor(.1,device='cuda'),torch.tensor(-.1,device='cuda'),torch.tensor(.01,device='cuda')]
    upstream=torch.randn(b,t,128,device='cuda');outputs=[];gradients=[]
    for reference in (True,False):
        xs=[x.detach().clone().requires_grad_(i in (1,2,3,4,5,6,7)) for i,x in enumerate(inputs)]
        if reference:out,reports=eager(*xs,cfg,policy,metadata)
        else:
            out,trace=NativeReplay.apply(cfg,policy,metadata,*xs)
            actual=native_reports(trace,cfg,metadata[3])
            for a,e in zip(actual,reports):
                for key in ('APPEND','POP','DECAY','CORRECT','MERGE','stale_retirements','natural_retirements','capacity_merges','duplicate_merges','array_requests','array_read_opportunities','array_slots_read','peak_occupancy','peak_array','peak_stream','final_array','final_stream'):
                    assert a[key]==e[key],(key,a[key],e[key])
        out.backward(upstream);outputs.append(out);gradients.append([x.grad for x in xs])
    torch.testing.assert_close(*outputs,atol=2e-5,rtol=2e-5)
    for a,e in zip(*gradients):
        if a is None or e is None:assert a is None and e is None
        else:torch.testing.assert_close(a,e,atol=2e-5,rtol=2e-5)

@pytest.mark.parametrize('control',['append_only','unbudgeted'])
def test_control_registered_shape_finite_and_checkpoint(control):
    torch.set_num_threads(1);torch.manual_seed(3401)
    model=StoreIntegrated(control).cuda().train();assert count(model)==2133700
    ids=torch.randint(0,256,(16,128),device='cuda')
    answers=torch.zeros(16,128,260,device='cuda');answers[...,97]=1;answers[...,256]=1
    lengths=torch.ones(16,128,device='cuda',dtype=torch.long)
    previous=None;grads=None
    for checkpointed in (False,True):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lp,_=model(ids,answers,lengths,checkpointed=checkpointed);loss=-lp[...,97].mean()
        loss.backward();current={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        assert model.device_input.weight.grad.abs().sum()>0
        assert all(torch.isfinite(g).all() for g in current.values())
        if previous is not None:
            assert torch.equal(lp,previous) and current.keys()==grads.keys()
            for n in current:torch.testing.assert_close(current[n],grads[n],atol=0,rtol=0)
        previous=lp;grads=current
        if control=='unbudgeted':
            assert all(r['budget']==257 and r['peak_occupancy']<257 and r['capacity_merges']==0 for layer in model.telemetry for r in layer)
    off,_=model(ids,answers,lengths,enabled=False)
    assert torch.equal(off,model.head(ids,torch.zeros_like(answers),False))
