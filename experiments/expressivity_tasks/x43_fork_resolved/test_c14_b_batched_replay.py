"""Independent scalar oracle versus batched orchestration, including gradients."""
from dataclasses import replace
import pytest
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_native import eager
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_batched_replay import replay
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_suffix import queried_suffix
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Policy

@pytest.mark.parametrize('budget',[1,2,32,64])
@pytest.mark.parametrize('write',[20.,-20.])
def test_batched_replay_values_and_gradients(budget,write):
    torch.set_num_threads(1);torch.manual_seed(3401)
    p=Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.)
    cfg=replace(p.configs()[0],budget=budget);policy=p.gate_policy(0)
    b,t=3,16
    raw=[b'abababababababab',b'aaaaaaaaaaaaaaaa',bytes(range(t))]
    windows=[(1,0),(2,0),(3,0)]
    owners=[[queried_suffix(windows[j],raw[j][:i+1],min(i+1,1+(i%4))) for i in range(t)] for j in range(b)]
    metadata=(owners,raw,windows,[t,t-2,0],[[i%3!=0 for i in range(t)] for _ in range(b)],[[98]*t,[97]*t,[0]*t])
    inputs=[torch.randn(b,t,d) for d in (64,64,128,128)]
    inputs += [torch.full((b,t),.8),torch.randn(b,t),torch.full((b,t),write)]
    inputs += [torch.tensor(.1),torch.tensor(-.1),torch.tensor(.01)]
    results=[];grads=[]
    for implementation in (eager,replay):
        xs=[x.clone().requires_grad_(i in (1,2,3,4,5,6,7)) for i,x in enumerate(inputs)]
        out,dependencies=implementation(*xs,cfg,policy,metadata)
        out.square().mean().backward()
        results.append(out)
        grads.append([x.grad if implementation is eager or dependencies & (1<<i) else None for i,x in enumerate(xs)])
    torch.testing.assert_close(*results,atol=2e-5,rtol=2e-5)
    for a,b in zip(*grads):
        if a is None or b is None:
            assert a is None and b is None
        else:torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

@pytest.mark.skipif(__import__('os').environ.get('GPU_LEASE_HELD') not in ('2','4'),reason='requires real GPU lease')
def test_graph_reuse_updates_values_ownership_padding_and_upstream_gradient():
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_native import NativeReplay
    torch.manual_seed(3401)
    p=Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.)
    cfg=replace(p.configs()[0],budget=2);policy=p.gate_policy(0)
    b,t=2,12;windows=[(101,0),(102,128)]
    for attempt in range(3):
        raw=[b'abababababab',b'aaaaaaaaaaaa'] if attempt==0 else [bytes(range(t)),b'bbbbbbbbbbbb']
        owners=[[queried_suffix(windows[j],raw[j][:i+1],1+(i%2)) if i else queried_suffix(windows[j],raw[j][:1],1) for i in range(t)] for j in range(b)]
        metadata=(owners,raw,windows,[t,t-attempt],[[attempt!=1]*t for _ in range(b)],[[97+attempt]*t for _ in range(b)])
        inputs=[torch.randn(b,t,d,device='cuda') for d in (64,64,128,128)]
        inputs += [torch.full((b,t),.2 if attempt==0 else .8,device='cuda'),torch.randn(b,t,device='cuda'),torch.full((b,t),20.,device='cuda')]
        inputs += [torch.tensor(.1,device='cuda'),torch.tensor(-.1,device='cuda'),torch.tensor(.01,device='cuda')]
        upstream=torch.randn(b,t,128,device='cuda');outputs=[];gradients=[]
        for native in (False,True):
            xs=[x.detach().clone().requires_grad_(i in (1,2,3,4,5,6,7)) for i,x in enumerate(inputs)]
            out=NativeReplay.apply(cfg,policy,metadata,*xs)[0] if native else eager(*xs,cfg,policy,metadata)[0]
            out.backward(upstream);outputs.append(out);gradients.append([x.grad for x in xs])
        torch.testing.assert_close(*outputs,atol=2e-5,rtol=2e-5)
        for expected,actual in zip(*gradients):
            if expected is None or actual is None:assert expected is None and actual is None
            else:torch.testing.assert_close(expected,actual,atol=2e-5,rtol=2e-5)
