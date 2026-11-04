"""Real leased-GPU native gates. Never silently substitute CPU/mock scans."""
import os
from dataclasses import replace
import pytest
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_native import native_forward, eager, NativeReplay, native_reports
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_suffix import queried_suffix
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Policy

pytestmark=pytest.mark.skipif(os.environ.get('GPU_LEASE_HELD') not in ('2','3','4','5','6'),reason='requires real GPU lease')


@pytest.mark.parametrize('budget',[1,32,64])
def test_native_lifecycle_read_and_replay_gradients(budget):
    torch.manual_seed(3401)
    policy=Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.).gate_policy(0)
    cfg=replace(Policy().configs()[0],budget=budget)
    b,t=2,12
    raw=[b'abababababab',b'aaaaaaaaaaaa']
    windows=[(1,0),(2,0)]
    owners=[[queried_suffix(windows[j],raw[j][:i+1],1) for i in range(t)] for j in range(b)]
    metadata=(owners,raw,windows,[t,t-2],[[True]*t for _ in range(b)],[[98]*t,[97]*t])
    inputs=[torch.randn(b,t,d,device='cuda') for d in (64,64,128,128)]
    inputs += [torch.full((b,t),.2,device='cuda'),torch.zeros(b,t,device='cuda'),torch.full((b,t),20.,device='cuda')]
    inputs += [torch.tensor(0.,device='cuda'),torch.tensor(0.,device='cuda'),torch.tensor(0.,device='cuda')]
    native,(state,trace)=native_forward(inputs,cfg,policy,metadata)
    reference,reports=eager(*inputs,cfg,policy,metadata)
    torch.testing.assert_close(native,reference,atol=2e-5,rtol=2e-5)
    native_stats=native_reports(trace,cfg,metadata[3])
    for j,(actual,expected) in enumerate(zip(native_stats,reports)):
        for name in ('APPEND','POP','DECAY','CORRECT','MERGE','stale_retirements','natural_retirements',
                     'capacity_merges','duplicate_merges','array_requests','array_read_opportunities','array_slots_read',
                     'peak_occupancy','peak_array','peak_stream','final_array','final_stream'):
            assert actual[name]==expected[name],(name,actual[name],expected[name])
        assert int(state[3][j,0].sum())==actual['final_array']+actual['final_stream']
    grad_inputs=[x.detach().requires_grad_(i in (1,2,3,4,5,6,7)) for i,x in enumerate(inputs)]
    out,_=NativeReplay.apply(cfg,policy,metadata,*grad_inputs)
    (out.square().mean()).backward()
    reference_inputs=[x.detach().requires_grad_(x.requires_grad) for x in grad_inputs]
    ref,_=eager(*reference_inputs,cfg,policy,metadata)
    ref.square().mean().backward()
    for a,b in zip(grad_inputs,reference_inputs):
        if a.grad is not None:
            assert torch.isfinite(a.grad).all()
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)


def test_model_disabled_exact_and_enabled_checkpoint_gradients():
    from experiments.expressivity_tasks.x43_fork_resolved.test_c14_b_enabled import fixture
    model,ids,answers,lengths=fixture()
    model=model.cuda();ids=ids.cuda();answers=answers.cuda();lengths=lengths.cuda()
    grads=None;previous=None
    for checkpointed in (False,True):
        model.zero_grad(set_to_none=True)
        lp,_=model(ids,answers,lengths,checkpointed=checkpointed)
        (-lp[...,99].mean()).backward()
        current={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(g).all() for g in current.values())
        assert current['device_input.weight'].abs().sum()>0
        if grads is not None:
            assert torch.equal(lp,previous)
            for n in current:torch.testing.assert_close(current[n],grads[n],atol=0,rtol=0)
        grads=current;previous=lp
    model.eval()
    disabled,_=model(ids,answers,lengths,enabled=False)
    dense=model.head(ids,torch.zeros_like(answers),False)
    assert torch.equal(disabled,dense)


def test_registered_training_shape_bf16_backward():
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_model import StoreIntegrated
    torch.set_num_threads(1);torch.manual_seed(3401)
    model=StoreIntegrated().cuda().train()
    ids=torch.randint(0,256,(16,128),device='cuda')
    answers=torch.zeros(16,128,260,device='cuda');answers[...,97]=1.;answers[...,256]=1.
    lengths=torch.ones(16,128,device='cuda',dtype=torch.long)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        lp,_=model(ids,answers,lengths)
        loss=-lp[...,97].mean()
    loss.backward()
    assert torch.isfinite(lp).all()
    assert model.device_input.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_no_array_native_stream_law_matches_standing_scan():
    from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import scan
    torch.manual_seed(3401)
    p=Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.)
    cfg=p.configs()[0];policy=p.gate_policy(0)
    b,t=2,16
    inputs=[torch.randn(b,t,d,device='cuda') for d in (64,64,128,128)]
    inputs += [torch.full((b,t),.2,device='cuda'),torch.zeros(b,t,device='cuda'),torch.full((b,t),2.,device='cuda')]
    inputs += [torch.tensor(0.,device='cuda'),torch.tensor(-1.,device='cuda'),torch.tensor(0.,device='cuda')]
    raw=[b'a'*t,b'b'*t];windows=[(0,0),(1,0)]
    owners=[[queried_suffix(windows[j],raw[j][:i+1],1) for i in range(t)] for j in range(b)]
    metadata=(owners,raw,windows,[t,t],[[False]*t for _ in range(b)],[[97]*t for _ in range(b)])
    with torch.no_grad():
        actual,_=native_forward(inputs,cfg,policy,metadata)
        k,q,v,av,pop,append,logit,coef,mc,ec=inputs
        result,_,_=scan(k.unsqueeze(2),q.unsqueeze(2),v.unsqueeze(2),torch.ones(b,t,1,device='cuda'),
            pop.unsqueeze(2),append.unsqueeze(2),cfg,coef[None],logit.unsqueeze(2),mc[None],ec[None],policy)
    torch.testing.assert_close(actual,result[0][:,:,0],atol=2e-5,rtol=2e-5)
