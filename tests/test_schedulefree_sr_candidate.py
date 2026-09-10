from copy import deepcopy
import io
import pytest
import torch
import schedulefree
from ndm.schedulefree_sr_candidate import ScheduleFreeSRCandidate, counter_round


def make(bucket=257,seed=9):
    p=torch.nn.Parameter(torch.linspace(-0.02,0.02,4096).bfloat16())
    q=torch.nn.Parameter(torch.ones(32,dtype=torch.bfloat16))
    opt=ScheduleFreeSRCandidate([('p',p),('q',q)],lr=2e-6,betas=(0.9,0.95),warmup_steps=2,
                                 seed=seed,bucket_numel=bucket,pin_memory=False)
    opt.train()
    return [p,q],opt


def steps(params,opt,start,stop):
    for n in range(start,stop):
        for p in params:
            p.grad=torch.full_like(p, 1 if n%3 else -0.5)
        opt.step()


def same(left,right):
    if torch.is_tensor(left):
        assert left.dtype==right.dtype and torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left: same(left[k],right[k])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right): same(a,b)
    else: assert left==right


def test_counter_round_bucket_invariance_and_mean():
    x=torch.full((65536,),float(torch.tensor(0.01).bfloat16())-2e-6)
    kwargs=dict(seed=9,step=7,parameter_id=17,stream=0x10001)
    full=counter_round(x,offset=0,**kwargs)
    pieces=torch.cat([counter_round(x[i:i+73],offset=i,**kwargs) for i in range(0,len(x),73)])
    assert torch.equal(full,pieces)
    assert abs(float(full.double().mean())-float(x[0]))<3e-7


def test_eval_train_is_bit_exact_and_does_not_advance_counter():
    params,opt=make(); steps(params,opt,0,5)
    before=[p.detach().clone() for p in params]; state=deepcopy(opt.state_dict())
    for _ in range(3):
        opt.eval(); x=[p.detach().clone() for p in params]
        opt.eval()
        for p,expected in zip(params,x): assert torch.equal(p,expected)
        assert opt.param_groups[0]['k']==5
        assert len(opt.state_dict()['eval_live_y'])==2
        opt.train()
        for p,expected in zip(params,before): assert torch.equal(p,expected)
        same(opt.state_dict(),state)
    assert opt.offloaded_state_bytes()==sum(p.numel()*4 for p in params)


@pytest.mark.parametrize('eval_checkpoint',[False,True])
def test_exact_resume_with_different_bucket_size(eval_checkpoint):
    params,opt=make(); steps(params,opt,0,7)
    if eval_checkpoint: opt.eval()
    buffer=io.BytesIO()
    torch.save({'model':[p.detach().clone() for p in params],'optimizer':opt.state_dict()},buffer)
    opt.train(); steps(params,opt,7,12)
    buffer.seek(0); saved=torch.load(buffer,weights_only=False)
    restored,other=make(bucket=1031)
    for p,value in zip(restored,saved['model']): p.data.copy_(value)
    other.load_state_dict(saved['optimizer']); other.train()
    steps(restored,other,7,12)
    for p,q in zip(params,restored): assert torch.equal(p,q)
    same(opt.state_dict(),other.state_dict())


@pytest.mark.parametrize('damage',['seed','layout','algorithm','schema','missing_y','dtype','negative_v','negative_warmup'])
def test_corrupt_checkpoint_rejected_before_model_mutation(damage):
    params,opt=make(); steps(params,opt,0,1); opt.eval()
    state=deepcopy(opt.state_dict())
    if damage in ('seed','algorithm','schema'):
        state['precision_identity'][damage]='corrupted'
    elif damage=='layout': state['precision_identity']['layout_sha256']='0'*64
    elif damage=='missing_y': state['eval_live_y'].clear()
    elif damage=='dtype': state['state'][0]['z']=state['state'][0]['z'].float()
    elif damage=='negative_v': state['state'][0]['exp_avg_sq'].fill_(-1)
    else: state['param_groups'][0]['warmup_steps']=-1
    new,other=make(); before=[p.detach().clone() for p in new]
    with pytest.raises(ValueError): other.load_state_dict(state)
    for p,q in zip(new,before): assert torch.equal(p,q)


def test_failed_step_cannot_publish_partial_state():
    params,opt=make()
    for p in params: p.grad=torch.full_like(p,float('nan'))
    with pytest.raises(ValueError): opt.step()
    with pytest.raises(RuntimeError): opt.state_dict()
    with pytest.raises(RuntimeError): opt.step()


def test_constant_gradient_mean_tracks_fp32_reference_and_norm_moves():
    p=torch.nn.Parameter(torch.ones(32768,dtype=torch.bfloat16))
    q=torch.nn.Parameter(p.detach().float().clone())
    sr=ScheduleFreeSRCandidate([('norm',p)],lr=2e-6,betas=(0.9,0.95),pin_memory=False)
    ref=schedulefree.AdamWScheduleFree([q],lr=2e-6,betas=(0.9,0.95),foreach=False)
    sr.train(); ref.train()
    for _ in range(64):
        p.grad=torch.ones_like(p); q.grad=torch.ones_like(q)
        sr.step(); ref.step()
    assert bool((p!=1).any())
    assert abs(float(p.detach().double().mean())-float(q.detach().double().mean()))<2e-5
    assert all(v.dtype==torch.bfloat16 and v.device.type=='cpu' for state in sr.state.values() for v in state.values())
