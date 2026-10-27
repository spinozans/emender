"""Certify recurrent gate decisions, then full-shape X31 replay parity."""
from dataclasses import asdict
import os
from unittest.mock import patch
import torch
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig, tape_scan as oracle
from ..x30_tape_cell.naive_reference import init_state
from ..x34_lm_pilot.cli import replay_present, write
from .semantic import scan
from .usage import REC


def certify():
    torch.manual_seed(3909)
    cfg=TapeHeadConfig(n_heads=4,budget=2944,key_dim=64,value_dim=128)
    b,t,h=64,512,4
    k=torch.randn(b,t,h,64,device='cuda')*.3
    q=torch.randn_like(k)*.3
    v=torch.randn(b,t,h,128,device='cuda')
    pop=torch.rand(b,t,h,device='cuda')
    append=torch.randn(b,t,h,device='cuda')
    logits=torch.randn_like(append)+.5
    coef=torch.full((h,),-1.,device='cuda')
    err=torch.zeros(h,device='cuda')
    mask=torch.ones_like(append);mask[:,::17]=0
    beta=torch.empty_like(mask);evidence=torch.empty_like(mask)
    state=None
    # Sequential unchanged X31 operations supply the independent state used
    # for next-token evidence. PyTorch reduction tolerance is explicit.
    with torch.no_grad():
        for i in range(t):
            if state is None:
                e=torch.zeros(b,h,device='cuda')
            else:
                native=state.native
                scores=(native.keys*q[:,i,:,None,:]).sum(-1)*(2/(64**.5))
                e=scores.masked_fill(~native.live,-float('inf')).amax(-1)
                e=torch.where(native.live.any(-1),e,0.)
            evidence[:,i]=e
            beta[:,i]=((logits[:,i]+coef*e).sigmoid()>.5).float()*mask[:,i]
            _,state,_,_=oracle(k[:,i:i+1],q[:,i:i+1],v[:,i:i+1],beta[:,i:i+1],
                pop[:,i:i+1],append[:,i:i+1],state,cfg,err)
        del state
    results=[]
    for wired in (False,True):
        xs=[x.detach().clone().requires_grad_(True) for x in (q,v,pop,append,err)]
        if wired:
            (out,st,tr,p),ev,sig=scan(k,xs[0],xs[1],mask,xs[2],xs[3],None,cfg,xs[4],logits,coef)
            torch.testing.assert_close(ev,evidence,rtol=2e-5,atol=2e-6)
            assert torch.equal(tr.w_on,beta.bool())
        else:
            with patch('experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused.tape_scan',oracle):
                out,st,tr,p=oracle(k,xs[0],xs[1],beta,xs[2],xs[3],None,cfg,xs[4])
        values=(out,st.native.keys,st.native.values,st.native.strength)
        loss=sum(x.square().sum() for x in values)
        if wired: replay_present(loss)
        grads=torch.autograd.grad(loss,xs)
        results.append((tuple(x.detach().clone() for x in values),grads,tr,p.detach().clone()))
        del values,out,st,loss,grads,xs
    a,z=results
    for group in (0,1):
        for x,y in zip(a[group],z[group]):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
    for name in ('case','matched','alloc_target','pop_target','pop_fired','freed','w_on'):
        assert torch.equal(getattr(a[2],name),getattr(z[2],name)),name
    assert torch.equal(a[3],z[3])
    result=dict(status='passed',batch=b,context=t,configuration=asdict(cfg),
        physical_gpu=os.environ['GPU_LEASE_HELD'],
        evidence='Sequential X31 state; pre-read PyTorch maximum scores, empty=0; rtol 2e-5 atol 2e-6',
        gate_decisions='exact',forward_state_trace_pressure_gradients='bitwise X31',replay='ReplayBackward asserted',
        limits='One seeded production shape; no claim of exhaustive adversarial coverage')
    write(REC/'part9_semantic_parity.json',result)
    print(result,flush=True)
    return result

if __name__=='__main__':certify()
