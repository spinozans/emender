"""Empty-import sparse replay and grouped head scoring regressions."""
import pytest
import torch
from .test_parity import device
from experiments.expressivity_tasks.x30_tape_cell import naive_reference as oracle
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import tape_scan


@pytest.mark.parametrize('slots',[736,11776])
@pytest.mark.parametrize('heads',[1,4,16])
def test_empty_sparse_replay_heads(slots,heads):
    dev=device();torch.manual_seed(731)
    cfg=oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=heads,budget=slots,
                             decay_rate=.5,free_threshold=.2)
    b,t=1,6
    base=[torch.randn(b,t,heads,4,device=dev),torch.randn(b,t,heads,4,device=dev),
          torch.randn(b,t,heads,5,device=dev),torch.ones(b,t,heads,device=dev),
          torch.rand(b,t,heads,device=dev),torch.randn(b,t,heads,device=dev)]
    a=[x.clone().requires_grad_() for x in base]
    e=[x.clone().requires_grad_() for x in base]
    ao,af,at,_=tape_scan(*a,None,cfg)
    eo,ef,et=oracle.tape_scan(*e,oracle.init_state(b,cfg,dev),cfg)
    actual=(ao,af.native.keys,af.native.values,af.native.strength)
    expected=(eo,ef.keys,ef.values,ef.strength)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    for field in ('case','matched','pop_target','alloc_target','freed'):
        assert torch.equal(getattr(at,field),getattr(et,field))
    weights=[torch.randn_like(x) for x in actual]
    ga=torch.autograd.grad(actual,a,weights,allow_unused=True,retain_graph=True)
    again=torch.autograd.grad(actual,a,weights,allow_unused=True)
    ge=torch.autograd.grad(expected,e,weights,allow_unused=True)
    for x,y,z,base_x in zip(ga,again,ge,base):
        x=torch.zeros_like(base_x) if x is None else x
        y=torch.zeros_like(base_x) if y is None else y
        z=torch.zeros_like(base_x) if z is None else z
        torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(x,z,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('slots',[736,11776])
def test_production_h16_score_resources(slots):
    dev=device();torch.manual_seed(732)
    cfg=oracle.TapeHeadConfig(key_dim=64,value_dim=128,n_heads=16,budget=slots)
    k,q=[torch.randn(1,3,16,64,device=dev) for _ in range(2)]
    v=torch.randn(1,3,16,128,device=dev)
    on=torch.ones(1,3,16,device=dev);off=torch.zeros_like(on)
    actual,final,trace,_=tape_scan(k,q,v,on,off,off,None,cfg)
    expected,state,reference=oracle.tape_scan(k,q,v,on,off,off,oracle.init_state(1,cfg,dev),cfg)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    for field in ('keys','values','strength','age','live'):
        torch.testing.assert_close(getattr(final.native,field),getattr(state,field),atol=1e-5,rtol=1e-5)
    for field in ('case','matched','alloc_target','pop_target','freed'):
        assert torch.equal(getattr(trace,field),getattr(reference,field))
