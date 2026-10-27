"""Live scan integration, including the production journal's restored reads."""
import pytest
import torch
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig, init_state
from experiments.expressivity_tasks.x31_tape_cell_fused.tests.test_parity import device
from experiments.expressivity_tasks.x35_tape_cell_v3.cell_fused import tape_scan, FusedState
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import tape_scan as reference_scan


@pytest.mark.parametrize('slots',[1,7,33])
def test_live_scan_recorded_replay_and_public_rollback(slots):
    dev=device()
    cfg=TapeHeadConfig(key_dim=4,value_dim=5,n_heads=2,budget=slots,decay_rate=.8,free_threshold=.2)
    torch.manual_seed(3503)
    native=init_state(2,cfg,dev)
    native.keys.normal_();native.values.normal_();native.strength.uniform_()
    native.live.copy_(native.strength>.3);native.age.random_(0,4)
    inputs=[torch.randn(2,8,2,d,device=dev) for d in (4,4,5)]
    inputs += [torch.rand(2,8,2,device=dev),torch.rand(2,8,2,device=dev),torch.randn(2,8,2,device=dev)]
    inputs[3][:,2]=0
    a=[x.clone().requires_grad_() for x in inputs]
    b=[x.clone().requires_grad_() for x in inputs]
    actual,state,trace,_=tape_scan(*a,native,cfg)
    expected,other,reftrace=reference_scan(*b,native,cfg)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    for name in ('case','matched','pop_target','alloc_target','freed'):
        assert torch.equal(getattr(trace,name),getattr(reftrace,name))
    upstream=torch.randn_like(actual)
    ga=torch.autograd.grad(actual,a,upstream)
    gb=torch.autograd.grad(expected,b,upstream,allow_unused=True)
    for x,y in zip(ga,gb):
        torch.testing.assert_close(x,torch.zeros_like(x) if y is None else y,atol=1e-5,rtol=1e-5)
    with torch.no_grad():
        _,final,_,_=tape_scan(*inputs,native,cfg,record_all=True)
        restored=final.all_field_journal.undo_(final)
        assert isinstance(restored,FusedState)
        for name in ('keys','values','strength','live','age'):
            assert torch.equal(getattr(restored.native,name),getattr(native,name))
        tape_scan(*inputs,restored,cfg)


def test_pop_argmax_below_mask_sentinel_in_a_full_tile():
    dev=device()
    cfg=TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=32,decay_rate=1.)
    native=init_state(1,cfg,dev)
    native.keys.fill_(1e16);native.live.fill_(True);native.strength.fill_(1.)
    native.age.fill_(3)
    k=torch.zeros(1,1,1,4,device=dev)
    q=torch.full_like(k,-1e16)
    v=torch.zeros(1,1,1,5,device=dev)
    off=torch.zeros(1,1,1,device=dev)
    with torch.no_grad():
        _,_,trace,_=tape_scan(k,q,v,off,off+.75,off,native,cfg)
    assert trace.pop_target.item()==0
