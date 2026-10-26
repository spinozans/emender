"""Deferred replay ownership and single-writer journal regressions."""
import pytest
import torch
from .test_parity import device
from experiments.expressivity_tasks.x30_tape_cell import naive_reference as oracle
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import tape_scan


def test_deferred_replay_freezes_before_inference_continuation():
    dev=device();torch.manual_seed(831)
    cfg=oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=2,budget=37,
                             decay_rate=.8,free_threshold=.2)
    inputs=[torch.randn(1,6,2,4,device=dev),torch.randn(1,6,2,4,device=dev),
            torch.randn(1,6,2,5,device=dev),torch.ones(1,6,2,device=dev),
            torch.rand(1,6,2,device=dev),torch.randn(1,6,2,device=dev)]
    a=[x.clone().requires_grad_() for x in inputs]
    e=[x.clone().requires_grad_() for x in inputs]
    out,state,_,_=tape_scan(*a,None,cfg)
    expected,_,_=oracle.tape_scan(*e,oracle.init_state(1,cfg,dev),cfg)
    storage=state.replay_storage
    assert not storage.frozen
    with torch.no_grad():
        tape_scan(*inputs,state,cfg)
    assert storage.frozen
    actual=torch.autograd.grad(out.sum(),a,retain_graph=True,allow_unused=True)
    repeated=torch.autograd.grad(out.sum(),a,allow_unused=True)
    reference=torch.autograd.grad(expected.sum(),e,allow_unused=True)
    for x,y,z,base in zip(actual,repeated,reference,inputs):
        x=torch.zeros_like(base) if x is None else x
        y=torch.zeros_like(base) if y is None else y
        z=torch.zeros_like(base) if z is None else z
        torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(x,z,atol=1e-5,rtol=1e-5)


def test_deferred_replay_rejects_public_inplace_mutation():
    dev=device()
    cfg=oracle.TapeHeadConfig(key_dim=2,value_dim=2,n_heads=1,budget=7)
    k=torch.ones(1,2,1,2,device=dev,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    on=torch.ones(1,2,1,device=dev);off=torch.zeros_like(on)
    out,state,_,_=tape_scan(k,k,v,on,off,off,None,cfg)
    state.native.live.fill_(False)
    with pytest.raises(RuntimeError,match='modified in place'):
        out.sum().backward()


@pytest.mark.parametrize('slots',[97,736])
def test_single_writer_materialization_multitile_rollback(slots):
    dev=device()
    cfg=oracle.TapeHeadConfig(key_dim=2,value_dim=3,n_heads=2,budget=slots,decay_rate=1.)
    initial=oracle.init_state(1,cfg,dev)
    initial.live.fill_(True);initial.strength.fill_(.7);initial.age.fill_(3)
    k=torch.zeros(1,1,2,2,device=dev)
    v=torch.zeros(1,1,2,3,device=dev);off=torch.zeros(1,1,2,device=dev)
    _,state,_,_=tape_scan(k,k,v,off,off,off,initial,cfg,record_all=True)
    assert state.native.age.eq(4).all()
    restored=state.all_field_journal.undo_(state)
    for field in ('keys','values','strength','age','live'):
        assert torch.equal(getattr(restored.native,field),getattr(initial,field))
