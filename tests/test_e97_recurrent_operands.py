import copy
import importlib
import torch
import pytest
from scripts.e97_tensor_capsule import pack,unpack
from scripts.e97_recurrent_call_capture import Capture,describe
from scripts.diagnose_e97_recurrent_operands import eligible,delta


def test_capsule_values_strides_offsets_aliases_none_and_empty():
    base=torch.arange(96,dtype=torch.float32).bfloat16().reshape(2,3,16)
    tensors=dict(k=base[...,4:8].transpose(0,1),q=base[...,:4].transpose(0,1),
        v=base[...,8:12].transpose(0,1),none=None,empty=torch.empty(0,3),scalar=torch.tensor(3.))
    c=pack(tensors);out=unpack(c,'cpu')
    for k,t in tensors.items():
        if t is None:assert out[k] is None;continue
        assert torch.equal(out[k],t) and out[k].dtype==t.dtype
        assert out[k].stride()==t.stride() and out[k].storage_offset()==t.storage_offset()
    assert out['q'].untyped_storage().data_ptr()==out['v'].untyped_storage().data_ptr()==out['k'].untyped_storage().data_ptr()
    assert out['q'].untyped_storage().data_ptr()!=base.untyped_storage().data_ptr()
    before=base.clone();out['q'][0,0,0]=99;assert torch.equal(base,before)
    assert describe(c)==describe(copy.deepcopy(c))


def test_capsule_bounds_are_checked_before_destination_copy(monkeypatch):
    c=pack(dict(x=torch.ones(2,3)))
    with pytest.raises(ValueError):pack(dict(x=torch.ones(20)),max_bytes=1)
    c['tensors']['x']['offset']=1000
    def forbidden(*args,**kwargs):raise AssertionError('copied before validation')
    monkeypatch.setattr(torch.Tensor,'to',forbidden)
    with pytest.raises(ValueError,match='exceeds storage'):unpack(c,'cpu')


def test_capture_preserves_result_and_observes_only_first_call(monkeypatch):
    owner=importlib.import_module('ndm.triton.e88_triton_optimized')
    x=torch.ones(2,1,2,4,dtype=torch.bfloat16);state=torch.zeros(1,2,4,4);ret=(x,state)
    def original(S0,k,v,q,decay,g=None,recurrent_state_precision='legacy'):return ret
    monkeypatch.setattr(owner,'e88_triton',original)
    probe=Capture()
    try:
        assert owner.e88_triton(state,x,x,x,torch.ones(2,1,2)) is ret
        assert owner.e88_triton(state,x,x,x,torch.ones(2,1,2)) is ret
    finally:probe.close()
    assert owner.e88_triton is original
    evidence=probe.finish();assert evidence['total_calls']==2 and evidence['inputs_unchanged']
    assert torch.equal(unpack(probe.outputs,'cpu')['out'],x)


def test_matrix_eligibility_rejects_nonzero_state_interior_reset_and_changed_operands():
    state=torch.zeros(1,2,4,4);x=torch.ones(2,1,2,4,dtype=torch.bfloat16)
    a=dict(S0=state,k=x,q=x,v=x,reset_before=None,valid_mask=None)
    b=dict(a,reset_before=torch.tensor([[True],[False]]),valid_mask=torch.ones(2,1,dtype=torch.bool))
    ma=dict(recurrent_state_precision='legacy',normalize_kq=True);mb=dict(ma,recurrent_state_precision='fp32')
    assert all(eligible(a,b,ma,mb).values())
    bad=dict(b,S0=torch.ones_like(state));assert not all(eligible(a,bad,ma,mb).values())
    bad=dict(b,reset_before=torch.ones(2,1,dtype=torch.bool));assert not eligible(a,bad,ma,mb)['masks_equivalent_on_zero_state']
    bad=dict(b,k=x+1);assert not eligible(a,bad,ma,mb)['numeric_values_equal']


def test_delta_reports_true_nd_coordinate_and_signed_zero():
    a=torch.zeros(2,1,2,4,dtype=torch.bfloat16);b=a.clone();b[1,0,1,3]=-0.
    d=delta(a,b);assert not d['exact'] and d['first_index']==[1,0,1,3] and d['absolute_max']==0
