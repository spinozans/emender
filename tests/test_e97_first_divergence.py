import torch
import pytest
from scripts.e97_first_divergence import Rows,compare,digest
from scripts.diagnose_e97_first_divergence import exact_scores


class Mixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for n in ('qkv_proj','a_proj','g_proj','erase_gate_proj','value_write_gate_proj','o_proj'):
            setattr(self,n,torch.nn.Linear(8,8))
    def forward(self,x):
        # Real projection subchunks, but no recurrent state in this mapping fixture.
        outputs=[]
        for c in x.split(2,dim=1):
            z=self.qkv_proj(c)+self.a_proj(c)+self.g_proj(c)+self.erase_gate_proj(c)+self.value_write_gate_proj(c)
            outputs.append(self.o_proj(z))
        return torch.cat(outputs,dim=1),None


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w1=torch.nn.Linear(8,8);self.w2=torch.nn.Linear(8,8);self.w3=torch.nn.Linear(8,8)
    def forward(self,x):return self.w3(self.w1(x)*self.w2(x))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.mixer=Mixer();self.norm_2=torch.nn.Identity();self.mlp=MLP()
    def forward(self,x):
        y,_=self.mixer(x);return y+self.mlp(self.norm_2(x+y)),None


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(32,8);self.layers=torch.nn.ModuleList([Block(),Block()])
    def forward(self,ids):
        x=self.embedding(ids)
        for b in self.layers:x,_=b(x)
        return x


@pytest.mark.parametrize('layer',[None,0,1])
def test_logical_rows_across_calls_projection_chunks_and_tail(layer):
    torch.manual_seed(7);model=Model().double();tokens=torch.arange(8)[None]
    expected=model(tokens);probe=Rows(model,5,layer)
    try:
        outputs=[model(x) for x in (tokens[:,:4],tokens[:,4:6],tokens[:,6:])]
        bank=probe.finish()
    finally:probe.close()
    assert torch.allclose(torch.cat(outputs,1),expected,atol=1e-14,rtol=0)
    assert all(v.shape==(5,8) for v in bank.values())
    probe=Rows(model,5,layer)
    try:model(tokens);other=probe.finish()
    finally:probe.close()
    assert list(bank)==list(other)
    assert all(torch.allclose(v,other[k],atol=1e-14,rtol=0) for k,v in bank.items())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_bitwise_first_site_token_and_signed_zero():
    a=dict(first=torch.zeros(4,3,dtype=torch.bfloat16),second=torch.ones(4,3,dtype=torch.bfloat16))
    b={k:v.clone() for k,v in a.items()};b['first'][2,1]=-0.;b['second'][0,0]=2
    result=compare(a,b)
    assert not result[0]['exact'] and result[0]['first_token']==2 and result[0]['first_feature']==1
    assert result[0]['absolute_max']==0 and result[0]['differing_elements']==1
    assert digest(a['first'])!=digest(b['first'])
    assert not exact_scores([0.],[-0.]) and exact_scores([1.],[1.])
    assert not exact_scores([float('nan')],[0.])


def test_bounds_and_incomplete_coverage():
    model=Model();probe=Rows(model,5,max_bytes=1)
    try:
        with pytest.raises(ValueError,match='budget'):model(torch.arange(5)[None])
    finally:probe.close()
    probe=Rows(model,5)
    try:
        model(torch.arange(3)[None])
        with pytest.raises(ValueError,match='coverage'):probe.finish()
    finally:probe.close()


def test_fused_norm_wrapper_preserves_objects_and_restores(monkeypatch):
    import importlib
    owner=importlib.import_module('ndm.models.ladder_lm');model=Model()
    model.fused_add_norm=True;model.layer_norms=torch.nn.ModuleList([torch.nn.LayerNorm(8),torch.nn.LayerNorm(8)])
    x=torch.ones(1,5,8);result=(x,x.float())
    def original(*args,**kwargs):return result
    monkeypatch.setattr(owner,'rms_norm_fn',original,raising=False)
    probe=Rows(model,5,0)
    try:
        assert owner.rms_norm_fn(x,model.layer_norms[0].weight,None,residual=None) is result
        assert probe.bank['outer-norm.residual'] is None
        assert torch.equal(probe.bank['outer-norm.normalized'][0],x[0])
    finally:probe.close()
    assert owner.rms_norm_fn is original
