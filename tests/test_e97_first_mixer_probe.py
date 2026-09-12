import pytest
import torch
from scripts.e97_first_mixer_probe import FirstMixerProbe,STAGES


class Mixer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.key_dim=self.value_dim=3
        for name,n in [('qkv_proj',9),('a_proj',1),('g_proj',3),('erase_gate_proj',3),('value_write_gate_proj',3),('o_proj',3)]:
            setattr(self,name,torch.nn.Linear(3,n,bias=False))
        with torch.no_grad():
            for p in self.parameters():p.fill_(.125)
    def forward(self,x):
        rows=[]
        for part in x.split(2,dim=1):
            q,k,v=self.qkv_proj(part).split(3,dim=-1)
            z=q+k+v+self.a_proj(part)+self.g_proj(part)+self.erase_gate_proj(part)+self.value_write_gate_proj(part)
            rows.append(self.o_proj(z))
        return torch.cat(rows,dim=1),[]


class Layer(torch.nn.Module):
    def __init__(self):super().__init__();self.mixer=Mixer()
    def forward(self,x):return self.mixer(x*.5)


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Layer()])
    def forward(self,x):return self.layers[0](x)[0]


def capture(model,parts,length):
    probe=FirstMixerProbe(model,length)
    try:
        with torch.no_grad():outputs=[model(x) for x in parts]
        return torch.cat(outputs,dim=1),probe.finish(),probe.chunk_lengths
    finally:probe.close()


def test_entire_prefix_alignment_with_chunking_padding_and_unused_last_step():
    model=Model().to(dtype=torch.bfloat16);x=torch.arange(18,dtype=torch.bfloat16).reshape(1,6,3)/64
    a,full,full_chunks=capture(model,[x[:,:5]],5)
    b,segmented,seg_chunks=capture(model,[x[:,:3],x[:,3:4],x[:,4:5],x[:,5:]],5)
    _,padded,_=capture(model,[torch.cat([x[:,:5],torch.zeros_like(x[:,:3])],dim=1)],5)
    assert torch.equal(a,b[:,:5]) and list(full)==list(STAGES)
    assert all(torch.equal(full[k],segmented[k]) and torch.equal(full[k],padded[k]) for k in STAGES)
    assert full_chunks==[2,2,1] and seg_chunks==[2,1,1,1,1]
    assert not torch.equal(full['wrapper-input'],full['mixer-input'])
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_hooks_preserve_outputs_and_parameters():
    model=Model();x=torch.ones(1,5,3);saved=[p.detach().clone() for p in model.parameters()]
    with torch.no_grad():baseline=model(x)
    output,_,_=capture(model,[x],5)
    assert torch.equal(baseline,output)
    assert all(torch.equal(a,b) for a,b in zip(saved,model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_component_budget_failure_removes_hooks_on_close():
    model=Model();probe=FirstMixerProbe(model,5,max_bytes=1)
    try:
        with pytest.raises(ValueError,match='budget'):model(torch.ones(1,5,3))
    finally:probe.close()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_missing_projection_rejected_before_hooks_are_added():
    model=Model();model.layers[0].mixer.a_proj=None
    with pytest.raises(ValueError,match='projections'):FirstMixerProbe(model,5)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_incomplete_causal_prefix_rejected():
    model=Model();probe=FirstMixerProbe(model,5)
    try:
        with torch.no_grad():model(torch.ones(1,3,3))
        with pytest.raises(ValueError,match='row coverage'):probe.finish()
    finally:probe.close()
