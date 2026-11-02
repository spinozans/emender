import torch
from unittest.mock import patch
from experiments.expressivity_tasks.x43_fork_resolved.c11 import Boosted

def test_device_input_dense_gradient_and_hook_cleanup():
    class Tiny(torch.nn.Module):
        def __init__(self,*args):
            super().__init__()
            self.base=torch.nn.Module()
            self.base.embed=torch.nn.Embedding(256,256)
        def forward(self,ids,features,aware):
            assert not aware
            return self.base.embed(ids).log_softmax(-1)
    with patch('experiments.expressivity_tasks.x43_fork_resolved.c11.Head',Tiny):
        m=Boosted('emender',1)
    ids=torch.tensor([[1,2]])
    features=torch.randn(1,2,260)
    lp,g=m(ids,features,torch.ones(1,2))
    (-lp[...,3].mean()).backward()
    assert m.device_input.weight.grad.abs().sum()>0
    assert m.head.base.embed.weight.grad[1].abs().sum()>0
    assert m.head.base.embed.weight.grad[2].abs().sum()>0
    assert not m.head.base.embed._forward_hooks
    with torch.no_grad():
        zero,_=m(ids,torch.zeros_like(features),torch.ones(1,2))
        assert torch.equal(zero,m.head(ids,features,False))
