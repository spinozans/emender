import torch
from experiments.expressivity_tasks.x43_fork_resolved.c7b import Boosted


def test_floor_never_starves_probability():
    # Exercise real forward routing with a tiny differentiable neural head.
    class Head(torch.nn.Module):
        def forward(self,ids,features,aware):
            return torch.zeros((*ids.shape,256)).log_softmax(-1)
    m=Boosted.__new__(Boosted); torch.nn.Module.__init__(m)
    m.head=Head(); m.gate=torch.nn.Linear(4,1,bias=False)
    m.gate.weight.data.fill_(-100)
    features=torch.ones(1,2,260); features[:,1,:256]=0
    _,g=m(torch.zeros(1,2,dtype=torch.long),features,torch.ones(1,2))
    assert g[0,0]>=.3 and g[0,1]==1


def test_fixed_true_gradient_survives_perfect_index():
    logits=torch.zeros(2,256,requires_grad=True)
    lp=logits.log_softmax(-1); teacher=torch.zeros_like(lp); teacher[:,1]=1
    true=-lp[:,0]; shaping=-(teacher*lp).sum(-1)
    covered=torch.tensor([True,False])
    loss=torch.where(covered,.5*true+.5*shaping,true).sum()
    loss.backward()
    assert logits.grad[0,0]<-.49
    assert logits.grad[1,0]<-.99
