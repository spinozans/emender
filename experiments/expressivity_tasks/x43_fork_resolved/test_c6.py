"""C6 probability contract and forced-loss exclusion (CPU)."""
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c6 import branch_logp


def test_branch_ood_mass_and_gradient():
    byte=torch.zeros(3,256,requires_grad=True)
    ood=torch.zeros(3,1,requires_grad=True)
    support=torch.zeros(3,256,dtype=torch.bool)
    support[0,:2]=True
    support[2,:]=True
    lp=branch_logp(byte,ood,support)
    p=lp.exp()
    torch.testing.assert_close(p.sum(-1),torch.ones(3))
    torch.testing.assert_close(p[0,:2],torch.full((2,),1/3))
    torch.testing.assert_close(p[0,2:].sum(),torch.tensor(1/3))
    torch.testing.assert_close(p[1],torch.full((256,),1/256))
    torch.testing.assert_close(p[2],torch.full((256,),1/256))
    (-lp[0,0]).backward()
    assert torch.isfinite(byte.grad).all() and torch.isfinite(ood.grad).all()
    assert byte.grad[0,2:].eq(0).all()


def test_target_alignment_and_forced_gradient():
    # Conditioning for input j is for next target j+1, including window offsets.
    starts=torch.tensor([3,9])
    at=starts[:,None]+torch.arange(4)
    fork=torch.tensor([False,True]*10)
    logits=torch.randn(2,4,256,requires_grad=True)
    targets=torch.arange(20)%256
    loss=-logits.log_softmax(-1).gather(-1,targets[at+1,None])[...,0]
    mask=fork[at+1]
    loss[mask].mean().backward()
    assert logits.grad[~mask].eq(0).all()
    assert logits.grad[mask].abs().sum()>0
