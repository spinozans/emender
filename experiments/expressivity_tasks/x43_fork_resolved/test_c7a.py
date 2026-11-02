"""CPU objective receipts: gradients, normalization, and empty abstention."""
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c7a import Boosted


def test_dense_mixture_retains_forced_gradient_and_fork_gradient():
    logits=torch.tensor([[0.,0.],[0.,0.]],requires_grad=True)
    gate_logits=torch.zeros(2,requires_grad=True)
    p=logits.softmax(-1)[:,0]; q=torch.tensor([1.,.25]); g=gate_logits.sigmoid()
    loss=-(g*p+(1-g)*q).log().mean()
    loss.backward()
    assert (logits.grad.abs().sum(-1)>0).all()
    assert (gate_logits.grad.abs()>0).all()
    # Confident index anchor attenuates rather than deletes the neural gradient.
    assert logits.grad[0].abs().sum()<logits.grad[1].abs().sum()


def test_empty_support_abstains_with_no_gate_gradient():
    class Fake(torch.nn.Module):
        def forward(self,ids,features,aware):
            return torch.zeros((*ids.shape,256)).log_softmax(-1)
    model=Boosted.__new__(Boosted); torch.nn.Module.__init__(model)
    model.head=Fake(); model.gate=torch.nn.Linear(4,1,bias=False)
    ids=torch.zeros((1,2),dtype=torch.long); features=torch.zeros((1,2,260))
    lp,g=model(ids,features,torch.ones_like(ids,dtype=torch.float32))
    assert torch.equal(g,torch.ones_like(g))
    g.sum().backward(); assert (model.gate.weight.grad==0).all()
    assert torch.allclose(lp.exp().sum(-1),torch.ones_like(g))


def test_mixture_is_normalized_and_joint_gradients_are_finite():
    logits=torch.randn(3,256,requires_grad=True); gate=torch.zeros(3,requires_grad=True)
    q=torch.rand(3,256); q=q/q.sum(-1,keepdim=True)
    g=gate.sigmoid()[:,None]; fused=g*logits.softmax(-1)+(1-g)*q
    assert torch.allclose(fused.sum(-1),torch.ones(3))
    (-fused[:,0].log().mean()).backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(gate.grad).all()
