import pytest
import torch
from experiments.expressivity_tasks.x36_tape_perf.steady import FiniteGradients


def test_clip_and_optimizer_exact():
    torch.manual_seed(36)
    a=torch.nn.Parameter(torch.randn(97,device='cuda'))
    b=torch.nn.Parameter(a.detach().clone())
    opts=[torch.optim.AdamW([p],lr=3e-4,betas=(.9,.95),weight_decay=.1) for p in (a,b)]
    monitor=FiniteGradients(a.device)
    for _ in range(3):
        gradient=torch.randn_like(a)*4
        a.grad=gradient.clone();b.grad=gradient.clone()
        old=torch.nn.utils.clip_grad_norm_([a],1.,error_if_nonfinite=True)
        new=monitor.clip([b])
        assert torch.equal(old.reshape(1).view(torch.uint8),new.reshape(1).view(torch.uint8))
        assert torch.equal(a.grad.view(torch.uint8),b.grad.view(torch.uint8))
        for opt in opts:opt.step()
        assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    monitor.check()


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_is_reported_at_boundary(bad):
    p=torch.nn.Parameter(torch.ones(3,device='cuda'))
    p.grad=torch.tensor([1.,bad,2.],device='cuda')
    monitor=FiniteGradients(p.device)
    monitor.clip([p])
    with pytest.raises(RuntimeError,match='nonfinite gradient norm'):
        monitor.check()


def test_bounded_projection_matches_dense():
    from experiments.expressivity_tasks.x36_tape_perf.loss import ChunkedLoss
    torch.manual_seed(36)
    h=torch.randn(2,17,16,device='cuda',requires_grad=True)
    w=torch.randn(37,16,device='cuda',requires_grad=True)
    y=torch.randint(37,(2,17),device='cuda')
    dense=torch.nn.functional.cross_entropy(torch.nn.functional.linear(h,w).flatten(0,1),y.flatten())
    bounded=ChunkedLoss.apply(h,w,y,7)
    torch.testing.assert_close(bounded,dense,atol=1e-6,rtol=1e-6)
    for a,b in zip(torch.autograd.grad(dense,(h,w)),torch.autograd.grad(bounded,(h,w))):
        torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-6)
