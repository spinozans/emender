import torch
from experiments.expressivity_tasks.x38_diversity_objective.model import read_diversity


def test_identical_reads():
    x = torch.ones(2,3,4,8,requires_grad=True)
    penalty,pairs,coverage = read_diversity(x)
    torch.testing.assert_close(penalty,torch.tensor(6.))
    assert (coverage == 1).all()
    penalty.backward()
    assert torch.isfinite(x.grad).all()


def test_zero_reads_are_missing_not_similar():
    x = torch.zeros(2,3,4,8,requires_grad=True)
    penalty,pairs,coverage = read_diversity(x)
    assert penalty == 0 and not coverage.any()
    penalty.backward()
    assert torch.isfinite(x.grad).all()


def test_signed_cosine_and_mask():
    x = torch.eye(4).view(1,1,4,4).repeat(1,2,1,1)
    assert read_diversity(x)[0] == 0
    x[:,:,1] = -x[:,:,0]
    torch.testing.assert_close(read_diversity(x)[0],torch.tensor(-1.))
    x[:,0] = 0
    _,pairs,coverage = read_diversity(x)
    torch.testing.assert_close(pairs[0],torch.tensor(-1.))
    assert (coverage == .5).all()


def test_gradcheck():
    # Production computes cosine in float32; finite difference at its scale.
    x = torch.randn(1,2,4,8,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a:read_diversity(a)[0],(x,),eps=.001,atol=.002,rtol=.02)
