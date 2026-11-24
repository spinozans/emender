import pytest
import torch

from .bank_read import BoundPage, reference_read


def scalar_oracle(q, k, v, ids, b):
    # Enumeration independent of the padded softmax/gather; only a test oracle.
    out = []
    for r, row in enumerate(ids.tolist()):
        entries = [(j, a) for j,a in enumerate(row) if a >= 0]
        if entries:
            scores = torch.stack([b[r,j] + torch.dot(q[r], k[a])/q.shape[1]**.5 for j,a in entries])
            out.append(sum(w*v[a] for w,(_,a) in zip(scores.softmax(0), entries)))
        else:
            out.append(q[r]*0)
    return torch.stack(out)


def _variable(x):
    return x.detach().clone().requires_grad_()


def test_banks_and_birth():
    ids = torch.tensor([[0,0,1], [2,-1,-1], [-1,-1,-1]])
    page = BoundPage.from_cpu(ids, 3)
    ids.fill_(1000)  # not an alias to the original input
    k = torch.tensor([[1.,0.], [0.,1.], [2.,3.]])
    v = torch.tensor([[1.,7.], [4.,1.], [8.,11.]])
    q = torch.zeros((3,2), requires_grad=True)
    b = torch.zeros((3,3), requires_grad=True)
    y = reference_read(q,k,v,page,b)
    torch.testing.assert_close(y, torch.tensor([[2.,5.],[8.,11.],[0.,0.]]))
    y[0,0].backward()
    assert q.grad[0].abs().sum() > 0 and q.grad[1:].count_nonzero() == 0
    assert b.grad[1:].count_nonzero() == 0
    perm = torch.tensor([2,0,1])
    y2 = reference_read(q[perm],k,v,BoundPage.from_cpu(page.ids[perm],3),b[perm])
    torch.testing.assert_close(y2, y[perm])


def test_page_refusals():
    for ids, n in [(torch.tensor([[-2]]),3),(torch.tensor([[2**32+1]]),2),
                   (torch.tensor([[3]]),3),(torch.zeros(1,65,dtype=torch.int32),1)]:
        with pytest.raises(ValueError):
            BoundPage.from_cpu(ids,n)


def test_reference_and_adjoint():
    g = torch.Generator().manual_seed(3411)
    page = BoundPage.from_cpu(torch.tensor([[0,1,1],[2,-1,0],[-1,-1,-1]]),3)
    q,k,v,b = [_variable(torch.randn(s, generator=g, dtype=torch.float64)) for s in [(3,5)]*3+[(3,3)]]
    y = reference_read(q,k,v,page,b)
    torch.testing.assert_close(y, scalar_oracle(q,k,v,page.ids,b))
    assert torch.autograd.gradcheck(lambda q,k,v,b: reference_read(q,k,v,page,b),(q,k,v,b))


def test_empty_table():
    page = BoundPage.from_cpu(torch.full((2,3),-1),0)
    q,k,v,b = [_variable(torch.randn(s)) for s in [(2,5),(0,5),(0,5),(2,3)]]
    y = reference_read(q,k,v,page,b)
    assert not y.count_nonzero()
    for grad in torch.autograd.grad(y.sum(),(q,k,v,b)):
        assert not grad.count_nonzero()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('n,f,d',[(0,7,17),(1,1,1),(8,7,31),(53,32,64),(89,64,128)])
def test_device_parity(n,f,d,dtype):
    from .b1_lease import approved
    if not approved() or not torch.cuda.is_available():
        pytest.skip('requires approved leased GPU; CPU oracle is separate')
    from .bank_read_triton import triton_read
    g = torch.Generator().manual_seed(3401); r = 7
    ids = torch.randint(max(n,1), (r,f), generator=g)
    if n:
        ids[::3] = -1
        ids[1, ::2] = 0
        ids[2, 1:] = -1
    else:
        ids.fill_(-1)
    page = BoundPage.from_cpu(ids,n,device='cuda')
    xs = [torch.randn(s,generator=g).to(dtype) for s in [(r,d),(n,d),(n,d)]]
    xs[0][1::3] = 0.  # birth query
    b = torch.randn(r,f,generator=g)
    b[4,0],b[5,0] = 80.,-80.
    gpu = [_variable(x.cuda()) for x in (*xs,b)]
    cpu = [_variable(x.double()) for x in (*xs,b)]
    u = torch.randn(r,d,generator=g)
    yt = triton_read(*gpu[:3],page,gpu[3]); yr = reference_read(*cpu[:3],BoundPage.from_cpu(ids,n),cpu[3])
    gt = torch.autograd.grad((yt*u.cuda()).sum(),gpu); gr = torch.autograd.grad((yr*u.double()).sum(),cpu)
    tol = dict(rtol=1e-4,atol=3e-5) if dtype==torch.float32 else dict(rtol=.03,atol=.02)
    errors = []
    for a,e in zip((yt,*gt),(yr,*gr)):
        a = a.detach().cpu().double(); e = e.detach()
        errors.append((a-e).abs().max().item() if a.numel() else 0.)
        torch.testing.assert_close(a,e,**tol)
    print('parity', n, f, d, str(dtype), 'max_abs[y,q,k,v,log_prior]', errors)
    assert yt.detach().cpu()[(ids < 0).all(1)].count_nonzero() == 0
    assert gt[-1].detach().cpu()[ids < 0].count_nonzero() == 0
