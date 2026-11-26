import math

import pytest
import torch

from .attention import Answer, Local, exhaustive, gaussian_features, indexed, prepare
from .plan import Layout, Plan


def fixture(dtype, device):
    rng = torch.Generator(device="cpu").manual_seed(5602)

    def rand(*shape, scale=1.0):
        return (
            (torch.randn(shape, generator=rng, dtype=dtype) * scale)
            .to(device)
            .requires_grad_()
        )

    B, N, D, V, R = 5, 109, 7, 11, 9
    q, k, v, b, o = rand(B, D), rand(N, D), rand(N, V), rand(N), rand(R, D)
    lk, lv, lb = rand(73, D), rand(73, V), rand(73)
    local = Local(lk, lv, lb, (0, 3, 67, 1, 2))
    null = (rand(B), rand(B, V))
    ranges = ([], [(1, 6), (3, 18), (19, 20)], [(0, N)], [(8, 73), (85, 99)], [(81, N)])
    p = Plan.build(Layout("invented", N, 8), ranges)
    return q, k, v, b, o, local, null, p, ranges


def literal(q, k, v, b, o, local, null, ranges):
    """No tree or tiling. Materialize the complete mixed-law score for every occurrence."""

    def features(x):
        a = x / math.sqrt(math.sqrt(x.shape[-1]))
        return a @ o.T - (a * a).sum(1, keepdim=True) * 0.5 - math.log(len(o)) * 0.5

    exact = q @ k.T / math.sqrt(q.shape[1])
    approx = (features(q)[:, None] + features(k)[None]).logsumexp(-1)
    mask = torch.zeros_like(exact, dtype=torch.bool)
    for i, rs in enumerate(ranges):
        for a, z in rs:
            mask[i, a:z] = True
    score = torch.where(mask, exact, approx) + b
    ans = []
    mass = []
    z = []
    lg = []
    begin = 0
    for i, n in enumerate(local.lengths):
        s = (
            local.keys[begin : begin + n] @ q[i] / math.sqrt(q.shape[1])
            + local.bias[begin : begin + n]
        )
        logits = torch.cat((score[i], s, null[0][i : i + 1]))
        prob = logits.softmax(0)
        ans.append(
            prob @ torch.cat((v, local.values[begin : begin + n], null[1][i : i + 1]))
        )
        mass.append(
            torch.stack(
                (prob[: len(k)].sum(), prob[len(k) : len(k) + n].sum(), prob[-1])
            )
        )
        z.append(logits.logsumexp(0))
        lg.append(torch.stack((score[i].logsumexp(0), s.logsumexp(0), null[0][i])))
        begin += n
    return Answer(*map(torch.stack, (ans, mass, z, lg)))


def objective(a):
    return (
        (a.read * torch.arange(1, 1 + a.read.shape[-1], device=a.read.device)).sum()
        + a.log_partition.sum() * 0.31
        + a.shares[:, 1].square().sum()
        + a.source_log_z[torch.isfinite(a.source_log_z)].sum() * 0.12
    )


def parity(dtype=torch.float64, device="cpu", backend="torch"):
    q, k, v, b, o, L, null, p, rs = fixture(dtype, device)
    bank = prepare(p.layout, k, v, b, o)
    args = (q, k, v, b, o, L.keys, L.values, L.bias, *null)
    a = indexed(q, bank, p, local=L, null=null, backend=backend)
    ref = literal(q, k, v, b, o, L, null, rs)
    tol = 2e-11 if dtype == torch.float64 else 2e-5
    for key in vars(a):
        torch.testing.assert_close(
            getattr(ref, key), getattr(a, key), rtol=tol, atol=tol
        )
    ga = torch.autograd.grad(objective(a), args)
    gr = torch.autograd.grad(objective(ref), args)
    for x, y in zip(ga, gr, strict=True):
        torch.testing.assert_close(x, y, rtol=tol * 4, atol=tol * 4)
    return max((x - y).abs().max().item() for x, y in zip(ga, gr, strict=True))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_ragged_indexed_matches_explicit_law_and_every_vjp(dtype):
    parity(dtype)


def test_empty_sources_many_weak_causes_and_log_scale():
    q = torch.ones(1, 1, dtype=torch.float64)
    k = q.new_zeros((100, 1))
    v = torch.eye(3, dtype=q.dtype)[[0] * 70 + [1] * 30]
    b = q.new_full((100,), 900.0)
    o = q.new_zeros((1, 1))
    layout = Layout("repeated", 100, 10)
    p = Plan.build(layout, [[(0, 100)]])
    bank = prepare(layout, k, v, b, o)
    local = Local(
        q.new_full((1, 1), math.log(100)),
        q.new_tensor([[0.0, 0.0, 1.0]]),
        q.new_tensor([900.0]),
        (1,),
    )
    a = indexed(q, bank, p, local=local)
    torch.testing.assert_close(a.shares, q.new_tensor([[0.5, 0.5, 0.0]]))
    torch.testing.assert_close(a.read, q.new_tensor([[0.35, 0.15, 0.5]]))
    torch.testing.assert_close(a.log_partition, q.new_tensor([900.0 + math.log(200)]))
    for corpus, loc in ((True, None), (False, local), (False, None)):
        ref = exhaustive(q, k, v, b, local=loc, corpus=corpus)
        ans = indexed(q, bank, p, local=loc, corpus=corpus)
        for key in vars(ans):
            torch.testing.assert_close(getattr(ref, key), getattr(ans, key))
    empty = Layout("empty", 0)
    ans = indexed(
        q,
        prepare(empty, k[:0], v[:0], b[:0], o),
        Plan.build(empty, [[]]),
        null=(q.flatten(), v[:1]),
    )
    assert ans.shares.tolist() == [[0, 0, 1]]
    with pytest.raises(ValueError, match="foreign"):
        indexed(q, bank, Plan.build(Layout("wrong", 100, 10), [[]]))
    k.add_(1.0)
    with pytest.raises(ValueError, match="mutated"):
        indexed(q, bank, p)


def test_gaussian_map_has_no_rng_side_effect_and_no_cancellation_of_the_tail():
    rng = torch.get_rng_state().clone()
    o = gaussian_features(1, 7)
    assert torch.equal(rng, torch.get_rng_state())
    # Replacing a huge background score by a tiny exact one must not erase the separate tail.
    q = torch.tensor([[8.0]])
    k = torch.tensor([[-4.0], [-6.0]])
    v = torch.eye(2)
    b = torch.tensor([20.0, 0.0])
    o = torch.full((1, 1), 16.0)
    p = Plan.build(Layout("hi", 2, 1), [[(0, 1)]])
    a = indexed(q, prepare(p.layout, k, v, b, o), p)
    truth = torch.tensor([-12.0, -18.0])
    torch.testing.assert_close(a.read[0], truth.softmax(0))
    torch.testing.assert_close(a.log_partition[0], truth.logsumexp(0))
