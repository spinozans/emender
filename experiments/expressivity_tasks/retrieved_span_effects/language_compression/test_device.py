"""Run first in the CPU interpreter and again on the leased card before any fit."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from ..corpus_windows.triton_mix import mix, csr
from ..corpus_windows.triton_parent import parents
from ..corpus_windows.tree import parent_reference
from .lm import ARMS, LanguageLM, Pack


@pytest.mark.parametrize("w", (7, 73))
def test_online_mix_fwd_vjp_ragged_and_empty_query(device, w):
    torch.manual_seed(5760)
    n = 71
    own = torch.arange(n, device=device) % 3
    own[-1] = 4
    p = torch.randn(n, device=device) * 20
    p[own == 0] = 1e8
    z = torch.rand(n, device=device).add_(0.1).requires_grad_()
    u = torch.randn(n, w, device=device, requires_grad=True)
    out = mix(p, z, u, csr(own, 5))
    ref = []
    for i in range(5):
        k = own == i
        if k.any():
            a = (p[k].double() - p[k].max()).exp()
            ref.append((a @ u[k].double()) / (a @ z[k].double()))
        else:
            ref.append(u.new_zeros(w).double())
    r = torch.stack(ref)
    t = torch.randn_like(out)
    torch.testing.assert_close(r, out.double(), atol=2e-5, rtol=2e-5)
    for x, y in zip(
        torch.autograd.grad(out, (z, u), t),
        torch.autograd.grad(r, (z, u), t.double()),
        strict=True,
    ):
        torch.testing.assert_close(x, y, atol=3e-5, rtol=5e-5)


def test_parent_odd_tail_and_saved_probabilities(device):
    torch.manual_seed(5761)
    z = torch.randn(5, 13, device=device, requires_grad=True)
    v = torch.randn(5, 13, 73, device=device, requires_grad=True)
    a = parents(z, v)
    b = parent_reference(z.double(), v.double())
    up = [torch.randn_like(x) for x in a]
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x.double(), y, atol=3e-6, rtol=3e-5)
    for x, y in zip(
        torch.autograd.grad(a, (z, v), up),
        torch.autograd.grad(b, (z, v), [x.double() for x in up]),
        strict=True,
    ):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=3e-5)
    q = torch.full((2, 1), 1e8, device=device, requires_grad=True)
    p, _ = parents(q, v[:2, :1].detach())
    torch.testing.assert_close(
        torch.autograd.grad(p.sum(), q)[0], q.new_full(q.shape, 0.5)
    )


@pytest.mark.parametrize("arm", ARMS)
def test_causal_lm_complete_updates_compare_to_eager(device, arm):
    rng = np.random.default_rng(5762)
    raw = rng.integers(0, 12, (4, 9), dtype=np.uint8)
    m = LanguageLM(5762, d=8, rank=5, leaf=4, time=4).to(device)
    ref = deepcopy(m)
    pack = Pack(raw, m, 2)
    opts = [torch.optim.SGD(v.parameters(), lr=0.03) for v in (m, ref)]
    for i in range(2):
        x = rng.integers(0, 12, (2, 5), dtype=np.uint8)
        x[0, :3] = raw[i, 1:4]
        batch = pack.bind(x)
        ls = []
        for mod, backend, opt in zip((m, ref), ("triton", "torch"), opts, strict=True):
            opt.zero_grad()
            loss = mod(batch, arm, backend=backend)
            loss.mean().backward()
            opt.step()
            ls.append(loss.detach())
        torch.testing.assert_close(*ls, rtol=1e-4, atol=3e-5)
        for p, q in zip(m.parameters(), ref.parameters(), strict=True):
            if p.grad is None:
                assert q.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=6e-4, atol=2e-5)
            torch.testing.assert_close(p, q, rtol=5e-5, atol=4e-6)
