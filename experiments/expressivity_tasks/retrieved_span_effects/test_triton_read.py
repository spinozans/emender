"""Run with TRITON_INTERPRET=1, or explicitly SPAN_TEST_DEVICE=cuda under lease."""

import copy
import math
import os

import pytest
import torch
from torch.nn import functional as F

from .accelerated import AcceleratedJointLM
from .joint_fixture import JointLM, episodes
from .triton_controls import pack
from .triton_read import dual_edges, merge, terminal, weights
from .triton_scan import pool, scan


@pytest.fixture(name="device")
def device():
    if os.getenv("TRITON_INTERPRET") == "1" and os.getenv("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"
    if (
        os.getenv("SPAN_TEST_DEVICE") == "cuda"
        and os.getenv("GPU_LEASE_HELD")
        and os.getenv("CUDA_VISIBLE_DEVICES") == os.getenv("GPU_LEASE_HELD")
    ):
        return "cuda"
    pytest.skip("requires explicit CPU interpreter or leased Triton qualification")


def prep(raw, k, v, family):
    a, d, e, w, x = raw.split([k, k, k, v, v], -1)
    a = F.normalize(a, dim=-1)
    d = (d + 2).sigmoid()
    if family == "e97":
        d = d[..., :1].expand_as(d)
    return torch.cat((a, d, a * e.sigmoid(), w.sigmoid() * x), -1)


def refstep(h, c, k, v, family):
    a, d, r, u = c.split([k, k, k, v], -1)
    z = d[:, None] * h
    err = u - r @ (h if family == "e97" else z)
    updated = z + a[:, None] * err[None, :]
    return updated.tanh() if family == "e97" else updated


def checked(outputs, expected, inputs, ref_inputs, atol=2.0e-5, rtol=2.0e-4):
    up = [torch.randn_like(y) for y in outputs]
    for y, z in zip(outputs, expected, strict=True):
        torch.testing.assert_close(y.double(), z.double(), atol=atol, rtol=rtol)
    got = torch.autograd.grad(outputs, inputs, up, allow_unused=True)
    want = torch.autograd.grad(
        expected,
        ref_inputs,
        [u.to(y.dtype) for u, y in zip(up, expected, strict=True)],
        allow_unused=True,
    )
    for i, (a, b, x) in enumerate(zip(got, want, inputs, strict=True)):
        a = torch.zeros_like(x) if a is None else a
        b = torch.zeros_like(x) if b is None else b
        torch.testing.assert_close(
            a.double(),
            b.double(),
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"input {i}: " + msg,
        )


@pytest.mark.parametrize(
    "family,backend", [("gdn2", "dual"), ("gdn2", "matrix"), ("e97", "matrix")]
)
@pytest.mark.parametrize("seeded", [False, True])
def test_masked_read_and_every_vjp(device, family, backend, seeded):
    torch.manual_seed(7701)
    g, p, t, k, v, r = 3, 5, 4, 3, 7, 6
    lens = [[0, 1, 4, -1, 2], [-1] * p, [t + 1, 3, 2, 2, 0]]
    lengths = torch.tensor(lens, dtype=torch.int32, device=device)

    def draw(*s):
        return torch.randn(s, device=device).requires_grad_()

    raw, q, s, addr, keys, bias = xs = (
        draw(g, p, t, 3 * k + 2 * v),
        draw(g, k),
        draw(g, k, v),
        draw(g, r),
        draw(g, p, r),
        draw(g, p),
    )
    rr, rq, rs, ra, rk, rb = refs = [x.detach().double().requires_grad_() for x in xs]
    c = pack(raw, k, v, family=family)
    pp = weights(addr, keys, bias, lengths, t)
    if backend == "dual":
        y = merge(dual_edges(c, q, lengths), pp, s, seeded)
    else:
        y = torch.einsum(
            "gk,gkv->gv",
            q,
            pool(terminal(c, s, lengths, family=family, seeded=seeded), pp),
        )
    ys = []
    ps = []
    cc = prep(rr, k, v, family)
    for i, ll in enumerate(lens):
        ids = [j for j, n in enumerate(ll) if 0 <= n <= t]
        z = torch.zeros(p, device=device, dtype=rr.dtype)
        ans = rs.new_zeros(v)
        if ids:
            z[ids] = torch.softmax(rk[i, ids] @ ra[i] / math.sqrt(r) + rb[i, ids], 0)
            for j in ids:
                h = rs[i] if seeded else torch.zeros_like(rs[i])
                for a in cc[i, j, : ll[j]]:
                    h = refstep(h, a, k, v, family)
                ans = ans + z[j] * (rq[i] @ h)
        ys.append(ans)
        ps.append(z)
    checked([y, pp], [torch.stack(ys), torch.stack(ps)], xs, refs)


@pytest.mark.parametrize("family", ["gdn2", "e97"])
def test_scan_with_both_output_cotangents(device, family):
    torch.manual_seed(7702)
    g, t, k, v = 3, 6, 5, 3
    ns = [6, 3, 0]
    lengths = torch.tensor(ns, device=device, dtype=torch.int32)[:, None]
    xs = [
        torch.randn(sh, device=device).requires_grad_()
        for sh in [(g, t, 3 * k + 2 * v), (g, t, k), (g, k, v)]
    ]
    rr, rq, rs = refs = [x.detach().double().requires_grad_() for x in xs]
    c = prep(rr, k, v, family)
    y, h = scan(pack(xs[0], k, v, family=family), xs[1], xs[2], lengths, family=family)
    ys = []
    hs = []
    for i, n in enumerate(ns):
        a = rs[i]
        reads = []
        for j in range(t):
            if j < n:
                a = refstep(a, c[i, j], k, v, family)
                reads.append(rq[i, j] @ a)
            else:
                reads.append(a.new_zeros(v))
        ys.append(torch.stack(reads))
        hs.append(a)
    checked([y, h], [torch.stack(ys), torch.stack(hs)], xs, refs)


@pytest.mark.parametrize("mode", ["state", "effect", "wrong_a"])
def test_joint_model_logits_gradients_and_optimizer_step(device, mode):
    torch.manual_seed(7703)
    b = episodes(torch.Generator().manual_seed(7704), 3)
    a = JointLM().to(device)
    z = AcceleratedJointLM().to(device)
    z.load_state_dict(copy.deepcopy(a.state_dict()))
    optim = [torch.optim.SGD(m.parameters(), lr=0.02) for m in (a, z)]
    for _ in range(2):
        out = []
        for model, opt in zip((a, z), optim, strict=True):
            opt.zero_grad(set_to_none=True)
            logits, p = model(b.prompt.to(device), b.pages.to(device), mode)
            F.cross_entropy(logits, b.answer(b.pages).to(device)).backward()
            out.append((logits, p))
            opt.step()
        for x, y in zip(*out, strict=True):
            torch.testing.assert_close(x, y, atol=3e-5, rtol=3e-4)
        for x, y in zip(a.parameters(), z.parameters(), strict=True):
            torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-4)
            torch.testing.assert_close(x, y, atol=3e-5, rtol=3e-4)
