"""No implicit GPU work. Interpreter or explicit device test under a valid local lease."""

from copy import deepcopy
import os

import pytest
import torch

from .attention import _torch_tiles
from .test_attention import parity
from .test_token_lm import example
from .token_lm import CorpusTokenLM
from .triton_tiles import reduce_tiles


@pytest.fixture
def device():
    if os.getenv("TRITON_INTERPRET") == "1" and os.getenv("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"
    if (
        os.getenv("CORPUS_KERNEL_DEVICE") == "cuda"
        and os.getenv("GPU_LEASE_HELD") in ("0", "1")
        and os.getenv("CUDA_VISIBLE_DEVICES") == os.getenv("GPU_LEASE_HELD")
    ):
        return "cuda"
    pytest.skip(
        "requires explicit Triton interpreter or device qualification under lease"
    )


def test_batched_resident_mem_all_depths_and_cached_plan(device):
    from .index import EndpointIndex
    from .triton_frontier import ResidentIndex
    from .batch import TokenArchive

    hi = 2**63 - 1
    idx = EndpointIndex(
        [(str(i), [i % 5, 1, 2, 1, 0, hi, 2]) for i in range(22)]
        + [("end", [0, 1, 0])],
        tokenizer="toy",
    )
    ri = ResidentIndex(idx, device)
    prefixes = [[], [0], [hi], [hi, 2], [2, 1, 0], [0, 8, 9], [8, 3, 1, 2], [0, hi]] * 3
    T = 9
    x = torch.tensor([p + [3] * (T - len(p)) for p in prefixes], device=device)
    lens = torch.tensor(list(map(len, prefixes)), device=device, dtype=torch.int32)
    all_got = ri.intervals(x, lens, all_depths=True).cpu().tolist()
    for i, p in enumerate(prefixes):
        f = idx.frontier(p)
        assert all_got[i] == [list(f.hit_range(d)) for d in range(1, T + 1)]
    for d in (1, 3, T + 1):
        assert ri.intervals(x, lens, min_match=d).cpu().tolist() == [
            list(idx.frontier(p).hit_range(d)) for p in prefixes
        ]
    arch, req = example()
    m = CorpusTokenLM(10, 9, 11, tokenizer="toy", leaf=3).to(device)
    f = TokenArchive(m, arch)
    cpu = f.batch(req)
    tri = f.batch(req, locator="triton")
    assert cpu.plan == tri.plan
    torch.testing.assert_close(
        cpu.predict(m).probabilities,
        tri.predict(m, backend="triton").probabilities,
        rtol=2e-5,
        atol=2e-6,
    )


def test_internal_mems_extend_to_frontier_before_scoring(device):
    from .index import EndpointIndex
    from .batch import TokenArchive
    from .test_internal import literal

    idx = EndpointIndex(
        [
            ("a", [2, 3, 4, 5, 8, 7]),
            ("b", [2, 3, 9, 1, 8, 6]),
            ("c", [2, 3, 4]),
            ("d", [2, 3, 4, 5, 8, 7]),
        ],
        tokenizer="toy",
    )
    req = [("live", [2, 3, 0, 5, 8]), ("live2", [9, 0, 4]), ("live3", [1])]
    torch.manual_seed(5612)
    m = CorpusTokenLM(10, 7, 11, tokenizer="toy", leaf=2).to(device)
    a = TokenArchive(m, idx)
    batch = a.batch(req, min_match=2, lookback=None, locator="triton")
    z = batch.predict(m, backend="triton")
    ref = m(idx, req, min_match=2, lookback=None)
    for i, (_, p) in enumerate(req):
        assert {
            r for j, lo, hi in batch.plan.exact if i == j for r in range(lo, hi)
        } == literal(idx, p, 2, None)
    torch.testing.assert_close(z.probabilities, ref.probabilities, rtol=3e-5, atol=2e-6)
    gr = []
    for ans in (z, ref):
        gr.append(
            torch.autograd.grad(
                -ans.probabilities[:, 8].log().mean(), tuple(m.parameters())
            )
        )
    for g, h in zip(*gr, strict=True):
        torch.testing.assert_close(g, h, rtol=3e-4, atol=4e-6)


def test_indexed_whole_read_and_every_vjp(device):
    parity(torch.float32, device, "triton")


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_partial_masks_large_value_tiles_and_shared_rows(mode, device):
    torch.manual_seed(2720)
    D = 27
    V = 71
    n = 81

    def rand(*s):
        return torch.randn(*s, device=device, requires_grad=True)

    q = rand(3, D)
    k = rand(n, D) if mode != 2 else None
    v = rand(n, V)
    b = rand(n)
    args = (q, k, v, b) if k is not None else (q, v, b)
    desc = torch.tensor(
        [[0, 0, 16], [0, 16, 11], [2, 0, 16], [1, 13, 3], [1, 72, 9]],
        device=device,
        dtype=torch.long,
    )
    up_z = rand(len(desc)).detach()
    up_u = rand(len(desc), V).detach()
    out = []
    grads = []
    for run in (_torch_tiles, reduce_tiles):
        p, z, u = run(mode, q, k, v, b, desc)
        out.append((p, z, u))
        grads.append(torch.autograd.grad((z, u), args, (up_z, up_u)))
    for a, c in zip(out[0] + grads[0], out[1] + grads[1], strict=True):
        torch.testing.assert_close(a, c, rtol=2e-5, atol=8e-6)


@pytest.mark.parametrize("kind", ["pointer", "vector"])
def test_live_optimizer_parity_across_changing_packets(device, kind):
    torch.manual_seed(2721)
    idx, req = example()
    a = CorpusTokenLM(10, 8, 7, tokenizer="toy", leaf=2, values=kind).to(device)
    b = deepcopy(a)
    opts = [torch.optim.AdamW(m.parameters(), lr=0.001) for m in (a, b)]
    for i in range(3):
        batch = [(name, p + [2 + i] if i else p) for name, p in req]
        outs = []
        for m, opt, impl in zip((a, b), opts, ("torch", "triton"), strict=True):
            opt.zero_grad()
            z = m(idx, batch, backend=impl)
            outs.append(z)
            loss = -z.probabilities[range(3), [3, 6, 8]].log().mean()
            loss.backward()
            opt.step()
        torch.testing.assert_close(
            outs[0].probabilities, outs[1].probabilities, rtol=1e-5, atol=2e-6
        )
        for x, y in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(x.grad, y.grad, rtol=4e-4, atol=2e-6)
            torch.testing.assert_close(x, y, rtol=2e-4, atol=3e-6)
