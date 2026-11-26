from copy import deepcopy
import os

import pytest
import torch

from .packet import Inputs
from .training import prefix_reference
from .tree import leaf_reference
from .triton_leaf import leaves
from .triton_prefix import prefix
from .test_windows import fixture, check_grads


@pytest.fixture
def device():
    if (
        os.getenv("TRITON_INTERPRET") == "1"
        and os.getenv("CUDA_VISIBLE_DEVICES") == ""
        and not os.getenv("GPU_LEASE_HELD")
    ):
        return "cpu"
    if (
        os.getenv("CORPUS_WINDOWS_TEST_DEVICE") == "cuda"
        and os.getenv("GPU_LEASE_HELD") in ("0", "1")
        and os.getenv("CUDA_VISIBLE_DEVICES") == os.getenv("GPU_LEASE_HELD")
        and os.getenv("TRITON_INTERPRET") != "1"
    ):
        return "cuda"
    pytest.skip("explicit CPU interpreter or admitted device test required")


def test_packed_prefix_mean_and_reverse_vjp(device):
    torch.manual_seed(5682)
    lengths = (1, 0, 7, 34, 97, 2)
    inp = Inputs.pack([tuple(range(n)) for n in lengths], device)
    x = torch.randn(45, sum(lengths), device=device).T.requires_grad_()
    up = torch.randn_like(x)
    r = prefix_reference(x, inp)
    z = prefix(x, inp)
    torch.testing.assert_close(z, r, rtol=3e-5, atol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(z, x, up)[0],
        torch.autograd.grad(r, x, up)[0],
        rtol=5e-5,
        atol=2e-6,
    )
    # The next document cannot inherit even a very large previous prefix.
    y = x.detach().clone()
    y[:8] += 1e6
    torch.testing.assert_close(prefix(y, inp)[8:], z[8:], rtol=0, atol=0)


@pytest.mark.parametrize(
    "n,r,w,leaf",
    [(1, 1, 1, 1), (79, 13, 47, 11), (257, 39, 65, 128), (205, 33, 9, 201)],
)
def test_leaf_statistics_and_independent_vjps(device, n, r, w, leaf):
    torch.manual_seed(5683)
    args = [
        torch.randn(shape, device=device).requires_grad_()
        for shape in ((n, r), (n,), (n, w))
    ]
    old = leaf_reference(*args, leaf)
    new = leaves(*args, leaf)
    up = [torch.randn_like(x) for x in old]
    for x, y in zip(old, new, strict=True):
        torch.testing.assert_close(x, y, rtol=7e-5, atol=2e-6)
    check_grads(
        torch.autograd.grad(new, args, up),
        torch.autograd.grad(old, args, up),
        rtol=9e-5,
        atol=8e-6,
    )


@pytest.mark.parametrize("v,offset", [(3, 1e8), (50281, 0.0)])
def test_native_nll_stable_at_shift_and_across_vocab_tiles(device, v, offset):
    from .triton_nll import nll

    torch.manual_seed(5684)
    x = (torch.randn(4, v, device=device) + offset).requires_grad_()
    y = torch.tensor([0, v - 1, v // 2, 1], dtype=torch.int64, device=device)
    r = torch.nn.functional.cross_entropy(x, y, reduction="sum")
    z = nll(x, y)
    torch.testing.assert_close(r, z, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(
        torch.autograd.grad(z, x)[0], torch.autograd.grad(r, x)[0], rtol=7e-5, atol=2e-7
    )


def test_streamed_triton_adjoint_matches_unbroken_graph(device):
    m, _, a, docs = fixture("vector", dtype=torch.float32, device=device)
    ref = deepcopy(m)
    p = a.windows(docs, chunk_targets=4, lookback=None)
    want = p.loss(ref)
    want.backward()
    got = p.backward_nll(m, backend="triton")
    torch.testing.assert_close(want, got, rtol=5e-5, atol=3e-6)
    check_grads(
        [x.grad for x in m.parameters()],
        [x.grad for x in ref.parameters()],
        rtol=5e-4,
        atol=4e-6,
    )


def test_extreme_equal_leaf_logits_preserve_normalized_backward(device):
    # logsumexp's +log(3) is rounded off in FP32. Backward must use saved
    # max/normalizer rather than exp(score - rounded_log_z).
    p = torch.full((3, 1), 1e8, device=device, requires_grad=True)
    v = torch.tensor([[1.0], [2.0], [6.0]], device=device, requires_grad=True)
    b = p.new_zeros(3, requires_grad=True)
    z, mu = leaves(p, b, v, 3)
    gs = torch.autograd.grad(z.sum() + mu.sum(), (p, b, v))
    torch.testing.assert_close(mu, p.new_tensor([[[3.0]]]))
    want = (1 + v.detach() - 3) / 3
    torch.testing.assert_close(gs[0], want)
    torch.testing.assert_close(gs[1], want[:, 0])
    torch.testing.assert_close(gs[2], v.new_full(v.shape, 1 / 3))


@pytest.mark.parametrize(
    "kind,mode,ck",
    [
        ("pointer", "indexed", False),
        ("vector", "indexed", True),
        ("vector", "dense", True),
    ],
)
def test_training_step_parity_across_changing_windows(device, kind, mode, ck):
    m, _, a, docs = fixture(kind, dtype=torch.float32, device=device)
    other = deepcopy(m)
    opts = [torch.optim.AdamW(x.parameters(), lr=0.001) for x in (m, other)]
    for i in range(3):
        req = docs if i == 0 else [(n, p + [i] if p else p) for n, p in docs]
        p = a.windows(req, chunk_targets=5, min_match=2, lookback=2, locator="triton")
        q = a.windows(req, chunk_targets=7, min_match=2, lookback=2)
        ref = q.predict(other, mode=mode)
        out = p.predict(m, backend="triton", mode=mode)
        torch.testing.assert_close(
            out.probabilities, ref.probabilities, rtol=5e-5, atol=2e-6
        )
        for k in vars(ref.attention):
            torch.testing.assert_close(
                getattr(out.attention, k),
                getattr(ref.attention, k),
                rtol=7e-5,
                atol=3e-6,
            )
        ls = []
        for model, pkt, backend, opt in zip(
            (m, other), (p, q), ("triton", "torch"), opts, strict=True
        ):
            opt.zero_grad(set_to_none=True)
            loss = pkt.loss(model, mode=mode, backend=backend, checkpoint_blocks=ck)
            ls.append(loss.detach())
            loss.backward()
            opt.step()
        torch.testing.assert_close(*ls, rtol=4e-5, atol=1e-6)
        for x, y in zip(m.parameters(), other.parameters(), strict=True):
            torch.testing.assert_close(x.grad, y.grad, rtol=5e-4, atol=3e-6)
            torch.testing.assert_close(x, y, rtol=3e-4, atol=8e-6)
