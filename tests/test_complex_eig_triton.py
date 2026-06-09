"""Parity + structural tests for the FUSED-TRITON complex gated-delta kernel.

Verifies ``complex_gated_delta_chunked_triton`` (real/imag-decomposed @triton.jit
fwd + bwd, NO torch.complex in the hot path) matches the eager per-step reference
(``complex_gated_delta_reference``) in forward AND backward at T=128/512/1024, that
the eigenvalue reductions (theta=0 real-decay, theta=pi reflection) hold, and the
structural mandate: >=2 @triton.jit kernels, no torch.complex inside them.
REAL data only (torch.randn), mirroring tests/test_complex_eig.py.
"""
import inspect
import math
import re

import pytest
import torch

from ndm.triton import complex_eig_chunked_autograd as cea
from ndm.triton.complex_eig_chunked import complex_gated_delta_reference
from ndm.triton.complex_eig_chunked_autograd import complex_gated_delta_chunked_triton

CUDA = torch.cuda.is_available()


def _mk(B, T, H, N, V, device, dtype=torch.float32, seed=0, decay_lo=0.85, decay_hi=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g) * 0.5
    q = torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g) * 0.5
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g) * 0.5
    P = N // 2
    r = torch.rand(B, T, H, P, device=device, dtype=dtype, generator=g) * (decay_hi - decay_lo) + decay_lo
    log_r = r.clamp_min(1e-6).log()
    theta = torch.randn(B, T, H, P, device=device, dtype=dtype, generator=g) * 0.7
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype, generator=g)) * 2.0
    return q, k, v, log_r, theta, beta


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-6)


# --------------------------------------------------------------------------
# Structural mandate: real fused Triton, no torch.complex in the hot path.
# --------------------------------------------------------------------------
def test_two_jit_kernels_and_no_torch_complex_in_kernels():
    src = inspect.getsource(cea)
    n_jit = src.count("@triton.jit")
    assert n_jit >= 2, f"expected >=2 @triton.jit kernels, found {n_jit}"
    # isolate the @triton.jit kernel bodies (everything from the first @triton.jit
    # up to the autograd.Function class) and assert no torch.complex there.
    kernel_region = src[src.index("@triton.jit"):src.index("class ComplexEigChunkedFn")]
    assert "torch.complex" not in kernel_region, "torch.complex must not appear in the fused kernels"
    assert "torch.polar" not in kernel_region
    # the kernels must use tl.dot (real tensor-core matmuls)
    assert "tl.dot" in kernel_region


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T,C", [(128, 32), (512, 32), (1024, 32), (130, 16), (256, 32)])
def test_fused_matches_reference_fwd(T, C):
    dev = "cuda"
    B, H, N, V = 2, 4, 32, 32
    q, k, v, log_r, theta, beta = _mk(B, T, H, N, V, dev, seed=0)
    ref, refS = complex_gated_delta_reference(q, k, v, log_r, theta, beta)
    out, outS = complex_gated_delta_chunked_triton(q, k, v, log_r, theta, beta, chunk_size=C)
    assert _rel(out, ref) < 3e-3, f"fwd rel err {_rel(out, ref)}"
    assert _rel(outS, refS) < 3e-3, f"S_final rel err {_rel(outS, refS)}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [128, 512, 1024])
def test_fused_matches_reference_bwd(T):
    dev = "cuda"
    B, H, N, V = 2, 3, 32, 32
    q, k, v, log_r, theta, beta = _mk(B, T, H, N, V, dev, seed=7)

    def run(fn):
        ts = [t.clone().requires_grad_(True) for t in (q, k, v, log_r, theta, beta)]
        out, _ = fn(*ts)
        (out * out).sum().backward()
        return [t.grad.clone() for t in ts]

    g_ref = run(lambda *a: complex_gated_delta_reference(*a))
    g_fus = run(lambda *a: complex_gated_delta_chunked_triton(*a, chunk_size=32))
    for nm, a, b in zip(["q", "k", "v", "log_r", "theta", "beta"], g_ref, g_fus):
        assert torch.isfinite(b).all(), f"non-finite grad {nm}"
        assert _rel(b, a) < 2e-2, f"grad {nm} rel err {_rel(b, a)}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("name,th_val", [("theta0", 0.0), ("thetapi", math.pi)])
def test_fused_reductions(name, th_val):
    dev = "cuda"
    B, T, H, N, V = 2, 64, 3, 32, 32
    q, k, v, log_r, _, beta = _mk(B, T, H, N, V, dev, seed=1)
    theta = torch.full((B, T, H, N // 2), th_val, device=dev)
    ref, _ = complex_gated_delta_reference(q, k, v, log_r, theta, beta)
    out, _ = complex_gated_delta_chunked_triton(q, k, v, log_r, theta, beta, chunk_size=32)
    assert _rel(out, ref) < 3e-3, f"{name} chunked-fused vs eager rel {_rel(out, ref)}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_bf16_tf32_path_parity():
    """bf16 inputs -> TF32 tensor-core matmuls (production autocast path); looser tol."""
    dev = "cuda"
    B, T, H, N, V = 2, 256, 4, 32, 32
    q, k, v, log_r, theta, beta = _mk(B, T, H, N, V, dev, seed=5)
    ref, _ = complex_gated_delta_reference(q, k, v, log_r, theta, beta)
    out, _ = complex_gated_delta_chunked_triton(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), log_r, theta, beta,
        chunk_size=32, allow_tf32=True)
    assert _rel(out.float(), ref) < 2e-2, f"bf16/TF32 fwd rel err {_rel(out.float(), ref)}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_head_runs_on_fused_triton_path():
    """ComplexEigHeadLayer drives the fused Triton kernel (fused_triton=True default)
    and trains a step with finite grads."""
    from ndm.models.complex_eig_head import ComplexEigHeadLayer
    dev = "cuda"
    torch.manual_seed(0)
    layer = ComplexEigHeadLayer(dim=256, n_state=32, n_heads=8,
                                nonlin_subset_frac=0.0).to(dev).train()
    assert layer.fused_triton is True
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    x = torch.randn(2, 96, 256, device=dev)
    out = layer(x)
    assert out.shape == x.shape
    loss = (out * out).mean()
    opt.zero_grad(); loss.backward()
    for p in layer.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    opt.step()
    assert torch.isfinite(loss).all()


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_vs_torch_complex_chunked_agree():
    """The fused kernel and the torch-complex chunked reference agree (both are
    chunked; this isolates the fused-kernel arithmetic from chunking)."""
    from ndm.triton.complex_eig_chunked import complex_gated_delta_chunked
    dev = "cuda"
    B, T, H, N, V = 2, 384, 4, 32, 32
    q, k, v, log_r, theta, beta = _mk(B, T, H, N, V, dev, seed=11)
    o_torch, _ = complex_gated_delta_chunked(q, k, v, log_r, theta, beta, chunk_size=32)
    o_fused, _ = complex_gated_delta_chunked_triton(q, k, v, log_r, theta, beta, chunk_size=32)
    assert _rel(o_fused, o_torch) < 3e-3


if __name__ == "__main__":
    test_two_jit_kernels_and_no_torch_complex_in_kernels()
    print("structural OK")
    if CUDA:
        for T, C in [(128, 32), (512, 32), (1024, 32), (130, 16), (256, 32)]:
            test_fused_matches_reference_fwd(T, C)
        print("fwd parity OK")
        for T in [128, 512, 1024]:
            test_fused_matches_reference_bwd(T)
        print("bwd parity OK")
        for nm, tv in [("theta0", 0.0), ("thetapi", math.pi)]:
            test_fused_reductions(nm, tv)
        print("reductions OK")
        test_bf16_tf32_path_parity(); print("bf16/TF32 OK")
        test_head_runs_on_fused_triton_path(); print("head fused-path OK")
        test_fused_vs_torch_complex_chunked_agree(); print("fused==torch-chunked OK")
        print("ALL FUSED-TRITON TESTS OK")
