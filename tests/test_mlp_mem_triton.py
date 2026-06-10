"""Parity tests for the fused sequential mlp-mem Triton kernels.

Verifies the fused fwd+bwd (ndm.triton.mlp_mem_fused.mlp_mem_triton) matches the
eager PyTorch reference recurrence (mlp_mem_torch_reference) in BOTH forward and
backward (grads wrt k, q, v, eta, gamma), fp32, at T = 128 / 512 / 1024.

The recurrence is the nonlinear MLP-memory cell of NONLIN_MEMORY_SPEC.md: the state
is the params of a 1-hidden-layer MLP, written by one gated inner gradient step per
token. It is non-associative (no chunked scan), so the kernel is a fused sequential
scan with sparse forward checkpoints + reverse-replay BPTT.
"""
import pytest
import torch

from ndm.triton.mlp_mem_fused import mlp_mem_triton, mlp_mem_torch_reference

CUDA = torch.cuda.is_available()


def _mk(B, T, NH, N, V, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(B, T, NH, N, device=device, dtype=dtype, generator=g) * 0.5
    q = torch.randn(B, T, NH, N, device=device, dtype=dtype, generator=g) * 0.5
    v = torch.randn(B, T, NH, V, device=device, dtype=dtype, generator=g) * 0.5
    # eta = inner LR / write strength >= 0 (softplus-like, capped small for stability)
    eta = torch.sigmoid(torch.randn(B, T, NH, device=device, dtype=dtype, generator=g)) * 0.5
    # gamma = forget gate in (0,1)
    gamma = torch.sigmoid(torch.randn(B, T, NH, device=device, dtype=dtype, generator=g)) * 0.3 + 0.6
    return k, q, v, eta, gamma


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [128, 512, 1024])
def test_forward_parity_fp32(T):
    dev = 'cuda'
    B, NH, N, V, HID = 2, 4, 32, 32, 32
    k, q, v, eta, gamma = _mk(B, T, NH, N, V, dev, torch.float32)
    ref_out, ref_W1, ref_W2 = mlp_mem_torch_reference(k, q, v, eta, gamma, HID)
    out, W1, W2 = mlp_mem_triton(k, q, v, eta, gamma, HID, ckpt_interval=16)
    err = (out - ref_out).abs().max().item()
    w1err = (W1 - ref_W1).abs().max().item()
    w2err = (W2 - ref_W2).abs().max().item()
    scale = ref_out.abs().max().item() + 1e-6
    assert torch.isfinite(out).all(), "fwd out has non-finite"
    assert err / scale < 2e-4, f"fwd out rel err {err/scale} (abs {err}) @ T={T}"
    assert w1err < 2e-4, f"W1_final err {w1err} @ T={T}"
    assert w2err < 2e-4, f"W2_final err {w2err} @ T={T}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [128, 512, 1024])
def test_backward_parity_fp32(T):
    dev = 'cuda'
    B, NH, N, V, HID = 2, 4, 32, 32, 32
    k, q, v, eta, gamma = _mk(B, T, NH, N, V, dev, torch.float32, seed=1)

    def run(fn):
        kk = k.clone().requires_grad_(True)
        qq = q.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True)
        ee = eta.clone().requires_grad_(True)
        gg = gamma.clone().requires_grad_(True)
        out, _, _ = fn(kk, qq, vv, ee, gg, HID)
        # deterministic non-trivial scalar loss
        loss = (out * torch.sin(out)).sum()
        loss.backward()
        return out, (kk.grad, qq.grad, vv.grad, ee.grad, gg.grad)

    ref_out, ref_g = run(lambda *a: mlp_mem_torch_reference(*a))
    tri_out, tri_g = run(lambda *a: mlp_mem_triton(*a, ckpt_interval=16))

    names = ['dk', 'dq', 'dv', 'deta', 'dgamma']
    for nm, rg, tg in zip(names, ref_g, tri_g):
        assert torch.isfinite(tg).all(), f"{nm} has non-finite grad @ T={T}"
        err = (rg - tg).abs().max().item()
        scale = rg.abs().max().item() + 1e-6
        assert err / scale < 3e-3, f"{nm} rel err {err/scale} (abs {err}) @ T={T}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_grads_finite_unaligned_T():
    """T not a multiple of ckpt_interval -> wrapper pads; grads must stay finite."""
    dev = 'cuda'
    B, NH, N, V, HID, T = 2, 3, 32, 32, 16, 130
    k, q, v, eta, gamma = _mk(B, T, NH, N, V, dev, torch.float32, seed=2)
    for t in (k, q, v, eta, gamma):
        t.requires_grad_(True)
    out, _, _ = mlp_mem_triton(k, q, v, eta, gamma, HID, ckpt_interval=16)
    assert out.shape == (B, T, NH, V)
    out.pow(2).sum().backward()
    for t, nm in [(k, 'k'), (q, 'q'), (v, 'v'), (eta, 'eta'), (gamma, 'gamma')]:
        assert torch.isfinite(t.grad).all(), f"{nm} grad non-finite"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_gdn_degenerate_corner_finite():
    """HID != N but small; just a smoke that varied HID runs + is finite/parity."""
    dev = 'cuda'
    B, NH, N, V, HID, T = 1, 2, 16, 16, 64, 256
    k, q, v, eta, gamma = _mk(B, T, NH, N, V, dev, torch.float32, seed=3)
    ref_out, _, _ = mlp_mem_torch_reference(k, q, v, eta, gamma, HID)
    out, _, _ = mlp_mem_triton(k, q, v, eta, gamma, HID, ckpt_interval=16)
    err = (out - ref_out).abs().max().item()
    scale = ref_out.abs().max().item() + 1e-6
    assert err / scale < 2e-4, f"HID=64 rel err {err/scale}"
