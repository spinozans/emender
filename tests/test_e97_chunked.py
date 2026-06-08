"""Parity tests for the chunked-parallel E97 split-edit delta kernel.

Verifies the chunked form (ndm.triton.e97_chunked.e97_delta_chunked) matches the
sequential reference recurrence (e88_torch_reference, linear_state=True,
raw_write=False, split-edit gates) in BOTH forward and backward, fp32 and bf16.
"""
import pytest
import torch

from ndm.triton.e97_chunked import e97_delta_chunked
from ndm.triton.e97_chunked_autograd import e97_delta_chunked_triton
from ndm.triton.e88_triton_forward import e88_torch_reference

CUDA = torch.cuda.is_available()


def _ref_forward(S0, k, v, q, decay, e, w):
    """Reference: [B,T,H,*] -> uses e88_torch_reference which wants [T,B,H,*]."""
    kT = k.transpose(0, 1).contiguous()
    vT = v.transpose(0, 1).contiguous()
    qT = q.transpose(0, 1).contiguous()
    dT = decay.transpose(0, 1).contiguous()
    eT = e.transpose(0, 1).contiguous()
    wT = w.transpose(0, 1).contiguous()
    out, S_final, _ = e88_torch_reference(
        S0, kT, vT, qT, dT, linear_state=True, raw_write=False,
        erase_gate=eT, value_write_gate=wT,
    )
    return out.transpose(0, 1).contiguous(), S_final  # [B,T,H,V], [B,H,N,V]


def _mk(B, T, H, N, V, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g) * 0.5
    q = torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g) * 0.5
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g) * 0.5
    # decay in (0,1) like Mamba2 exp-decay
    decay = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype, generator=g)) * 0.4 + 0.5
    e = torch.sigmoid(torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g))
    w = torch.sigmoid(torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g))
    return k, v, q, decay, e, w


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T,C", [(64, 64), (128, 64), (256, 64), (192, 64), (130, 64)])
def test_forward_parity_fp32(T, C):
    dev = 'cuda'
    B, H, N, V = 2, 4, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.float32)
    S0 = torch.zeros(B, H, N, V, device=dev)
    ref_out, ref_S = _ref_forward(S0, k, v, q, decay, e, w)
    out, S = e97_delta_chunked(k, v, q, decay, e, w, S0=S0, chunk_size=C)
    err = (out - ref_out).abs().max().item()
    serr = (S - ref_S).abs().max().item()
    assert err < 2e-4, f"fwd out err {err}"
    assert serr < 2e-4, f"S_final err {serr}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_forward_parity_bf16():
    dev = 'cuda'
    B, T, H, N, V = 2, 256, 4, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.bfloat16)
    S0 = torch.zeros(B, H, N, V, device=dev, dtype=torch.bfloat16)
    ref_out, ref_S = _ref_forward(S0, k, v, q, decay, e, w)
    out, S = e97_delta_chunked(k, v, q, decay, e, w, S0=S0, chunk_size=64)
    # bf16 inputs, fp32 compute internally; compare in fp32
    err = (out.float() - ref_out.float()).abs().max().item()
    rel = err / (ref_out.float().abs().max().item() + 1e-6)
    assert rel < 0.05, f"bf16 fwd rel err {rel} (abs {err})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_backward_parity_fp32():
    dev = 'cuda'
    B, T, H, N, V = 2, 128, 3, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=7)
    S0 = torch.zeros(B, H, N, V, device=dev)

    def run(fn):
        kk = k.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True)
        qq = q.clone().requires_grad_(True)
        dd = decay.clone().requires_grad_(True)
        ee = e.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True)
        out, _ = fn(kk, vv, qq, dd, ee, ww)
        loss = (out * out).sum()
        loss.backward()
        return [t.grad.clone() for t in (kk, vv, qq, dd, ee, ww)]

    g_ref = run(lambda kk, vv, qq, dd, ee, ww: _ref_forward(S0, kk, vv, qq, dd, ee, ww))
    g_chk = run(lambda kk, vv, qq, dd, ee, ww: e97_delta_chunked(kk, vv, qq, dd, ee, ww, S0=S0, chunk_size=64))
    names = ['k', 'v', 'q', 'decay', 'e', 'w']
    for nm, a, b in zip(names, g_ref, g_chk):
        err = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-6
        assert err / scale < 1e-3, f"grad {nm} rel err {err/scale} (abs {err})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [64, 128, 256, 96])
def test_fused_triton_forward_parity_fp32(T):
    """Fused fwd+bwd Triton autograd kernel — forward parity, fp32 (C=32)."""
    dev = 'cuda'
    B, H, N, V = 2, 4, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=3)
    S0 = torch.zeros(B, H, N, V, device=dev)
    ref_out, ref_S = _ref_forward(S0, k, v, q, decay, e, w)
    out, S = e97_delta_chunked_triton(k, v, q, decay, e, w, chunk_size=32)
    assert (out - ref_out).abs().max().item() < 2e-4
    assert (S - ref_S).abs().max().item() < 2e-4


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_triton_backward_parity_fp32():
    """Fused Triton kernel — gradient parity vs reference recurrence, fp32."""
    dev = 'cuda'
    B, T, H, N, V = 2, 128, 3, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=11)
    S0 = torch.zeros(B, H, N, V, device=dev)

    def run(fn):
        ts = [t.clone().requires_grad_(True) for t in (k, v, q, decay, e, w)]
        out, _ = fn(*ts)
        (out * out).sum().backward()
        return [t.grad.clone() for t in ts]

    g_ref = run(lambda kk, vv, qq, dd, ee, ww: _ref_forward(S0, kk, vv, qq, dd, ee, ww))
    g_chk = run(lambda kk, vv, qq, dd, ee, ww: e97_delta_chunked_triton(kk, vv, qq, dd, ee, ww, chunk_size=32))
    for nm, a, b in zip(['k', 'v', 'q', 'decay', 'e', 'w'], g_ref, g_chk):
        err = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-6
        assert err / scale < 3e-3, f"fused grad {nm} rel err {err/scale} (abs {err})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_triton_backward_parity_bf16():
    """Fused Triton kernel — bf16 fwd + grad parity (TF32 tensor-core dots)."""
    dev = 'cuda'
    B, T, H, N, V = 2, 256, 3, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.bfloat16, seed=5)
    S0 = torch.zeros(B, H, N, V, device=dev, dtype=torch.bfloat16)

    def run(fn):
        ts = [t.clone().requires_grad_(True) for t in (k, v, q, decay, e, w)]
        out, _ = fn(*ts)
        (out.float() * out.float()).sum().backward()
        return out.detach(), [t.grad.clone() for t in ts]

    o_ref, g_ref = run(lambda kk, vv, qq, dd, ee, ww: _ref_forward(S0, kk, vv, qq, dd, ee, ww))
    o_chk, g_chk = run(lambda kk, vv, qq, dd, ee, ww: e97_delta_chunked_triton(kk, vv, qq, dd, ee, ww, chunk_size=32))
    frel = (o_chk.float() - o_ref.float()).abs().max().item() / (o_ref.float().abs().max().item() + 1e-6)
    assert frel < 0.05, f"bf16 fwd rel err {frel}"
    for nm, a, b in zip(['k', 'v', 'q', 'decay', 'e', 'w'], g_ref, g_chk):
        err = (a.float() - b.float()).abs().max().item()
        scale = a.float().abs().max().item() + 1e-6
        assert err / scale < 0.06, f"bf16 fused grad {nm} rel err {err/scale}"


if __name__ == '__main__':
    if not CUDA:
        print("no CUDA; skipping")
    else:
        test_forward_parity_fp32(256, 64)
        print("staged fwd fp32 OK")
        test_forward_parity_bf16()
        print("staged fwd bf16 OK")
        test_backward_parity_fp32()
        print("staged bwd fp32 OK")
        test_fused_triton_forward_parity_fp32(256)
        print("fused fwd fp32 OK")
        test_fused_triton_backward_parity_fp32()
        print("fused bwd fp32 OK")
        test_fused_triton_backward_parity_bf16()
        print("fused bwd bf16 OK")
        print("ALL PARITY OK")
