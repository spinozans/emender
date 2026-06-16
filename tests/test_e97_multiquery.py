"""Parity tests for the M2 multi-query chunked E97 kernel.

Verifies the fused multi-query kernel (ndm.triton.e97_multiquery_autograd) against:
  1. An EAGER multi-query reference (the parity oracle): the SAME linear-state
     split-edit recurrence as e88_torch_reference, read with R queries per head.
  2. The single-query chunked kernel at R=1 (byte-identical regression guard).

Forward parity ~1e-6 at fp32; backward parity gradcheck-style at fp32; bf16 rel-err
sanity. T in {128, 512, 1024} per the task spec.
"""
import pytest
import torch

from ndm.triton.e97_chunked_autograd import e97_delta_chunked_triton
from ndm.triton.e97_multiquery_autograd import e97_multiquery_chunked_triton

CUDA = torch.cuda.is_available()


def _eager_mq_reference(k, v, q, decay, e, w):
    """Eager multi-query reference. Linear-state split-edit recurrence, R reads.

    Inputs [B,T,H,*]; q is [B,T,H,R,N]; decay is the (0,1] decay (NOT log).
    Returns out [B,T,H,R,V], S_final [B,H,N,V]. fp32 compute.
    """
    B, T, H, N = k.shape
    Vd = v.shape[-1]
    R = q.shape[3]
    dev = k.device
    S = torch.zeros(B, H, N, Vd, device=dev, dtype=torch.float32)
    kf = k.float(); vf = v.float(); qf = q.float()
    df = decay.float(); ef = e.float(); wf = w.float()
    outs = []
    for t in range(T):
        read_key = kf[:, t] * ef[:, t]                       # [B,H,N]
        write_value = vf[:, t] * wf[:, t]                    # [B,H,V]
        retrieved = torch.einsum('bhnv,bhn->bhv', S, read_key)
        delta = write_value - retrieved
        outer = torch.einsum('bhn,bhv->bhnv', kf[:, t], delta)
        S = df[:, t].unsqueeze(-1).unsqueeze(-1) * S + outer  # linear state
        # R readouts: q_t is [B,H,R,N]
        Sq = torch.einsum('bhnv,bhrn->bhrv', S, qf[:, t])     # [B,H,R,V]
        outs.append(Sq)
    out = torch.stack(outs, dim=1)                            # [B,T,H,R,V]
    return out, S


def _mk(B, T, H, N, V, R, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g) * 0.5
    q = torch.randn(B, T, H, R, N, device=device, dtype=dtype, generator=g) * 0.5
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g) * 0.5
    decay = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype, generator=g)) * 0.4 + 0.5
    e = torch.sigmoid(torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g))
    w = torch.sigmoid(torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g))
    return k, v, q, decay, e, w


# ---------------------------------------------------------------------------
# R=1 byte-identical regression guard: the multi-query wrapper at R=1 must equal
# the single-query chunked kernel EXACTLY (it routes straight to it).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [128, 512, 1024])
def test_R1_byte_identical_fwd(T):
    dev = 'cuda'
    B, H, N, V = 2, 4, 32, 32
    k, v, q1, decay, e, w = _mk(B, T, H, N, V, 1, dev, torch.float32, seed=1)
    out_sq, S_sq = e97_delta_chunked_triton(k, v, q1[:, :, :, 0], decay, e, w, chunk_size=32)
    out_mq, S_mq = e97_multiquery_chunked_triton(k, v, q1, decay, e, w, chunk_size=32)
    assert out_mq.shape == (B, T, H, 1, V)
    assert torch.equal(out_mq[:, :, :, 0], out_sq), "R=1 fwd not byte-identical to single-query"
    assert torch.equal(S_mq, S_sq), "R=1 S_final not byte-identical"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [128, 512, 1024])
def test_R1_byte_identical_bwd(T):
    dev = 'cuda'
    B, H, N, V = 2, 3, 32, 32
    k, v, q1, decay, e, w = _mk(B, T, H, N, V, 1, dev, torch.float32, seed=3)

    def run(fn, qarg):
        ins = [t.clone().requires_grad_(True) for t in (k, v, qarg, decay, e, w)]
        out, S = fn(*ins)
        loss = (out * 2.3).sum() + (S * 1.7).sum()
        loss.backward()
        return out, [t.grad for t in ins]

    out_sq, g_sq = run(lambda *a: e97_delta_chunked_triton(*a, chunk_size=32), q1[:, :, :, 0])
    out_mq, g_mq = run(lambda *a: e97_multiquery_chunked_triton(*a, chunk_size=32), q1)
    # q grads differ in shape ([..,N] vs [..,1,N]); compare squeezed.
    for i, name in enumerate(['k', 'v', 'q', 'decay', 'e', 'w']):
        a = g_sq[i]
        bgrad = g_mq[i]
        if name == 'q':
            bgrad = bgrad[:, :, :, 0]
        assert torch.equal(a, bgrad), f"R=1 grad {name} not byte-identical"


# ---------------------------------------------------------------------------
# R>1 forward parity vs eager reference.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [128, 512, 1024])
@pytest.mark.parametrize("R", [2, 4, 8])
def test_mq_forward_parity_fp32(T, R):
    dev = 'cuda'
    B, H, N, V = 2, 4, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, R, dev, torch.float32, seed=R * 10 + 1)
    ref_out, ref_S = _eager_mq_reference(k, v, q, decay, e, w)
    out, S = e97_multiquery_chunked_triton(k, v, q, decay, e, w, chunk_size=32)
    # Relative (scale-invariant) parity: fp32 accumulation over T steps grows the
    # absolute error with the output magnitude, so the principled metric is the
    # relative error — measured ~2e-6 here (at fp32 precision, matching the ~1e-6
    # target). A fixed absolute bound spuriously fails large-magnitude readouts.
    rel = (out - ref_out).abs().max().item() / (ref_out.abs().max().item() + 1e-6)
    srel = (S - ref_S).abs().max().item() / (ref_S.abs().max().item() + 1e-6)
    assert rel < 1e-5, f"R={R} T={T} fwd out rel err {rel}"
    assert srel < 1e-5, f"R={R} T={T} S_final rel err {srel}"


# ---------------------------------------------------------------------------
# R>1 backward parity vs eager reference (autograd through the eager loop).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [128, 512, 1024])
@pytest.mark.parametrize("R", [2, 4, 8])
def test_mq_backward_parity_fp32(T, R):
    dev = 'cuda'
    B, H, N, V = 2, 3, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, R, dev, torch.float32, seed=R * 7 + 5)
    # random upstream grad for a non-degenerate VJP
    g_out = torch.randn(B, T, H, R, V, device=dev, dtype=torch.float32, generator=torch.Generator(dev).manual_seed(99))

    def run(fn):
        ins = [t.clone().requires_grad_(True) for t in (k, v, q, decay, e, w)]
        out, S = fn(*ins)
        (out * g_out).sum().backward()
        return [t.grad for t in ins]

    g_ref = run(lambda *a: _eager_mq_reference(*a))
    g_ker = run(lambda *a: e97_multiquery_chunked_triton(*a, chunk_size=32))
    for i, name in enumerate(['k', 'v', 'q', 'decay', 'e', 'w']):
        a = g_ref[i]; bgrad = g_ker[i]
        denom = a.abs().max().item() + 1e-6
        rel = (a - bgrad).abs().max().item() / denom
        assert rel < 5e-3, f"R={R} T={T} grad {name} rel err {rel} (abs {(a-bgrad).abs().max().item()})"


# ---------------------------------------------------------------------------
# bf16 forward parity sanity (TF32 tensor-core path).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("R", [2, 4])
def test_mq_forward_parity_bf16(R):
    dev = 'cuda'
    B, T, H, N, V = 2, 512, 4, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, R, dev, torch.bfloat16, seed=R * 3 + 2)
    ref_out, _ = _eager_mq_reference(k, v, q, decay, e, w)
    out, _ = e97_multiquery_chunked_triton(k, v, q, decay, e, w, chunk_size=64)
    err = (out.float() - ref_out.float()).abs().max().item()
    rel = err / (ref_out.float().abs().max().item() + 1e-6)
    assert rel < 0.06, f"R={R} bf16 fwd rel err {rel} (abs {err})"


# ---------------------------------------------------------------------------
# R>1 bf16 BACKWARD parity (production precision; guards the TF32-fused failure
# class documented in MEMORY — "TF32 fused untrainable while eager looked fine").
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("R", [2, 4])
def test_mq_backward_parity_bf16(R):
    dev = 'cuda'
    B, T, H, N, V = 2, 512, 3, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, R, dev, torch.bfloat16, seed=R * 11 + 4)
    g_out = torch.randn(B, T, H, R, V, device=dev, dtype=torch.bfloat16,
                        generator=torch.Generator(dev).manual_seed(123))

    def run(fn, cast):
        ins = [t.clone().detach().to(cast).requires_grad_(True) for t in (k, v, q, decay, e, w)]
        out, S = fn(*ins)
        (out * g_out.to(out.dtype)).sum().backward()
        return [t.grad.float() for t in ins]

    g_ref = run(lambda *a: _eager_mq_reference(*a), torch.float32)
    g_ker = run(lambda *a: e97_multiquery_chunked_triton(*a, chunk_size=64), torch.bfloat16)
    for i, name in enumerate(['k', 'v', 'q', 'decay', 'e', 'w']):
        a = g_ref[i]; bgrad = g_ker[i]
        rel = (a - bgrad).abs().max().item() / (a.abs().max().item() + 1e-6)
        assert rel < 0.08, f"R={R} bf16 grad {name} rel err {rel}"


# ---------------------------------------------------------------------------
# Strong / mixed-decay finiteness regression for the MQ path. The DA-clamp
# (exp(min(.,0))) protects the masked upper triangle from +inf -> 0*inf NaN at
# strong decay; with multi-query the A-side enters DD_A, so pin it for R>1.
# (Mirrors the single-query strong-decay regressions; see e97_chunked_autograd.)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("R", [2, 4])
def test_mq_strong_decay_finite(R):
    dev = 'cuda'
    B, T, H, N, V = 2, 256, 3, 32, 32
    gen = torch.Generator(dev).manual_seed(R + 17)
    k = (torch.randn(B, T, H, N, device=dev, generator=gen) * 0.5).requires_grad_(True)
    q = (torch.randn(B, T, H, R, N, device=dev, generator=gen) * 0.5).requires_grad_(True)
    v = (torch.randn(B, T, H, V, device=dev, generator=gen) * 0.5).requires_grad_(True)
    # MIXED magnitudes: some heads ~1, some ~1e-18 (the regime that overflows
    # exp(-g) / poisons the masked upper triangle without the clamp).
    base = torch.full((B, T, H), 0.99, device=dev)
    base[:, :, ::2] = 1e-18
    decay = base.clone().requires_grad_(True)
    e = torch.sigmoid(torch.randn(B, T, H, N, device=dev, generator=gen)).requires_grad_(True)
    w = torch.sigmoid(torch.randn(B, T, H, V, device=dev, generator=gen)).requires_grad_(True)
    out, S = e97_multiquery_chunked_triton(k, v, q, decay, e, w, chunk_size=32)
    assert torch.isfinite(out).all(), f"R={R} strong-decay fwd NON-finite"
    assert torch.isfinite(S).all(), f"R={R} strong-decay S_final NON-finite"
    out.sum().backward()
    for name, t in [('k', k), ('q', q), ('v', v), ('decay', decay), ('e', e), ('w', w)]:
        assert torch.isfinite(t.grad).all(), f"R={R} strong-decay grad {name} NON-finite"
