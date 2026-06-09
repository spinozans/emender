"""Parity + reduction tests for the complex-eigenvalue gated-delta scan.

Verifies the chunked-parallel complex scan (complex_gated_delta_chunked) matches
the eager per-step complex reference (complex_gated_delta_reference, the spec
recurrence) in forward and backward, and that the eigenvalue reductions hold:
theta=0 -> real-positive decay (GDN regime), theta=pi -> reflection (negative
eigenvalue).  REAL data only (torch.randn), mirroring tests/test_e97_chunked.py.
"""
import math

import pytest
import torch

from ndm.triton.complex_eig_chunked import (
    complex_gated_delta_chunked,
    complex_gated_delta_reference,
)
from ndm.triton.complex_eig_chunked_autograd import complex_gated_delta_chunked_triton

CUDA = torch.cuda.is_available()


def _mk(B, T, H, N, V, device, dtype=torch.float32, seed=0, decay_lo=0.85, decay_hi=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g) * 0.5
    q = torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g) * 0.5
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g) * 0.5
    P = N // 2
    # per-channel magnitude r in [decay_lo, decay_hi) -> log_r <= 0
    r = torch.rand(B, T, H, P, device=device, dtype=dtype, generator=g) * (decay_hi - decay_lo) + decay_lo
    log_r = r.clamp_min(1e-6).log()
    theta = torch.randn(B, T, H, P, device=device, dtype=dtype, generator=g) * 0.7
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype, generator=g)) * 2.0
    return q, k, v, log_r, theta, beta


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T,C", [(32, 32), (64, 32), (128, 32), (96, 32), (130, 32), (256, 64)])
def test_chunked_matches_eager_fp32(T, C):
    dev = "cuda"
    B, H, N, V = 2, 4, 32, 32
    q, k, v, log_r, theta, beta = _mk(B, T, H, N, V, dev, seed=0)
    ref, refS = complex_gated_delta_reference(q, k, v, log_r, theta, beta)
    out, outS = complex_gated_delta_chunked(q, k, v, log_r, theta, beta, chunk_size=C)
    err = (out - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    assert err / scale < 3e-3, f"fwd rel err {err/scale} (abs {err})"
    serr = (outS - refS).abs().max().item() / (refS.abs().max().item() + 1e-6)
    assert serr < 3e-3, f"S_final rel err {serr}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("decay_lo,decay_hi", [(0.01, 0.2), (1e-4, 0.02), (1e-6, 1e-3)])
@pytest.mark.parametrize("C", [32, 64])
def test_chunked_stable_small_lambda(decay_lo, decay_hi, C):
    """HARDENING REGRESSION (task fix-harden-complex): when the model drives the
    eigenvalue magnitude |lambda| << 1 WITHIN a chunk, the cumulative-decay fold
    KR = k / cp = exp(-Gprev) blows past fp32 range and the old magnitude clamp
    silently erased the (banded) near-diagonal entries -> chunked scan returned
    garbage (rel err -> 1), corrupting training.  After the adaptive sub-chunking
    fix the chunked scan must stay finite AND match the exact eager reference even
    at extreme decay.  This regime is OUTSIDE the [0.85,1.0] band the other parity
    tests cover, so it is the test that would have caught the bug."""
    dev = "cuda"
    B, T, H, N, V = 2, 128, 4, 32, 32
    for fn in (complex_gated_delta_chunked, complex_gated_delta_chunked_triton):
        q, k, v, log_r, theta, beta = _mk(B, T, H, N, V, dev, seed=3,
                                          decay_lo=decay_lo, decay_hi=decay_hi)
        ref, _ = complex_gated_delta_reference(q, k, v, log_r, theta, beta)
        out, _ = fn(q, k, v, log_r, theta, beta, chunk_size=C)
        assert torch.isfinite(out).all(), f"{fn.__name__} non-finite at decay<{decay_hi}"
        scale = ref.abs().max().item() + 1e-12
        rel = (out - ref).abs().max().item() / scale
        assert rel < 3e-3, f"{fn.__name__} rel err {rel} at decay [{decay_lo},{decay_hi}) C={C}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_chunked_backward_finite_small_lambda():
    """Backward must also stay finite at |lambda| << 1 (the training failure mode
    was loss=nan, i.e. a non-finite gradient propagating from this kernel)."""
    dev = "cuda"
    B, T, H, N, V = 2, 128, 3, 32, 32
    for fn in (complex_gated_delta_chunked, complex_gated_delta_chunked_triton):
        q, k, v, log_r, theta, beta = _mk(B, T, H, N, V, dev, seed=5,
                                          decay_lo=1e-5, decay_hi=1e-2)
        ts = [t.clone().requires_grad_(True) for t in (q, k, v, log_r, theta, beta)]
        out, _ = fn(*ts, chunk_size=32)
        (out * out).sum().backward()
        for nm, t in zip(["q", "k", "v", "log_r", "theta", "beta"], ts):
            assert torch.isfinite(t.grad).all(), f"{fn.__name__} grad {nm} non-finite at small |lambda|"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_reduction_theta0_real_positive():
    """theta == 0 -> eigenvalue lambda = r is real-positive (imag == 0). GDN regime."""
    dev = "cuda"
    B, T, H, N, V = 2, 64, 3, 32, 32
    q, k, v, log_r, _, beta = _mk(B, T, H, N, V, dev, seed=1)
    theta = torch.zeros(B, T, H, N // 2, device=dev)
    lam = torch.polar(torch.exp(log_r), theta)
    assert lam.imag.abs().max().item() < 1e-6, "theta=0 must give imaginary part 0"
    assert (lam.real > 0).all(), "theta=0 must give positive real eigenvalue"
    # chunked still matches eager exactly in this regime
    ref, _ = complex_gated_delta_reference(q, k, v, log_r, theta, beta)
    out, _ = complex_gated_delta_chunked(q, k, v, log_r, theta, beta, chunk_size=32)
    rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert rel < 3e-3, f"theta=0 chunked vs eager rel {rel}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_reduction_thetapi_reflection():
    """theta == pi -> eigenvalue lambda = -r real-negative (reflection / neg-eigval)."""
    dev = "cuda"
    B, T, H, N, V = 2, 64, 3, 32, 32
    q, k, v, log_r, _, beta = _mk(B, T, H, N, V, dev, seed=2)
    theta = torch.full((B, T, H, N // 2), math.pi, device=dev)
    lam = torch.polar(torch.exp(log_r), theta)
    assert lam.imag.abs().max().item() < 1e-5, "theta=pi must give imaginary part ~0"
    assert (lam.real < 0).all(), "theta=pi must give negative real eigenvalue (reflection)"
    ref, _ = complex_gated_delta_reference(q, k, v, log_r, theta, beta)
    out, _ = complex_gated_delta_chunked(q, k, v, log_r, theta, beta, chunk_size=32)
    rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert rel < 3e-3, f"theta=pi chunked vs eager rel {rel}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_backward_finite_and_parity_fp32():
    dev = "cuda"
    B, T, H, N, V = 2, 128, 3, 32, 32
    q, k, v, log_r, theta, beta = _mk(B, T, H, N, V, dev, seed=7)

    def run(fn):
        ts = [t.clone().requires_grad_(True) for t in (q, k, v, log_r, theta, beta)]
        out, _ = fn(*ts)
        (out * out).sum().backward()
        return [t.grad.clone() for t in ts]

    g_ref = run(lambda *a: complex_gated_delta_reference(*a))
    g_chk = run(lambda *a: complex_gated_delta_chunked(*a, chunk_size=32))
    for nm, a, b in zip(["q", "k", "v", "log_r", "theta", "beta"], g_ref, g_chk):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), f"non-finite grad {nm}"
        err = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-6
        assert err / scale < 1e-2, f"grad {nm} rel err {err/scale} (abs {err})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_phi_subset_reference_runs_and_bounds_state():
    """The nonlinear-subset path (per-step hardtanh) runs fwd+bwd and bounds |S|."""
    dev = "cuda"
    B, T, H, N, V = 2, 64, 3, 32, 32
    q, k, v, log_r, theta, beta = _mk(B, T, H, N, V, dev, seed=9, decay_lo=0.99, decay_hi=1.0)
    ts = [t.clone().requires_grad_(True) for t in (q, k, v, log_r, theta, beta)]
    out, S = complex_gated_delta_reference(*ts, phi="hardtanh")
    assert S.real.abs().max().item() <= 1.0 + 1e-5 and S.imag.abs().max().item() <= 1.0 + 1e-5
    (out * out).sum().backward()
    for nm, t in zip(["q", "k", "v", "log_r", "theta", "beta"], ts):
        assert torch.isfinite(t.grad).all(), f"phi-subset grad {nm} non-finite"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("frac", [0.0, 0.25, 1.0])
def test_head_layer_builds_runs_and_trains(frac):
    """ComplexEigHeadLayer builds in the substrate, runs fwd+bwd, finite grads,
    and a smoke optimizer step succeeds; config selects the nonlinear-subset frac."""
    from ndm.models.complex_eig_head import ComplexEigHeadLayer
    dev = "cuda"
    torch.manual_seed(0)
    layer = ComplexEigHeadLayer(dim=256, n_state=32, n_heads=8, nonlin_subset_frac=frac).to(dev).train()
    alloc = layer.head_alloc()
    assert alloc["n_complex_hardtanh"] == round(frac * 8)
    assert alloc["n_complex_chunked"] + alloc["n_complex_hardtanh"] == 8
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    for _ in range(3):
        x = torch.randn(2, 64, 256, device=dev)
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
def test_complex_eig_selects_in_typed_head_mixture():
    """The substrate config selects complex-everywhere + nonlinear-subset frac."""
    from ndm.models.typed_head_mixture import TypedHeadMixtureLayer
    dev = "cuda"
    torch.manual_seed(0)
    lay = TypedHeadMixtureLayer(dim=256, n_state=32, n_heads=12,
                               complex_eig=True, nonlin_subset_frac=0.25).to(dev).train()
    a = lay.head_alloc()
    assert a["n_heads"] == 12 and a["n_complex_hardtanh"] == 3 and a["n_complex_chunked"] == 9
    x = torch.randn(2, 64, 256, device=dev)
    out = lay(x)
    assert out.shape == x.shape
    (out * out).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in lay.parameters() if p.grad is not None)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_lm_substrate_smoke_train_step():
    """LadderLM 'typed-gdn2-lm' with complex_eig runs a real fwd+bwd+optimizer step
    (no NaN/Inf) on byte tokens — the end-to-end substrate smoke gate."""
    from ndm.models.ladder_lm import LadderLM
    dev = "cuda"
    torch.manual_seed(0)
    model = LadderLM(vocab_size=256, dim=256, depth=2, level="typed-gdn2-lm",
                     n_heads=8, n_state=32, use_gate=True,
                     layer_kwargs={"complex_eig": True, "nonlin_subset_frac": 0.25}).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        toks = torch.randint(0, 256, (2, 128), device=dev)
        out = model(toks)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 256), toks[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        assert torch.isfinite(loss).all(), "LM smoke-train loss non-finite"
        for n, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"


if __name__ == "__main__":
    if not CUDA:
        print("no CUDA; skip")
    else:
        for T, C in [(32, 32), (64, 32), (128, 32), (96, 32), (130, 32), (256, 64)]:
            test_chunked_matches_eager_fp32(T, C)
        print("chunked==eager fp32 OK")
        test_reduction_theta0_real_positive(); print("theta=0 real-positive OK")
        test_reduction_thetapi_reflection(); print("theta=pi reflection OK")
        test_backward_finite_and_parity_fp32(); print("backward parity OK")
        test_phi_subset_reference_runs_and_bounds_state(); print("phi-subset OK")
        for frac in (0.0, 0.25, 1.0):
            test_head_layer_builds_runs_and_trains(frac)
        print("head layer build/run/train OK")
        test_complex_eig_selects_in_typed_head_mixture(); print("substrate config-select OK")
        test_lm_substrate_smoke_train_step(); print("LM substrate smoke-train OK")
        print("ALL COMPLEX-EIG PARITY OK")
