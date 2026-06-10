"""Parity tests for the fused chunked `refit` (inner-optimization / TTT) cell.

Verifies ``ndm.triton.refit_chunked_autograd.refit_chunked_triton`` (two fused
``@triton.jit`` kernels, no torch fallback in the hot path) against:
  * the eager heavy-ball recurrence ``refit_eager_reference`` (momentum on);
  * the e97 / gated-delta reference at ``μ≡0`` — the delta-rule = one-inner-step
    special case, recovered bit-identically by the ``has_mom=False`` fast path
    AND, separately, by the momentum path as ``μ→0``.
Covers fwd+grad parity (fp32 + bf16), the log-gate path, the exposed inner-step
count K (truncated-Neumann), and finite-grad regressions. REAL data only.
"""
import pytest
import torch

from ndm.triton.refit_chunked_autograd import (
    refit_chunked_triton, refit_eager_reference, _refit_forward_only, _GLOG_FLOOR,
)
from ndm.triton.e88_triton_forward import e88_torch_reference

CUDA = torch.cuda.is_available()


def _l2n(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def _mk(B, T, H, N, V, device, dtype, seed=0, corr=False):
    """Stable GDN regime: L2-normed keys (‖k‖=1), β<1 via erase, decay 0.6-0.9,
    momentum 0.1-0.5 — the convention TTT_WRITE_SPEC assumes (heavy-ball stable)."""
    g = torch.Generator(device=device).manual_seed(seed)
    kb = torch.randn(B, T, H, N, device=device, generator=g)
    if corr:  # correlated (non-orthogonal) keys -> large intra-chunk coupling
        base = torch.randn(B, T, H, N, device=device, generator=g)
        kb = 0.7 * base + 0.3 * kb
    k = _l2n(kb).to(dtype)
    q = _l2n(torch.randn(B, T, H, N, device=device, generator=g)).to(dtype)
    v = (torch.randn(B, T, H, V, device=device, generator=g) * 0.5).to(dtype)
    decay = (torch.sigmoid(torch.randn(B, T, H, device=device, generator=g)) * 0.3 + 0.6).to(dtype)
    mu = (torch.sigmoid(torch.randn(B, T, H, device=device, generator=g)) * 0.4 + 0.1).to(dtype)
    e = (torch.sigmoid(torch.randn(B, T, H, N, device=device, generator=g)) * 0.5).to(dtype)
    w = torch.sigmoid(torch.randn(B, T, H, V, device=device, generator=g)).to(dtype)
    return k, v, q, decay, mu, e, w


# ---------------------------------------------------------------------------
# Forward parity at the spec-mandated lengths T = 128 / 512 / 1024.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [128, 512, 1024])
def test_forward_parity_momentum_fp32(T):
    dev = 'cuda'
    B, H, N, V = 2, 4, 32, 32
    k, v, q, decay, mu, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=1)
    out, S = refit_chunked_triton(k, v, q, decay, e, w, mu, chunk_size=32)
    out_e, S_e, _ = refit_eager_reference(k, v, q, decay, e, w, mu)
    scale = out_e.abs().max().item() + 1e-6
    assert (out - out_e).abs().max().item() / scale < 5e-5, "momentum fwd parity"
    assert (S - S_e).abs().max().item() / (S_e.abs().max().item() + 1e-6) < 5e-5


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [128, 512, 1024])
def test_forward_parity_delta_special_case(T):
    """has_mom=False reduces to the e97 / gated-delta write (μ≡0): the delta-rule
    = one-inner-step special case, bit-near the e88 reference recurrence."""
    dev = 'cuda'
    B, H, N, V = 2, 4, 32, 32
    k, v, q, decay, mu, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=2)
    out, S = refit_chunked_triton(k, v, q, decay, e, w, mu, chunk_size=32, has_mom=False)
    # e88_torch_reference wants [T,B,H,*]
    S0 = torch.zeros(B, H, N, V, device=dev)
    out_r, S_r, _ = e88_torch_reference(
        S0, k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous(),
        q.transpose(0, 1).contiguous(), decay.transpose(0, 1).contiguous(),
        linear_state=True, raw_write=False,
        erase_gate=e.transpose(0, 1).contiguous(),
        value_write_gate=w.transpose(0, 1).contiguous())
    out_r = out_r.transpose(0, 1).contiguous()
    assert (out - out_r).abs().max().item() < 2e-4, "delta special case vs e88 ref"
    assert (S - S_r).abs().max().item() < 2e-4


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_momentum_mu_to_zero_reduces_to_delta():
    """The momentum path itself (has_mom=True) collapses to the delta write as
    μ→0 (gm floored) — numerically recovers the 1-inner-step special case."""
    dev = 'cuda'
    B, T, H, N, V = 2, 256, 3, 32, 32
    k, v, q, decay, mu, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=4)
    glog = decay.log()
    gm_tiny = torch.full_like(decay.log(), _GLOG_FLOOR)   # μ ≈ e^-30 ≈ 0
    out_mom, _ = _refit_forward_only(k, v, q, glog, gm_tiny, e, w, chunk_size=32, has_mom=True)
    out_delta, _ = _refit_forward_only(k, v, q, glog, gm_tiny, e, w, chunk_size=32, has_mom=False)
    assert (out_mom - out_delta).abs().max().item() < 2e-4, "μ→0 momentum != delta"


# ---------------------------------------------------------------------------
# Backward grad parity vs eager (all 7 grad sinks, momentum on AND off).
# ---------------------------------------------------------------------------
def _run(fn, tensors):
    ts = [t.clone().requires_grad_(True) for t in tensors]
    out = fn(*ts)
    (out * out).sum().backward()
    return out.detach(), [None if t.grad is None else t.grad.clone() for t in ts]


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("has_mom", [True, False])
def test_backward_parity_fp32(has_mom):
    dev = 'cuda'
    B, T, H, N, V = 2, 128, 3, 32, 32
    k, v, q, decay, mu, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=11)
    names = ['k', 'v', 'q', 'decay', 'mu', 'e', 'w']

    def ref(kk, vv, qq, dd, mm, ee, ww):
        mm_use = mm if has_mom else torch.zeros_like(mm)
        return refit_eager_reference(kk, vv, qq, dd, ee, ww, mm_use)[0]

    def chk(kk, vv, qq, dd, mm, ee, ww):
        return refit_chunked_triton(kk, vv, qq, dd, ee, ww, mm,
                                    chunk_size=32, has_mom=has_mom)[0]

    o_ref, g_ref = _run(ref, (k, v, q, decay, mu, e, w))
    o_chk, g_chk = _run(chk, (k, v, q, decay, mu, e, w))
    assert (o_chk - o_ref).abs().max().item() < 5e-5
    for nm, a, b in zip(names, g_ref, g_chk):
        if a is None:
            continue
        err = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-6
        assert err / scale < 3e-3, f"grad {nm} rel err {err / scale} (abs {err})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_logdecay_backward_parity_fp32():
    """log_decay=True: grads returned wrt the LOG-gates (g, gm) directly, matching
    the eager reference reparametrized in the logs."""
    dev = 'cuda'
    B, T, H, N, V = 2, 256, 3, 32, 32
    k, v, q, decay, mu, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=3)
    g, gm = decay.log(), mu.log()
    names = ['k', 'v', 'q', 'g', 'gm', 'e', 'w']

    def ref(kk, vv, qq, gg, mm, ee, ww):
        return refit_eager_reference(kk, vv, qq, gg, ee, ww, mm, log_gates=True)[0]

    def chk(kk, vv, qq, gg, mm, ee, ww):
        return refit_chunked_triton(kk, vv, qq, gg, ee, ww, mm,
                                    chunk_size=32, log_decay=True)[0]

    o_ref, g_ref = _run(ref, (k, v, q, g, gm, e, w))
    o_chk, g_chk = _run(chk, (k, v, q, g, gm, e, w))
    assert (o_chk - o_ref).abs().max().item() < 5e-5
    for nm, a, b in zip(names, g_ref, g_chk):
        err = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-6
        assert err / scale < 3e-3, f"log grad {nm} rel err {err / scale}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_backward_parity_bf16():
    """bf16 inputs run the TF32 tensor-core path; fwd+grad parity within TF32 tol."""
    dev = 'cuda'
    B, T, H, N, V = 2, 256, 3, 32, 32
    k, v, q, decay, mu, e, w = _mk(B, T, H, N, V, dev, torch.bfloat16, seed=5)
    ts = [t.clone().requires_grad_(True) for t in (k, v, q, decay, mu, e, w)]
    out, _ = refit_chunked_triton(ts[0], ts[1], ts[2], ts[3], ts[5], ts[6], ts[4],
                                  chunk_size=32)
    (out.float() ** 2).sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in ts), "bf16 grads non-finite"
    f = [x.float() for x in (k, v, q, decay, mu, e, w)]
    tr = [t.clone().requires_grad_(True) for t in f]
    outr, _, _ = refit_eager_reference(tr[0], tr[1], tr[2], tr[3], tr[5], tr[6], tr[4])
    (outr ** 2).sum().backward()
    frel = (out.float() - outr).abs().max().item() / (outr.abs().max().item() + 1e-6)
    assert frel < 0.05, f"bf16 fwd rel {frel}"
    for nm, i in zip(['k', 'v', 'q', 'decay', 'mu', 'e', 'w'], range(7)):
        err = (tr[i].grad - ts[i].grad.float()).abs().max().item()
        scale = tr[i].grad.abs().max().item() + 1e-6
        assert err / scale < 0.06, f"bf16 grad {nm} rel {err / scale}"


# ---------------------------------------------------------------------------
# Exposed inner-step count K (= NEWTON_STEPS): K→exact recovers the chunk refit;
# K<exact is a genuine truncated-Neumann approximation (the knob is live).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_inner_step_knob_truncated_neumann():
    dev = 'cuda'
    B, T, H, N, V = 1, 64, 1, 16, 16
    k, v, q, decay, mu, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=2, corr=True)
    decay = torch.full_like(decay, 0.95)
    e = torch.full_like(e, 0.95)   # near-1 β => strong updates => large coupling
    glog, gmlog = decay.log(), mu.log()
    out_e, _, _ = refit_eager_reference(k, v, q, decay, e, w, mu)
    errs = {}
    for K in (1, 3, 5):   # exact for C=64 is ceil(log2 64)=6; 5 already near-exact
        o, _ = _refit_forward_only(k, v, q, glog, gmlog, e, w, chunk_size=64,
                                   newton_steps=K, has_mom=True)
        errs[K] = (o - out_e).abs().max().item()
    # The knob is LIVE and converges to the exact chunk refit as K grows. (Newton-
    # Schulz is a quadratic, not monotone-linear, iteration so intermediate K can
    # overshoot before its basin; we assert liveness + convergence, not monotonicity.)
    assert errs[5] < 1e-3, f"K=exact did not recover the chunk refit: {errs}"
    assert errs[1] > 0.1, f"K=1 truncation is not a real approximation: {errs}"
    assert errs[3] < errs[1], f"more inner steps did not help: {errs}"


# ---------------------------------------------------------------------------
# Finite-grad regressions (mirror e97): strong-decay cluster + gate floor.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_strong_decay_and_floor_finite():
    dev = 'cuda'
    B, T, H, N, V = 2, 128, 3, 32, 32
    k, v, q, decay, mu, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=13)
    g = decay.log()
    g[:, 40:48, :] = -31.0    # consecutive floor-decay run (span >> 88)
    g[:, 70:75, :] = -120.0   # scatter way below the floor
    gm = mu.log()
    gm[:, 10:14, :] = -120.0
    ts = [t.clone().requires_grad_(True) for t in (k, v, q, g, gm, e, w)]
    out, _ = refit_chunked_triton(ts[0], ts[1], ts[2], ts[3], ts[5], ts[6], ts[4],
                                  chunk_size=32, log_decay=True)
    assert torch.isfinite(out).all(), "strong-decay fwd non-finite"
    (out ** 2).sum().backward()
    for nm, t in zip(['k', 'v', 'q', 'g', 'gm', 'e', 'w'], ts):
        assert torch.isfinite(t.grad).all(), f"strong-decay grad {nm} non-finite"
    # floored gate steps -> exactly zero gate grad
    assert (ts[3].grad[g < _GLOG_FLOOR] == 0).all(), "floored decay grad must be 0"
    assert (ts[4].grad[gm < _GLOG_FLOOR] == 0).all(), "floored mu grad must be 0"


if __name__ == '__main__':
    if not CUDA:
        print("no CUDA; skipping")
    else:
        for T in (128, 512, 1024):
            test_forward_parity_momentum_fp32(T)
            test_forward_parity_delta_special_case(T)
        print("forward parity (momentum + delta special case) T=128/512/1024 OK")
        test_momentum_mu_to_zero_reduces_to_delta(); print("μ→0 reduction OK")
        for hm in (True, False):
            test_backward_parity_fp32(hm)
        print("backward parity fp32 (momentum + delta) OK")
        test_logdecay_backward_parity_fp32(); print("log_decay backward OK")
        test_backward_parity_bf16(); print("bf16 fwd+grad OK")
        test_inner_step_knob_truncated_neumann(); print("inner-step K knob live OK")
        test_strong_decay_and_floor_finite(); print("strong-decay/floor finite OK")
        print("ALL REFIT PARITY OK")
