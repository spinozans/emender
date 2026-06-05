"""Correctness gate for the UNIFIED parameterized matrix-recurrence cell.

The Triton forward+backward kernels must match the pure-PyTorch reference
recurrence (whose gradients come from autograd) to <1e-4 relative error, across
random inputs/shapes/knobs and ALL phi modes, including the un-cribbed
lambda>=1 regime.

Run: pytest tests/test_unified_cell.py -v   (requires CUDA + Triton)
"""
import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="CUDA required for Triton unified-cell tests"),
]

from ndm.triton.unified_cell_forward import (
    unified_cell_forward, unified_cell_torch_reference,
    PHI_IDENTITY, PHI_TANH, PHI_GAMMA_MIX, PHI_RELU, PHI_SOFTPLUS,
)
from ndm.triton.unified_cell_backward import unified_cell

PHI_MODES = [
    (PHI_IDENTITY, 'identity'),
    (PHI_TANH, 'tanh'),
    (PHI_GAMMA_MIX, 'gamma_mix'),
    (PHI_RELU, 'relu'),
    (PHI_SOFTPLUS, 'softplus'),
]

TOL = 1e-4


def _make(T, B, H, N, V, seed, lam_hi=1.3, requires_grad=False, device='cuda'):
    g = torch.Generator(device=device).manual_seed(seed)
    S0 = 0.05 * torch.randn(B, H, N, V, device=device, generator=g)
    k = torch.randn(T, B, H, N, device=device, generator=g)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q = torch.randn(T, B, H, N, device=device, generator=g)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v = 0.3 * torch.randn(T, B, H, V, device=device, generator=g)
    # lambda spans (0.3, lam_hi) -> includes the un-cribbed >=1 regime
    lam = 0.3 + (lam_hi - 0.3) * torch.rand(T, B, H, device=device, generator=g)
    beta = torch.rand(T, B, H, device=device, generator=g)
    igain = 0.5 + torch.rand(T, B, H, device=device, generator=g)
    gamma = torch.rand(H, device=device, generator=g)
    ts = [S0, k, v, q, lam, beta, igain, gamma]
    if requires_grad:
        ts = [t.detach().clone().requires_grad_(True) for t in ts]
    return ts


def _relerr(a, b):
    return (a - b).abs().max().item() / b.abs().max().clamp_min(1e-8).item()


@pytest.mark.parametrize("phi_mode,name", PHI_MODES)
@pytest.mark.parametrize("shape", [(32, 2, 4, 24, 20), (64, 1, 8, 16, 16), (16, 3, 5, 32, 32)])
def test_forward_matches_reference(phi_mode, name, shape):
    T, B, H, N, V = shape
    S0, k, v, q, lam, beta, igain, gamma = _make(T, B, H, N, V, seed=hash((phi_mode, shape)) & 0xffff)
    out, S_final, S_ckpt = unified_cell_forward(S0, k, v, q, lam, beta, igain, gamma, phi_mode=phi_mode)
    out_ref, S_ref, ckpt_dense = unified_cell_torch_reference(S0, k, v, q, lam, beta, igain, gamma, phi_mode=phi_mode)
    assert _relerr(out, out_ref) < TOL, f"{name}: out relerr too high"
    assert _relerr(S_final, S_ref) < TOL, f"{name}: S_final relerr too high"
    # sparse checkpoint (interval 16) must match the dense subsample.
    ckpt_int = 16
    if T % ckpt_int == 0:
        for slot in range(S_ckpt.shape[0]):
            dense_idx = 0 if slot == 0 else slot * ckpt_int
            assert _relerr(S_ckpt[slot], ckpt_dense[dense_idx]) < TOL


@pytest.mark.parametrize("phi_mode,name", PHI_MODES)
def test_backward_matches_autograd(phi_mode, name):
    T, B, H, N, V = 48, 3, 5, 24, 20
    seed = 1000 + phi_mode
    tri = _make(T, B, H, N, V, seed=seed, requires_grad=True)
    ref = _make(T, B, H, N, V, seed=seed, requires_grad=True)
    S0, k, v, q, lam, beta, igain, gamma = tri
    out, _ = unified_cell(k, v, q, lam, beta, igain, gamma, S0, phi_mode=phi_mode)
    go = torch.randn_like(out)
    out.backward(go)
    S0r, kr, vr, qr, lamr, betar, igr, gammar = ref
    out2, _, _ = unified_cell_torch_reference(S0r, kr, vr, qr, lamr, betar, igr, gammar, phi_mode=phi_mode)
    out2.backward(go)

    def cmp(a, b):
        if b is None or (b is not None and b.abs().max() == 0):
            return 0.0 if (a is None or a.abs().max() < 1e-6) else float('inf')
        return _relerr(a, b)

    pairs = [('out', out, out2), ('dk', k.grad, kr.grad), ('dv', v.grad, vr.grad),
             ('dq', q.grad, qr.grad), ('dlam', lam.grad, lamr.grad),
             ('dbeta', beta.grad, betar.grad), ('dig', igain.grad, igr.grad),
             ('dgamma', gamma.grad, gammar.grad), ('dS0', S0.grad, S0r.grad)]
    for nm, a, b in pairs:
        e = cmp(a, b)
        assert e < TOL, f"{name}: grad {nm} relerr {e:.2e} >= {TOL}"


def test_e88_equivalence():
    """E88's recurrence is the unified cell at lambda=decay, beta=1, igain=1, tanh."""
    from ndm.triton.e88_triton_forward import e88_torch_reference
    T, B, H, N, V = 32, 2, 4, 16, 16
    device = 'cuda'
    g = torch.Generator(device=device).manual_seed(123)
    S0 = 0.05 * torch.randn(B, H, N, V, device=device, generator=g)
    k = torch.randn(T, B, H, N, device=device, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    q = torch.randn(T, B, H, N, device=device, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    v = 0.3 * torch.randn(T, B, H, V, device=device, generator=g)
    decay = 0.3 + 0.6 * torch.rand(T, B, H, device=device, generator=g)
    # E88 reference
    out_e88, S_e88, _ = e88_torch_reference(S0, k, v, q, decay, linear_state=False)
    # Unified cell: beta=1, igain=1, tanh
    one = torch.ones_like(decay)
    gamma = torch.zeros(H, device=device)
    out_u, S_u, _ = unified_cell_torch_reference(S0, k, v, q, decay, one, one, gamma, phi_mode=PHI_TANH)
    assert _relerr(out_u, out_e88) < TOL
    assert _relerr(S_u, S_e88) < TOL


def test_high_lambda_stability():
    """lambda>1 with tanh self-bounds (latch corner): |S| stays <= 1, no NaN/Inf."""
    T, B, H, N, V = 64, 2, 4, 16, 16
    S0, k, v, q, lam, beta, igain, gamma = _make(T, B, H, N, V, seed=42, lam_hi=1.5)
    lam = torch.full_like(lam, 1.3)   # runaway gain
    beta = torch.zeros_like(beta)     # pure accumulation
    out, S_final, _ = unified_cell_forward(S0, k, v, q, lam, beta, igain, gamma, phi_mode=PHI_TANH)
    assert torch.isfinite(out).all() and torch.isfinite(S_final).all()
    assert S_final.abs().max() <= 1.0 + 1e-4
