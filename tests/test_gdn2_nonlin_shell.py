"""Tests for the Triton-fused GDN-2 nonlinear shell."""
from __future__ import annotations

import pytest
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused GDN shell tests"),
]

from ndm.models.gdn2_nonlin_shell import GDN2NonlinShellLayer, nonlinear_gated_delta_scan
from ndm.triton.gdn2_nonlin_fused import (
    fused_nonlinear_gated_delta_scan,
    nonlinear_gated_delta_torch_reference,
)


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item() / b.abs().max().clamp_min(1e-8).item()


def _scan_inputs(
    *,
    B: int = 2,
    T: int = 48,
    H: int = 3,
    K: int = 16,
    V: int = 16,
    seed: int = 0,
    requires_grad: bool = False,
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, T, H, K, device="cuda", generator=gen)
    k = torch.randn(B, T, H, K, device="cuda", generator=gen)
    v = 0.3 * torch.randn(B, T, H, V, device="cuda", generator=gen)
    g = -0.01 - 0.08 * torch.rand(B, T, H, device="cuda", generator=gen)
    beta = 2.0 * torch.rand(B, T, H, device="cuda", generator=gen)
    tensors = (q, k, v, g, beta)
    if requires_grad:
        tensors = tuple(t.detach().clone().requires_grad_(True) for t in tensors)
    return tensors


def test_identity_reproduces_native_gdn():
    """phi=identity, fused scan == FLA native full-sequence chunk_gated_delta_rule."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    dim, n_state, n_heads, T = 128, 32, 4, 128
    layer = GDN2NonlinShellLayer(
        dim=dim, n_state=n_state, n_heads=n_heads,
        state_nonlin="identity", state_chunk=16,
    ).to("cuda")
    layer.eval()
    x = torch.randn(2, T, dim, device="cuda")
    with torch.no_grad():
        q, k, v, g, beta = layer._project(x)
        o_native, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        o_fused = nonlinear_gated_delta_scan(
            q, k, v, g, beta,
            state_chunk=16, state_nonlin="identity",
        )
    rel = _relerr(o_fused, o_native)
    assert torch.isfinite(o_fused).all()
    assert rel < 1e-4, f"identity fused shell must reproduce native GDN (rel={rel:.3e})"


def test_low_level_fused_identity_matches_recurrent_reference():
    q, k, v, g, beta = _scan_inputs(T=40, seed=9)
    out = fused_nonlinear_gated_delta_scan(q, k, v, g, beta, state_chunk=8, state_nonlin="identity")
    ref = nonlinear_gated_delta_torch_reference(q, k, v, g, beta, state_chunk=8, state_nonlin="identity")
    assert _relerr(out.float(), ref.float()) < 1e-4


@pytest.mark.parametrize("state_nonlin", ["tanh", "relu", "softplus_c"])
def test_fused_forward_matches_torch_reference(state_nonlin):
    q, k, v, g, beta = _scan_inputs(T=40, seed=10)
    out = nonlinear_gated_delta_scan(q, k, v, g, beta, state_chunk=8, state_nonlin=state_nonlin)
    ref = nonlinear_gated_delta_torch_reference(q, k, v, g, beta, state_chunk=8, state_nonlin=state_nonlin)
    assert _relerr(out.float(), ref.float()) < 1e-4


@pytest.mark.parametrize("state_nonlin", ["tanh", "softplus_c"])
def test_fused_backward_matches_torch_reference(state_nonlin):
    tri = _scan_inputs(T=32, seed=20, requires_grad=True)
    ref = tuple(t.detach().clone().requires_grad_(True) for t in tri)
    out = nonlinear_gated_delta_scan(*tri, state_chunk=8, state_nonlin=state_nonlin)
    out_ref = nonlinear_gated_delta_torch_reference(*ref, state_chunk=8, state_nonlin=state_nonlin)
    gen = torch.Generator(device="cuda").manual_seed(21)
    grad = torch.randn(out.shape, device="cuda", generator=gen)
    out.backward(grad)
    out_ref.backward(grad)
    names = ("q", "k", "v", "g", "beta")
    for name, got, want in zip(names, tri, ref):
        rel = _relerr(got.grad.float(), want.grad.float())
        assert rel < 5e-4, f"{state_nonlin} d{name} relerr {rel:.2e}"


def test_nonlin_differs_and_is_finite():
    q, k, v, g, beta = _scan_inputs(T=64, seed=30)
    out_id = nonlinear_gated_delta_scan(q, k, v, g, beta, state_chunk=8, state_nonlin="identity")
    for kind in ("tanh", "relu", "softplus_c"):
        out = nonlinear_gated_delta_scan(q, k, v, g, beta, state_chunk=8, state_nonlin=kind)
        assert torch.isfinite(out).all()
        assert (out - out_id).abs().mean().item() > 1e-6


def test_profiler_shows_single_fused_forward_launch():
    q, k, v, g, beta = _scan_inputs(B=1, T=96, H=2, K=16, V=16, seed=40)
    nonlinear_gated_delta_scan(q, k, v, g, beta, state_chunk=8, state_nonlin="tanh")
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        nonlinear_gated_delta_scan(q, k, v, g, beta, state_chunk=8, state_nonlin="tanh")
        torch.cuda.synchronize()
    keys = [event.key for event in prof.key_averages()]
    fused = [key for key in keys if "gdn2_nonlin_fwd" in key]
    chunk = [key for key in keys if "chunk_gated_delta" in key]
    assert len(fused) == 1, keys
    assert chunk == []


def test_layer_backward_flows_to_all_params():
    torch.manual_seed(50)
    layer = GDN2NonlinShellLayer(
        dim=96, n_state=32, n_heads=3,
        state_nonlin="tanh", state_chunk=8,
    ).to("cuda")
    x = torch.randn(2, 48, 96, device="cuda", requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape
    out.float().pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    missing = [
        name for name, param in layer.named_parameters()
        if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all())
    ]
    assert missing == []
