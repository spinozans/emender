"""Tests for TypedHeadMixtureLayer (typed-gdn-2-head).

Covers the deterministic allocation contract (softmax + largest-remainder, zeros
allowed, exact sum) and that the layer instantiates the right NATIVE sub-blocks
and runs a real forward/backward on GPU.
"""
import math

import pytest
import torch

from ndm.models.typed_head_mixture import (
    TypedHeadMixtureLayer, allocate_types, largest_remainder_counts, TYPE_NAMES,
)


def test_largest_remainder_sums_exactly():
    for H in (1, 7, 30, 48, 64):
        for fr in ([1, 1, 1, 1, 1], [3, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0.0, 0.4]):
            counts = largest_remainder_counts(H, fr)
            assert sum(counts) == H
            assert all(c >= 0 for c in counts)


def test_allocation_allows_zero():
    # extreme logits -> one type takes everything, others get zero (reported honestly)
    a = allocate_types(48, [10.0, -10.0, -10.0, -10.0, -10.0])
    assert a['counts']['gdn2_recall'] == 48
    assert a['n_unified'] == 0
    assert all(a['counts'][t] == 0 for t in TYPE_NAMES[1:])


def test_legacy_5_logits_pad_shell_off():
    # a 5-entry call (pre-shell callers) must reproduce the 5-type allocation
    # EXACTLY, with the 6th (shell) slot padded off to 0 heads.
    a5 = allocate_types(48, [0.5, 0.2, 0.1, 0.1, 0.1])
    a6 = allocate_types(48, [0.5, 0.2, 0.1, 0.1, 0.1, float('-inf')])
    assert a5['counts'] == a6['counts']
    assert a5['counts']['gdn2_nonlin_shell'] == 0
    assert a5['n_shell'] == 0
    assert sum(a5['counts'].values()) == 48


def test_allocation_uniform_balanced():
    # explicit full 8-vector -> ~uniform across all eight types (48/8 == 6 each)
    a = allocate_types(48, [0.0] * len(TYPE_NAMES))
    assert sum(a['counts'].values()) == 48
    assert all(5 <= a['counts'][t] <= 7 for t in TYPE_NAMES)
    # legacy 6-vector -> the six historical types balance, the two new e97 fused
    # types stay 0 (padded off), reproducing the pre-e97 allocation.
    a6 = allocate_types(48, [0.0] * 6)
    assert a6['counts']['e97_raw'] == 0 and a6['counts']['e97_delta'] == 0
    assert all(7 <= a6['counts'][t] <= 9 for t in TYPE_NAMES[:6])
    # legacy 5-vector -> the five active types balance, shell + e97 stay 0
    a5 = allocate_types(48, [0.0] * 5)
    assert a5['counts']['gdn2_nonlin_shell'] == 0
    assert a5['counts']['e97_raw'] == 0 and a5['counts']['e97_delta'] == 0
    assert all(9 <= a5['counts'][t] <= 10 for t in TYPE_NAMES[:5])


def test_shell_allocation_6_logits():
    # 6-vector that asks for ~1/3 shell heads -> a real shell allocation
    a = allocate_types(102, [math.log(2 / 3), -30.0, -30.0, -30.0, -30.0, math.log(1 / 3)])
    assert a['n_shell'] > 0
    assert a['counts']['gdn2_nonlin_shell'] == a['n_shell']
    assert a['counts']['gdn2_recall'] + a['n_shell'] == 102
    assert a['n_unified'] == 0


def test_allocation_e97_fused_types():
    # full 8-vector asking for an explicit e97_raw + e97_delta slice
    a = allocate_types(48, [0.0, -30, -30, -30, -30, -30, math.log(2), math.log(2)])
    assert a['n_e97_raw'] > 0 and a['n_e97_delta'] > 0
    assert a['counts']['e97_raw'] == a['n_e97_raw']
    assert a['counts']['e97_delta'] == a['n_e97_delta']
    # e97 heads + the gdn head account for everything (unified/shell starved off)
    assert a['n_gdn'] + a['n_e97_raw'] + a['n_e97_delta'] == 48
    assert a['n_unified'] == 0 and a['n_shell'] == 0


def test_allocation_softmax_fractions_sum_to_one():
    a = allocate_types(48, [0.5, 0.2, 0.1, 0.1, 0.1])
    assert math.isclose(sum(a['fractions'].values()), 1.0, rel_tol=1e-6)
    # gdn has the largest logit -> the most heads
    assert a['counts']['gdn2_recall'] == max(a['counts'].values())


def test_logits_length_validated():
    with pytest.raises(ValueError):
        allocate_types(48, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        allocate_types(48, [0.0] * 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU/FLA kernels")
def test_forward_backward_mixed_population():
    dev = 'cuda'
    layer = TypedHeadMixtureLayer(
        dim=128, n_state=32, n_heads=20,
        head_type_logits=[0.5, 0.2, 0.1, 0.1, 0.1]).to(dev)
    # both native sub-blocks present
    assert layer.gdn is not None and layer.unified is not None
    x = torch.randn(2, 64, 128, device=dev, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 64, 128)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU/FLA kernels")
def test_unified_personalities_are_frozen():
    # fixed_pop => the recurrence knobs are buffers, NOT trainable parameters.
    layer = TypedHeadMixtureLayer(
        dim=128, n_state=32, n_heads=20,
        head_type_logits=[-2.0, 1.0, 1.0, 1.0, 1.0]).to('cuda')
    knob_suffixes = ('lam_raw', 'beta_raw', 'igain_raw', 'gamma_raw')
    trainable_knobs = [n for n, p in layer.named_parameters()
                       if p.requires_grad and any(n.endswith(s) for s in knob_suffixes)]
    assert trainable_knobs == [], f"knobs should be frozen, got {trainable_knobs}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU/FLA kernels")
def test_all_gdn_and_all_unified_extremes():
    dev = 'cuda'
    x = torch.randn(2, 48, 96, device=dev)
    all_gdn = TypedHeadMixtureLayer(
        dim=96, n_state=32, n_heads=12,
        head_type_logits=[10.0, -10, -10, -10, -10]).to(dev)
    assert all_gdn.gdn is not None and all_gdn.unified is None
    assert all_gdn(x).shape == (2, 48, 96)
    all_track = TypedHeadMixtureLayer(
        dim=96, n_state=32, n_heads=12,
        head_type_logits=[-10.0, 10, -10, -10, -10]).to(dev)
    assert all_track.gdn is None and all_track.unified is not None
    assert all_track(x).shape == (2, 48, 96)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU/Triton kernels")
def test_e97_fused_heads_no_eager_fallback():
    # A heterogeneous bf16 layer with e97_raw + e97_delta heads must route BOTH
    # through the fused split-edit Triton kernel during training — never the eager
    # T-scan (the wire-fused-e97 bug). Count real kernel calls.
    import ndm.triton.e88_triton_optimized as e88t
    dev = 'cuda'
    layer = TypedHeadMixtureLayer(
        dim=256, n_state=32, n_heads=48,
        head_type_logits=[0.3, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.6],
        use_triton_e97=True, cast_recurrent_bf16=True,
    ).to(device=dev, dtype=torch.bfloat16)
    layer.train()
    assert layer.e97_raw is not None and layer.e97_delta is not None
    n_fused = int(layer.alloc['n_e97_raw'] > 0) + int(layer.alloc['n_e97_delta'] > 0)

    calls = {'n': 0}
    orig = e88t.e88_triton_optimized_apply
    e88t.e88_triton_optimized_apply = lambda *a, **k: (calls.__setitem__('n', calls['n'] + 1), orig(*a, **k))[1]
    try:
        x = torch.randn(2, 128, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        out = layer(x)
        out.float().pow(2).mean().backward()
    finally:
        e88t.e88_triton_optimized_apply = orig
    assert calls['n'] == n_fused, f"expected {n_fused} fused kernel calls, saw {calls['n']}"
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU/Triton kernels")
def test_e97_fused_loud_guard_no_silent_eager():
    # With the bf16 cast disabled, an fp32 training input must RAISE rather than
    # silently fall back to the eager T-scan.
    layer = TypedHeadMixtureLayer(
        dim=128, n_state=32, n_heads=16,
        head_type_logits=[-30, -30, -30, -30, -30, -30, 1.0, 1.0],
        use_triton_e97=True, cast_recurrent_bf16=False,
    ).to(device='cuda', dtype=torch.float32)
    layer.train()
    x = torch.randn(2, 64, 128, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        layer(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU/Triton kernels")
def test_e97_within_layer_parity_unaligned_T():
    # _run_e97 zero-pads the causal time axis to the checkpoint interval and
    # truncates back, so padded-fused must match eager at an UNALIGNED T=511.
    dev = 'cuda'
    torch.manual_seed(1)
    layer = TypedHeadMixtureLayer(
        dim=256, n_state=32, n_heads=16,
        head_type_logits=[-30, -30, -30, -30, -30, -30, 1.0, 1.0],
        use_triton_e97=True, cast_recurrent_bf16=True,
    ).to(device=dev, dtype=torch.bfloat16)
    layer.train()
    x0 = torch.randn(2, 511, 256, device=dev, dtype=torch.bfloat16)
    xf = x0.clone().requires_grad_(True)
    of = layer(xf); of.float().pow(2).mean().backward(); gf = xf.grad.clone()
    layer.e97_raw.use_triton = False
    layer.e97_delta.use_triton = False
    xe = x0.clone().requires_grad_(True)
    oe = layer(xe); oe.float().pow(2).mean().backward(); ge = xe.grad.clone()
    fwd = (torch.norm((of - oe).float()) / (torch.norm(oe.float()) + 1e-12)).item()
    grad = (torch.norm((gf - ge).float()) / (torch.norm(ge.float()) + 1e-12)).item()
    assert of.shape[1] == 511
    assert fwd < 3e-2 and grad < 3e-2, f"unaligned-T parity fwd={fwd:.2e} grad={grad:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU/FLA kernels")
def test_forward_backward_with_shell_heads():
    # gdn2_recall + gdn2_nonlin_shell mixture: the native GDN sub-block AND the
    # fused nonlinear-state shell sub-block both present and trainable.
    dev = 'cuda'
    layer = TypedHeadMixtureLayer(
        dim=128, n_state=32, n_heads=24,
        head_type_logits=[math.log(2 / 3), -30, -30, -30, -30, math.log(1 / 3)],
        shell_state_nonlin='tanh', shell_state_chunk=64).to(dev)
    assert layer.gdn is not None and layer.shell is not None and layer.unified is None
    assert layer.alloc['n_shell'] > 0
    x = torch.randn(2, 64, 128, device=dev, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 64, 128)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
