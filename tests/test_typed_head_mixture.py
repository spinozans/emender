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
    # explicit 6-vector -> ~uniform across all six types (48/6 == 8 each)
    a = allocate_types(48, [0.0] * 6)
    assert sum(a['counts'].values()) == 48
    assert all(7 <= a['counts'][t] <= 9 for t in TYPE_NAMES)
    # legacy 5-vector -> the five active types balance, shell stays 0
    a5 = allocate_types(48, [0.0] * 5)
    assert a5['counts']['gdn2_nonlin_shell'] == 0
    assert all(9 <= a5['counts'][t] <= 10 for t in TYPE_NAMES[:5])


def test_shell_allocation_6_logits():
    # 6-vector that asks for ~1/3 shell heads -> a real shell allocation
    a = allocate_types(102, [math.log(2 / 3), -30.0, -30.0, -30.0, -30.0, math.log(1 / 3)])
    assert a['n_shell'] > 0
    assert a['counts']['gdn2_nonlin_shell'] == a['n_shell']
    assert a['counts']['gdn2_recall'] + a['n_shell'] == 102
    assert a['n_unified'] == 0


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
