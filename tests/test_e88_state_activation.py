"""Tests for the E88 ``state_activation`` knob (e88-nonsat task).

The hypothesis under test: E88's default ``tanh`` state SATURATES, bounding the
matrix-state magnitude (finite-state expressivity, cannot count). Swapping in a
NON-SATURATING nonlinearity (relu / softplus) lets the state grow without bound,
so a counter can accumulate. These tests verify the implementation is real and
that the saturating-vs-non-saturating distinction holds at the state-update level.

All CPU, fp32 — the PyTorch reference recurrence path that the counting/S5
experiments use (bf16 CUDA/Triton kernels are skipped under disable_autocast).
"""
import math

import pytest
import torch

from ndm.models.e88_fla_hybrid import E88FLAHybrid


def test_state_activation_resolution():
    """Backwards-compat + explicit resolution of the knob."""
    # Default (unset) keeps historical behaviour.
    assert E88FLAHybrid(dim=64, n_state=16, n_heads=4).state_activation == 'tanh'
    assert E88FLAHybrid(dim=64, n_state=16, n_heads=4,
                        linear_state=True).state_activation == 'identity'
    # 'linear'/'affine' alias -> identity, and identity forces linear_state True.
    m = E88FLAHybrid(dim=64, n_state=16, n_heads=4, state_activation='linear')
    assert m.state_activation == 'identity' and m.linear_state is True
    # Non-saturating flags.
    for act in ('relu', 'softplus'):
        m = E88FLAHybrid(dim=64, n_state=16, n_heads=4, state_activation=act)
        assert m.state_activation == act and m._nonsat_state is True
    for act in ('tanh', 'identity'):
        m = E88FLAHybrid(dim=64, n_state=16, n_heads=4, state_activation=act)
        assert m._nonsat_state is False
    with pytest.raises(ValueError):
        E88FLAHybrid(dim=64, n_state=16, n_heads=4, state_activation='sigmoid')


def test_apply_state_activation_matches_definition():
    m = E88FLAHybrid(dim=64, n_state=16, n_heads=4)
    x = torch.linspace(-5, 5, 101)
    for act, ref in [
        ('tanh', torch.tanh),
        ('identity', lambda z: z),
        ('relu', torch.relu),
        ('softplus', torch.nn.functional.softplus),
    ]:
        m.state_activation = act
        assert torch.allclose(m._apply_state_activation(x), ref(x), atol=1e-6)


def test_saturating_vs_nonsaturating_accumulation():
    """The load-bearing property: with decay=1 and a constant positive write,
    tanh SATURATES to a bounded ceiling while relu/softplus/identity accumulate
    without bound (a counter can grow). This is the Weiss-Goldberg-Yahav (2018)
    distinction between bounded (finite-state) and unbounded (counter) cells.
    """
    m = E88FLAHybrid(dim=64, n_state=16, n_heads=4)
    inc = torch.full((1, 1, 4, 4), 0.5)
    steps = 500
    finals = {}
    for act in ('tanh', 'identity', 'relu', 'softplus'):
        m.state_activation = act
        S = torch.zeros(1, 1, 4, 4)
        for _ in range(steps):
            S = m._apply_state_activation(1.0 * S + inc)  # decay = 1
        finals[act] = S.max().item()

    # tanh is bounded by 1 (saturating) no matter how many increments.
    assert finals['tanh'] <= 1.0 + 1e-4
    # The non-saturating cells grow ~linearly with the number of steps, far
    # beyond the tanh ceiling -> they hold an unbounded count.
    assert finals['relu'] > 100.0
    assert finals['softplus'] > 100.0
    assert finals['identity'] > 100.0
    # relu accumulates the exact integer count (0.5 * steps) since every
    # pre-activation is positive.
    assert math.isclose(finals['relu'], 0.5 * steps, rel_tol=1e-4)


@pytest.mark.parametrize('act', ['tanh', 'identity', 'relu', 'softplus'])
def test_forward_and_backward_run(act):
    """Full layer forward+backward in fp32 for every state activation."""
    torch.manual_seed(0)
    m = E88FLAHybrid(dim=64, n_state=16, n_heads=4, state_activation=act)
    m.disable_autocast = True
    x = torch.randn(2, 48, 64, requires_grad=True)
    out, hidden = m(x)
    assert out.shape == (2, 48, 64)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_nonsat_rejects_triton_path():
    """Non-saturating state must fail loudly on the bf16 kernel route rather
    than silently running tanh."""
    m = E88FLAHybrid(dim=64, n_state=16, n_heads=4,
                     state_activation='relu', use_triton=True)
    x = torch.randn(1, 8, 64)
    with pytest.raises(RuntimeError, match='reference recurrence'):
        m(x)
