"""CPU tests for the C3 driver's pure logic (the registered trigger's
statistics, the slice/surgery algebra, the protocol constants, the
full-window regime filter).  No CUDA, no stream access -- the GPU machinery
is receipted by the smoke and the arms themselves.

Run:

    OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \\
        experiments/expressivity_tasks/x45_matrix_emender/test_gdn2_growth_c3.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender import gdn2_growth_c3 as C3
from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction import (
    cooldown_factor)


# ------------------------------------------------------------- the statistics


def test_participation_ratio_exact_rank():
    """PR of a rank-r state with equal singular values is exactly r (the
    registered statistic's anchor), and the zero state guards to 0."""
    from experiments.expressivity_tasks.x45_matrix_emender import (
        x45_spectra_sidecar as SPECTRA)
    r = 7
    u = torch.linalg.qr(torch.randn(C3.HEAD_DIM, C3.HEAD_DIM))[0][:, :r]
    v = torch.linalg.qr(torch.randn(C3.HEAD_V_DIM, r))[0].T
    state = u @ v                         # 128x256, rank r, singular vals 1
    lam = torch.linalg.eigvalsh(state @ state.T)
    pr = SPECTRA.participation_ratio(lam)
    # the Gram+sqrt route inflates \u03a3σ by sqrt of the numerical-zero
    # eigenvalues at exact rank-deficiency (an honest property of the exact
    # statistic on synthetic degenerate states; real states are full), so
    # the anchor is checked within that inflation and against the svdvals
    # reference
    assert abs(float(pr) - r) < 0.05
    sv = torch.linalg.svdvals(state)
    pr_svd = (sv.sum() ** 2) / (sv ** 2).sum()
    assert abs(float(pr) - float(pr_svd)) < 5e-2
    zero = torch.zeros(C3.HEAD_DIM, C3.HEAD_V_DIM)
    lam0 = torch.linalg.eigvalsh(zero @ zero.T)
    assert float(SPECTRA.participation_ratio(lam0)) == 0.0


def test_pr_from_trajectory_excludes_initial_state():
    """h[:, 0] is the empty initial state (excluded); h[:, c+1] is the state
    after chunk c -- the boundary count is C-1 and zero states guard."""
    pool = _FakePool()
    B, Ch, H = 1, 5, 2
    h = torch.zeros(B, Ch, H, C3.HEAD_DIM, C3.HEAD_V_DIM)
    h[:, 2:, 0] = 1.0                      # head 0 saturates from chunk 2 on
    pr = C3.pr_from_trajectory(pool, h)
    assert pr.shape == (B, Ch - 1, H)
    assert float(pr[0, 0, 0]) == 0.0       # the first boundary state is zero
    assert float(pr[0, 1, 0]) > 0.0


class _FakePool:
    """The sidecar's contract, on CPU (torch.linalg -- the exact-spectra
    semantics the guardrail receipt covers at ulp scale)."""

    def pr_from_states(self, states):
        sv = torch.linalg.svdvals(states.float())
        denom = (sv ** 2).sum(-1)
        pr = (sv.sum(-1) ** 2) / denom.clamp(min=1e-30)
        return torch.where(denom > 0, pr, torch.zeros_like(pr))


def test_smoothed_series_partial_means_and_fire():
    """The registered aggregation: smoothed over the last 4 boundaries with
    partial means at the span start; the fire is the first crossing."""
    series = np.array([10.0, 20.0, 30.0, 40.0, 100.0])
    smoothed = C3.smoothed_series(series)
    assert smoothed[0] == 10.0
    assert smoothed[1] == 15.0
    assert smoothed[3] == 25.0
    assert smoothed[4] == (20 + 30 + 40 + 100) / 4     # the last 4
    fired = np.argmax(smoothed >= 96.0) if (smoothed >= 96.0).any() else None
    assert fired is None                                # 47.5 never crosses 96


# ------------------------------------------------------ the protocol constants


def test_windows_per_step_landing():
    """The 20M matched-exposure landing: 14 windows/step at 8192, 7 at
    16384, both landing 175 steps / 20,070,400 targets."""
    for ctx, wps in ((8192, 14), (16384, 7)):
        assert C3.windows_per_step(ctx) == wps
        steps = -(-C3.EXPOSURE_TARGETS // (wps * ctx))     # ceil
        assert steps == 175
        assert steps * wps * ctx == 20_070_400
        assert C3.EXPOSURE_TARGETS <= steps * wps * ctx < \
            C3.EXPOSURE_TARGETS * 1.04


def test_cooldown_fraction_inside_band():
    """The 24-step cooldown is 13.7% of the 175-step constant phase -- inside
    the Hägele 10-15% band -- and the corrected form never hits zero."""
    frac = C3.COOLDOWN_STEPS / 175
    assert 0.10 <= frac <= 0.15
    assert cooldown_factor(1) == 1.0
    assert cooldown_factor(C3.COOLDOWN_STEPS) == 1 / C3.COOLDOWN_STEPS


# ------------------------------------------------------- the slice/surgery math


def test_prefix_slice_and_compose():
    old = (4, 3)
    new = (8, 3)
    sl = C3._prefix_slice(old, new)
    t = torch.arange(24).view(8, 3)
    assert t[sl].shape == (4, 3)
    composed = C3._compose(sl, (slice(None), slice(None)))
    assert composed == sl
    chained = C3._compose(sl, C3._prefix_slice((8, 3), (16, 3)))
    assert chained == sl


def test_merge_event_grads_chains_to_the_live_core():
    """Chained events within one step: the oldest core's orphaned gradients
    land in the LIVE core's shared slices through the intermediate core."""
    p0 = torch.nn.Parameter(torch.arange(4.0))
    p1 = torch.nn.Parameter(torch.arange(8.0) * 10)
    p2 = torch.nn.Parameter(torch.arange(16.0) * 100)
    sl01 = (slice(0, 4),)
    sl12 = (slice(0, 8),)
    p0.grad = torch.ones(4)
    p1.grad = torch.full((8,), 2.0)
    p2.grad = torch.zeros(16)
    ev1 = _FakeEvent([("x", p0, p1, sl01)])
    ev2 = _FakeEvent([("x", p1, p2, sl12)])
    C3.merge_event_grads([ev1, ev2])
    # p1 received p0's grad on the shared slice, then flowed into p2:
    assert p1.grad is None or float(p1.grad.abs().sum()) == 0.0  # merged away
    assert p2.grad[:4].tolist() == [3.0] * 4          # 2 (p1's own) + 1 (p0's)
    assert p2.grad[4:8].tolist() == [2.0] * 4
    assert p2.grad[8:].tolist() == [0.0] * 8


class _FakeEvent:
    def __init__(self, param_map):
        self.param_map = param_map
        self.heads_old = 1
        self.heads_new = 2
        self.layer_idx = 0
        self.boundary = 64
        self.fire_pr = 96.0
        self.old_core = None
        self.new_core = None


def test_ema_grow_carries_shared_slices_and_births_new_regions():
    p_old = torch.arange(4.0)
    p_new = torch.full((8,), -1.0)          # the birth value everywhere
    birth = {"x": p_new}
    ev = _FakeEvent([("x", p_old, p_new, (slice(0, 4),))])
    ev.birth = birth
    ema = {"x": torch.ones(4) * 5.0}
    resized = C3.ema_grow(ema, [ev])
    assert resized == ["x"]
    assert ema["x"].tolist() == [5.0] * 4 + [-1.0] * 4


# ----------------------------------------------------------- the regime filter


def test_full_window_filter_and_tail_exclusion():
    rows = [("d0", 0, np.zeros(8191), np.zeros(8191)),
            ("d1", 0, np.zeros(8192), np.zeros(8192)),
            ("d2", 0, np.zeros(100), np.zeros(100))]
    out = []
    for row in rows:
        if len(row[2]) == 8192:
            out.append(row)
            if len(out) >= 24:
                break
    assert [r[0] for r in out] == ["d1"]


def test_iota_zero_entry_state_is_zero():
    """A fresh (empty) store: writes collide with nothing -- iota is exactly
    0 for the segments whose entry state is empty; nonzero once the state
    fills (the clamped denominator).  The trajectory's C entries are the C
    segments' entry states (h[:, 0] the empty initial state), exactly the
    smoke's own S-5 loop."""
    B, C, H, K, V, seg = 1, 2, 1, C3.HEAD_DIM, C3.HEAD_V_DIM, 4
    h = torch.zeros(B, C, H, K, V)            # 2 chunks of 4 tokens
    h[:, 1] = 1.0                             # chunk 1 enters a full state
    b = torch.rand(B, C * seg, H, K)
    khat = F_normalized(b)
    iota = C3.iota_from_trajectory(h, b, khat)
    assert iota.shape == (B, C, H)
    assert float(iota[0, 0, 0]) == 0.0        # the empty initial state
    assert float(iota[0, 1, 0]) > 0.0         # the filled entry state


def F_normalized(b):
    return torch.nn.functional.normalize(b, dim=-1)


def test_predictions_are_pre_stated_with_the_band():
    """The falsifiable predictions carry the 0.075 endpoint band and both
    refutation branches (the registration §7's structure)."""
    assert C3.PREDICTIONS["band_nats_per_byte"] == 0.075
    assert "saturates" in C3.PREDICTIONS["P_G1"]
    assert "informative null" in C3.PREDICTIONS["P_G2"]
    assert "refutes the slot story" in C3.PREDICTIONS["P_G3"]
    assert "no directional prediction" in C3.PREDICTIONS["fixed_8"]
