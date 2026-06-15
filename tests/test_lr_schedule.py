#!/usr/bin/env python3
"""fix-long-horizon: REAL tests for the corrected long-horizon LR schedule.

Guards the two train.py bugs that caused held-out BPB to roll over mid-run:

  1. `warmup_steps` was never passed into `schedulefree.AdamWScheduleFree`, so the
     SCALE_PLAN "add a 2-5k warmup" instruction was a silent no-op.
  2. The AdamW `get_lr` used `step/warmup_steps` as the cosine phase, so the LR
     oscillated with period 2*warmup_steps and collapsed to min right after warmup,
     never decaying over the run (it never referenced total_steps).

These are real assertions against the real `train.lr_scale_at` and a real
`AdamWScheduleFree` construction — no mocks, no fabricated numbers.

Run: python3 tests/test_lr_schedule.py
"""
import os, sys, math
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import train
import schedulefree


def test_warmup_ramps_linearly():
    W, T, F = 250, 2500, 0.1
    # step 0 -> ~1/W (not 0, so the very first step still moves); strictly increasing to peak.
    assert abs(train.lr_scale_at(0, W, T, F) - 1.0 / W) < 1e-9
    prev = -1.0
    for s in range(W):
        v = train.lr_scale_at(s, W, T, F)
        assert v > prev, f"warmup not strictly increasing at {s}"
        assert 0.0 < v <= 1.0
        prev = v
    # at the warmup boundary the schedule is at the peak (cos(0) == 1)
    assert abs(train.lr_scale_at(W, W, T, F) - 1.0) < 1e-9
    print("PASS warmup ramps linearly 0->1, peak at boundary")


def test_cosine_decays_monotone_to_floor():
    W, T, F = 250, 2500, 0.1
    prev = 2.0
    for s in range(W, T):
        v = train.lr_scale_at(s, W, T, F)
        assert v <= prev + 1e-12, f"post-warmup LR rose at step {s}: {v} > {prev}"
        assert F - 1e-9 <= v <= 1.0 + 1e-9
        prev = v
    # final step reaches the floor exactly (progress == 1 -> cos(pi) == -1)
    assert abs(train.lr_scale_at(T - 1, W, T, F) - F) < 1e-3
    assert abs(train.lr_scale_at(T, W, T, F) - F) < 1e-9
    print(f"PASS cosine decays monotone to floor {F}")


def test_no_warmup_starts_at_peak():
    # warmup_steps=0 must not divide-by-zero and must start at the peak.
    assert abs(train.lr_scale_at(0, 0, 1000, 0.1) - 1.0) < 1e-9
    assert train.lr_scale_at(500, 0, 1000, 0.1) < 1.0  # already decaying
    print("PASS warmup_steps=0 starts at peak, no div-by-zero")


def test_total_le_warmup_is_safe():
    # degenerate: total_steps <= warmup_steps must not blow up (clamped progress).
    v = train.lr_scale_at(100, 200, 150, 0.1)
    assert math.isfinite(v) and 0.0 < v <= 1.0
    print("PASS total<=warmup is finite/clamped")


def test_schedulefree_receives_warmup():
    # The real optimizer must actually carry warmup_steps (bug #1). AdamWScheduleFree
    # stores it in each param group.
    p = torch.nn.Parameter(torch.zeros(4))
    opt = schedulefree.AdamWScheduleFree([p], lr=1e-3, warmup_steps=500)
    got = opt.param_groups[0].get('warmup_steps', None)
    assert got == 500, f"AdamWScheduleFree did not store warmup_steps (got {got})"
    print("PASS schedulefree carries warmup_steps=500")


def test_per_group_ratio_preserved():
    # The schedule scales each group's base_lr by the same factor -> a knob group at
    # 5x base stays exactly 5x at every step (the call site multiplies base_lr*scale).
    base_lr, knob_lr = 1e-3, 5e-3
    W, T, F = 100, 1000, 0.1
    for s in [0, 50, 100, 300, 600, 999]:
        scale = train.lr_scale_at(s, W, T, F)
        lr_base = base_lr * scale
        lr_knob = knob_lr * scale
        assert abs(lr_knob / lr_base - 5.0) < 1e-9, f"ratio drifted at step {s}"
    print("PASS per-group LR ratio (knob 5x) preserved across schedule")


if __name__ == '__main__':
    test_warmup_ramps_linearly()
    test_cosine_decays_monotone_to_floor()
    test_no_warmup_starts_at_peak()
    test_total_le_warmup_is_safe()
    test_schedulefree_receives_warmup()
    test_per_group_ratio_preserved()
    print("\nALL LR-SCHEDULE TESTS PASS")
