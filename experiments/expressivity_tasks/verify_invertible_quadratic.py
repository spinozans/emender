"""CPU-only, seeded independent-trajectory suffix-cap measurements; no torch.

Run from the repo root with `python -m experiments.expressivity_tasks.verify_invertible_quadratic`.
Use --output to retain JSON; each trajectory contributes ONE terminal observation.
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np

from experiments.expressivity_tasks.tasks.invertible_quadratic import (
    PILOT_PRIMES, PRODUCTION_P, InvertibleQuadraticOneStepTask,
    InvertibleQuadraticTask, OldParitySuffixBaseline, SuffixBaselinePredictor,
    trajectory,
)


def measure(p: int, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    baseline = SuffixBaselinePredictor(p)
    old = OldParitySuffixBaseline()
    hits = {"suffix_0": 0, "suffix_1": 0, "suffix_7": 0, "old_parity_suffix_transfer": 0}
    for start in range(0, samples, 10_000):
        B = min(10_000, samples - start)
        # Fix the KNOWN initial state: chance accuracy must come from hidden
        # drivers, not merely averaging over randomized initialization.
        initial = np.ones((B, 2), dtype=np.int64)
        drivers = rng.integers(p, size=(B, 15))
        truth = trajectory(initial, drivers, p)[:, -1]
        for k in (0, 1, 7):
            suffix = drivers[:, 15 - k:]
            hits[f"suffix_{k}"] += int(np.all(baseline.predict(suffix) == truth, axis=1).sum())
        prediction = old.predict_invertible(initial, drivers, p)
        hits["old_parity_suffix_transfer"] += int(np.all(prediction == truth, axis=1).sum())
    cap = 1 / p**2
    sigma = sqrt(samples * cap * (1 - cap))
    rows = {}
    for name, count in hits.items():
        # A fixed, predeclared six-sigma sampling check, NOT proof of the cap.
        consistent = abs(count - samples * cap) <= 6 * sigma + 1
        rows[name] = dict(hits=count, accuracy=count / samples,
                          binomial_z=(count - samples * cap) / sigma,
                          within_six_sigma=consistent)
        if not consistent:
            raise AssertionError(f"p={p} {name} inconsistent with cap: {rows[name]}")
    # Data/oracle pilots, not learned performance: full prefix + driver flow at
    # train, x4 and x8 DEPTHS. Token lengths include two initialization tokens.
    pilots = []
    for depth in (128, 512, 1024):
        inputs, targets, mask = InvertibleQuadraticTask(p).generate_batch(32, depth + 2, rng)
        expected = trajectory(inputs[:, :2] - [p, 2 * p], inputs[:, 2:], p)
        exact = np.array_equal(targets[:, 2:], expected[..., 0] * p + expected[..., 1])
        if not exact or mask.sum() != 32 * depth:
            raise AssertionError("CPU pilot failed")
        pilots.append(dict(drivers=depth, tokens=depth + 2, oracle_exact=exact))
    inputs, targets, _ = InvertibleQuadraticOneStepTask(p).generate_batch(10_000, 3, rng)
    state = trajectory(inputs[:, :2] - [p, 2 * p], inputs[:, 2:], p)[:, 0]
    if not np.array_equal(targets[:, 2], state[:, 0] * p + state[:, 1]):
        raise AssertionError("one-step oracle failed")
    return dict(p=p, states=p**2, samples=samples, seed=seed,
                drivers=15, known_initial_state=[1, 1], theoretical_cap=cap, predictors=rows,
                cpu_data_pilots=pilots, one_step_oracle_cases=10_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 100_000:
        parser.error("use >=100000 independent trajectories for the cap check")
    result = dict(stage=1, gpu_work=False, training_launched=False,
                  measurements=[measure(p, args.samples, args.seed + p)
                                for p in (*PILOT_PRIMES, PRODUCTION_P)])
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
