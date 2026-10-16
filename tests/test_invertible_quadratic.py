"""Source-named CPU tests for the invertible quadratic benchmark contract."""
from itertools import product

import numpy as np
import pytest

from experiments.expressivity_tasks.tasks import ALL_TASKS
from experiments.expressivity_tasks.tasks.invertible_quadratic import (
    PILOT_PRIMES, PRODUCTION_P, InvertibleQuadraticOneStepTask,
    InvertibleQuadraticTask, OldParitySuffixBaseline, SuffixBaselinePredictor,
    forward, inverse, trajectory,
)
from experiments.expressivity_tasks.verify_invertible_quadratic import measure


def all_states(p):
    return np.array(list(product(range(p), repeat=2)), dtype=np.int64)


@pytest.mark.parametrize("p", [3, 5, 7])
def test_exact_inverse_exhaustive_small_fields(p):
    states = all_states(p)
    for c in range(p):
        np.testing.assert_array_equal(inverse(forward(states, c, p), c, p), states)
        np.testing.assert_array_equal(forward(inverse(states, c, p), c, p), states)


@pytest.mark.parametrize("p", [13, 31, PRODUCTION_P])
def test_exact_inverse_10000_random_larger_fields(p):
    rng = np.random.default_rng(510 + p)
    states = rng.integers(p, size=(10_000, 2))
    c = rng.integers(p, size=10_000)
    np.testing.assert_array_equal(inverse(forward(states, c, p), c, p), states)
    np.testing.assert_array_equal(forward(inverse(states, c, p), c, p), states)


@pytest.mark.parametrize("p", [3, 5, 7])
def test_no_collision_exhaustive_each_driver_and_shared_sequence(p):
    states = all_states(p)
    for c in range(p):
        assert len(np.unique(forward(states, c, p), axis=0)) == p**2
    for c in np.random.default_rng(81).integers(p, size=128):
        states = forward(states, c, p)
        assert len(np.unique(states, axis=0)) == p**2


def test_no_collision_randomized_production_prime():
    p = PRODUCTION_P
    rng = np.random.default_rng(818)
    for c in rng.integers(p, size=100):
        states = all_states(p)[rng.choice(p**2, size=2000, replace=False)]
        assert len(np.unique(forward(states, c, p), axis=0)) == len(states)
    states = all_states(p)[rng.choice(p**2, size=2000, replace=False)]
    for c in rng.integers(p, size=128):
        states = forward(states, c, p)
        assert len(np.unique(states, axis=0)) == len(states)


@pytest.mark.parametrize("p", [3, 5, 7])
def test_two_unknown_drivers_uniform_and_suffix_map_accuracy_exact(p):
    unknown = all_states(p)  # all equally likely (c1,c2)
    for initial in all_states(p):
        states = trajectory(np.broadcast_to(initial, (p**2, 2)), unknown, p)[:, -1]
        assert len(np.unique(states, axis=0)) == p**2
        for k in (0, 1, 7):
            suffix = np.random.default_rng(k).integers(p, size=k)
            result = states.copy()
            for c in suffix:
                result = forward(result, c, p)
            prediction = SuffixBaselinePredictor(p).predict(np.broadcast_to(suffix, (p**2, k)))
            assert np.all(result == prediction, axis=1).sum() == 1


@pytest.mark.parametrize("p", [*PILOT_PRIMES, PRODUCTION_P])
def test_empirical_suffix_cap_and_old_parity_transfer(p):
    # Independent terminal observations; no overlapping correlated suffixes.
    result = measure(p, samples=200_000, seed=20261016 + p)
    assert all(row["within_six_sigma"] for row in result["predictors"].values())
    assert result["predictors"]["old_parity_suffix_transfer"]["accuracy"] < 0.02


def test_old_parity_suffix_exact_mod256_every_trajectory_position():
    rng = np.random.default_rng(256)
    initial = rng.integers(256, size=4096)
    drivers = rng.integers(256, size=(4096, 128))
    old = OldParitySuffixBaseline()
    state = initial.copy()
    for t in range(128):
        state = (state * state + drivers[:, t]) % 256
        np.testing.assert_array_equal(old.predict_old(initial, drivers[:, :t + 1]), state)
    # Original task used x0=1: explicitly calibrate that stream too.
    state = np.ones(4096, dtype=np.int64)
    for c in drivers.T:
        state = (state * state + c) % 256
    np.testing.assert_array_equal(old.predict_old(np.ones(4096, dtype=np.int64), drivers), state)


@pytest.mark.parametrize("p", [13, 31, PRODUCTION_P])
def test_old_parity_summary_cannot_reproduce_new_trajectories(p):
    # Fixed, known initial state; vary ONLY two omitted full-field drivers.
    # The old summary has <=2 outputs, while the new true state has p^2.
    prefix = all_states(p)
    suffix = np.broadcast_to(np.arange(7) % p, (p**2, 7))
    drivers = np.concatenate((prefix, suffix), axis=1)
    initial = np.ones((p**2, 2), dtype=np.int64)
    truth = trajectory(initial, drivers, p)[:, -1]
    guess = OldParitySuffixBaseline().predict_invertible(initial, drivers, p)
    assert len(np.unique(truth, axis=0)) == p**2
    assert len(np.unique(guess, axis=0)) <= 2
    assert np.all(guess == truth, axis=1).sum() <= 2


@pytest.mark.parametrize("p", [13, 31, PRODUCTION_P])
def test_full_state_targets_initial_once_no_teacher_forcing_and_determinism(p):
    task = InvertibleQuadraticTask(p)
    seed = 122
    stream_a, stream_b = np.random.default_rng(seed), np.random.default_rng(seed)
    actual = task.generate_batch(32, 130, stream_a)
    repeat = task.generate_batch(32, 130, stream_b)
    for a, b in zip(actual, repeat):
        np.testing.assert_array_equal(a, b)
    for B, T in ((7, 3), (8, 514), (2, 1026)):
        for a, b in zip(task.generate_batch(B, T, stream_a), task.generate_batch(B, T, stream_b)):
            np.testing.assert_array_equal(a, b)
    inputs, targets, mask = actual
    rng = np.random.default_rng(seed)
    initial = rng.integers(p, size=(32, 2), dtype=np.int64)
    drivers = rng.integers(p, size=(32, 128), dtype=np.int64)
    np.testing.assert_array_equal(inputs[:, :2], initial + [p, 2 * p])
    np.testing.assert_array_equal(inputs[:, 2:], drivers)
    assert not mask[:, :2].any() and mask[:, 2:].all()
    assert targets.min() >= 0 and targets.max() < p**2
    assert inputs.dtype == targets.dtype == np.int64 and mask.dtype == bool
    # Independent scalar oracle, not the implementation's vector recurrence.
    for b in range(32):
        x, y = map(int, initial[b])
        for t, c in enumerate(drivers[b], start=2):
            x, y = y, (x + y**2 + int(c)) % p
            assert divmod(int(targets[b, t]), p) == (x, y)
    assert task.random_baseline_acc() == 1 / p**2
    assert task.vocab_size >= max(inputs.max(), targets.max()) + 1


@pytest.mark.parametrize("p", [13, 31, PRODUCTION_P])
def test_full_field_driver_and_initial_coordinate_uniformity(p):
    B = 200_000
    inputs, _, _ = InvertibleQuadraticTask(p).generate_batch(B, 3, np.random.default_rng(902 + p))
    for values in (inputs[:, 0] - p, inputs[:, 1] - 2 * p, inputs[:, 2]):
        counts = np.bincount(values, minlength=p)
        sigma = np.sqrt(B / p * (1 - 1 / p))
        assert len(counts) == p and counts.min() > 0
        assert np.max(np.abs(counts - B / p)) < 6 * sigma


@pytest.mark.parametrize("p", [13, 31, PRODUCTION_P])
def test_one_step_diagnostic_full_state_oracle(p):
    task = InvertibleQuadraticOneStepTask(p)
    inputs, targets, mask = task.generate_batch(10_000, 3, np.random.default_rng(18))
    x, y, c = inputs[:, 0] - p, inputs[:, 1] - 2 * p, inputs[:, 2]
    np.testing.assert_array_equal(targets[:, 2] // p, y)
    np.testing.assert_array_equal(targets[:, 2] % p, (x + y * y + c) % p)
    assert mask.sum() == 10_000
    with pytest.raises(ValueError, match="T=3"):
        task.generate_batch(1, 4, np.random.default_rng(18))


@pytest.mark.parametrize("name,p,T", [
    ("invertible_quadratic", PRODUCTION_P, 130),
    ("invertible_quadratic_p13", 13, 130),
    ("invertible_quadratic_p31", 31, 130),
    ("invertible_quadratic_one_step", PRODUCTION_P, 3),
    ("invertible_quadratic_p13_one_step", 13, 3),
    ("invertible_quadratic_p31_one_step", 31, 3),
])
def test_task_variants_registered(name, p, T):
    task = ALL_TASKS[name]()
    assert task.p == p
    inputs, targets, mask = task.generate_batch(4, T, np.random.default_rng(1))
    assert inputs.shape == targets.shape == mask.shape == (4, T)


@pytest.mark.parametrize("p", [1, 2, 4, 9, 15, 256])
def test_non_prime_configurations_fail_closed(p):
    with pytest.raises(ValueError, match="odd prime"):
        InvertibleQuadraticTask(p)


@pytest.mark.parametrize("B,T", [(0, 3), (1, 2)])
def test_invalid_batch_shapes_fail_closed(B, T):
    with pytest.raises(ValueError):
        InvertibleQuadraticTask().generate_batch(B, T, np.random.default_rng(1))
