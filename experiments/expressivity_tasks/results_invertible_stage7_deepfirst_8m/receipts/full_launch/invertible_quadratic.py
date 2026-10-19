"""Invertible nonlinear full-state tracking on F_p^2 (CPU-only data).

Inputs: two coordinate tokens ONCE, then T-2 full-field drivers. Targets
at driver positions encode the ENTIRE resulting state as p*x+y. Target states
are never fed back; initial coordinates use a disjoint driver-token range. T counts
all tokens, including the two-token prefix, as in the generic task interface.
See ../INVERTIBLE_QUADRATIC.md for the suffix lemma and interpretation limits.
"""
from __future__ import annotations

from math import isqrt

import numpy as np


PRODUCTION_P = 61
PILOT_PRIMES = (13, 31)


def _check_prime(p: int) -> None:
    if not isinstance(p, int) or p < 3 or any(p % d == 0 for d in range(2, isqrt(p) + 1)):
        raise ValueError("p must be an odd prime")


def forward(state, driver, p: int):
    """Vectorized (x,y),c -> (y,x+y*y+c) mod p; last axis is coordinates."""
    state = np.asarray(state, dtype=np.int64)
    x, y = state[..., 0], state[..., 1]
    return np.stack((y, (x + y * y + driver) % p), axis=-1)


def inverse(state, driver, p: int):
    """Recover old y=x_next and old x=y_next-x_next^2-c exactly."""
    state = np.asarray(state, dtype=np.int64)
    x_next, y_next = state[..., 0], state[..., 1]
    return np.stack(((y_next - x_next * x_next - driver) % p, x_next), axis=-1)


def trajectory(initial, drivers, p: int):
    """Return both coordinates after every driver, shape [B,L,2]."""
    state = np.asarray(initial, dtype=np.int64).copy()
    drivers = np.asarray(drivers, dtype=np.int64)
    states = np.empty((*drivers.shape, 2), dtype=np.int64)
    for t in range(drivers.shape[1]):
        state = forward(state, drivers[:, t], p)
        states[:, t] = state
    return states


class InvertibleQuadraticTask:
    def __init__(self, p: int = PRODUCTION_P):
        _check_prime(p)
        self.p = p
        self.name = f"invertible_quadratic_p{p}"
        # Drivers: [0,p); initial x: [p,2p); initial y: [2p,3p).
        # Targets: [0,p^2), jointly encode both coordinates, not a marginal.
        self.vocab_size = max(p * p, 3 * p)

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        if B < 1 or T < 3:
            raise ValueError("need B>=1 and T>=3 (two prefix tokens and a driver)")
        initial = rng.integers(self.p, size=(B, 2), dtype=np.int64)
        drivers = rng.integers(self.p, size=(B, T - 2), dtype=np.int64)
        inputs = np.concatenate((initial + np.array([self.p, 2 * self.p]), drivers), axis=1)
        states = trajectory(initial, drivers, self.p)
        targets = np.zeros((B, T), dtype=np.int64)
        targets[:, 2:] = states[..., 0] * self.p + states[..., 1]
        mask = np.ones((B, T), dtype=bool)
        mask[:, :2] = False
        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.p**2


class InvertibleQuadraticOneStepTask(InvertibleQuadraticTask):
    """Only (x0,y0),c -> (x1,y1); requires exactly three input tokens."""
    def __init__(self, p: int = PRODUCTION_P):
        super().__init__(p)
        self.name += "_one_step"

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        if T != 3:
            raise ValueError("one-step diagnostic requires T=3")
        return super().generate_batch(B, T, rng)


class SuffixBaselinePredictor:
    """A MAP tie-break: propagate canonical (0,0) through ONLY the suffix.

    Once >=2 prefix drivers are hidden, the conditional full-state posterior
    is uniform, even given the initial state; every deterministic guess is MAP.
    This interface deliberately cannot inspect targets or forgotten drivers.
    """
    def __init__(self, p: int = PRODUCTION_P):
        _check_prime(p)
        self.p = p

    def predict(self, suffix):
        suffix = np.asarray(suffix, dtype=np.int64)
        state = np.zeros((suffix.shape[0], 2), dtype=np.int64)
        for c in suffix.T:
            state = forward(state, c, self.p)
        return state


class OldParitySuffixBaseline:
    """Exact old x'=x^2+c mod 256 via one parity bit plus seven drivers.

    This is NOT suffix-only: parity summarizes the omitted prefix. The port
    to F_p^2 below is an empirical negative control, not covered by the lemma.
    """
    suffix_length = 7

    def predict_old(self, initial, drivers):
        drivers = np.asarray(drivers, dtype=np.int64)
        initial = np.asarray(initial, dtype=np.int64)
        split = max(0, drivers.shape[1] - self.suffix_length)
        if split:
            state = (initial + drivers[:, :split].sum(axis=1)) % 2
        else:
            state = initial.copy()
        for c in drivers[:, split:].T:
            state = (state * state + c) % 256
        return state

    def predict_invertible(self, initial, drivers, p: int):
        """Transfer the old one-bit summary, not the actual hidden new state."""
        drivers = np.asarray(drivers, dtype=np.int64)
        split = max(0, drivers.shape[1] - self.suffix_length)
        if split:
            bit = (np.asarray(initial).sum(axis=1) + drivers[:, :split].sum(axis=1)) % 2
            state = np.stack((bit, np.zeros_like(bit)), axis=-1)
        else:
            state = np.asarray(initial, dtype=np.int64).copy()
        for c in drivers[:, split:].T:
            state = forward(state, c, p)
        return state
