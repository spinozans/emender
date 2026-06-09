"""Positional clock — the purest native-clock / relative-position witness.

Each sequence is handed a random START PHASE s in {0..K-1} at position 0 (token
s); every later position is a filler token. The target at position t is the
running clock value

    y_t = (s + t) mod K

So the model must read the initial phase and then ADVANCE A MOD-K CLOCK with no
further input — there is nothing to count except elapsed position. The only state
that solves this exactly is a rotation by theta = 2*pi/K per step: a complex
eigenvalue lambda = e^{i 2*pi/K} returns to phase 0 every K steps and visits K
distinct phases, so its (Re, Im) read-out lands on K distinguishable points of
the unit circle. A REAL eigenvalue can only realize period <= 2 (lambda>0 is
monotone; lambda<0 alternates = mod-2), so a real-eigenvalue cell is capped at
parity and cannot track a mod-K>2 clock. This is the cleanest test of whether the
complex/rotation axis unlocks a capability the real cell cannot reach
(task complex-eig-capability).

Token layout: 0..K-1 = start-phase symbol at t=0 AND the K target classes;
K = filler token (input only). vocab = K+1. Random start phase per sequence
prevents pure memorization and makes batches non-degenerate.

REAL deterministic generator; no mocks. Exactly solvable autonomous mod-K clock.
"""
import numpy as np


class PositionalClockTask:
    def __init__(self, K: int = 6):
        assert K >= 2
        self.K = int(K)
        self.name = "positional_clock"
        self.filler = self.K
        self.vocab_size = self.K + 1

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        s = rng.integers(0, self.K, size=(B,)).astype(np.int64)      # start phase
        inputs = np.full((B, T), self.filler, dtype=np.int64)
        inputs[:, 0] = s                                              # phase handed at t=0
        t = np.arange(T, dtype=np.int64)[None, :]                    # [1,T]
        targets = (s[:, None] + t) % self.K                          # [B,T] clock value
        mask = np.ones((B, T), dtype=bool)                           # running supervision
        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.K
