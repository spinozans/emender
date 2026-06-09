"""Periodic pattern detection — predict-the-next-symbol on a repeating motif.

Each sequence draws a hidden PERIOD p uniformly from {2..K} and a random MOTIF
of length p over an alphabet of A symbols; the visible stream is that motif
repeated to fill T:

    stream[t] = motif[t mod p]
    y_t       = stream[t+1]        # next-symbol prediction (shifted)

To predict the next symbol the model must lock onto the (per-sequence, unknown)
period and phase. A complex eigenvalue lambda = r*e^{i*theta} supplies an
input-dependent phase clock (theta from theta_proj selects the period, the
rotation tracks the phase), so it can phase-lock to an arbitrary period; a real
eigenvalue cell has no phase and can only exploit period <= 2. Supervision starts
at t = K (one full max-period motif has been observed, so every period in {2..K}
is determinable) to keep the comparison fair across periods.

Token layout: 0..A-1 = motif symbols (= target classes). vocab = A. With A=K the
alphabet matches the period range. Random motif + period per sequence -> batches
are non-degenerate and cannot be memorized.

REAL deterministic generator; no mocks. Exactly solvable periodic automaton.
"""
import numpy as np


class PeriodicPatternTask:
    def __init__(self, K: int = 6, A: int = 0):
        assert K >= 2
        self.K = int(K)                       # maximum period
        self.A = int(A) if A and A >= 2 else self.K   # alphabet size (default = K)
        self.name = "periodic_pattern"
        self.vocab_size = self.A
        self._warmup = self.K                 # first supervised position

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        periods = rng.integers(2, self.K + 1, size=(B,)).astype(np.int64)   # p in [2,K]
        stream = np.zeros((B, T + 1), dtype=np.int64)
        t = np.arange(T + 1, dtype=np.int64)[None, :]                       # [1,T+1]
        for b in range(B):
            p = int(periods[b])
            motif = rng.integers(0, self.A, size=(p,)).astype(np.int64)
            stream[b] = motif[t[0] % p]
        inputs = stream[:, :T].copy()
        targets = stream[:, 1:].copy()                                      # next-symbol
        mask = np.zeros((B, T), dtype=bool)
        warm = min(self._warmup, T - 1)
        mask[:, warm:] = True                                               # predictable region
        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.A
