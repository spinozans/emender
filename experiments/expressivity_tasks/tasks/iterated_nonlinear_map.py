"""PROBE 3 — iterated nonlinear map (state-nonlinearity witness).

A hidden scalar h_t evolves under an input-DRIVEN nonlinear recurrence (the
logistic / quadratic map):

    h_0   = 0.5
    a_t   = a_lo + (d_t / (D-1)) * (a_hi - a_lo)      # control param from input token
    h_t   = a_t * h_{t-1} * (1 - h_{t-1})            # logistic map (quadratic nonlinearity)
    y_t   = floor(h_t * n_bins)                       # binned target

The model sees the driver tokens d_0..d_{T-1} and must predict the binned value
y_t of the internally iterated map at every position (running supervision).

Why this separates state-nonlinearity: the update contains h*(1-h), a genuine
quadratic in the state. A LINEAR-state recurrence (phi=identity) cannot realise
``h - h^2`` in its hidden state and is capped near the random baseline; a
state-NONLINEAR cell (tanh / relu / gamma-mix phi) can approximate the curved
map and bin it correctly. a_t is kept in [2.5, 3.5] (period-1..period-2, NOT
chaotic) so the target trajectory is deterministic and learnable -- the
discriminator is the quadratic nonlinearity, not chaos.
"""
import numpy as np


class IteratedNonlinearMapTask:
    def __init__(self, n_drivers: int = 5, n_bins: int = 10,
                 a_lo: float = 2.6, a_hi: float = 3.6):
        self.n_drivers = n_drivers
        self.n_bins = n_bins
        self.a_lo = a_lo
        self.a_hi = a_hi
        self.name = "iterated_nonlinear_map"
        self.vocab_size = max(n_drivers, n_bins)

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        d = rng.integers(0, self.n_drivers, size=(B, T)).astype(np.int64)
        a = self.a_lo + (d.astype(np.float64) / max(self.n_drivers - 1, 1)) * (self.a_hi - self.a_lo)
        h = np.full((B,), 0.5, dtype=np.float64)
        targets = np.zeros((B, T), dtype=np.int64)
        for t in range(T):
            h = a[:, t] * h * (1.0 - h)
            h = np.clip(h, 0.0, 1.0 - 1e-9)
            targets[:, t] = np.floor(h * self.n_bins).astype(np.int64)
        targets = np.clip(targets, 0, self.n_bins - 1)
        inputs = d
        mask = np.ones((B, T), dtype=bool)
        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.n_bins
