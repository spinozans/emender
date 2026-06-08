"""Modular quadratic map — arithmetic non-invertible / non-contracting witness.

A hidden state x_t in Z_p evolves under an input-driven NONLINEAR, NON-INVERTIBLE
recurrence over a finite field:

    x_0 = 1
    x_t = (x_{t-1}^2 + c_t) mod p        # c_t in {0..G-1} is the input token
    y_t = x_t                            # running supervision (every position)

This is the finite-field cousin of the logistic / iterated_nonlinear_map probe,
constructed to REMOVE the two confounds that let a linear-recurrence solve the
logistic version:

  * NON-contracting: the orbit lives on a finite set Z_p driven by fresh c_t
    every step, so it never settles to a fixed point / short cycle the way the
    contracting logistic map (a in [2.6,3.6], period-1/2) does. h_t there forgets
    its history -> bounded effective composition depth -> a linear-state cell +
    MLP solves it. Here x_t genuinely depends on ALL of c_1..c_t.
  * NON-invertible & NONLINEAR: squaring mod p is 2-to-1 (a non-group monoid
    action) and quadratic, so the per-step transition is neither a permutation
    (handled by gdn-neg's group tracking) nor representable by the contraction
    the logistic map exploited.

Linear control (`square=False`): x_t = (x_{t-1} + c_t) mod p, i.e. the modular
counter — an INVERTIBLE, LINEAR group action. gdn-neg solves this for free, so
no gap is expected; it is the linearizable control for the quadratic test.

Theory anchor: Weiss, Goldberg & Yahav (2018) on counting vs saturating
nonlinearity; Merrill et al. (2024) "Illusion of State" (linear SSMs in TC0);
the separating signature is a LENGTH-EXTRAPOLATION cliff — the readout MLP has
fixed O(depth) nonlinear depth and cannot supply the O(T) nested squarings.

REAL generable data; no mocks. Exactly solvable finite automaton on Z_p.
"""
import numpy as np


class ModularQuadraticTask:
    def __init__(self, p: int = 7, G: int = 5, square: bool = True,
                 mode: str = "running"):
        assert p >= 3 and G >= 2
        assert mode in ("running", "final")
        self.p = p
        self.G = G
        self.square = bool(square)
        self.mode = mode
        self.name = "modular_quadratic" + ("" if square else "_lin")
        self.vocab_size = max(G, p)

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        c = rng.integers(0, self.G, size=(B, T)).astype(np.int64)
        x = np.ones((B,), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        for t in range(T):
            if self.square:
                x = (x * x + c[:, t]) % self.p
            else:
                x = (x + c[:, t]) % self.p
            targets[:, t] = x
        if self.mode == "running":
            mask = np.ones((B, T), dtype=bool)
        else:
            mask = np.zeros((B, T), dtype=bool)
            mask[:, T - 1] = True
        return c, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.p


class ModularQuadraticLinTask(ModularQuadraticTask):
    """Linear (modular-counter) control: x_t = (x_{t-1} + c_t) mod p."""
    def __init__(self, **kw):
        kw.pop("square", None)
        super().__init__(square=False, **kw)
