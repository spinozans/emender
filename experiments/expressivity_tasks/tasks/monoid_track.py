"""Non-group MONOID composition tracking — the capability-gap witness.

This is the non-invertible analogue of s5_permutation. A hidden state
s_t in {0..N-1} evolves under input-selected fixed functions:

    s_0 = 0
    s_t = g_{x_t}(s_{t-1})        # x_t in {0..G-1} is the input token
    y_t = s_t                     # running supervision (every position)

The G functions g_0..g_{G-1} : [N] -> [N] are FIXED (seeded once with
`func_seed`, independent of the per-batch rng, so they are a stable property of
the task and identical across training seeds and across eval lengths).

Two regimes (the `invertible` flag is the controlled knob):

  * invertible=False  (DEFAULT, the GAP TEST): the functions are general random
    maps [N]->[N], generically NON-injective -> the transition semigroup is a
    NON-GROUP MONOID. Composing them collapses states; the running product is a
    genuinely nonlinear, non-invertible composition of depth t. This is the
    "non-group monoid product" family: the per-step transition is a rank-
    deficient 0/1 selection matrix, NOT a product of a few Householder
    reflections, so a linear-recurrence / DeltaProduct-style state-transition
    (gdn-neg) cannot represent it the way it represents group composition,
    while a state-NONLINEAR (tanh) recurrence can realise the finite automaton.

  * invertible=True   (the LINEARIZABLE CONTROL): the functions are random
    PERMUTATIONS -> the transition group is a subgroup of S_N (a GROUP). This is
    the same structure as s5_permutation; gdn-neg's negative-eigenvalue /
    orthogonal transitions already solve group composition, so NO gap is
    expected here. Identical task shape, only invertibility differs -> a clean
    A/B that isolates "non-group monoid" as the discriminator.

Theory anchor: Merrill, Petty & Sabharwal, "The Illusion of State in
State-Space Models" (ICML 2024) — linear SSMs live in TC0 and cannot compose
arbitrary finite-state transitions; Deletang et al. (2023) Chomsky-hierarchy
for RNNs; Grazzi et al. (2025) negative eigenvalues give group/reflection
tracking (handled by the invertible control), NOT general monoid composition.

REAL generable data; no mocks. Both regimes are exactly solvable (deterministic
finite automaton), so the discriminator is representational, not noise.
"""
import numpy as np


class MonoidTrackTask:
    def __init__(self, N: int = 8, G: int = 4, invertible: bool = False,
                 func_seed: int = 0, mode: str = "running"):
        assert N >= 2 and G >= 2
        assert mode in ("running", "final")
        self.N = N
        self.G = G
        self.invertible = bool(invertible)
        self.mode = mode
        self.name = "monoid_track" + ("_inv" if invertible else "")
        # Fixed function table  funcs[g] : [N] -> [N], seeded independently of the
        # per-batch rng so the automaton is identical across training seeds/lengths.
        frng = np.random.default_rng(func_seed)
        funcs = np.zeros((G, N), dtype=np.int64)
        for g in range(G):
            if self.invertible:
                funcs[g] = frng.permutation(N)
            else:
                funcs[g] = frng.integers(0, N, size=N)
        # Guarantee the non-invertible regime is actually non-injective (a random
        # map can occasionally come out a permutation for small N): force a
        # collapse in at least one function.
        if not self.invertible:
            collapsed = any(len(np.unique(funcs[g])) < N for g in range(G))
            if not collapsed:
                funcs[0, 1] = funcs[0, 0]
        self.funcs = funcs
        self.vocab_size = max(G, N)

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        x = rng.integers(0, self.G, size=(B, T)).astype(np.int64)
        s = np.zeros((B,), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        for t in range(T):
            s = self.funcs[x[:, t], s]
            targets[:, t] = s
        if self.mode == "running":
            mask = np.ones((B, T), dtype=bool)
        else:
            mask = np.zeros((B, T), dtype=bool)
            mask[:, T - 1] = True
        return x, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.N


class MonoidTrackInvTask(MonoidTrackTask):
    """Invertible (group) control: functions are permutations."""
    def __init__(self, **kw):
        kw.pop("invertible", None)
        super().__init__(invertible=True, **kw)
