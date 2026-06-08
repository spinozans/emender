"""Counting-with-comparison probe (PROBE 1, LINEAR_VS_NONLINEAR_TASK_DESIGN.md §3).

Two REAL counter-language tasks with dense per-position supervision and
length extrapolation. The load-bearing operation in each is a *comparison*
on a running, unbounded counter — exactly the non-saturating conditional
update that additive ReLU / LSTM cells implement and that squashing (tanh)
RNNs / bounded linear-state recurrences provably cannot at unbounded length
[Weiss, Goldberg & Yahav 2018; Délétang et al. 2022].

  1b  DyckDepthTask           — running Dyck-1 nesting depth; the floor at 0
                                (max(depth-1, 0)) is the zero-test comparison.
  1a  AnBnCnViabilityTask     — per-position "is this prefix still a viable
                                prefix of some a^n b^n c^n?"; the count
                                comparisons nb<=na, nb==na, nc<=na are the
                                load-bearing nonlinearity.

Both follow the harness task contract:
    generate_batch(B, T, rng) -> (input[B,T] int64, target[B,T] int64,
                                   loss_mask[B,T] bool)
    random_baseline_acc() -> float
Trains at T=128, evaluates at T in {128,256,512,1024} via --eval_lengths.
"""
import numpy as np


class DyckDepthTask:
    """Variant 1b — running Dyck-1 (single bracket type) nesting depth.

    Input stream over {'(' = 0, ')' = 1}. The nesting depth is a non-negative
    counter:

        depth_0 = 0
        depth_t = max(depth_{t-1} + (+1 if '(' else -1), 0)

    The max(., 0) floor IS the load-bearing comparison (a ')' at depth 0 is a
    no-op, not a negative count). The target at each position is the *current*
    depth, capped to [0, cap] for the cross-entropy loss; the underlying depth
    is left unbounded so the counter must keep counting past the display cap.

    A slight negative drift (p_open < 0.5) keeps the walk positive-recurrent
    near the floor, so the zero-test is exercised a constant fraction of the
    time at EVERY length — length extrapolation therefore measures pure
    counting stamina, not a shifting target distribution.
    """
    name = 'dyck_depth'

    def __init__(self, cap: int = 15, p_open: float = 0.45):
        assert cap >= 2
        assert 0.0 < p_open < 0.5, "p_open<0.5 gives negative drift -> floor exercised"
        self.cap = cap
        self.p_open = p_open
        # vocab covers both input symbols {0,1} and target depths {0..cap}.
        self.vocab_size = cap + 1

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        # +1 for '(' (token 0), -1 for ')' (token 1).
        opens = rng.random((B, T)) < self.p_open          # True -> '('
        inputs = np.where(opens, 0, 1).astype(np.int64)    # 0='(', 1=')'
        steps = np.where(opens, 1, -1).astype(np.int64)

        # Reflected (clamped-at-0) running sum, computed exactly per position.
        depth = np.zeros((B, T), dtype=np.int64)
        cur = np.zeros(B, dtype=np.int64)
        for t in range(T):
            cur = np.maximum(cur + steps[:, t], 0)
            depth[:, t] = cur

        targets = np.minimum(depth, self.cap)
        mask = np.ones((B, T), dtype=bool)
        return inputs, targets, mask

    def random_baseline_acc(self):
        # Always-predict-0 (the majority class). Under the reflected biased
        # walk the stationary distribution is pi_k = (1-rho) rho^k with
        # rho = p_open / (1 - p_open), so pi_0 = 1 - rho.
        rho = self.p_open / (1.0 - self.p_open)
        return 1.0 - rho


class DyckDepthUnboundedTask(DyckDepthTask):
    """UNBOUNDED-magnitude counting (the genuine Weiss-Goldberg-Yahav separator).

    Same floored-counter dynamics as DyckDepthTask, but with POSITIVE drift
    (p_open > 0.5) and a high cap, so the running depth GROWS roughly linearly
    with t and at long eval lengths reaches magnitudes NEVER SEEN at the
    training length. Trained at T=128 the model sees depths up to ~30; evaluated
    at T=2048 it must emit depths up to ~200. This is the regime the survey
    identifies as the learnable-and-separating sweet spot: the state is
    UNBOUNDED + MONOTONE (non-fading, escapes contraction) yet non-chaotic
    (lambda bounded, hence learnable).

    The discriminator is whether the cell maintains a NON-SATURATING additive
    count that the readout can decode past the trained magnitude band. A
    bounded/saturating state (tanh) compresses large counts into a fixed range
    and a fixed-depth MLP readout has only finitely many decision regions, so
    accuracy degrades monotonically once depth exceeds the training band; a
    non-saturating additive counter (LSTM/relu-state cell) extrapolates. A
    LINEAR integrating recurrence (eigenvalue 1) is ALSO non-saturating, so this
    task tests whether nonlinearity-IN-TIME helps OR whether linear integration
    already suffices.
    """
    name = 'dyck_depth_unbounded'

    def __init__(self, cap: int = 256, p_open: float = 0.55):
        assert cap >= 2
        assert 0.5 < p_open < 1.0, "p_open>0.5 gives POSITIVE drift -> unbounded growth"
        self.cap = cap
        self.p_open = p_open
        self.vocab_size = cap + 1

    def random_baseline_acc(self):
        # Positive-drift walk has no stationary mass concentration; the depth
        # spreads over a growing range, so majority-class accuracy -> ~0. Report
        # the conservative 1/cap uniform baseline.
        return 1.0 / self.cap


class AnBnCnViabilityTask:
    """Variant 1a — per-position viability for the language a^n b^n c^n.

    Input stream over {'a'=0, 'b'=1, 'c'=2}. At each position the target is a
    binary label: is the prefix read so far still a *viable* prefix of some
    a^n b^n c^n (n>=1)? Viability is an exact online property of three
    unbounded counters (na, nb, nc) and a phase, decided by COUNT COMPARISONS:

        - a's must come first; a 'b' requires phase A (na>=1) or B, never C
        - in the b-phase, nb must never exceed na   (comparison nb <= na)
        - a 'c' may start only once nb == na          (comparison nb == na)
        - in the c-phase, nc must never exceed na    (comparison nc <= na)

    Once viability is lost it is sticky (target stays 0). The comparisons fire
    deep in the sequence — e.g. a^n b^{n+1} stays viable through all n a's and
    n b's and only flips to 0 at the (n+1)-th b, which requires having COUNTED
    na and compared. n scales with T, so at long T the counters must reach
    large magnitude: a bounded-state device cannot.

    Generation mixes valid a^n b^n c^n, off-by-one / perturbed-count blocks,
    and fully random {a,b,c} streams; per-position targets are computed by the
    exact checker for whatever stream is produced, so every label is real.
    """
    name = 'anbncn_viability'

    def __init__(self, p_valid: float = 0.4, p_perturb: float = 0.4):
        assert 0.0 <= p_valid + p_perturb <= 1.0
        self.p_valid = p_valid
        self.p_perturb = p_perturb
        self.vocab_size = 3  # a,b,c; targets are the subset {0,1}

    @staticmethod
    def _viability(seq: np.ndarray) -> np.ndarray:
        """Exact per-position viability for one int sequence over {0,1,2}.

        Returns an int64 array (1 viable, 0 dead) of the same length.
        phase: 0=A (reading a's), 1=B (reading b's), 2=C (reading c's).
        """
        T = seq.shape[0]
        out = np.zeros(T, dtype=np.int64)
        na = nb = nc = 0
        phase = 0
        dead = False
        for t in range(T):
            s = seq[t]
            if not dead:
                if s == 0:        # 'a'
                    if phase == 0:
                        na += 1
                    else:
                        dead = True
                elif s == 1:      # 'b'
                    if phase == 0:
                        if na >= 1:
                            phase = 1
                            nb = 1
                        else:
                            dead = True      # 'b' with no preceding 'a'
                    elif phase == 1:
                        nb += 1
                        if nb > na:
                            dead = True      # more b's than a's
                    else:
                        dead = True          # 'b' after a 'c'
                else:             # 'c'
                    if phase == 1 and nb == na:
                        phase = 2
                        nc = 1
                    elif phase == 2:
                        nc += 1
                        if nc > na:
                            dead = True      # more c's than a's
                    else:
                        dead = True          # 'c' before b's complete
            out[t] = 0 if dead else 1
        return out

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        inputs = np.empty((B, T), dtype=np.int64)
        max_n = max(1, (T - 1) // 3)
        for b in range(B):
            r = rng.random()
            if r < self.p_valid:
                n = int(rng.integers(1, max_n + 1))
                block = ([0] * n) + ([1] * n) + ([2] * n)
            elif r < self.p_valid + self.p_perturb:
                n = int(rng.integers(1, max_n + 1))
                # off-by-one / perturbed counts (and occasional wrong order)
                da = int(rng.integers(-2, 3))
                db = int(rng.integers(-2, 3))
                dc = int(rng.integers(-2, 3))
                na = max(1, n + da)
                nb = max(0, n + db)
                nc = max(0, n + dc)
                block = ([0] * na) + ([1] * nb) + ([2] * nc)
            else:
                block = []
            seq = np.array(block[:T], dtype=np.int64)
            if seq.shape[0] < T:
                tail = rng.integers(0, 3, size=T - seq.shape[0]).astype(np.int64)
                seq = np.concatenate([seq, tail]) if seq.shape[0] else tail
            inputs[b] = seq

        targets = np.empty((B, T), dtype=np.int64)
        for b in range(B):
            targets[b] = self._viability(inputs[b])
        mask = np.ones((B, T), dtype=bool)
        return inputs, targets, mask

    def random_baseline_acc(self):
        # Binary label; uniform-guess chance level. (The viable class is the
        # majority early in sequences and the dead class dominates the tails,
        # so 0.5 is the conservative uniform baseline reported in the design.)
        return 0.5
