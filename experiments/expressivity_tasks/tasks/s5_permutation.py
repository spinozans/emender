"""Permutation-composition tracking tasks.

These are finite-state state-tracking tasks over symmetric groups. S3 is a
solvable-group ladder task. S5 is the smallest symmetric-group witness with a
non-solvable transition group, and is the right empirical companion to the
Barrington/NC1 framing: train on prefix products, then test length
generalization.

Tokens 0..n-2 represent adjacent transposition generators. The model sees a
sequence of generators and predicts the running group product, encoded as the
lexicographic permutation id. The dense "running" mode gives every prefix as a
supervised state target; "final" only supervises the final product.
"""
from itertools import permutations as _perm

import numpy as np


def _all_perms(n: int):
    return list(_perm(range(n)))


def _apply_swap(perm, swap):
    """Apply a transposition (swap=(i,j)) to a permutation tuple."""
    p = list(perm)
    i, j = swap
    p[i], p[j] = p[j], p[i]
    return tuple(p)


class PermutationCompositionTask:
    """Track running composition in S_n using adjacent transpositions."""

    def __init__(self, n: int, mode: str = "running"):
        assert n >= 2
        assert mode in ("running", "final")
        self.n = n
        self.mode = mode
        self.name = f"s{n}_permutation"
        self.generators = [(i, i + 1) for i in range(n - 1)]
        self.perms = _all_perms(n)
        self.perm_to_id = {p: i for i, p in enumerate(self.perms)}
        self.n_gen = len(self.generators)
        self.n_classes = len(self.perms)
        # Vocabulary covers input generators and output permutation ids.
        self.vocab_size = max(self.n_gen + 1, self.n_classes)

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        gen_ids = rng.integers(0, self.n_gen, size=(B, T)).astype(np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        identity = tuple(range(self.n))

        for b in range(B):
            cur = identity
            for t in range(T):
                cur = _apply_swap(cur, self.generators[gen_ids[b, t]])
                targets[b, t] = self.perm_to_id[cur]

        inputs = gen_ids
        if self.mode == "running":
            mask = np.ones((B, T), dtype=bool)
        else:
            mask = np.zeros((B, T), dtype=bool)
            mask[:, T - 1] = True
        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.n_classes


class S3PermutationTask(PermutationCompositionTask):
    def __init__(self, mode: str = "running"):
        super().__init__(n=3, mode=mode)


class S5PermutationTask(PermutationCompositionTask):
    def __init__(self, mode: str = "running"):
        super().__init__(n=5, mode=mode)
