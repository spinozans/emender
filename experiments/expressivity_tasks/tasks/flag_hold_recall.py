"""PROBE 4 — flag-hold-and-recall (latch / long-range hold witness).

Set K bits early, sit through a long distractor gap, then recall a queried bit.

Sequence (single causal stream):
    [ write_0, write_1, ..., write_{K-1},  filler * gap,  query_q ]

    write_j  carries (slot j, bit b_j)   -> token 1 + 2*j + b_j        (1 .. 2K)
    filler                                -> token 0
    query_q  asks for slot q              -> token 1 + 2K + q           (2K+1 .. 3K)

Only the FINAL position (the query) is supervised; its target is the bit b_q
written into the queried slot at the very start. T sets the gap = T - K - 1, so
evaluating at T in {128, 256, 512, 1024} measures hold-distance directly.

Why this separates the self-loop gain: holding a bit unchanged across a gap of
length L needs an eigenvalue with magnitude ~1 (or a bistable >1 attractor). A
CRIBBED leaky cell (lambda<1, the E88 regime) decays the stored bit by
lambda^L -> 0 and fails at large gaps; a LATCH (lambda>=1 + bistable tanh) or a
gated counter (LSTM) holds it. The model never knows which slot will be queried,
so it must retain all K bits.
"""
import numpy as np


class FlagHoldRecallTask:
    def __init__(self, n_keys: int = 4):
        self.K = n_keys
        self.name = "flag_hold_recall"
        # tokens: 0 filler; 1..2K writes; 2K+1..3K queries. targets: bits {0,1}.
        self.vocab_size = max(3 * self.K + 1, 2)

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        K = self.K
        assert T >= K + 1, f"T={T} too short for K={K}"
        inputs = np.zeros((B, T), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)

        bits = rng.integers(0, 2, size=(B, K)).astype(np.int64)   # stored bits
        # write phase: positions 0..K-1 write slot j with its bit
        for j in range(K):
            inputs[:, j] = 1 + 2 * j + bits[:, j]
        # filler phase: positions K..T-2 stay 0 (filler)
        # query phase: last position queries a random slot
        q = rng.integers(0, K, size=(B,)).astype(np.int64)
        inputs[:, T - 1] = 1 + 2 * K + q
        targets[:, T - 1] = bits[np.arange(B), q]
        mask[:, T - 1] = True
        return inputs, targets, mask

    def random_baseline_acc(self):
        return 0.5
