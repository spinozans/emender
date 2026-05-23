"""Associative recall: model sees a stream of (key, value) pairs, then a
single query key, and must output the value associated with that key.

Format:
  [k1, v1, k2, v2, ..., kN, vN, QUERY_MARKER, kq, ANSWER_SLOT]

Standard test of "in-context lookup" — transformers and Mamba2 both win;
linear scans struggle when N is large.

Vocab: V content tokens + pad + QUERY_MARKER.
Keys and values share the content vocabulary V.
"""
import numpy as np


class AssocRecallTask:
    name = 'assoc_recall'

    def __init__(self, n_pairs: int = 8, vocab: int = 16):
        self.n_pairs = n_pairs
        self.vocab = vocab  # number of distinct content tokens
        # ids: 0 = pad, 1 = QUERY_MARKER, 2..2+vocab-1 = content
        self.vocab_size = 2 + vocab

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        N = self.n_pairs
        assert T >= 2 * N + 3, f"T={T} too small for {N} pairs + query"

        inputs = np.zeros((B, T), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)

        for b in range(B):
            # Sample N distinct keys
            keys = rng.choice(self.vocab, size=N, replace=False) + 2
            values = rng.integers(2, 2 + self.vocab, size=N)
            # Place k1 v1 k2 v2 ... at start
            for i in range(N):
                inputs[b, 2 * i] = keys[i]
                inputs[b, 2 * i + 1] = values[i]
            # Add QUERY_MARKER then a random key from the list, then answer slot
            qpos = 2 * N
            inputs[b, qpos] = 1  # QUERY_MARKER
            qi = rng.integers(N)
            inputs[b, qpos + 1] = keys[qi]
            inputs[b, qpos + 2] = 0  # answer slot is pad in input
            targets[b, qpos + 2] = values[qi]
            mask[b, qpos + 2] = True
            # Remaining T positions are 0 (pad)

        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.vocab
