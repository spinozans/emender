"""MQAR — Multi-Query Associative Recall (E98 five-corner recall probe).

Standard multi-query associative-recall task (Zoology / Based "MQAR"): a sequence
first STORES K random key->value bindings, then QUERIES a subset of those keys; at
each query position the model must emit the value bound to the queried key.

This is the canonical witness for the *leaky-linear associative-memory* regime
(lambda<1, small beta -> positive along-key eigenvalue in (0,1), phi=identity):
the fading key-value store. Each (k,v) pair is written as an outer product into the
matrix state; a query key reads it back via the q-readout. The exotic corners
(track/count/latch/nonlin) do NOT implement content-addressable recall and should
fail here.

Layout (total length T, P = T//4 pairs, P queries — length-EXTRAPOLATING: longer
sequences store strictly more bindings, the actual hardness of MQAR):

    [ k1 v1 k2 v2 ... kP vP ][ q1 . q2 . ... qP . ]
      (------ store -------)  (------ query ------)

    * keys are DISTINCT content tokens (sampled without replacement); values are
      i.i.d. content tokens (may repeat). Keys & values share one content vocab.
    * a query re-presents one of the stored keys; the per-position target AT the
      query-key position is the bound value (position-aligned with how
      train_hybrid computes the masked cross-entropy — no shift). The trailing '.'
      is an unsupervised pad gap separating queries.
    * mask is True only on the P query positions (dense multi-query supervision).

REAL generator, no mocks. Pairs scale with T so the eval grid {128,256,512,1024}
is a genuine memory-capacity / length-extrapolation sweep.
"""
import numpy as np


class MQARRecallTask:
    name = 'mqar_recall'

    def __init__(self, vocab: int = 64, query_frac: float = 1.0, min_pairs: int = 2):
        self.vocab = int(vocab)            # distinct content tokens (keys & values share)
        self.query_frac = float(query_frac)
        self.min_pairs = int(min_pairs)
        # ids: 0 = pad / query-gap slot, 1..vocab = content tokens
        self.vocab_size = self.vocab + 1

    def _n_pairs(self, T: int) -> int:
        # Pairs scale GENTLY with length so train length (T=128 -> 8 pairs) is
        # learnable, while extrapolation (T=1024 -> 64 pairs) is a genuine
        # memory-capacity sweep: more distinct bindings to store in the fixed
        # N-dim key space (the actual hardness of associative recall). Storage +
        # queries occupy 4P <= T//4 tokens; the rest is trailing pad.
        P = T // 16
        return max(self.min_pairs, min(self.vocab, P))

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        P = self._n_pairs(T)
        Q = max(1, int(round(P * self.query_frac)))

        inputs = np.zeros((B, T), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)

        for b in range(B):
            keys = rng.choice(self.vocab, size=P, replace=False) + 1      # distinct, [1, vocab]
            values = rng.integers(0, self.vocab, size=P) + 1              # i.i.d.,  [1, vocab]
            # store region: k1 v1 k2 v2 ...
            for i in range(P):
                inputs[b, 2 * i] = keys[i]
                inputs[b, 2 * i + 1] = values[i]
            # query region: re-present a random subset of keys, predict their value
            qpos = 2 * P
            qidx = rng.permutation(P)[:Q]
            for j, qi in enumerate(qidx):
                p = qpos + 2 * j
                if p >= T:
                    break
                inputs[b, p] = keys[qi]            # re-present the queried key
                targets[b, p] = values[qi]         # target AT this position = bound value
                mask[b, p] = True
                # p+1 stays 0 (pad gap), unsupervised
        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.vocab
