"""Boolean-association recall — the NON-BILINEAR association witness (task
nlmem-capability, NONLIN_MEMORY_SPEC.md §8.1 probe 1).

THE question this isolates: can a recurrent memory store a key->value association
whose READ is a NON-bilinear function of the query, of the form the spec's canonical
``k=[a,b] -> v = a XOR b`` example? A GDN/delta linear matrix memory ``S`` reads
``out = S q`` — *linear* in the query — so from a single composite query it can only
produce a *linear* combination of stored bits; it physically cannot emit their XOR
(``a XOR b = (a+b) mod 2`` needs the product term ``ab``, a non-bilinear coupling).
The ``mlp-mem`` cell reads ``out = W2 sigma(W1 q)`` — nonlinear in the query — so a
1-hidden-layer memory CAN represent the XOR from one composite key.

Layout (one sequence):

    STORE:  [ FEAT_0 BIT_{b0} ][ FEAT_1 BIT_{b1} ] ... [ FEAT_{M-1} BIT_{b_{M-1}} ]
    QUERY:  ( PAIR_{ij}  ANSWER_SLOT ) x Q

  * Per sequence, each of the M features gets a fresh random bit ``b[f] in {0,1}``,
    so the (feature -> bit) map is NOT learnable in the weights — it must be WRITTEN
    into the recurrent memory in-context during the STORE phase.
  * A QUERY is a single PAIR token ``PAIR_{ij}`` (one key/query projection) naming an
    unordered feature pair (i, j), followed by an answer slot. The per-position target
    AT the answer slot is:
        op='xor' :  b[i] XOR b[j]      (NON-bilinear  — the separator)
        op='lin' :  b[i]               (linear single-bit recall — matched control)
    The PAIR token carries no information about the bits (they are random per
    sequence), so the model must route the query through the stored memory: align the
    query projection of ``PAIR_{ij}`` with the stored key directions of features i and
    j, retrieve their bits, and combine. For 'xor' the combine is non-bilinear.

  * mask is True only on the Q answer slots.

Why the matched 'lin' control matters: it shares the EXACT format and storage demand
(same M features, same PAIR tokens, same routing) but its answer is a *linear* readout
of one stored bit. Both the linear GDN memory and the nonlinear mlp-mem memory should
solve 'lin'; only the nonlinear read can (in principle) solve 'xor' from one query when
the fixed post-head MLP readout is removed. Running both arms on both ops at
``mlp_ratio in {2.0, 0}`` localizes whether any XOR separation comes from the MEMORY
read-nonlinearity or merely from the model-level SwiGLU readout (the convergent-loss-null
mechanism this lab keeps finding).

REAL, exactly-solvable generator (a deterministic boolean function on stored bits).
No mocks.
"""
import numpy as np
from itertools import combinations


class BooleanAssocTask:
    def __init__(self, n_features: int = 8, op: str = 'xor', queries: int = 8):
        assert op in ('xor', 'lin')
        assert n_features >= 2
        self.M = int(n_features)
        self.op = op
        self.queries = int(queries)
        self.name = 'boolean_assoc' + ('' if op == 'xor' else '_lin')
        # all unordered feature pairs (i<j)
        self._pairs = list(combinations(range(self.M), 2))
        self.n_pairs = len(self._pairs)
        # token ids:
        #   0          = pad / answer slot input
        #   1          = BIT 0
        #   2          = BIT 1
        #   3..3+M-1   = FEAT_f
        #   feat_off+M .. +n_pairs-1 = PAIR_{ij}
        self.bit_off = 1
        self.feat_off = 3
        self.pair_off = self.feat_off + self.M
        self.vocab_size = self.pair_off + self.n_pairs

    def _store_len(self) -> int:
        return 2 * self.M

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        store_len = self._store_len()
        # how many query (PAIR, ANSWER) blocks fit after the store region
        max_q = (T - store_len) // 2
        Q = max(1, min(self.queries, max_q))
        if Q < 1:
            raise ValueError(f"T={T} too small for M={self.M} features (need >= {store_len + 2})")

        inputs = np.zeros((B, T), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)

        for b in range(B):
            bits = rng.integers(0, 2, size=self.M)                 # fresh per sequence
            # STORE: FEAT_f BIT_{b[f]}
            for f in range(self.M):
                inputs[b, 2 * f] = self.feat_off + f
                inputs[b, 2 * f + 1] = self.bit_off + int(bits[f])
            # QUERY: Q distinct (or sampled) pairs, each PAIR token then answer slot
            qpos = store_len
            if Q <= self.n_pairs:
                pair_ids = rng.choice(self.n_pairs, size=Q, replace=False)
            else:
                pair_ids = rng.integers(0, self.n_pairs, size=Q)
            for j, pid in enumerate(pair_ids):
                p = qpos + 2 * j
                if p + 1 >= T:
                    break
                i_feat, j_feat = self._pairs[pid]
                inputs[b, p] = self.pair_off + pid
                # answer slot (p+1) stays pad=0 in the input
                if self.op == 'xor':
                    ans = int(bits[i_feat]) ^ int(bits[j_feat])
                else:  # 'lin' — single-bit linear recall control
                    ans = int(bits[i_feat])
                targets[b, p + 1] = self.bit_off + ans
                mask[b, p + 1] = True

        return inputs, targets, mask

    def random_baseline_acc(self):
        return 0.5  # 2-class (bit 0 / bit 1)
