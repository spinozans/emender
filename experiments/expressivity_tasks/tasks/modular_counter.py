"""Modular counter: input is a stream of integers in [0, K), model predicts
the running sum modulo K at each position.

For K=2 this reduces to parity. For K>2 it's a generalized counting task —
also outside TC0 for unbounded length.

Tokens 0..K-1 are the input symbols and the targets. Padding/marker handled
by reserving extra ids if needed (we keep it minimal).
"""
import numpy as np


class ModularCounterTask:
    name = 'modular_counter'

    def __init__(self, K: int = 5, mode: str = 'running'):
        assert mode in ('running', 'final')
        assert K >= 2
        self.K = K
        self.mode = mode
        self.vocab_size = K + 2  # extras for marker/padding when needed

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        symbols = rng.integers(0, self.K, size=(B, T)).astype(np.int64)
        running = np.cumsum(symbols, axis=1) % self.K

        if self.mode == 'running':
            inputs = symbols.copy()
            targets = running
            mask = np.ones((B, T), dtype=bool)
        else:
            inputs = np.full((B, T), self.K + 1, dtype=np.int64)  # padding
            inputs[:, :T - 1] = symbols[:, :T - 1]
            inputs[:, T - 1] = self.K  # predict marker
            targets = np.zeros((B, T), dtype=np.int64)
            targets[:, T - 1] = running[:, T - 2]
            mask = np.zeros((B, T), dtype=bool)
            mask[:, T - 1] = True

        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.K
