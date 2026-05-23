"""Parity: input is a stream of bits, model predicts running parity.

The classic linear-RNN / TC0-circuit-impossible task. A pure linear model
cannot maintain parity over unbounded sequence length; nonlinear RNNs can.

Vocabulary (4 tokens encoded as bytes 0-3):
  0 = bit 0
  1 = bit 1
  2 = "predict here" marker (placeholder, target is parity-of-bits-so-far)
  3 = padding / unused

Two variants:
  - 'running': at every position, target = parity of all bits so far
  - 'final':   target appears only at last position (after a marker)

Default 'running' mode gives dense supervision (target at every position).
"""
import numpy as np


class ParityTask:
    name = 'parity'
    vocab_size = 4

    def __init__(self, mode='running'):
        assert mode in ('running', 'final')
        self.mode = mode

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        # Random bit stream
        bits = rng.integers(0, 2, size=(B, T)).astype(np.int64)
        running_parity = np.cumsum(bits, axis=1) % 2

        if self.mode == 'running':
            inputs = bits.copy()
            # Target at position t is parity of bits[0..t]
            # We treat input as the stream and supervise predicting running parity at each step
            targets = running_parity
            mask = np.ones((B, T), dtype=bool)
        else:  # 'final'
            # Last 1 position is the answer slot.
            inputs = np.full((B, T), 3, dtype=np.int64)  # padding token
            inputs[:, :T - 1] = bits[:, :T - 1]
            inputs[:, T - 1] = 2  # "predict" marker
            targets = np.zeros((B, T), dtype=np.int64)
            targets[:, T - 1] = running_parity[:, T - 2]
            mask = np.zeros((B, T), dtype=bool)
            mask[:, T - 1] = True

        return inputs, targets, mask

    def random_baseline_acc(self):
        return 0.5  # 2-class
