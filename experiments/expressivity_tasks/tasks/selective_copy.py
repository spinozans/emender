"""Selective copy: from the Mamba paper. The model sees a sequence of
"tokens" interspersed with "noise" markers, and must output the tokens
in order at the end (after a "go" marker) ignoring the noise.

This task heavily favors attention/selective scan models. Pure linear
recurrence (S4) fails here; transformers and Mamba succeed.

Layout:
  Input:  [t1] [pad] [pad] [t2] [pad] [t3] ... [GO] [pad pad pad ...]
  Output: zeros until GO, then [t1, t2, t3, ...]

Vocabulary: K content tokens + pad + GO + answer tokens (we just reuse
content token ids for outputs).
"""
import numpy as np


class SelectiveCopyTask:
    name = 'selective_copy'

    def __init__(self, n_content: int = 16, n_to_copy: int = 4):
        """n_content = vocabulary of content tokens (excluding pad/go).
        n_to_copy = number of content tokens to copy."""
        self.n_content = n_content
        self.n_to_copy = n_to_copy
        # ids: 0 = pad, 1 = GO, 2..2+n_content-1 = content tokens
        self.vocab_size = 2 + n_content

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        K = self.n_to_copy
        assert T >= 2 * K + 2, f"T={T} too small for {K} tokens to copy"
        # Randomly pick K positions in the first half for content; others = pad.
        # Last K+1 positions: GO then K answer slots.
        inputs = np.zeros((B, T), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)

        first_half_len = T - K - 1  # GO + K answer slots reserved at end
        for b in range(B):
            # Choose K positions in [0, first_half_len)
            content_pos = rng.choice(first_half_len, size=K, replace=False)
            content_pos.sort()
            content_tokens = rng.integers(2, 2 + self.n_content, size=K)
            for cp, ct in zip(content_pos, content_tokens):
                inputs[b, cp] = ct
            inputs[b, first_half_len] = 1  # GO marker
            # Answer slots: the K content tokens in order
            for i, ct in enumerate(content_tokens):
                slot = first_half_len + 1 + i
                inputs[b, slot] = 0  # pad in input
                targets[b, slot] = ct
                mask[b, slot] = True

        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.n_content
