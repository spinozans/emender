"""Dyck-1: balanced parens. Input is a sequence of '(' and ')' tokens;
model predicts the current nesting depth at each position.

Tracking depth requires an integer counter. Linear/TC0 models cannot
do this for unbounded depth. We cap depth at MAX_DEPTH so it remains
a finite-state task within our test range, but T grows much larger
than MAX_DEPTH so the task still tests counter dynamics.

Tokens:
  0 = '('
  1 = ')'
  2 = pad / clamp marker

Targets are the depth at each position (clipped to [0, MAX_DEPTH]).
Targets are token-encoded as small integers; we add an offset so target
ids don't collide with input ids when models share an embedding.
"""
import numpy as np


class DyckTask:
    name = 'dyck'

    def __init__(self, max_depth: int = 8, p_open: float = 0.5):
        self.max_depth = max_depth
        self.p_open = p_open
        # Vocabulary covers input tokens (0,1,2) AND target depths (0..max_depth)
        # Use offsetting: input ids 0..2, target ids represent depth directly.
        # Models predict next token = depth at current position.
        self.vocab_size = max(3, max_depth + 1)

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        # Generate balanced-ish strings: open with prob p_open, close otherwise
        # but only if depth>0 (else open). Cap at max_depth (force close).
        inputs = np.zeros((B, T), dtype=np.int64)
        depths = np.zeros((B, T), dtype=np.int64)

        for b in range(B):
            d = 0
            for t in range(T):
                if d == 0:
                    inputs[b, t] = 0  # must open
                    d = 1
                elif d >= self.max_depth:
                    inputs[b, t] = 1  # must close
                    d -= 1
                else:
                    if rng.random() < self.p_open:
                        inputs[b, t] = 0
                        d += 1
                    else:
                        inputs[b, t] = 1
                        d -= 1
                depths[b, t] = d

        targets = depths
        mask = np.ones((B, T), dtype=bool)
        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / (self.max_depth + 1)
