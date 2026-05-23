"""Dyck-2: balanced parens with TWO bracket types '()' and '[]'.

The hard test for context-free language modeling. Requires a real LIFO stack:
when closing, must close the SAME type that was last opened. Linear-scan SSMs
cannot do this in principle (no stack), even with bounded depth.

Tokens:
  0 = '('
  1 = ')'
  2 = '['
  3 = ']'

Targets at each position: predict the next token. Specifically when at depth>0:
  - if next must close: which closer? (the type matching the most recent unmatched opener)
  - if next can open or close: model can choose

For evaluation we focus on positions where the choice is FORCED — i.e., depth at MAX
(must close) or stack depth=0 (must open) — and check correctness of bracket type.

Simpler protocol: generate strings where at random positions we FORCE a close, and
check whether the model predicts the correct closing bracket type.

Tokens:
  0 = '('   (open type 0)
  1 = ')'   (close type 0)
  2 = '['   (open type 1)
  3 = ']'   (close type 1)
  4 = pad

Target: next token id (so model must predict ')' vs ']' correctly when forced to close).
Mask only forced-close positions for cleaner separation.
"""
import numpy as np


class Dyck2Task:
    name = 'dyck2'

    def __init__(self, max_depth: int = 8, p_open: float = 0.5):
        self.max_depth = max_depth
        self.p_open = p_open
        self.vocab_size = 5  # ( ) [ ] pad

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        inputs = np.zeros((B, T), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)

        for b in range(B):
            stack = []  # list of bracket types (0 or 1)
            for t in range(T):
                d = len(stack)
                # Decide action: open or close
                if d == 0:
                    # Must open. Random type.
                    typ = int(rng.integers(0, 2))
                    inputs[b, t] = 0 if typ == 0 else 2
                    stack.append(typ)
                elif d >= self.max_depth:
                    # Must close. Type forced by stack top.
                    typ = stack.pop()
                    inputs[b, t] = 1 if typ == 0 else 3
                else:
                    if rng.random() < self.p_open:
                        # Open random type
                        typ = int(rng.integers(0, 2))
                        inputs[b, t] = 0 if typ == 0 else 2
                        stack.append(typ)
                    else:
                        # Close. Type forced.
                        typ = stack.pop()
                        inputs[b, t] = 1 if typ == 0 else 3

                # Target = NEXT token (the one model predicts at position t+1)
                # We'll fill at end: targets[t] = inputs[t+1]
            # Shift: targets are next-token at each position
            targets[b, :T-1] = inputs[b, 1:]
            targets[b, T-1] = 4  # pad for last
            # Mask: only positions where the next token is FORCED to be a close
            # i.e., next position is at max_depth and the action is close
            # Equivalently: target token is a closer (1 or 3) AND position-after has only one valid choice
            # Simpler: mask positions where next token is closer (1 or 3) — these are where bracket-type matters
            for t in range(T-1):
                nxt = inputs[b, t+1]
                if nxt in (1, 3):
                    mask[b, t] = True

        return inputs, targets, mask

    def random_baseline_acc(self):
        # Random over 4 token types when forced to close (only types 1 or 3 are valid closers)
        # Conditional: given we know it's a closer, random pick = 1/2
        return 0.5
