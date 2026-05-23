"""Keyed finite-state memory.

This task combines the two axes that distinguish E88/NDM from M2RNN:

1. Keyed mutable memory: many independent keys must be updated and queried.
2. State-dependent updates: some operations transform the old value at a key.

Token layout:
  0 = pad / answer slot input
  1 = QUERY marker
  2..2+n_keys-1 = key tokens
  key_offset+n_keys..+n_states-1 = SET-state operations and answer tokens
  add_offset..add_offset+n_add_ops-1 = ADD-r operations modulo n_states

Each operation is encoded as two tokens: KEY OP. At the end, the model sees
QUERY KEY and must emit the current state token for that key.
"""
import numpy as np


class KeyedFSMMemoryTask:
    name = 'keyed_fsm_memory'

    def __init__(self, n_keys: int = 8, n_states: int = 8, n_ops: int = 48):
        assert n_keys >= 2
        assert n_states >= 2
        assert n_ops >= 4
        self.n_keys = n_keys
        self.n_states = n_states
        self.n_ops = n_ops
        self.n_add_ops = min(n_states - 1, 4)
        self.key_offset = 2
        self.state_offset = self.key_offset + n_keys
        self.add_offset = self.state_offset + n_states
        self.vocab_size = self.add_offset + self.n_add_ops

    def _key_token(self, key):
        return self.key_offset + key

    def _state_token(self, state):
        return self.state_offset + state

    def _add_token(self, amount):
        assert 1 <= amount <= self.n_add_ops
        return self.add_offset + amount - 1

    def _apply_op(self, state, op):
        if self.state_offset <= op < self.state_offset + self.n_states:
            return op - self.state_offset
        if self.add_offset <= op < self.add_offset + self.n_add_ops:
            amount = op - self.add_offset + 1
            return (state + amount) % self.n_states
        raise ValueError(f"invalid op token {op}")

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        N = self.n_ops
        assert T >= 2 * N + 3, f"T={T} too small for {N} ops + query"

        inputs = np.zeros((B, T), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)

        for b in range(B):
            states = rng.integers(self.n_states, size=self.n_keys)

            # Ensure every key gets at least one operation, then fill the rest.
            keys = np.empty(N, dtype=np.int64)
            keys[:self.n_keys] = np.arange(self.n_keys)
            keys[self.n_keys:] = rng.integers(self.n_keys, size=N - self.n_keys)
            rng.shuffle(keys)

            for i, key in enumerate(keys):
                # Mix absolute writes and old-state-dependent modular updates.
                if rng.random() < 0.35:
                    new_state = int(rng.integers(self.n_states))
                    op = self._state_token(new_state)
                    states[key] = new_state
                else:
                    amount = int(rng.integers(1, self.n_add_ops + 1))
                    op = self._add_token(amount)
                    states[key] = (states[key] + amount) % self.n_states

                inputs[b, 2 * i] = self._key_token(int(key))
                inputs[b, 2 * i + 1] = op

            query_key = int(rng.integers(self.n_keys))
            qpos = 2 * N
            inputs[b, qpos] = 1
            inputs[b, qpos + 1] = self._key_token(query_key)
            targets[b, qpos + 2] = self._state_token(int(states[query_key]))
            mask[b, qpos + 2] = True

        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.n_states
