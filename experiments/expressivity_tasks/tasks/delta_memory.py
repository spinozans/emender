"""Delta-memory tasks for testing overwrite and reset semantics.

These tasks are designed to hit the difference between raw associative writes
and error-correcting delta writes.

Token layout:
  0 = pad / answer slot input
  1 = QUERY marker
  2 = RESET marker
  3..3+n_keys-1 = key tokens
  3+n_keys..3+n_keys+n_values-1 = value tokens
  last token = NONE answer for reset/no-value
"""
import numpy as np


class OverwriteRecallTask:
    name = 'overwrite_recall'

    def __init__(self, n_keys: int = 8, n_values: int = 16, n_writes: int = 24):
        assert n_keys >= 2
        assert n_values >= 2
        assert n_writes >= 3
        self.n_keys = n_keys
        self.n_values = n_values
        self.n_writes = n_writes
        self.key_offset = 3
        self.value_offset = self.key_offset + n_keys
        self.none_token = self.value_offset + n_values
        self.vocab_size = self.none_token + 1

    def _key_token(self, key):
        return self.key_offset + key

    def _value_token(self, value):
        return self.value_offset + value

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        N = self.n_writes
        assert T >= 2 * N + 3, f"T={T} too small for {N} writes + query"

        inputs = np.zeros((B, T), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)

        for b in range(B):
            target_key = rng.integers(self.n_keys)
            first_value = rng.integers(self.n_values)
            final_value = rng.integers(self.n_values - 1)
            if final_value >= first_value:
                final_value += 1

            # Fill with non-target distractors, then place two target writes.
            # The final target write is not the final sequence write, so the
            # model has to keep keyed latest-value state through later writes.
            first_pos = int(rng.integers(0, max(1, N // 2)))
            final_pos = int(rng.integers(first_pos + 1, N - 1))
            keys = rng.integers(self.n_keys - 1, size=N)
            keys = np.where(keys >= target_key, keys + 1, keys)
            values = rng.integers(self.n_values, size=N)
            keys[first_pos] = target_key
            values[first_pos] = first_value
            keys[final_pos] = target_key
            values[final_pos] = final_value

            for i in range(N):
                inputs[b, 2 * i] = self._key_token(keys[i])
                inputs[b, 2 * i + 1] = self._value_token(values[i])

            qpos = 2 * N
            inputs[b, qpos] = 1
            inputs[b, qpos + 1] = self._key_token(target_key)
            targets[b, qpos + 2] = self._value_token(final_value)
            mask[b, qpos + 2] = True

        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.n_values


class ResetRecallTask:
    name = 'reset_recall'

    def __init__(self, n_keys: int = 8, n_values: int = 16, n_ops: int = 24):
        assert n_keys >= 2
        assert n_values >= 2
        assert n_ops >= 3
        self.n_keys = n_keys
        self.n_values = n_values
        self.n_ops = n_ops
        self.key_offset = 3
        self.value_offset = self.key_offset + n_keys
        self.none_token = self.value_offset + n_values
        self.vocab_size = self.none_token + 1

    def _key_token(self, key):
        return self.key_offset + key

    def _value_token(self, value):
        return self.value_offset + value

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        N = self.n_ops
        assert T >= 2 * N + 3, f"T={T} too small for {N} ops + query"

        inputs = np.zeros((B, T), dtype=np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)

        for b in range(B):
            target_key = rng.integers(self.n_keys)
            initial_value = rng.integers(self.n_values)
            reset_target = bool(rng.integers(2))

            first_pos = int(rng.integers(0, max(1, N // 2)))
            final_pos = int(rng.integers(first_pos + 1, N - 1))
            non_target_keys = rng.integers(self.n_keys - 1, size=N)
            non_target_keys = np.where(non_target_keys >= target_key, non_target_keys + 1, non_target_keys)
            ops = [
                (self._key_token(int(key)), self._value_token(int(rng.integers(self.n_values))))
                for key in non_target_keys
            ]

            ops[first_pos] = (self._key_token(target_key), self._value_token(initial_value))
            if reset_target:
                expected = self.none_token
                ops[final_pos] = (2, self._key_token(target_key))
            else:
                final_value = rng.integers(self.n_values - 1)
                if final_value >= initial_value:
                    final_value += 1
                expected = self._value_token(final_value)
                ops[final_pos] = (self._key_token(target_key), expected)

            for i, (a, b_tok) in enumerate(ops):
                inputs[b, 2 * i] = a
                inputs[b, 2 * i + 1] = b_tok

            qpos = 2 * N
            inputs[b, qpos] = 1
            inputs[b, qpos + 1] = self._key_token(target_key)
            targets[b, qpos + 2] = expected
            mask[b, qpos + 2] = True

        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / (self.n_values + 1)
