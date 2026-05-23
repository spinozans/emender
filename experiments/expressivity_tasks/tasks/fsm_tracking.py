"""FSM tracking: simulate a small finite state machine where the input is
a stream of "actions" and the target is the current state.

Default FSM: 4-state direction tracker.
  States: 0=N, 1=E, 2=S, 3=W
  Actions:
    0 = turn left  (rotate CCW)
    1 = turn right (rotate CW)
    2 = no-op
  At each step, model predicts current state.

Linear models cannot maintain state under modular rotations over arbitrary
length (similar to parity / counting).
"""
import numpy as np


class FSMTrackingTask:
    name = 'fsm_tracking'

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        # 3 actions encode rotations; output is current state.
        # Vocabulary covers actions (0,1,2) AND state ids (0..n_states-1).
        self.vocab_size = max(3, n_states)

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        actions = rng.integers(0, 3, size=(B, T)).astype(np.int64)
        states = np.zeros((B, T), dtype=np.int64)
        s = np.zeros(B, dtype=np.int64)
        for t in range(T):
            a = actions[:, t]
            # left = -1, right = +1, noop = 0 (mod n_states)
            delta = np.where(a == 0, -1, np.where(a == 1, 1, 0))
            s = (s + delta) % self.n_states
            states[:, t] = s
        return actions, states, np.ones((B, T), dtype=bool)

    def random_baseline_acc(self):
        return 1.0 / self.n_states
