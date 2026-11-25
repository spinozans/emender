"""Online greedy decode: one external byte in, full next-byte logits out. No future input buffer."""

from dataclasses import dataclass

import numpy as np
import torch

from experiments.expressivity_tasks.x50_read_tape.program import ARMS, D, T, log_prior


@dataclass(frozen=True)
class Prediction:
    logits: torch.Tensor  # 256 unnormalized next-byte scores
    extension: int | None  # hypothetical operand, NOT an emitted/observed byte


class ReadSession:
    """A single growing tape for one document/window and one fixed immutable mount.

    Use with fixed CPU weights. `observe(b)` is the only API that advances the
    external transcript. A READ reply is projected to a row and remains visible
    through its K/V. `reset()` discards the entire neural tape and observed prefix.
    """

    def __init__(self, model, bank, arm="ONE"):
        if arm not in ARMS or next(model.parameters()).device.type != "cpu":
            raise ValueError("expected a registered CPU reader")
        self.model, self.bank, self.arm = model, bank, arm
        self.reset()

    def reset(self):
        self.observed = b""
        self.rows = 0
        self.keys, self.values = (torch.empty(1, 3 * T, D) for _ in range(2))

    def _deposit(self, kind, byte, t, pdf=None):
        m = self.model
        row = m.row(kind, torch.tensor([byte]), torch.tensor(t), pdf)
        i = self.rows
        self.keys[:, i], self.values[:, i] = m.k(row), m.v(row)
        self.rows += 1
        return row

    def _attend(self, row, *, observations_only=False):
        # BLIND has exactly three typed rows per past frontier; take OBS only.
        step = 3 if observations_only else 1
        return self.model.frontier(
            row, self.keys[:, : self.rows : step], self.values[:, : self.rows : step]
        )

    @torch.no_grad()
    def observe(self, byte):
        if type(byte) is not int or not 0 <= byte <= 255:
            raise ValueError("observe exactly one real external byte (0..255)")
        t = len(self.observed)
        if t >= T:
            raise OverflowError(
                "trained position range exhausted; reset at a new window"
            )
        # Do not partially commit a frontier after a failed source read. Inert
        # cache cells beyond the cursor are overwritten if the caller resets.
        prefix = self.observed + bytes([byte])
        old_rows = self.rows
        try:
            first = self.bank.read([prefix])
            p = torch.from_numpy(first.pdf)
            prior = log_prior(p)
            obs = self._deposit(0, byte, t)
            row = self._deposit(1, byte, t, p)
            a = None
            if self.arm != "ONE":
                if self.arm == "GREEDY":
                    act = p
                elif self.arm == "BLIND":
                    act = self.model.act(self._attend(obs, observations_only=True))
                else:
                    act = self.model.act(self._attend(row)) + self.model.scale * prior
                a = int(act.argmax(-1).item())
                reply = self.bank.follow(first, np.array([a], np.uint8))
                row = self._deposit(2, a, t, torch.from_numpy(reply.pdf))
            logits = (prior + self.model.emit(self._attend(row)))[0]
        except BaseException:
            self.rows = old_rows
            raise
        self.observed = prefix
        return Prediction(logits, a)
