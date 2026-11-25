"""X49 same-address A/B reference head. One window of actual causal host features at a time."""

from __future__ import annotations

import torch
from torch import nn

from ndm.memory_program.direct import DirectTrust
from .compact import evidence


class LiveValueHead(nn.Module):
    """A and B have the same trainable tensors. Only the writer's host input differs on a hit."""

    def __init__(self, direct: DirectTrust):
        super().__init__()
        self.direct = direct
        self.byte = nn.Embedding(256, 16)
        self.writer = nn.Linear(direct.host_dim + 16, 32)
        self.mix = nn.Linear(direct.host_dim + 32 + 16 + 11 + 256, 64)
        self.delta = nn.Linear(64, 256)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        word,
        hidden,
        logits,
        *,
        mode,
        horizon=4096,
        work=None,
        inspect_rotation=False,
        age_out=None,
        hit_out=None,
    ):
        """word contains inputs only. A caller may score shifted targets AFTER this returns.

        `rotated` uses B's weights with a causal, byte- and age-bin-preserving wrong binding.
        The private plan is derived here; no interface accepts teacher-provided addresses.
        """
        if mode not in ("A", "B", "rotated"):
            raise ValueError("unknown literal value mode")
        if hidden.shape != (len(word), self.direct.host_dim) or logits.shape != (
            len(word),
            256,
        ):
            raise ValueError(
                "one aligned host row and 256 logits per observed byte required"
            )
        dev = hidden.device
        f, query, source, donor, counts = evidence(
            word,
            horizon,
            device=dev,
            rotate=mode == "rotated",
            inspect_rotation=inspect_rotation,
        )
        if hit_out is not None:
            # Output only: expose exactly the routes B actually uses, never accept addresses.
            if mode != "B" or type(hit_out) is not dict or hit_out:
                raise ValueError("only an empty B hit output is supported")
            hit_out.update(
                query=query, latest=source, depth=(8 * f[0][query, 7]).long()
            )
        if (
            age_out is not None
        ):  # output diagnostics only; never read to select a record
            ages = torch.zeros(len(word), dtype=torch.long, device=dev)
            ages[query] = query - source
            age_out["literal"] = ages
        if work is not None:
            work.update(
                counts,
                literal_value_writes=0 if mode == "A" else len(word),
                literal_value_reads=0 if mode == "A" else len(query),
            )
        base, _ = self.direct(hidden, logits, f)
        ids = torch.tensor(list(word), dtype=torch.long, device=dev)
        # B actually writes the value when observing j; the batched map is positionwise,
        # and is never detached. It cannot access later inputs. A has no historical writes.
        written = (
            None
            if mode == "A"
            else self.writer(torch.cat((hidden, self.byte(ids)), -1)).tanh()
        )
        if not len(query):
            return base
        code = self.byte(ids[source])
        if mode == "A":
            payload = self.writer(torch.cat((hidden[query], code), -1)).tanh()
        else:
            payload = written[donor]
        return self.answer(base, hidden, f, query, code, payload)

    def answer(self, base, hidden, f, query, code, payload):
        """The same outer B readout for all matched writer variants."""
        features = torch.cat(
            (hidden[query], payload, code, f[0][query], f[1][query, 1]), -1
        )
        correction = self.delta(self.mix(features).tanh())
        # Preserve the original log mass (and bit-identical incumbent at zero correction).
        original = base[query]
        dz = (original + correction).logsumexp(-1, keepdim=True) - original.logsumexp(
            -1, keepdim=True
        )
        return base.index_copy(0, query, original + (correction - dz))
