"""Read at the writer, exactly one extra hop within one real window. No caller-supplied addresses."""

from __future__ import annotations

import torch
from torch import nn
from .compact import evidence
from .values import LiveValueHead


class WriteHead(nn.Module):
    def __init__(self, literal: LiveValueHead, *, arm, seed):
        super().__init__()
        if arm not in ("P", "L") or seed not in (4901, 4902, 4903):
            raise ValueError("unknown writer or seed")
        self.arm = arm
        self.literal = literal
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(49000 + seed)
            self.inner = nn.Linear(80, 64)
            self.out = nn.Linear(64, 32)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        word,
        hidden,
        logits,
        *,
        mode,
        horizon=4096,
        inner_sham=False,
        outer_sham=False,
        inspect_rotation=False,
        work=None,
        age_out=None,
    ):
        if mode != self.arm or (inner_sham and (mode != "L" or outer_sham)):
            raise ValueError("wrong predictor or incompatible writer intervention")
        b = self.literal
        n = len(word)
        dev = hidden.device
        if hidden.shape != (n, b.direct.host_dim) or logits.shape != (n, 256):
            raise ValueError("misaligned writer inputs")
        donors = outer_sham or inner_sham or inspect_rotation
        f, q, src, alt, counts = evidence(word, horizon, device=dev, rotate=donors)
        base, _ = b.direct(hidden, logits, f)
        ids = torch.tensor(list(word), device=dev, dtype=torch.long)
        emb = b.byte(ids)
        written = b.writer(torch.cat((hidden, emb), dim=-1)).tanh()
        # q are all hit frontiers, src are their earlier follower slots. Compute an enriched write
        # at q from a BASE write at src, never from an already-enriched (recursive) bank.
        changed = 0
        used = 0
        two = 0
        old128 = 0
        old1024 = 0
        answer = base
        if len(q):
            code = emb[src]
            inner_operand = (
                written[alt if inner_sham else src]
                if mode == "L"
                else b.writer(torch.cat((hidden[q], code), -1)).tanh()
            )
            corr = self.out(
                self.inner(torch.cat((written[q], code, inner_operand), -1)).tanh()
            )
            stored = written.index_copy(0, q, written[q] + corr)
            outer = alt if outer_sham else src
            answer = b.answer(base, hidden, f, q, code, stored[outer])
        if work is not None or age_out is not None:
            # Diagnostics use the true links; altered donors never become new logical addresses.
            age = torch.zeros(n, device=dev, dtype=torch.long)
            age[q] = q - src
            inner_age = age[src]
            hit2 = inner_age > 0
            effective = torch.zeros_like(age)
            effective[q[hit2]] = age[q[hit2]] + inner_age[hit2]
            two = int(hit2.sum())
            old128 = int((effective > 128).sum())
            old1024 = int((effective > 1024).sum())
            if age_out is not None:
                age_out.update(literal=age, linked=effective)
            if inner_sham:
                different = torch.zeros(n, dtype=torch.bool, device=dev)
                different[q] = src != alt
                changed = int(different.sum())
                used = int(different[src].sum())
        if work is not None:
            counts["literal_payloads_reassigned"] = (
                int((src != alt).sum()) if outer_sham else 0
            )
            work.update(
                counts,
                literal_value_writes=n,
                literal_value_reads=len(q),
                writer_lookups_reused=n,
                writer_hit_enrichments=len(q),
                writer_old_value_reads=len(q) if mode == "L" else 0,
                writer_current_recomputations=len(q) if mode == "P" else 0,
                linked_uses=two,
                linked_misses=n - two,
                linked_effective_age_gt128=old128,
                linked_effective_age_gt1024=old1024,
                inner_writes_reassigned=changed,
                inner_reassignments_consumed=used,
            )
        return answer
