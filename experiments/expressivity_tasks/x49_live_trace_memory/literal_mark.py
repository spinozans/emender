"""LS: a literal arrival-surprise tape and matched fields on B's unmodified hit path. No targets."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from math import floor, log
from time import perf_counter
import torch
from . import surprise as u

DEPTHS = (1, 2, 4, 8)
DIMS = {"N": 3, "T": 4, "H": 4, "J": 5, "S": 5}


@dataclass(slots=True)
class Followers:
    n: int = 0
    counts: dict[int, int] = field(default_factory=dict)


class LiteralHistory:
    """One document window's interpolation tables. Only the last eight literal bytes are needed as keys."""

    def __init__(self):
        self.n = 0
        self.order_zero = [0] * 256
        self.tables: dict[bytes, Followers] = {}
        self.tail = b""
        self.lookups = 0
        self.seen_lookups = 0
        self.pairs = 0
        self.updates = 0

    def _rows(self):
        keys = (self.tail[-d:] for d in DEPTHS if d <= self.n)
        return [(key, self.tables.get(key)) for key in keys]

    def _probability(self, b, rows):
        if type(b) is not int or not 0 <= b < 256:
            raise ValueError("expected an actual byte")
        p = (self.order_zero[b] + 0.5) / (self.n + 128)
        for _, row in rows:
            if row is not None:
                distinct = len(row.counts)
                p = (row.counts.get(b, 0) + distinct * p) / (row.n + distinct)
        if not 0 < p <= 1:
            raise ValueError("invalid literal forecast")
        return p

    def probability(self, b):
        """Non-mutating hypothetical component, used by the dense-prefix oracle tests."""
        return self._probability(b, self._rows())

    def observe(self, b):
        if self.n >= u.HORIZON:
            raise ValueError("start a fresh window")
        rows = self._rows()
        p = self._probability(b, rows)
        code = min(255, floor(16 * max(0.0, -log(p)) + 0.5)) if self.n else 0
        # No follower count, not even the order-zero count, includes the scored arrival yet.
        self.lookups += len(rows)
        for k, row in rows:
            if row is None:
                self.tables[k] = row = Followers()
            else:
                self.seen_lookups += 1
            count = row.counts.get(b, 0)
            row.counts[b] = count + 1
            row.n += 1
            self.pairs += int(count == 0)
            self.updates += 1
        self.order_zero[b] += 1
        self.n += 1
        self.tail = (self.tail + bytes((b,)))[-DEPTHS[-1] :]
        return code


def text_tape(word):
    if type(word) is not bytes or not 1 <= len(word) <= u.HORIZON:
        raise ValueError("one observed byte window required")
    hist = LiteralHistory()
    tape = torch.frombuffer(bytearray(hist.observe(b) for b in word), dtype=torch.uint8)
    return tape, dict(
        text_mark_storage_bytes=tape.nbytes,
        text_arrivals_encoded=hist.n - 1,
        text_order_zero_updates=hist.n,
        text_context_lookups=hist.lookups,
        text_context_updates=hist.updates,
        text_seen_context_lookups=hist.seen_lookups,
        text_distinct_contexts=len(hist.tables),
        text_distinct_pairs=hist.pairs,
        text_marks_saturated=int((tape[1:] == 255).sum()),
    )


@torch.inference_mode()
def fields(head, word, hidden, logits):
    # U's N/U/S reference computes B exactly once. Its wrong binding moves only the neural surprise.
    p = u.fields(head, word, hidden, logits)
    tic = perf_counter()
    tape, w = text_tape(word)
    clock = dict(p.seconds, text_build_s=perf_counter() - tic)
    tic = perf_counter()
    tc = tape[p.latest].long()
    n = p.phi["N"]
    h = p.phi["U"]
    s = p.phi["S"]
    t = torch.cat([n, tc.double()[:, None] / 64], dim=1)
    phi = dict(
        N=n,
        T=t,
        H=h,
        J=torch.cat([t, h[:, -1:]], dim=1),
        S=torch.cat([t, s[:, -1:]], dim=1),
    )
    clock["text_read_joint_fields_s"] = perf_counter() - tic
    # Exact integer sufficient sums across hits; not a persistent per-target record or a fit feature.
    hc = (h[:, -1] * 64).long()
    w.update(
        text_mark_reads=len(tc),
        text_read_code_saturated=int((tc == 255).sum()),
        text_vs_neural_changed=int((tc != hc).sum()),
        text_code_sum=int(tc.sum()),
        neural_code_sum=int(hc.sum()),
        text_code_square_sum=int(tc @ tc),
        neural_code_square_sum=int(hc @ hc),
        text_neural_code_product_sum=int(tc @ hc),
    )
    return replace(p, phi=phi, work=dict(p.work, **w), seconds=clock)


def response(packet, arm, coefficients):
    # H and U have identical fields, but only H is fitted by this study.
    key = "H" if arm == "U" else arm
    beta = u.checked_vector(DIMS, key, coefficients)
    return u.copy_response(packet, packet.phi[key], beta)
