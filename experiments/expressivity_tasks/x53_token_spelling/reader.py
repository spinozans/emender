"""Causal prediction from a packed token response or its first-byte projection."""

from collections import Counter
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from .archive import U32, WORD

ARMS = ("BYTE", "FIRST", "LEX", "SPELL")
PRE, REFRESH, INSIDE = range(3)


def bands(raw):
    active = False
    tags = np.empty(len(raw), np.uint8)
    for t, b in enumerate(raw):
        if not WORD[b]:
            active = True
            tags[t] = REFRESH
        else:
            tags[t] = INSIDE if active else PRE
    return tags


class Saved:
    def __init__(self, wire, arm, lex, root_prior):
        self.cum = None
        if wire == b"R":
            local = 0
        elif wire[:1] == b"C" and arm == "SPELL":
            pairs = np.frombuffer(wire, U32, offset=1).reshape(-1, 2)
            self.ids = pairs[:, 0].copy()
            self.cum = np.empty(len(pairs) + 1, np.int64)
            self.cum[0] = 0
            np.cumsum(pairs[:, 1], dtype=np.int64, out=self.cum[1:])
            local = np.zeros(256, np.int64)
            np.add.at(local, lex.first[self.ids], pairs[:, 1])
        elif wire[:1] == b"C" and len(wire) == 1025:
            local = np.frombuffer(wire, U32, offset=1)
        else:
            raise ValueError("malformed source response")
        first = local + root_prior
        self.first = first / first.sum()

    def inside(self, row, lex, node, counts):
        if node < 0:
            counts["unsupported_gap"] += 1
            return
        sl = lex.span(node)
        root = lex.mass[sl]
        labels = lex.labels[sl]
        if not len(labels):
            raise ValueError("unexpected complete token in a word prefix")
        if self.cum is None:
            mass = root
        else:
            lo = self.ids.searchsorted(lex.lo[sl])
            hi = self.ids.searchsorted(lex.hi[sl])
            local = self.cum[hi] - self.cum[lo]
            n = local.sum()
            counts["context_supported_gap"] += int(n > 0)
            # When local mass vanishes, use the root law exactly (not a rounded rescale).
            mass = local + (4.0 / lex.total) * root if n else root
        counts["lex_supported_gap"] += 1
        row *= 0.5
        row[labels] += 0.5 * mass / mass.sum()


@dataclass
class Prediction:
    pdf: np.ndarray
    # A route is saved for the audit, never read as a model input. -1 off schedule.
    route: np.ndarray | None
    counts: Counter


class Reader:
    def __init__(self, arm, base, *, service=None, root_first=None, lex=None):
        if (
            arm not in ARMS
            or (service is None) != (arm == "BYTE")
            or (root_first is None) != (arm == "BYTE")
            or (lex is not None) != (arm in ("LEX", "SPELL"))
        ):
            raise ValueError("wrong reader capability")
        if service and service.full != (arm == "SPELL"):
            raise ValueError("wrong response format")
        self.arm = arm
        self.base = base
        self.service = service
        self.lex = lex
        self.root_prior = (
            None if root_first is None else (4.0 / int(root_first.sum())) * root_first
        )

    def predict(self, x):
        if type(x) is not bytes or not x:
            raise ValueError("expected a nonempty byte history")
        rows = self.base.predict(x)
        counts = Counter()
        if self.service is None:
            return Prediction(rows, None, counts)
        route = np.full((len(x), 2), -1, np.int64)
        start = None
        h = None
        node = 0
        answer = None
        for t, b in enumerate(x):
            if not WORD[b]:
                tick = perf_counter_ns()
                token = None if start is None else x[start : t + 1]
                h, wire = self.service.refresh(h, token, counts)
                counts["lookup_ns"] += perf_counter_ns() - tick
                route[t] = h._state, h.length
                start = t + 1
                node = 0
                answer = Saved(wire, self.arm, self.lex, self.root_prior)
                rows[t] = 0.5 * (rows[t] + answer.first)
            elif start is not None and self.lex is not None:
                node = self.lex.step(node, b)
                answer.inside(rows[t], self.lex, node, counts)
        return Prediction(rows, route, counts)


def reader(arm, atlas):
    from .archive import Service

    if arm == "BYTE":
        return Reader("BYTE", atlas.base)
    return Reader(
        arm,
        atlas.base,
        service=Service(atlas, arm == "SPELL"),
        root_first=atlas.lex.first_counts,
        lex=atlas.lex if arm in ("LEX", "SPELL") else None,
    )
