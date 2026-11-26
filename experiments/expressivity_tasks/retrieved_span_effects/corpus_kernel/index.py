"""Occurrence index of predictive boundaries, ordered by *reversed* causal context.

Equal substrings do not collapse occurrences. A nested family of suffix intervals
contains the longest frontier-ending exact match for each endpoint. Rank doubling
uses bounded document offsets, never materializing all prefix strings.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from functools import cached_property
from operator import index as integer

import numpy as np


def tokens(seq):
    out = []
    for t in seq:
        if isinstance(t, (bool, np.bool_)):
            raise ValueError("boolean token")
        t = integer(t)
        if not 0 <= t < 2**63:
            raise ValueError("expected nonnegative signed int64 token")
        out.append(t)
    return tuple(out)


def _version_field(h, blob):
    h.update(len(blob).to_bytes(8, "little"))
    h.update(blob)


@dataclass(frozen=True)
class Frontier:
    version: str
    intervals: tuple[
        tuple[int, int], ...
    ]  # entry l-1 = endpoints matching last l tokens

    def hit_range(self, min_length=1):
        if min_length < 1:
            raise ValueError("use an explicit full range to refine all rows")
        return (
            self.intervals[min_length - 1]
            if min_length <= len(self.intervals)
            else (0, 0)
        )

    def maximal_runs(self):
        """Disjoint (lo,hi,length): longest left extension at each matching occurrence."""
        for depth, (a, b) in enumerate(self.intervals, 1):
            c, d = self.intervals[depth] if depth < len(self.intervals) else (b, b)
            if a < c:
                yield a, c, depth
            if d < b:
                yield d, b, depth


@dataclass(frozen=True)
class Trail:
    version: str
    rows: (
        range | tuple[int, ...]
    )  # ORIGINAL endpoints: an extension doesn't create votes
    depth: int = 0


class EndpointIndex:
    def __init__(self, documents: Iterable, *, tokenizer: str):
        if not isinstance(tokenizer, str) or not tokenizer:
            raise ValueError("bind tokenizer identity")
        h = sha256(b"corpus-attention/endpoints/v1")
        _version_field(h, tokenizer.encode())
        self.tokenizer = tokenizer
        names = []
        docs = []
        seen = set()
        for name, seq in documents:
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("duplicate/invalid document identity")
            doc = tokens(seq)
            seen.add(name)
            names.append(name)
            docs.append(doc)
            _version_field(h, name.encode())
            _version_field(h, np.asarray(doc, dtype="<i8").tobytes())
        self.version = h.hexdigest()
        self.names = tuple(names)
        self.documents = tuple(docs)
        self.sources = frozenset(names)
        lengths = np.asarray(list(map(len, docs)), dtype=np.int64)
        nt = int(lengths.sum())
        start = np.cumsum(lengths) - lengths
        docid = np.repeat(np.arange(len(docs)), lengths)
        pos = np.arange(nt, dtype=np.int64)
        offset = pos - np.repeat(start, lengths)
        flat = np.fromiter((t for doc in docs for t in doc), np.int64, count=nt)
        rank = (
            np.unique(flat, return_inverse=True)[1] + 1
        )  # 0 is structural start, not token id 0
        span = 1
        while span < int(lengths.max(initial=0)):
            second = np.zeros_like(rank)
            ok = offset >= span
            second[ok] = rank[pos[ok] - span]
            order = np.lexsort((second, rank))
            r, s = rank[order], second[order]
            rank[order] = np.cumsum(np.r_[True, (r[1:] != r[:-1]) | (s[1:] != s[:-1])])
            span *= 2
        eligible = pos[offset + 1 < lengths[docid]]
        self.order = np.argsort(rank[eligible], kind="stable")
        p = eligible[self.order]
        self.row_doc = docid[p]
        self.row_offset = offset[p]
        self.successor = flat[p + 1]
        self._flat = flat
        self._ends = p
        for a in (
            self.order,
            self.row_doc,
            self.row_offset,
            self.successor,
            self._flat,
            self._ends,
        ):
            a.flags.writeable = False

    def __len__(self):
        return len(self.order)

    @cached_property
    def inverse_order(self):
        """Only internal-anchor alignment needs this document-order -> sorted row map."""
        rank = np.empty_like(self.order)
        rank[self.order] = np.arange(len(rank))
        rank.flags.writeable = False
        return rank

    def check_live(self, name):
        if not isinstance(name, str) or not name or name in self.sources:
            raise ValueError("live source must be provisioned outside the archive")

    def frontier(self, observed):
        lo, hi = 0, len(self)
        intervals = []
        for depth, t in enumerate(reversed(tokens(observed))):
            # All rows in [lo,hi) agree through depth-1, so the next older token is sorted.
            def bound(a, b, target):
                while a < b:
                    m = (a + b) // 2
                    u = (
                        int(self._flat[self._ends[m] - depth])
                        if self.row_offset[m] >= depth
                        else -1
                    )
                    if u < target:
                        a = m + 1
                    else:
                        b = m
                return a

            lo = bound(lo, hi, t)
            hi = bound(lo, hi, t + 1)
            if lo == hi:
                break
            intervals.append((lo, hi))
        return Frontier(self.version, tuple(intervals))

    def trail(self, frontier, min_length=1):
        if frontier.version != self.version:
            raise ValueError("foreign frontier")
        return Trail(self.version, range(*frontier.hit_range(min_length)))

    def branch(self, path):
        """Partition original row identities by ONE more corpus token, or structural end."""
        if path.version != self.version or path.depth < 0:
            raise ValueError("foreign/invalid trail")
        by_token = {}
        ends = []
        for r in path.rows:
            if not 0 <= r < len(self):
                raise ValueError("invalid occurrence")
            doc = self.documents[self.row_doc[r]]
            p = self.row_offset[r] + path.depth + 1
            if p == len(doc):
                ends.append(r)
            elif p < len(doc):
                by_token.setdefault(doc[p], []).append(r)
            else:
                raise ValueError("trail extended past document end")
        return {
            t: Trail(self.version, tuple(rows), path.depth + 1)
            for t, rows in by_token.items()
        }, tuple(ends)
