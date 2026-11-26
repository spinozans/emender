"""Plan ragged work without an N-row mask per query or a maximum number of hits.

Unmatched ranges use whole dyadic feature summaries and explicit cheap rows at
partial edge leaves. All contributions are disjoint and positive: no subtraction
of a large approximate mass from a nearly equal total.
"""

from dataclasses import dataclass
from operator import index as integer


def ceildiv(a, b):
    return (a + b - 1) // b


@dataclass(frozen=True)
class Layout:
    version: str
    rows: int
    leaf: int = 64

    def __post_init__(self):
        if not self.version or self.rows < 0 or self.leaf < 1:
            raise ValueError("invalid row layout")

    @property
    def widths(self):
        n = ceildiv(self.rows, self.leaf)
        out = []
        while n:
            out.append(n)
            if n == 1:
                break
            n = ceildiv(n, 2)
        return tuple(out)

    def cover(self, lo, hi):
        """Maximal whole nodes covering [lo,hi) in leaf units; truncated right spine allowed."""
        widths = self.widths
        if lo == hi:
            return ()
        offsets = [0]
        for w in widths:
            offsets.append(offsets[-1] + w)
        n = widths[0]
        out = []
        todo = [(len(widths) - 1, 0)]
        while todo:
            level, i = todo.pop()
            a = i << level
            b = min((i + 1) << level, n)
            if b <= lo or hi <= a:
                continue
            if lo <= a and b <= hi:
                out.append(offsets[level] + i)
                continue
            todo.append((level - 1, 2 * i))
            if 2 * i + 1 < widths[level - 1]:
                todo.append((level - 1, 2 * i + 1))
        return tuple(out)


def union(ranges, n):
    spans = []
    for a, b in sorted((integer(a), integer(b)) for a, b in ranges):
        if not 0 <= a <= b <= n:
            raise ValueError("out-of-range row interval")
        if a == b:
            continue
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))
    return spans


@dataclass(frozen=True)
class Plan:
    layout: Layout
    batch: int
    exact: tuple  # (query, row start, exclusive stop)
    edge: tuple  # unrefined rows in a partial leaf: explicit feature score
    nodes: tuple  # (query, node id): summarized unrefined rows

    @classmethod
    def build(cls, layout, hits):
        hits = tuple(hits)
        exact = []
        edge = []
        nodes = []
        n = layout.rows
        size = layout.leaf
        if not hits:
            raise ValueError("expected at least one query")
        for q, rs in enumerate(hits):

            def tail(a, b):
                if a == b:
                    return
                first = ceildiv(a, size)
                stop = ceildiv(n, size) if b == n else b // size
                if first >= stop:
                    edge.append((q, a, b))
                    return
                if a < first * size:
                    edge.append((q, a, first * size))
                nodes.extend((q, i) for i in layout.cover(first, stop))
                if stop * size < b:
                    edge.append((q, stop * size, b))

            prev = 0
            for a, b in union(rs, n):
                tail(prev, a)
                exact.append((q, a, b))
                prev = b
            tail(prev, n)
        return cls(layout, len(hits), tuple(exact), tuple(edge), tuple(nodes))

    @classmethod
    def from_frontiers(cls, index, prefixes, *, leaf=64, min_length=1):
        return cls.build(
            Layout(index.version, len(index), leaf),
            [[index.frontier(p).hit_range(min_length)] for p in prefixes],
        )


def tiles(ranges, size):
    return tuple(
        (q, i, min(size, b - i)) for q, a, b in ranges for i in range(a, b, size)
    )
