import numpy as np
import pytest
from ndm.memory_program.m20 import MaterializedBranches, TerminalBranches
from ndm.memory_program.runtime import Unavailable


class Pages:
    def __init__(self):
        words = [b"", b"\x00", b"\x00\x1e", b"\xff"] + [b"a" * n for n in range(1, 9)]
        self.ids = {s: i for i, s in enumerate(words, 1)}
        self.root = 1
        self.nodes = len(words)
        self.r = []
        self.s = []
        self.fault = None

    def data(self, id_):
        key = next(k for k, v in self.ids.items() if v == id_)
        child = np.array([self.ids.get(key + bytes([i]), 0) for i in range(256)], dtype="<u4")
        h = np.zeros(257, "<u8")
        h[:256][child > 0] = 30
        h[256] = 4
        if key in (b"a" * 8, b"\x00"):
            h[255] = 12  # true following bytes; no materialized child
        return dict(h=h, child=child)

    def rows(self, ids):
        self.r.append(tuple(ids))
        return [self.data(i) for i in ids]

    def suffix(self, key):
        self.s.append(key)
        ids = np.zeros((len(key), 8), "<u4")
        for i in range(len(key)):
            for j in range(1, min(i + 1, 8) + 1):
                ids[i, j - 1] = self.ids.get(key[i + 1 - j : i + 1], 0)
        if self.fault == "shape":
            return ids[:, 1:]
        if self.fault == "type":
            return ids.astype("<i4")
        if self.fault == "range":
            ids[-1, 0] = self.nodes + 1
        if self.fault == "gap":
            ids[0, 0] = 0
        if self.fault == "child":
            ids[-1, len(key) - 1] = 0
        return ids


def make(cls, p):
    return cls("archive", "immutable-v1", p, scope="whole", cache_pages=2)


def test_terminal_returns_the_same_257_bins_and_refusals_with_only_decisive_rows():
    p = Pages()
    a = make(TerminalBranches, p)
    b = make(MaterializedBranches, Pages())
    keys = [
        b"a" * 7,
        b"\x00\x1e",
        b"\x00\xffhello",
        b"a" * 8 + b"\xffx",
        b"a" * 8 + b"\x00",
        b"\x00\x1e",
        b"",
        b"\xff",
        b"\xfe",
    ]
    rows = a.histograms(keys)
    assert (
        rows == b.histograms(keys)
        and rows[2] == rows[3] == Unavailable("unmaterialized branch")
        and rows[4] == rows[8] == (0,) * 257
    )
    assert rows[1][256] == 4 and rows[-2][256] == 4
    assert p.s == [
        k[:8] for k in dict.fromkeys(keys) if k
    ]  # never pack literals together, incl identical truncated prefixes
    needed = [b"a" * 7, b"\x00\x1e", b"\x00", b"a" * 8, b"", b"\xff"]
    assert p.r == [tuple(p.ids[k] for k in needed)] and len(a.cache) <= 2
    assert a.histograms([]) == [] and len(p.r) == 1


@pytest.mark.parametrize("fault", ["shape", "type", "range", "gap", "child"])
def test_terminal_rejects_malformed_metadata_and_parent_contradiction(fault):
    p = Pages()
    p.fault = fault
    a = make(TerminalBranches, p)
    with pytest.raises(ValueError):
        a.histograms([b"aa"])
