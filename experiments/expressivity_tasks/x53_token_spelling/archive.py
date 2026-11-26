"""BUILD-only byte counts, reproducible token-index image and packed responses."""

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import struct

import numpy as np

from experiments.expressivity_tasks.token_suffix_fanout.index import TokenIndex

WORD = bytes(
    int(65 <= b <= 90 or 97 <= b <= 122 or 48 <= b <= 57 or b == 95) for b in range(256)
)
U32 = np.dtype("<u4")


def pieces(raw):
    """Ignore the opening fragment through its delimiter, and the unclosed tail."""
    start = None
    for i, b in enumerate(raw):
        if not WORD[b]:
            if start is not None:
                yield raw[start : i + 1]
            start = i + 1


class Builder:
    def __init__(self):
        self.c0 = np.zeros(256, U32)
        self.c1 = np.zeros((256, 256), U32)
        self.c2 = np.zeros((65536, 256), U32)
        self.words = {}
        self.docs = []

    def add(self, raw):
        if type(raw) is not bytes or len(raw) < 2:
            raise ValueError("need a real shifted window")
        x = np.frombuffer(raw, np.uint8).astype(np.int32)
        np.add.at(self.c0, x[1:], 1)
        np.add.at(self.c1, (x[:-1], x[1:]), 1)
        np.add.at(self.c2, (x[:-2] * 256 + x[1:-1], x[2:]), 1)
        self.docs.append(
            np.fromiter(
                (self.words.setdefault(p, len(self.words)) for p in pieces(raw)),
                dtype=U32,
            )
        )

    def save(self, path):
        words = sorted(self.words)
        if not words:
            raise ValueError("BUILD has no closed tokens")
        ids = np.empty(len(words), U32)
        ids[[self.words[p] for p in words]] = np.arange(len(words), dtype=U32)

        def offsets(seq):
            return np.array(
                [0, *np.cumsum([len(w) for w in seq], dtype=np.int64)], dtype=U32
            )

        with path.open("xb") as f:
            np.savez(
                f,
                c0=self.c0,
                c1=self.c1,
                c2=self.c2,
                spell=np.frombuffer(b"".join(words), np.uint8),
                offsets=offsets(words),
                docs=offsets(self.docs),
                tokens=ids[np.concatenate(self.docs)],
            )


class ByteModel:
    def __init__(self, c0, c1, c2):
        if (
            (c0.shape, c1.shape, c2.shape) != ((256,), (256, 256), (65536, 256))
            or any(v.dtype != U32 for v in (c0, c1, c2))
            or not np.array_equal(c0, c1.sum(0))
        ):
            raise ValueError("bad byte count image")
        self.zero = (c0 + 0.5) / (int(c0.sum()) + 128)
        self.one = (c1 + 16 * self.zero) / (c1.sum(1)[:, None] + 16)
        self.c2 = c2
        self.total = c2.sum(1)

    def predict(self, x):
        """Full 256-way laws; only adjacent observed bytes index a count row."""
        v = np.frombuffer(x, np.uint8).astype(np.int32)
        out = self.one[v]
        key = v[:-1] * 256 + v[1:]
        out[1:] = (self.c2[key] + 16 * out[1:]) / (self.total[key, None] + 16)
        return out


class Lexicon:
    """A local spelling dictionary. All descendants of a prefix occupy a sorted ID interval."""

    def __init__(self, words, freq):
        if not words or len(freq) != len(words) or min(freq) <= 0:
            raise ValueError("empty lexical source")
        self.children = [{}]
        lo = [0]
        hi = [len(words)]
        self.first = np.fromiter((w[0] for w in words), np.uint8)
        for i, w in enumerate(words):
            if (
                (i and w <= words[i - 1])
                or not w
                or WORD[w[-1]]
                or not all(WORD[b] for b in w[:-1])
            ):
                raise ValueError("invalid complete token or lexical order")
            node = 0
            for b in w:
                dest = self.children[node].get(b)
                if dest is None:
                    dest = len(lo)
                    self.children[node][b] = dest
                    self.children.append({})
                    lo.append(i)
                    hi.append(i + 1)
                else:
                    hi[dest] = i + 1
                node = dest
        ptr = [0]
        labels = []
        start = []
        stop = []
        for children in self.children:
            for b, d in children.items():
                labels.append(b)
                start.append(lo[d])
                stop.append(hi[d])
            ptr.append(len(labels))
        self.ptr = np.asarray(ptr, np.int64)
        self.labels = np.asarray(labels, np.uint8)
        self.lo = np.asarray(start, U32)
        self.hi = np.asarray(stop, U32)
        cs = np.r_[0, np.cumsum(freq, dtype=np.int64)]
        self.mass = cs[self.hi] - cs[self.lo]
        self.total = int(cs[-1])
        self.first_counts = np.zeros(256, U32)
        np.add.at(self.first_counts, self.first, freq.astype(U32))

    def step(self, node, b):
        return -1 if node < 0 else self.children[node].get(b, -1)

    def span(self, node):
        return slice(self.ptr[node], self.ptr[node + 1])


class Atlas:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as im:
            if set(im.files) != {
                "c0",
                "c1",
                "c2",
                "spell",
                "offsets",
                "docs",
                "tokens",
            }:
                raise ValueError("unknown image")
            self.base = ByteModel(im["c0"], im["c1"], im["c2"])
            raw = im["spell"].tobytes()
            offsets = im["offsets"]
            ds = im["docs"]
            tokens = im["tokens"]
        if not (
            offsets[0] == ds[0] == 0
            and offsets[-1] == len(raw)
            and ds[-1] == len(tokens)
            and np.all(offsets[1:] > offsets[:-1])
            and np.all(ds[1:] >= ds[:-1])
            and tokens.size
            and tokens.max() < len(offsets) - 1
        ):
            raise ValueError("bad token image")
        words = [raw[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
        self.ids = {p: i for i, p in enumerate(words)}
        freq = np.bincount(tokens.astype(np.int64), minlength=len(words))
        self.lex = Lexicon(words, freq)
        self.index = TokenIndex(
            tokens[a:b].tolist() for a, b in zip(ds[:-1], ds[1:], strict=True)
        )
        root = self.index.branch(self.index.root)
        if (
            root.ends != len(ds) - 1
            or [e.token for e in root.edges] != list(range(len(words)))
            or not np.array_equal([e.count for e in root.edges], freq)
        ):
            raise ValueError("root differs from direct token census")
        self.geometry = dict(
            documents=len(ds) - 1,
            tokens=len(tokens),
            spelling_bytes=len(raw),
            token_types=len(words),
            trie_nodes=len(self.lex.children),
            suffix_states=len(self.index._states),
            suffix_edges=sum(len(s.edges) for s in self.index._states),
            byte_targets=int(self.base.total.sum()) + len(ds) - 1,
        )

    def digest(self):
        h = sha256()
        buf = bytearray()
        for st in self.index._states:
            buf.extend(
                struct.pack("<IiII", st.longest, st.link, st.occurrences, len(st.edges))
            )
            for t, j in sorted(st.edges.items()):
                buf.extend(struct.pack("<iI", t, j))
            if len(buf) > 65536:
                h.update(buf)
                buf.clear()
        h.update(buf)
        return h.hexdigest()


@dataclass(frozen=True, slots=True)
class Entry:
    wire: bytes
    fanout: int


class Service:
    """One arm's navigator and immutable source-response cache. No cross-arm answer reuse."""

    def __init__(self, atlas, full, capacity=4096):
        self.atlas = atlas
        self.full = full
        self.cache = OrderedDict()
        self.capacity = capacity

    def refresh(self, h, token, counts):
        a = self.atlas
        idx = a.index
        if token is None:
            h = idx.root  # first real delimiter anchors the window
        else:
            t = a.ids.get(token)
            counts["commits"] += 1
            counts["known_commits"] += int(t is not None)
            h = idx.root if t is None else idx.advance(h, t)
        counts["reads"] += 1
        counts["suffix_length_sum"] += h.length
        if h == idx.root:
            counts["root_reads"] += 1
            entry = Entry(b"R", len(a.ids))
        else:
            key = h._state
            entry = self.cache.get(key)
            if entry is None:
                counts["nonroot_misses"] += 1
                answer = idx.branch(h)
                n = len(answer.edges)
                if self.full:
                    arr = np.asarray(
                        [(e.token, e.count) for e in answer.edges], dtype=U32
                    ).reshape(-1, 2)
                else:
                    arr = np.zeros(256, U32)
                    for e in answer.edges:
                        arr[a.lex.first[e.token]] += e.count
                entry = Entry(b"C" + arr.tobytes(), n)
                self.cache[key] = entry
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(key)
            counts["nonroot_empty"] += int(entry.fanout == 0)
        # Simulate a real in-process response boundary: no candidate array or cache
        # identity crosses it. This is copied payload volume, NOT network traffic.
        wire = memoryview(entry.wire).tobytes()
        counts["logical_entries"] += entry.fanout
        counts["wire_bytes"] += len(wire)
        counts["returned_entries"] += (len(wire) - 1) // (8 if self.full else 4)
        return h, wire
