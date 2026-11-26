"""Cache raw IDs and route descriptors, NEVER learned values, across training updates.

Power-of-two length buckets keep each document on its own time axis. Positions
beyond the end neither enter a real hidden state nor become a key, value or label.
"""

import numpy as np
import torch

from .anchors import proposals
from .attention import Local
from .index import tokens
from .plan import Layout, Plan
from .work import bind


class _Documents:
    def __init__(self, docs, order, vocab, device):
        self.lengths = tuple(max(0, len(d) - 1) for d in docs)
        offset = np.cumsum((0, *self.lengths))
        groups = {}
        rows = []
        last_order = []
        self.groups = []

        def tensor(x):
            return torch.tensor(x, device=device, dtype=torch.long)

        for i, d in enumerate(docs):
            if d:
                if min(d) < 0 or max(d) >= vocab:
                    raise ValueError("token outside vocabulary")
                groups.setdefault((len(d) - 1).bit_length(), []).append(i)
        for members in groups.values():
            width = max(len(docs[j]) for j in members)
            ids = np.zeros((len(members), width), np.int64)
            take = []
            last = []
            for a, j in enumerate(members):
                d = docs[j]
                ids[a, : len(d)] = d
                last.append(a * width + len(d) - 1)
                last_order.append(j)
                take.extend(range(a * width, a * width + len(d) - 1))
                rows.extend(range(offset[j], offset[j + 1]))
            self.groups.append((tensor(ids), tensor(take), tensor(last)))
        inverse = np.argsort(np.array(rows, dtype=np.int64))
        self.order = tensor(inverse[order])
        self.last_order = tensor(np.argsort(last_order))
        successors = np.fromiter(
            (t for d in docs for t in d[1:]), np.int64, count=int(offset[-1])
        )
        self.successors = tensor(successors[order])
        self.device = device

    def encode(self, m, *, need_rows, need_last):
        rows = []
        last = []
        for ids, take, end in self.groups:
            z = m._states(ids).reshape(-1, m.dim)
            if need_rows:
                rows.append(z[take])
            if need_last:
                last.append(z[end])
        h = m.embed.weight.new_empty((0, m.dim))
        v = m.embed.weight.new_empty((0, m.width))
        if need_rows and rows:
            h = m.key(torch.cat(rows)[self.order])
            v = m._values(self.successors)
        return h, v, torch.cat(last)[self.last_order] if need_last else None


class TokenArchive:
    def __init__(self, model, index):
        if model.tokenizer != index.tokenizer:
            raise ValueError("foreign tokenizer")
        self.device = model.embed.weight.device
        self.index = index
        self.tokenizer = model.tokenizer
        self.vocab = model.vocab
        self.rank = len(model.omega)
        self.layout = Layout(index.version, len(index), model.leaf)
        self.docs = _Documents(index.documents, index.order, self.vocab, self.device)

    def route(self, prefixes, min_match, locator, lookback):
        if locator not in ("cpu", "triton"):
            raise ValueError("unknown locator")
        return proposals(
            self.index,
            prefixes,
            min_match=min_match,
            lookback=lookback,
            seek=None if locator == "cpu" else self._intervals,
        )

    def _intervals(self, windows, depth):
        from .triton_frontier import ResidentIndex

        if not hasattr(self, "_resident"):
            self._resident = ResidentIndex(self.index, self.device)
        intervals = self._resident.intervals(
            torch.tensor(windows, dtype=torch.int64, device=self.device),
            torch.full((len(windows),), depth, dtype=torch.int64, device=self.device),
            min_match=depth,
        )
        # Only compact interval boundaries return to the host alignment/union planner.
        return intervals.cpu().tolist()

    def batch(
        self,
        requests,
        *,
        min_match=1,
        lookback=0,
        local=True,
        corpus=True,
        allow_skip=True,
        locator="cpu",
    ):
        return TokenBatch(
            self, requests, min_match, lookback, local, corpus, allow_skip, locator
        )

    def check(self, m):
        if (m.tokenizer, m.vocab, len(m.omega), m.leaf, m.embed.weight.device) != (
            self.tokenizer,
            self.vocab,
            self.rank,
            self.layout.leaf,
            self.device,
        ):
            raise ValueError("host/integer packet mismatch")

    def rows(self, m):
        self.check(m)
        k, v, _ = self.docs.encode(m, need_rows=True, need_last=False)
        return self.layout, k, v, k.new_zeros(len(k))


class TokenBatch:
    def __init__(
        self, archive, requests, min_match, lookback, local, corpus, allow_skip, locator
    ):
        self.archive = archive
        self.corpus = corpus
        self.local = local
        self.skip = allow_skip
        prefix = []
        for name, p in requests:
            archive.index.check_live(name)
            seq = tokens(p)
            if not seq:
                raise ValueError("empty query prefix")
            prefix.append(seq)
        if not prefix:
            raise ValueError("empty batch")
        ranges = (
            archive.route(prefix, min_match, locator, lookback)
            if corpus
            else [() for _ in prefix]
        )
        self.layout = (
            archive.layout
            if corpus
            else Layout(archive.index.version, 0, archive.layout.leaf)
        )
        self.plan = Plan.build(self.layout, ranges)
        self.live = _Documents(
            prefix,
            np.arange(sum(len(p) - 1 for p in prefix), dtype=np.int64),
            archive.vocab,
            archive.device,
        )
        if (
            not allow_skip
            and not self.layout.rows
            and not (local and all(self.live.lengths))
        ):
            raise ValueError("no supported next-token law")
        self.work = bind(
            self.plan, archive.rank, self.live.lengths if local else (), archive.device
        )

    def predict(self, m, *, mode="indexed", backend="torch"):
        self.archive.check(m)
        lk, lv, x = self.live.encode(m, need_rows=self.local, need_last=True)
        local = (
            Local(lk, lv, x.new_zeros(len(lk)), self.live.lengths)
            if self.local
            else None
        )
        rows = (
            self.archive.rows(m)
            if self.corpus
            else (self.layout, x[:0], x.new_empty((0, m.width)), x.new_empty(0))
        )
        return m._read(
            x,
            local,
            rows,
            None,
            self.plan,
            self.corpus,
            self.skip,
            mode,
            backend,
            self.work,
        )
