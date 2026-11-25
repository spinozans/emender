"""READ(source, string): exact byte-valued reduction for the materialized suffix kernel.

The native transport owns suffix traversal and immutable histograms. A parent returns
child *identifiers*, not child answers. `follow` fetches only selected suffix pages.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

DEPTH = 8
ROW = np.dtype([("h", "<u8", (257,)), ("child", "<u4", (256,))])
WEIGHTS = 32.0 ** np.arange(DEPTH + 1)


@dataclass(frozen=True)
class Answer:
    mount: object
    # Logical native suffix path in this pinned immutable source, shortest first.
    ids: np.ndarray
    pdf: np.ndarray

    def at(self, t):
        return Answer(self.mount, self.ids[:, t], self.pdf[:, t])


class Kernel:
    def __init__(self, pages, *, scope="whole", cache=8192):
        if scope != "whole":
            raise PermissionError(
                "this transport cannot filter documents before counting"
            )
        if not 1 <= pages.root <= pages.nodes or cache < 1:
            raise ValueError("invalid page mount")
        self.pages = pages
        self.cache_limit = cache
        self.lru = OrderedDict()
        self.tag = object()
        self.counts = dict(
            parent_reads=0, child_reads=0, page_rows=0, page_batches=0, prefix_frames=0
        )

    def _rows(self, ids):
        # Lifetime of the returned compact table is independent of subsequent eviction.
        unique, inv = np.unique(ids, return_inverse=True)
        records = np.zeros(len(unique), dtype=ROW)
        missing = []
        for pos, u in enumerate(unique.tolist()):
            if u in self.lru:
                records[pos] = self.lru[u]
                self.lru.move_to_end(u)
            elif u:  # zero is absence of materialization, never an actual page
                missing.append(pos)
        if missing:
            need = unique[missing]
            fresh = self.pages.rows(need)
            if (
                fresh.shape != (len(need),)
                or fresh.dtype != ROW
                or np.any(fresh["child"] > self.pages.nodes)
                or np.any(fresh["child"] == self.pages.root)
                or np.any((fresh["child"] != 0) & (fresh["h"][:, :256] == 0))
            ):
                raise ValueError("malformed branch table")
            records[missing] = fresh
            for u, rec in zip(need.tolist(), fresh, strict=True):
                self.lru[u] = rec.copy()
                if len(self.lru) > self.cache_limit:
                    self.lru.popitem(last=False)
            self.counts["page_rows"] += len(need)
            self.counts["page_batches"] += (len(need) + 4095) // 4096
        return records, inv.reshape(ids.shape)

    def _answer(self, ids):
        if (
            ids.dtype != np.dtype("<u4")
            or ids.shape[-1] != DEPTH
            or ids.max() > self.pages.nodes
            or np.any(ids == self.pages.root)
        ):
            raise ValueError("bad suffix resolver frame")
        roots = np.full((*ids.shape[:-1], 1), self.pages.root, dtype="<u4")
        records, index = self._rows(np.concatenate((roots, ids), axis=-1))
        # Do not materialize a [batch,time,depth,257] FP64 archive tensor.
        hist = records["h"][:, :256].astype(np.float64)
        mass = hist[index[..., 0]]
        for depth, weight in enumerate(WEIGHTS[1:], 1):
            mass += weight * hist[index[..., depth]]
        total = mass.sum(axis=-1, keepdims=True)
        if not np.all(np.isfinite(total) & (total > 0)):
            raise ValueError("source has no byte mass or invalid counts")
        return Answer(self.tag, ids, (mass / total).astype(np.float32))

    def _prefix(self, raw):
        self.counts["prefix_frames"] += 1
        ids = self.pages.suffix(raw)
        if ids.dtype != np.dtype("<u4") or ids.shape != (len(raw), DEPTH):
            raise ValueError("bad prefix frame shape or dtype")
        return ids

    def read(self, keys):
        """Independent literal READs (root for an empty key); no concatenating strings."""
        if not keys or any(type(k) is not bytes for k in keys):
            raise ValueError("READ requires literal byte strings")
        ids = np.stack(
            [
                self._prefix(k[-DEPTH:])[-1] if k else np.zeros(DEPTH, "<u4")
                for k in keys
            ]
        )
        self.counts["parent_reads"] += len(keys)
        return self._answer(ids)

    def scan(self, inputs):
        """Batch independent first READs only. Row t is identical to READ(inputs[:t+1])."""
        if inputs.dtype != np.uint8 or inputs.ndim != 2 or not inputs.size:
            raise ValueError("expected nonempty [document, time] observed bytes")
        ids = np.stack([self._prefix(row.tobytes()) for row in inputs])
        self.counts["parent_reads"] += inputs.size
        return self._answer(ids)

    def follow(self, parent, byte):
        """READ(parent.string + chosen byte). No other extension's histogram is read."""
        if (
            parent.mount is not self.tag
            or byte.dtype != np.uint8
            or byte.shape != parent.ids.shape[:-1]
            or parent.ids.shape[-1] != DEPTH
        ):
            raise ValueError("foreign parent or misaligned chosen bytes")
        roots = np.full((*byte.shape, 1), self.pages.root, dtype="<u4")
        records, index = self._rows(
            np.concatenate((roots, parent.ids[..., :-1]), axis=-1)
        )
        child = records["child"][index, byte[..., None]]
        self.counts["child_reads"] += byte.size
        return self._answer(child)
