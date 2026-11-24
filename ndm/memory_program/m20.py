"""Lookup of arbitrary chosen keys on a pinned, already-admitted W3BP0001 page transport."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .runtime import Unavailable, branch_reader


class MaterializedBranches:
    """An immutable whole-bank capability, never a general document filter.

    `pages` supplies root, nodes and rows(ids) yielding records with `h[257]` and
    `child[256]`. The mounting application verifies and owns the transport/lifetime.
    The instance and its bounded cache belong to exactly this mount.
    """

    def __init__(self, source, version, pages, *, scope, cache_pages=256):
        if scope != "whole":
            raise PermissionError("partial m20 aggregates unavailable")
        if not source or not version or cache_pages < 0 or not 1 <= pages.root <= pages.nodes:
            raise ValueError("invalid mount")
        self.pages = pages
        self.source, self.version = source, version
        self.capacity = cache_pages
        self.cache = OrderedDict()
        self.pages_fetched = 0
        self.page_batches = 0

    def _load(self, ids):
        # Hold just the current traversal level plus the bounded LRU; no fetch of
        # an unchosen *child page* just because its ID was returned by its parent.
        found = {}
        for i in dict.fromkeys(ids):
            if i in self.cache:
                found[i] = self.cache[i]
                self.cache.move_to_end(i)
        missing = tuple(i for i in dict.fromkeys(ids) if i not in found)
        if missing:
            records = self.pages.rows(missing)
            self.page_batches += 1
            self.pages_fetched += len(missing)
            for i, record in zip(missing, records, strict=True):
                h = tuple(record["h"].tolist())
                children = tuple(record["child"].tolist())
                if (
                    len(h) != 257
                    or len(children) != 256
                    or any(c < 0 for c in h)
                    or any(not 0 <= ch <= self.pages.nodes or (ch and not c) for c, ch in zip(h, children))
                ):
                    raise ValueError("invalid m20 node")
                found[i] = h, children
                self.cache[i] = found[i]
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
        return found

    def histograms(self, keys):
        hits = [None] * len(keys)
        live = {i: self.pages.root for i in range(len(keys))}
        depth = 0
        while live:
            nodes = self._load(live.values())
            following = {}
            for i, node in live.items():
                h, children = nodes[node]
                if len(keys[i]) == depth:
                    hits[i] = h
                else:
                    b = keys[i][depth]
                    if children[b]:
                        following[i] = children[b]
                    elif h[b]:
                        hits[i] = Unavailable("unmaterialized branch")
                    else:
                        hits[i] = (0,) * 257  # certified absent by the parent's full histogram
            live = following
            depth += 1
        return hits

    def reader(self, *, structural_end=False):
        return branch_reader(self.source, self.version, self.histograms, structural_end=structural_end)


class TerminalBranches(MaterializedBranches):
    """Depth-8 W3BP0001 transport: resolve one chosen literal on its own reset S frame.

    Only the prefix diagonal matters. On a missing transition we MUST still inspect its
    materialized parent's count to discriminate absent from positive-but-unmaterialized.
    No cache of clients' raw strings; full rows use this one mount's existing bounded LRU.
    """

    def histograms(self, keys):
        keys = tuple(keys)
        if any(type(k) is not bytes for k in keys):
            raise ValueError("literal bytes required")
        plans = {}
        for key in dict.fromkeys(keys):
            node = self.pages.root
            depth = 0
            if key:
                literal = key[:8]
                ids = self.pages.suffix(literal)
                if (
                    not isinstance(ids, np.ndarray)
                    or ids.shape != (len(literal), 8)
                    or ids.dtype != np.dtype("<u4")
                    or int(ids.max()) > self.pages.nodes
                ):
                    raise ValueError("invalid m20 suffix resolver frame")
                for j, raw in enumerate(ids.diagonal()):
                    child = int(raw)
                    if child:
                        if j != depth or child == self.pages.root:
                            raise ValueError("inconsistent literal prefix path")
                        node = child
                        depth += 1
            plans[key] = node, depth
        rows = self._load([node for node, _ in plans.values()])
        answers = {}
        for key, (node, depth) in plans.items():
            h, children = rows[node]
            if depth == len(key):
                answers[key] = h
            else:
                b = key[depth]
                if children[b]:
                    raise ValueError("missing transition contradicts materialized parent")
                answers[key] = Unavailable("unmaterialized branch") if h[b] else (0,) * 257
        return [answers[k] for k in keys]
