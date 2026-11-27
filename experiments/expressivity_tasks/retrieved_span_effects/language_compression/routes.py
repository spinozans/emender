"""Immutable byte-suffix routes. The three-byte interval only nominates exact scores.

The rest of the archive is covered positively by edge rows and entire feature nodes.
No learned table is prepared here. The specialized planner requires a complete tree.
"""

import numpy as np

from ..corpus_kernel.index import EndpointIndex
from ..corpus_kernel.attention import TILE

TOKENIZER = "raw-byte-identity-256-v1"


def codes(x, d):
    n = x.shape[1] - d + 1
    z = np.zeros((len(x), max(0, n)), np.int64)
    if n > 0:
        for j in range(d):
            z = z * 256 + x[:, j : j + n]
    return z


class Reference:
    def __init__(self, archive):
        if (
            archive.ndim != 2
            or archive.dtype != np.uint8
            or archive.shape[1] < 5
            or not len(archive)
        ):
            raise ValueError("need rectangular real byte documents")
        self.index = EndpointIndex(
            [(f"build:{i}", p.tolist()) for i, p in enumerate(archive)],
            tokenizer=TOKENIZER,
        )
        x, y = archive[:, :-1], archive[:, 1:]
        rank = self.index.inverse_order.reshape(x.shape)
        root = np.bincount(y.flatten(), minlength=256) + 0.5
        root /= root.sum()
        tables = [root[None]]
        start = 1
        self.levels = []
        for d in range(1, 5):
            val, inv = np.unique(codes(x, d).ravel(), return_inverse=True)
            inv = inv.ravel()
            target = y[:, d - 1 :].ravel()
            hist = np.zeros((len(val), 256), np.int32)
            np.add.at(hist, (inv, target), 1)
            count = hist.sum(1)
            lo = hi = None
            if d == 3:
                r = rank[:, d - 1 :].ravel()
                lo = np.full(len(val), len(self.index), np.int64)
                mx = np.full(len(val), -1, np.int64)
                np.minimum.at(lo, inv, r)
                np.maximum.at(mx, inv, r)
                hi = mx + 1
                if not np.array_equal(hi - lo, count):
                    raise ValueError("suffix posting interval is not contiguous")
            self.levels.append((val, start, lo, hi))
            tables.append((hist + 8 * root) / (count[:, None] + 8))
            start += len(val)
        self.probs = np.concatenate(tables).astype(np.float32)

    def lookup(self, x):
        if x.ndim != 2 or x.dtype != np.uint8:
            raise ValueError("expected observed byte windows")
        counts = np.zeros(x.shape, np.int32)
        bounds = np.zeros((*x.shape, 2), np.int32)
        for d, (v, start, lo, hi) in enumerate(self.levels, 1):
            if x.shape[1] < d:
                break
            z = codes(x, d)
            ids = np.minimum(v.searchsorted(z), len(v) - 1)
            keep = v[ids] == z
            counts[:, d - 1 :] = np.where(keep, ids + start, counts[:, d - 1 :])
            if d == 3:
                bounds[:, d - 1 :, 0] = np.where(keep, lo[ids], 0)
                bounds[:, d - 1 :, 1] = np.where(keep, hi[ids], 0)
        return counts, bounds.reshape(-1, 2)


def tiles(q, lo, hi):
    n = (hi - lo + TILE - 1) // TILE
    owners = np.repeat(q, n)
    starts = np.repeat(lo, n) + TILE * (
        np.arange(len(owners)) - np.repeat(n.cumsum() - n, n)
    )
    return np.stack(
        (owners, starts, np.minimum(TILE, np.repeat(hi, n) - starts)), 1
    ).astype(np.int32)


def compact(bounds, rows, leaf, rank):
    if (
        leaf < 1
        or rank < 1
        or rows < 1
        or rows % leaf
        or (rows // leaf) & (rows // leaf - 1)
        or rows * rank >= 2**31
    ):
        raise ValueError("regular complete-tree planner geometry")
    a, b = np.asarray(bounds, dtype=np.int64).T
    q = np.arange(len(a))
    nl = rows // leaf
    if np.any((a < 0) | (a > b) | (b > rows)):
        raise ValueError("invalid suffix interval")
    hit = a < b
    a, b = np.where(hit, a, 0), np.where(hit, b, 0)
    exact = tiles(q, a, b)
    left = a // leaf
    right = (b + leaf - 1) // leaf
    edge = tiles(np.tile(q, 2), np.r_[left * leaf, b], np.r_[a, right * leaf])
    nq = []
    ids = []
    off = 0
    s = 1
    while s <= nl:
        for width, prefix in ((left, True), (nl - right, False)):
            take = width & s != 0
            block = (
                width[take] // (2 * s) * 2
                if prefix
                else (nl // s - width[take] // (2 * s) * 2 - 1)
            )
            nq.append(q[take])
            ids.append(off + block)
        off += nl // s
        s *= 2
    node = np.concatenate(ids) * rank
    nodes = tiles(np.concatenate(nq), node, node + rank)
    return exact, edge, nodes
