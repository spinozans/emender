"""Window-local packed literal evidence. No model tensors, targets, serialized tape, or caller-supplied addresses."""

from __future__ import annotations

from array import array
from dataclasses import dataclass

import numpy as np
import torch

from .literal import Match, literal_rows, rotated_payload_index


@dataclass(frozen=True, slots=True)
class Packed:
    offsets: np.ndarray  # CSR: 256-way counts with no entry for an absent follower
    codes: np.ndarray
    multiplicities: np.ndarray
    latest: np.ndarray  # int16 with -1 = no address (NOT a byte symbol)
    depth: np.ndarray

    @property
    def nbytes(self):
        return sum(
            a.nbytes
            for a in (
                self.offsets,
                self.codes,
                self.multiplicities,
                self.latest,
                self.depth,
            )
        )


def pack(word, horizon):
    # At most a 4096-byte input, all follower counts/addresses fit in 16 bits. Use only
    # the current live-row accumulator, never a T x T incidence matrix or a per-step XSA clone.
    ends = array("I", [0])
    codes = array("B")
    mult = array("H")
    js = array("h")
    ds = array("B")
    for r in literal_rows(word, horizon):
        js.append(-1 if r.latest is None else r.latest)
        ds.append(r.depth)
        for c, n in r.counts:
            codes.append(c)
            mult.append(n)
        ends.append(len(codes))
    return Packed(*(np.asarray(x) for x in (ends, codes, mult, js, ds)))


def evidence(word, horizon, *, rotate, device, inspect_rotation=False):
    """Materialize only at the direct-head seam. Correct and sham eval can pay for the same donor map."""
    p = pack(word, horizon)
    n = len(word)
    at = np.flatnonzero(p.latest >= 0)
    js = p.latest[at].astype(np.int64)
    age = at - js
    ids = np.frombuffer(word, dtype=np.uint8)
    # DirectTrust takes dense FP32 distributions, so this layout reduces address storage,
    # not the size of DirectTrust's mathematical output or its dense activation tensors.
    probs = np.zeros((n, 3, 256), np.float32)
    stats = np.zeros((n, 11), np.float32)
    ok = np.zeros((n, 3), bool)
    if len(at):
        byrow = np.repeat(np.arange(n), np.diff(p.offsets).astype(np.intp))
        start = p.offsets[at]
        counts = p.multiplicities
        total = np.add.reduceat(counts, start, dtype=np.int64)
        high = np.maximum.reduceat(counts, start)
        probs[byrow, 1, p.codes] = counts / np.repeat(
            total, np.diff(p.offsets)[at].astype(np.intp)
        )
        probs[at, 2, ids[js]] = 1
        ok[at, 1:] = True
        stats[at, 7] = p.depth[at] / 8
        stats[at, 8] = np.log1p(total) / 8
        stats[at, 9] = high / total
        stats[at, 10] = np.log1p(age) / 8
    alternate = js
    if rotate or inspect_rotation:
        alternate = np.fromiter(
            (
                rotated_payload_index(
                    word, int(t), Match(int(p.depth[t]), int(j), ()), horizon
                )
                for t, j in zip(at, js, strict=True)
            ),
            dtype=np.int64,
            count=len(at),
        )
    trips = dict(
        literal_queries=n,
        literal_hits=len(at),
        literal_misses=n - len(at),
        literal_age_gt128=int(np.count_nonzero(age > 128)),
        literal_age_gt1024=int(np.count_nonzero(age > 1024)),
        literal_hist_entries=len(p.codes),
        literal_packed_bytes=p.nbytes,
        literal_payloads_reassigned=int(np.count_nonzero(js != alternate))
        if rotate
        else 0,
    )
    arrays = (stats, probs, ok, at, js, alternate if rotate else js)
    f, d, v, q, s, k = (torch.from_numpy(a).to(device) for a in arrays)
    return (f, d, v), q, s, k, trips
