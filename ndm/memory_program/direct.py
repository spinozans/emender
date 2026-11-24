"""Direct retrieval comparator. This is a different predictor, not an implicit read by MemoryPolicy.

An owner supplies one admitted immutable branch capability. All local follower counts belong to one
bounded observation stream. The trainable head sees the host and direct proposals, never a label.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log, log1p

import torch
from torch import nn
from .runtime import Refused, Unavailable

ARCHIVE = (8, 6, 4, 2, 1, 0)
LOCAL = (8, 4, 2, 1, 0)
BYTE_BINS = 256


def suffixes(tail, depths):
    return tuple(tail[-n:] if n else b"" for n in depths if n <= len(tail))


@dataclass(slots=True)
class _Followers:
    counts: dict = field(default_factory=dict)
    last: int = 0
    at: int = 0


@dataclass(frozen=True, slots=True)
class DirectView:
    document: int
    start: int
    visible: int
    tail: bytes
    local_key: bytes
    counts: tuple[int, ...]
    latest: int | None
    age: int | None
    owner: object = field(repr=False)  # per-stream identity, never a feature or cross-client address


class DocumentCounts:
    """Incremental observed bigram/n-gram counts. No addresses into another stream or the future."""

    def __init__(self, document, start, *, byte_limit):
        if any(type(n) is not int or n < 0 for n in (document, start, byte_limit)) or byte_limit < 1:
            raise ValueError("invalid document origin or byte bound")
        self.document = document
        self.start = start
        self.byte_limit = byte_limit
        self._owner = object()
        self.visible = 0
        self.tail = b""
        self.rows = {}
        self.edges = 0
        self.increments = 0
        self.closed = False

    def _frontier(self, at):
        if self.closed or type(at) is not int or at != self.visible:
            raise ValueError("closed or stale document frontier")

    def observe(self, byte, *, at):
        self._frontier(at)
        if type(byte) is not int or not 0 <= byte < BYTE_BINS:
            raise ValueError("one observed raw byte required")
        if at >= self.byte_limit:
            raise Refused("direct document byte budget")
        # This is the known follower of each suffix of the *previous* prefix.
        for key in suffixes(self.tail, LOCAL):
            row = self.rows.get(key)
            if row is None:
                row = self.rows[key] = _Followers()
            self.edges += int(byte not in row.counts)
            row.counts[byte] = row.counts.get(byte, 0) + 1
            row.last = byte
            row.at = at + 1
            self.increments += 1
        self.tail = (self.tail + bytes([byte]))[-8:]
        self.visible += 1

    def view(self, *, at):
        self._frontier(at)
        if not at:
            raise ValueError("no observed frontier")
        for key in suffixes(self.tail, LOCAL):
            row = self.rows.get(key)
            if row is not None:
                return DirectView(
                    self.document,
                    self.start,
                    at,
                    self.tail,
                    key,
                    tuple(row.counts.get(i, 0) for i in range(BYTE_BINS)),
                    row.last if key else None,
                    at - row.at if key else None,
                    self._owner,
                )
        raise RuntimeError("missing unigram after observation")

    def close(self):
        self.rows.clear()
        self.tail = b""
        self.closed = True
        self.edges = 0


@dataclass(frozen=True, slots=True)
class Branch:
    source: str
    version: str
    probes: tuple  # (literal key, status, unavailable reason if any, END mass)
    key: bytes | None
    histogram: tuple[int, ...] | None


def direct_branches(bank, views, *, grouped, work=None):
    """Resolve only required probes, on ONE owner-verified, immutable whole-archive capability.

    Column 256 is structural end and not a predicted byte. Neither an unavailable probe nor
    an end-only row supplies negative/byte evidence; preserve the status and explicitly back off.
    The serial path is a scalar scheduling reference for precisely the same retrieval rule.
    """
    if not views or len(views) > 256 or len({v.owner for v in views}) != len(views):
        raise ValueError("one to 256 distinct private streams")
    work = Counter() if work is None else work
    source, version = bank.source, bank.version
    candidates = [suffixes(v.tail, ARCHIVE) for v in views]
    traces = [[] for _ in views]
    out = [None] * len(views)

    def resolve(active):
        while active:
            chosen = [candidates[i][len(traces[i])] for i in active]
            keys = tuple(dict.fromkeys(chosen))
            work["hist_calls"] += 1
            work["submitted_keys"] += len(keys)
            work["logical_probes"] += len(chosen)
            ans = dict(zip(keys, bank.histograms(keys), strict=True))
            again = []
            for i, k in zip(active, chosen, strict=True):
                raw = ans[k]
                why = None
                end = None
                hist = None
                if isinstance(raw, Unavailable):
                    status = "unavailable"
                    why = raw.reason
                else:
                    if len(raw) != 257 or any(type(n) is not int or not 0 <= n < 2**63 for n in raw):
                        raise ValueError("invalid materialized direct histogram")
                    end = raw[256]
                    status = "selected" if any(raw[:256]) else ("end_only" if end else "absent")
                    if status == "selected":
                        hist = tuple(raw)
                traces[i].append((k, status, why, end))
                work[status] += 1
                if hist is not None or len(traces[i]) == len(candidates[i]):
                    out[i] = Branch(source, version, tuple(traces[i]), k if hist is not None else None, hist)
                else:
                    again.append(i)
            active = again

    if grouped:
        resolve(list(range(len(views))))
    else:
        for i in range(len(views)):
            resolve([i])
    if (bank.source, bank.version) != (source, version):
        raise ValueError("changed archive binding")
    return out


def direct_features(views, branches, *, device="cpu"):
    """Return sufficient statistics for trust and three direct *byte* distributions; no host/target needed."""
    stats = []
    proposals = []
    masks = []
    for v, a in zip(views, branches, strict=True):
        h = (0,) * 257 if a.histogram is None else a.histogram
        arch = h[:256]
        u = sum(arch)
        local = v.counts
        n = sum(local)
        statuses = [p[1] for p in a.probes]
        stats.append(
            (
                len(a.key or b"") / 8,
                log1p(u) / 24,
                max(arch) / max(u, 1),
                h[256] / max(u + h[256], 1),
                statuses.count("unavailable") / 6,
                statuses.count("absent") / 6,
                statuses.count("end_only") / 6,
                len(v.local_key) / 8,
                log1p(n) / 8,
                max(local) / max(n, 1),
                log1p(v.age or 0) / 8,
            )
        )
        proposals.append(
            (
                [k / max(u, 1) for k in arch],
                [k / max(n, 1) for k in local],
                [int(j == v.latest) for j in range(256)],
            )
        )
        masks.append((bool(u), bool(n), v.latest is not None))
    return (
        torch.tensor(stats, device=device, dtype=torch.float32),
        torch.tensor(proposals, device=device, dtype=torch.float32),
        torch.tensor(masks, device=device, dtype=torch.bool),
    )


class DirectTrust(nn.Module):
    """Trainable trust over host, archive, local histogram, latest follower. No program/branch sampling."""

    def __init__(self, host_dim, width=64):
        super().__init__()
        self.host_dim = host_dim
        self.hidden = nn.Linear(host_dim + 11 + 3, width)
        self.vote = nn.Linear(width, 4)
        with torch.no_grad():
            self.vote.weight.mul_(0.1)
            self.vote.bias.copy_(self.vote.bias.new_tensor([0.0, 0.0, 0.0, -log(4.0)]))

    def forward(self, hidden, base, features):
        stats, p, ok = features
        b = len(hidden)
        if (
            hidden.ndim != 2
            or stats.shape != (b, 11)
            or p.shape != (b, 3, 256)
            or ok.shape != (b, 3)
            or ok.dtype != torch.bool
            or base.shape != (b, 256)
        ):
            raise ValueError("invalid direct mixture shape")
        host = base.float().log_softmax(-1)
        agreement = (host.exp()[:, None, :] * p).sum(-1)
        score = self.vote(self.hidden(torch.cat((hidden, stats, agreement), -1)).tanh())
        valid = torch.cat((ok.new_ones((b, 1)), ok), 1)
        w = score.masked_fill(~valid, -torch.inf).log_softmax(-1)
        # Every byte has host support. Exact zeros in an empirical expert stay zero.
        log_emp = p.clamp_min(torch.finfo(p.dtype).tiny).log().masked_fill(p == 0, -torch.inf)
        return (w[:, :, None] + torch.cat((host[:, None, :], log_emp), 1)).logsumexp(1), w.exp()
