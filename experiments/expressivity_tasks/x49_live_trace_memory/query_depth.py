"""X49-Q bounded left-context refinement of B's neural payload address. No target or alternative value inputs."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import torch
from .compact import evidence, pack
from .literal import DEPTHS, MIN_AGE
from .values import LiveValueHead

MAX_DEPTH = 32


@dataclass(frozen=True, slots=True)
class Hit:
    t: int
    depth: int
    latest: int
    extended: int
    selected: int
    latest_length: int
    longest: int
    matched: bool  # assessed only by M, true iff a distinct matched control exists


def _hits(word, horizon, rows, policy, work):
    if policy not in ("E", "M"):
        raise ValueError("only refinements need an extra address search")
    # Per-window postings keyed by (literal suffix, observed follower). A variable-length
    # bytes key is unambiguous: its last byte is the follower, all preceding bytes the key.
    postings = {}
    last_added = -1
    for t, d, j in rows:
        lo = max(0, t + 1 - horizon)
        for k in range(last_added + 1, t - MIN_AGE + 1):
            for dep in DEPTHS:
                if k >= dep:
                    postings.setdefault(word[k - dep : k + 1], []).append(k)
                    work["posting_inserts"] += 1
        last_added = t - MIN_AGE
        positions = postings[word[t + 1 - d : t + 1] + word[j : j + 1]]
        if positions[-1] != j:
            raise ValueError("inconsistent incumbent suffix/follower")

        def accepts(k, m):
            work["extension_tests"] += 1
            return k - m >= lo and word[k - m : k] == word[t + 1 - m : t + 1]

        def length(k, lower):
            upper = min(MAX_DEPTH, k - lo)
            while lower < upper:
                mid = (lower + upper + 1) // 2
                if accepts(k, mid):
                    lower = mid
                else:
                    upper = mid - 1
            return lower

        lj = length(j, d)
        best = lj
        e = j
        # Rightmost first: ties retain the more recent slot. Once k-lo<=best no earlier
        # slot can improve; if the current value is already maximal, visit no alternatives.
        for i in range(len(positions) - 2, -1, -1):
            k = positions[i]
            if best == MAX_DEPTH or k - lo <= best:
                break
            work["extension_slots_considered"] += 1
            if accepts(k, best + 1):
                best = length(k, best + 1)
                e = k
        s = e
        if policy == "M" and e != j:
            age_floor = 1 << ((t - e).bit_length() - 1)
            left = bisect_left(positions, max(lo + d, t - 2 * age_floor + 1))
            right = bisect_right(positions, min(j - 1, t - age_floor))
            ix = bisect_left(positions, e, left, right)
            low = ix - 1
            high = ix + 1
            # Visit nearest admissible positions only until a non-extended match is found.
            while low >= left or high < right:
                if high < right and (
                    low < left or positions[high] - e <= e - positions[low]
                ):
                    k = positions[high]
                    high += 1
                else:
                    k = positions[low]
                    low -= 1
                work["control_slots_considered"] += 1
                if (lj == d or accepts(k, lj)) and not accepts(k, lj + 1):
                    s = k
                    break
        yield Hit(t, d, j, e, s, lj, best, s != e)


def addresses(word, *, horizon=4096, policy):
    """Inspectable, no injected addresses; recomputes the base literal search in this window."""
    base = pack(word, horizon)
    rows = (
        (t, int(base.depth[t]), int(j)) for t, j in enumerate(base.latest) if j >= 0
    )
    return tuple(_hits(word, horizon, rows, policy, Counter()))


def score(
    head,
    word,
    hidden,
    logits,
    *,
    policy,
    horizon=4096,
    work=None,
    diagnostics=None,
    timing=None,
):
    """Score all 256 next bytes; diagnostics are output only. No caller-supplied payload slots.

    J calls the unmodified B. The same observed follower, direct evidence and head are
    used by E/M; their one neural read is gathered *after* their byte-only search.
    """
    if not isinstance(head, LiveValueHead) or policy not in ("J", "E", "M"):
        raise ValueError("Q requires a B reader and a registered address policy")
    if policy == "J":
        if timing is not None:
            timing["extra_address_s"] = 0.0
        return head(word, hidden, logits, mode="B", horizon=horizon, work=work)
    if hidden.shape != (len(word), head.direct.host_dim) or logits.shape != (
        len(word),
        256,
    ):
        raise ValueError("unaligned Q inputs")
    dev = hidden.device
    f, q, src, _, counts = evidence(word, horizon, device=dev, rotate=False)
    start = perf_counter() if timing is not None else None
    new = Counter()
    rows = zip(q.tolist(), (f[0][q, 7] * 8).long().tolist(), src.tolist(), strict=True)
    hits = list(_hits(word, horizon, rows, policy, new))
    slots = torch.tensor([r.selected for r in hits], device=dev, dtype=torch.long)
    if timing is not None:
        timing["extra_address_s"] = perf_counter() - start
    base, _ = head.direct(hidden, logits, f)
    ids = torch.tensor(list(word), dtype=torch.long, device=dev)
    written = head.writer(torch.cat((hidden, head.byte(ids)), dim=-1)).tanh()
    if diagnostics is not None:
        diagnostics["hits"] = hits
    if work is not None:
        work.update(
            counts,
            **new,
            literal_value_writes=len(word),
            literal_value_reads=len(hits),
            extended_moves=sum(r.latest != r.extended for r in hits),
            matched_controls=sum(r.matched for r in hits),
        )
    if not hits:
        return base
    return head.answer(base, hidden, f, q, head.byte(ids[src]), written[slots])
