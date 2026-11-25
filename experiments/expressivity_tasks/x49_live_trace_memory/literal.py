"""Stage 0: causal literal-address plan for ONE window. No archive, targets or neural state.

Each row is the prediction frontier *after* its input byte. Position integers are local to
this invocation; do not persist them or use them to index a different document's activations.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

DEPTHS = (8, 4, 2, 1)
HORIZONS = (128, 1024, 4096)
MIN_AGE = 16


@dataclass(frozen=True, slots=True)
class Match:
    depth: int
    latest: int | None  # index of a previously observed follower, NOT the key start
    counts: tuple[
        tuple[int, int], ...
    ]  # sorted (byte, multiplicity); no END bin needed for an open window


MISS = Match(0, None, ())


@dataclass(slots=True)
class _Bucket:
    js: deque[int] = field(default_factory=deque)
    counts: Counter[int] = field(default_factory=Counter)


def check_window(observed: bytes, horizon: int) -> None:
    if type(observed) is not bytes or not 1 <= len(observed) <= 4096:
        raise ValueError("one nonempty, at most 4096-byte input window required")
    if type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError("unregistered history horizon")


def literal_rows(observed: bytes, horizon: int):
    """Exact overlap counts. Insert j only when its follower is at least MIN_AGE old.

    A lookup evicts occurrences whose *key start* passed the window's left boundary.
    The whole temporary index is bounded by this window and discarded on return.
    This is a batched causal training implementation, not an XSA service benchmark.
    """
    check_window(observed, horizon)
    by_key: dict[bytes, _Bucket] = {}
    for t in range(len(observed)):
        j = t - MIN_AGE
        lo = max(0, t + 1 - horizon)
        for d in DEPTHS:
            if j - d >= lo:
                k = observed[j - d : j]
                if k not in by_key:
                    by_key[k] = _Bucket()
                b = by_key[k]
                b.js.append(j)
                b.counts[observed[j]] += 1
        for d in DEPTHS:
            if t + 1 - d < lo:
                continue
            b = by_key.get(observed[t + 1 - d : t + 1])
            if b is None:
                continue
            while b.js and b.js[0] - d < lo:
                x = observed[b.js.popleft()]
                b.counts[x] -= 1
                if not b.counts[x]:
                    del b.counts[x]
            if b.js:
                yield Match(d, b.js[-1], tuple(sorted(b.counts.items())))
                break
        else:
            yield MISS


def literal_plan(observed: bytes, horizon: int) -> tuple[Match, ...]:
    """Inspectable reference layout used by the prefix and native checks."""
    return tuple(literal_rows(observed, horizon))


def rotated_payload_index(observed: bytes, t: int, hit: Match, horizon: int) -> int:
    """Reference sham: change only the payload address, not the byte, age bin or direct evidence.

    Eligible donors were writable within the same window, at sufficient depth and age.
    They need not share the literal key: the point is to break that association. No future
    contents, other windows, or RNG. This CPU diagnostic is not a throughput implementation.
    """
    check_window(observed, horizon)
    j = hit.latest
    if j is None or not 0 <= t < len(observed) or hit.depth not in DEPTHS:
        raise ValueError("rotation needs an eligible hit")
    lo = max(0, t + 1 - horizon) + hit.depth
    if (
        not lo <= j <= t - MIN_AGE
        or observed[j - hit.depth : j] != observed[t + 1 - hit.depth : t + 1]
    ):
        raise ValueError("ineligible literal payload index")
    floor_age = 1 << ((t - j).bit_length() - 1)
    first = max(lo, t - (2 * floor_age - 1))
    last = t - max(MIN_AGE, floor_age)
    byte = observed[j]
    after = observed.find(bytes([byte]), j + 1, last + 1)
    if after >= 0:
        return after
    return observed.index(bytes([byte]), first, j + 1)  # j itself is always eligible
