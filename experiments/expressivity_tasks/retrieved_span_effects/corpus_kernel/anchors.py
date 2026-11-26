"""Optional internal MEM proposals, aligned to the prediction frontier.

An internal MEM ending `lag` observations ago proposes the archive *boundary*
`lag` tokens after its end. Its own successor is not a vote for the current
prediction. Alignment by offset is a proposal heuristic, not a new score law.
"""

from operator import index as integer

import numpy as np

from .plan import union


def _aligned(index, lo, hi, lag, observed_after):
    if lag == 0:
        return [(lo, hi)]
    # The earlier alignment is right-maximal only on mismatch with the next
    # ALREADY OBSERVED token. All left-maximal extensions of length >= the
    # threshold are represented by the threshold interval; no longest-only cap.
    anchor = np.flatnonzero(index.successor[lo:hi] != observed_after) + lo
    target = index.order[anchor] + lag
    keep = target < len(index)
    anchor = anchor[keep]
    target = target[keep]
    if not len(target):
        return []
    candidate = index.inverse_order[target]
    # `order` counts predictive boundaries in document order. A shift is valid
    # only within the same document (also excludes its last token without a successor).
    ids = np.sort(candidate[index.row_doc[anchor] == index.row_doc[candidate]])
    if not len(ids):
        return []
    gap = np.diff(ids) != 1
    return list(
        zip(
            ids[np.r_[True, gap]].tolist(),
            (ids[np.r_[gap, True]] + 1).tolist(),
            strict=True,
        )
    )


def proposals(index, prefixes, *, min_match=1, lookback=0, seek=None):
    """Return an interval union per prefix. `None` searches every earlier observed endpoint.

    The optional seeker takes distinct threshold-length observed substrings
    and returns their endpoint intervals in the same mounted index.
    """
    depth = integer(min_match)
    back = None if lookback is None else integer(lookback)
    if depth < 1 or (back is not None and back < 0):
        raise ValueError("invalid MEM proposal scope")
    uses = []
    windows = {}
    ranges = [[] for _ in prefixes]
    for i, p in enumerate(prefixes):
        last = len(p) - depth
        for lag in range(1 + (last if back is None else min(back, last))):
            end = len(p) - lag
            w = tuple(p[end - depth : end])
            slot = windows.setdefault(w, len(windows))
            uses.append((i, lag, slot))
    ws = list(windows)
    if not ws:
        return ranges
    spans = (
        [index.frontier(w).hit_range(depth) for w in ws]
        if seek is None
        else seek(ws, depth)
    )
    if len(spans) != len(ws):
        raise ValueError("wrong number of MEM intervals")
    for i, lag, j in uses:
        lo, hi = spans[j]
        if not 0 <= lo <= hi <= len(index):
            raise ValueError("invalid MEM interval")
        if lo < hi:
            ranges[i].extend(
                _aligned(index, lo, hi, lag, prefixes[i][-lag] if lag else None)
            )
    return [union(rs, len(index)) for rs in ranges]
