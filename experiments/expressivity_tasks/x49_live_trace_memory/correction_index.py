"""PC: query older mistaken (forecast, observed follower) pairs at the live selected suffix.

No target or external geometry. Correct and switched tables are separate calls. Neither
looks up an earlier host row: the single byte forecast already existed before the arrival.
"""

from array import array
from dataclasses import dataclass
import numpy as np
from .literal import DEPTHS, MIN_AGE, check_window


@dataclass(slots=True)
class _Key:
    pending: int
    size: int = 0
    # Allocate only after a mistake becomes OLDER than the pending latest occurrence.
    rows: dict | None = None


@dataclass(slots=True)
class _Switched(_Key):
    loose: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class Counts:
    latest: np.ndarray
    depth: np.ndarray
    size: np.ndarray
    position: np.ndarray  # sorted unique 256*t+b for the row of q_t only
    count: np.ndarray
    work: dict


def _inc(s, a, c):
    if s.rows is None:
        s.rows = {}
    row = s.rows.get(a)
    if row is None:
        s.rows[a] = {c: 1}
        return 1, 1
    old = row.get(c, 0)
    row[c] = old + 1
    return 0, int(old == 0)


def _real(s, a, c):
    nr, nc = _inc(s, a, c)
    return nr, nc, 0, 0, 0


def _switch(s, a, c):
    # Before the second mistake commits the first is observable at its true edge.
    if s.loose is None:
        s.loose = (a, c)
        nr, nc = _inc(s, a, c)
        return nr, nc, 0, 0, 1
    d, e = s.loose
    s.loose = None
    if a == d or c == e or a == e or d == c:
        nr, nc = _inc(s, a, c)
        return nr, nc, 1, 0, -1
    # Remove just this provisional d->e. The d row survives in the completed table.
    row = s.rows[d]
    left = row[e] - 1
    if left:
        row[e] = left
    else:
        del row[e]
    u, v = _inc(s, d, c)
    nr, nc = _inc(s, a, e)
    return u + nr, v + nc - int(not left), 1, 1, -1


def _scan(word, q, key_type, commit):
    check_window(word, 4096)
    if type(q) is not bytes or len(q) != len(word):
        raise ValueError("one aligned query-forecast byte per observed frontier")
    n = len(q)
    by = {}
    latest = np.full(n, -1, dtype=np.int16)
    depth = np.zeros(n, dtype=np.uint8)
    size = np.zeros(n, dtype=np.uint16)
    pos = array("I")
    vals = array("H")
    admissions = commits = mistakes = trials = legal = 0
    rows = cells = peak_cells = loose = peak_loose = 0
    active = uses = 0
    for t, query in enumerate(q):
        j = t - MIN_AGE
        for d in DEPTHS:
            if j < d:
                continue
            key = word[j - d : j]
            admissions += 1
            s = by.get(key)
            if s is None:
                by[key] = key_type(j)
                continue
            # j just became eligible. File only the previously pending record.
            k = s.pending
            c, a = word[k], q[k - 1]
            s.pending = j
            s.size += 1
            commits += 1
            if a != c:
                mistakes += 1
                dr, dc, attempted, exchanged, dl = commit(s, a, c)
                rows += dr
                cells += dc
                trials += attempted
                legal += exchanged
                loose += dl
                peak_cells = max(peak_cells, cells)
                peak_loose = max(peak_loose, loose)
        for d in DEPTHS:
            if t + 1 < d:
                continue
            s = by.get(word[t + 1 - d : t + 1])
            if s is None:
                continue
            latest[t], depth[t], size[t] = s.pending, d, s.size
            row = s.rows.get(query) if s.rows else None
            if row:
                active += 1
                for b in sorted(row):
                    pos.append(256 * t + b)
                    vals.append(row[b])
                    uses += row[b]
            break  # no shallower fallback, even if the current guess has no corrections
    at = np.asarray(pos, dtype=np.uint32)
    k = np.asarray(vals, dtype=np.uint16)
    if (
        admissions != commits + len(by)
        or (len(k) and (not k.min() or not np.all(at[1:] > at[:-1])))
        or int(k.sum()) != uses
    ):
        raise ValueError("lost or duplicate queried correction")
    work = dict(
        key_admissions=admissions,
        committed_records=commits,
        mistake_commits=mistakes,
        peak_keys=len(by),
        peak_guess_rows=rows,
        peak_pair_cells=peak_cells,
        final_pair_cells=cells,
        active_rows=active,
        matched_mistake_uses=uses,
        packed_cells=len(k),
        packed_bytes=sum(a.nbytes for a in (latest, depth, size, at, k)),
    )
    if key_type is _Switched:
        if trials * 2 + loose != mistakes or legal > trials:
            raise ValueError("incorrect switched-prefix pairing")
        work.update(
            attempted_exchanges=trials,
            legal_exchanges=legal,
            peak_loose_keys=peak_loose,
        )
    return Counts(latest, depth, size, at, k, work)


def true_pairs(word, q):
    return _scan(word, q, _Key, _real)


def switched_pairs(word, q):
    return _scan(word, q, _Switched, _switch)
