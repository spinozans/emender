"""PT's key-blind association: one delayed prefix table, weighted by permitted local missed alternatives.

The local key supplies only ERROR's marginals and one pending/latest position. No pair
counts per key, old occurrence enumeration, future label, external archive or host callback.
"""

from math import fsum
import numpy as np
from .literal import MIN_AGE, check_window


def distribution(word, q, tab, law):
    check_window(word, 4096)
    n = len(word)
    if (
        type(q) is not bytes
        or len(q) != len(tab.latest)
        or len(q) != len(law.h)
        or n != len(q)
    ):
        raise ValueError(
            "one aligned prefix query tape and internal ERROR geometry required"
        )
    cols = [0] * 256
    by = {}
    stops = np.cumsum(np.bincount(law.row, minlength=n))
    r = np.zeros(len(law.row), np.float64)
    lo = 0
    admit = mistakes = cells = matched = 0
    reads = nonzero = discount = col_discount = edge_discount = 0
    # Never index a record simply because some future query will use it. Admit all aged arrivals once.
    for t, a in enumerate(q):
        j = t - MIN_AGE
        if j >= 1:
            admit += 1
            g = q[j - 1]
            c = word[j]
            if g != c:
                mistakes += 1
                cols[c] += 1
                row = by.get(g)
                if row is None:
                    row = by[g] = {}
                prior = row.get(c, 0)
                cells += int(prior == 0)
                row[c] = prior + 1
        hi = int(stops[t])
        h = int(law.h[t])
        if not h:
            if hi != lo:
                raise ValueError("unmatched prefix query has alternatives")
            continue
        matched += 1
        k = int(tab.latest[t])
        row = by.get(a)
        if row is None or not 1 <= k <= j or hi == lo:
            raise ValueError("prefix query has no eligible supporting mistake")
        old = q[k - 1]
        last = word[k]
        miss = old != last
        discount += int(miss)
        mass = 0
        for i in range(lo, hi):
            count = int(law.alt[i])
            if count == 0:
                continue  # includes q itself; never a correction to its own mistake
            b = int(law.symbol[i])
            sub = int(miss and last == b)
            edge = sub * int(old == a)
            den = cols[b] - sub
            num = row.get(b, 0) - edge
            col_discount += sub
            edge_discount += edge
            reads += 1
            if b == a or den < count or not 0 <= num <= den:
                raise ValueError("prefix failed to cover older local alternatives")
            r[i] = count * num / den
            mass += num
            nonzero += int(num > 0)
        total = fsum(r[lo:hi])
        if mass < h or not total > 0:
            raise ValueError("prefix lost local matched evidence after latest discount")
        r[lo:hi] /= total
        lo = hi
    if (
        lo != len(r)
        or matched != len(law.active)
        or sum(cols) != mistakes
        or cells != sum(len(v) for v in by.values())
    ):
        raise ValueError("incorrect key-blind pair work")
    return r, dict(
        admitted_records=admit,
        mistake_writes=mistakes,
        peak_pair_cells=cells,
        peak_guess_rows=len(by),
        peak_nonempty_actual_columns=sum(c > 0 for c in cols),
        active_rows=matched,
        alternative_bin_lookups=reads,
        nonzero_queried_bins=nonzero,
        latest_mistake_discounts=discount,
        discounted_column_reads=col_discount,
        discounted_edge_reads=edge_discount,
        rate_bytes=r.nbytes,
    )
