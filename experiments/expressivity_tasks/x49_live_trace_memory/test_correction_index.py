"""PC's independent prefix oracle: re-enumerate occurrences, exchange complete adjacent pairs, keep both margins."""

from collections import Counter
from itertools import product
import numpy as np
import pytest
from experiments.expressivity_tasks.x49_live_trace_memory import correction_index as ix


def word():
    # The same 8-byte prefix followed by several answers, ASCII and binary. No delimiter parsing.
    return b"".join(
        b"abcd\xff\0\x1ek" + bytes([c]) + b"~more;!"
        for c in (10, 32, 0, 32, 10, 32, 80, 81, 81, 65, 10, 32, 255, 80, 10, 32)
    )


def queries(n):
    # No next byte is an input. Each motif alternates what the model expected at its 'k'.
    return bytes(
        (80 + (t // 16) % 2) if t % 16 == 7 else ((t * 7 + 31) % 5 if t % 4 else 0)
        for t in range(n)
    )


def switched(edges):
    out = list(edges)
    for i in range(0, len(edges) - 1, 2):
        (a, b), (c, d) = edges[i : i + 2]
        if (
            len({a, b, c, d}) == 4
        ):  # two off-diagonal mistakes; real change and both crossed edges legal
            out[i : i + 2] = [(a, d), (c, b)]
    assert Counter(a for a, b in out) == Counter(a for a, b in edges)
    assert Counter(b for a, b in out) == Counter(b for a, b in edges)
    assert all(a != b for a, b in out)
    return out


def oracle(w, q):
    n = len(w)
    latest = []
    depth = []
    sizes = []
    dr = np.zeros((n, 256), dtype=np.int64)
    ds = dr.copy()
    cols = np.zeros((n, 256, 3), dtype=np.float64)
    for t, g in enumerate(q):
        at = []
        use = 0
        for d in (8, 4, 2, 1):
            if t + 1 >= d:
                at = [
                    j for j in range(d, t - 15) if w[j - d : j] == w[t + 1 - d : t + 1]
                ]
                if at:
                    use = d
                    break
        latest.append(at[-1] if at else -1)
        depth.append(use)
        sizes.append(max(0, len(at) - 1))
        edges = [(q[j - 1], w[j]) for j in at[:-1] if q[j - 1] != w[j]]
        for i, r in ((dr, edges), (ds, switched(edges))):
            for a, b in r:
                if a == g:
                    i[t, b] += 1
        h = sum(a == g for a, b in edges)
        if h:
            correct = sum(w[j] == q[j - 1] == g for j in at[:-1])
            A = h / (1 + h + correct)
            missed = Counter(b for a, b in edges if b != g)
            den = sum(missed.values())
            assert den >= h
            v = np.array([missed[b] / den for b in range(256)])
            cols[t, :, 0] = A * v
            cols[t, g, 0] -= A
            cols[t, :, 1] = A * (dr[t] / h - v)
            cols[t, :, 2] = A * (ds[t] / h - v)
    return latest, depth, sizes, dr, ds, cols


def dense(z):
    v = np.zeros((len(z.size), 256), np.int64)
    p = z.position
    assert np.all(p[1:] > p[:-1]) and np.all(p < 256 * len(v)) and np.all(z.count > 0)
    v[p >> 8, p & 255] = z.count
    return v


def test_all_prefixes_older_only_current_query_correct_and_margin_switched_edges():
    w = word()
    q = queries(len(w))
    latest, depth, size, dr, ds, cols = oracle(w, q)
    assert (dr != ds).any() and np.abs(cols).max() > 0 and np.any(dr.sum(1) == 0)
    for make, want in [(ix.true_pairs, dr), (ix.switched_pairs, ds)]:
        c = make(w, q)
        np.testing.assert_array_equal(dense(c), want)
        np.testing.assert_array_equal(c.size, size)
        np.testing.assert_array_equal(c.latest, latest)
        np.testing.assert_array_equal(c.depth, depth)
        assert (
            c.work["active_rows"] == np.sum(want.sum(1) > 0)
            and c.work["matched_mistake_uses"] == want.sum()
        )
        assert (
            c.work["key_admissions"]
            == c.work["committed_records"] + c.work["peak_keys"]
        )
        assert c.work["packed_bytes"] == 5 * len(w) + 6 * len(c.count)
        for t in (1, 17, 33, 49, 65, 97, 114, len(w) - 1):
            small = make(w[:t], q[:t])
            np.testing.assert_array_equal(dense(small), want[:t])
    c = ix.switched_pairs(w, q)
    assert (
        0
        < c.work["legal_exchanges"]
        <= c.work["attempted_exchanges"]
        < c.work["mistake_commits"]
    )


def test_every_two_edge_switch_with_repeated_counts_provisional_odd_edges_and_diagonal():
    edges = [(a, b) for a, b in product(range(4), repeat=2) if a != b]
    for first, second in product(edges, repeat=2):
        k = ix._Switched(1)
        seen = []
        rows = cells = odd = trials = legal = 0
        for edge in [first, second, first, second, second, first, first]:
            seen.append(edge)
            r, c, t, exchanged, o = ix._switch(k, *edge)
            rows += r
            cells += c
            odd += o
            trials += t
            legal += exchanged
            want = Counter(switched(seen))
            actual = {(a, b): n for a, bs in k.rows.items() for b, n in bs.items()}
            assert (
                actual == want
                and cells == len(actual)
                and rows == len(k.rows)
                and all(k.rows.values())
            )
            assert (
                k.loose == (seen[-1] if odd else None)
                and odd == len(seen) % 2
                and trials == len(seen) // 2
                and legal <= trials
            )
    assert not hasattr(ix._Key(3), "loose")


def test_full_overlap_multiplicity_open_end_singleton_no_fallback_and_no_query_match():
    w = bytes(4096)
    q = b"\xff" * len(w)
    for make in (ix.true_pairs, ix.switched_pairs):
        c = make(w, q)
        v = dense(c)
        assert (c.latest[18], c.depth[18], c.size[18], sum(v[18])) == (2, 2, 0, 0)
        assert (c.latest[-1], c.size[-1], v[-1, 0]) == (
            4079,
            4071,
            4071,
        ) and not np.any(v[:, 1:])
        assert not np.any(dense(make(w[:128], w[:128])))
        # The current forecast can change this lookup, not retroactively any old edge.
        q2 = q[:-1] + b"\x1e"
        z = dense(make(w, q2))
        np.testing.assert_array_equal(v[:-1], z[:-1])
        assert not z[-1].any()
        for bad, bq in [
            (b"", b""),
            (w, q[:-1]),
            (w + b"z", q + b"z"),
            (w, bytearray(q)),
        ]:
            with pytest.raises(ValueError):
                make(bad, bq)
