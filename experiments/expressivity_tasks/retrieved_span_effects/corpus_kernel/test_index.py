import numpy as np
import pytest

from .index import EndpointIndex
from .plan import Layout, Plan


def test_all_frontier_mems_endpoints_and_extensions_vs_occurrences():
    rng = np.random.default_rng(2701)
    docs = (
        [[], [8], [0, 0, 4]]
        + [
            rng.integers(0, 7, size=int(n)).tolist()
            for n in rng.integers(12, 35, size=20)
        ]
        + [[3] * 76 + [i] for i in range(9)]
    )
    idx = EndpointIndex([(str(i), d) for i, d in enumerate(docs)], tokenizer="fixture")
    expect = sorted(
        ((i, j) for i, d in enumerate(docs) for j in range(len(d) - 1)),
        key=lambda p: docs[p[0]][: p[1] + 1][::-1],
    )
    assert list(zip(idx.row_doc, idx.row_offset, strict=True)) == expect
    for p in [[], [0], [6, 4, 0], [8], [7, 7], [3] * 76] + [d[-7:] for d in docs]:
        f = idx.frontier(p)
        matches = [0] * len(idx)
        for r, (i, j) in enumerate(expect):
            for x, y in zip(reversed(p), reversed(docs[i][: j + 1])):
                if x != y:
                    break
                matches[r] += 1
        spans = list(f.maximal_runs())
        actual = np.zeros(len(idx), int)
        for a, b, n in spans:
            assert not actual[a:b].any()
            actual[a:b] = n
        assert actual.tolist() == matches
        for n in (1, 2, 5, 48):
            rows = list(range(*f.hit_range(n)))
            assert rows == [r for r, v in enumerate(matches) if v >= n]
            path = idx.trail(f, n)
            branches, ends = idx.branch(path)
            assert not ends
            assert sum(map(lambda x: len(x.rows), branches.values())) == len(rows)
            for token, h in branches.items():
                assert all(
                    docs[expect[r][0]][expect[r][1] + 1] == token for r in h.rows
                )
                b, e = idx.branch(h)
                assert sorted([*e, *(r for v in b.values() for r in v.rows)]) == sorted(
                    h.rows
                )
                assert all(expect[r][1] + 2 == len(docs[expect[r][0]]) for r in e)
    f = idx.frontier([3] * 76)
    assert len(f.intervals) == 76 and len(idx.trail(f, 76).rows) == 9
    other = EndpointIndex([("Q", [1, 2])], tokenizer="fixture")
    with pytest.raises(ValueError):
        other.branch(idx.trail(f))
    with pytest.raises(ValueError):
        idx.check_live("1")
    assert len(EndpointIndex([], tokenizer="fixture")) == 0


def test_ragged_plan_is_a_positive_disjoint_cover_with_no_hit_limit():
    rng = np.random.default_rng(2702)
    for n in (0, 1, 7, 8, 9, 63, 100, 311):
        for leaf in (1, 8, 32):
            layout = Layout("invented", n, leaf)
            spans = [
                [],
                [(0, n)],
                [(0, n // 2), (0, n // 3), (n // 2 + 1, n)] if n > 2 else [],
            ] + [
                [tuple(sorted(rng.integers(0, n + 1, size=2))) for _ in range(4)]
                for _ in range(9)
            ]
            p = Plan.build(layout, spans)
            coverage = np.zeros((p.batch, n), int)
            exact = coverage.copy()
            ref = coverage.copy()
            for g, rs in enumerate(spans):
                for a, b in rs:
                    ref[g, a:b] = 1
            for g, a, b in p.exact:
                coverage[g, a:b] += 1
                exact[g, a:b] += 1
            for g, a, b in p.edge:
                coverage[g, a:b] += 1
            nodes = [
                (j * (1 << lev) * leaf, min(n, (j + 1) * (1 << lev) * leaf))
                for lev, count in enumerate(layout.widths)
                for j in range(count)
            ]
            for g, i in p.nodes:
                a, b = nodes[i]
                coverage[g, a:b] += 1
            np.testing.assert_array_equal(exact, ref)
            assert (coverage == 1).all()
    with pytest.raises(ValueError):
        Plan.build(Layout("a", 9), [[(0, 10)]])
