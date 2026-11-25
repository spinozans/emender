"""PT's first gate: invented streams, an enumerated prefix law, frozen answers and isolated read clocks."""

from collections import Counter
from copy import deepcopy
from fractions import Fraction
from itertools import permutations
from math import isclose
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    correction_transfer_score as pt,
    correction_transfer_reader as reader,
    paired_correction as pc,
    test_paired_correction as orig,
)

binary = orig.binary
BETA = dict(U=orig.UB, J=orig.JB, HISTORY=orig.HB, ERROR=orig.EB, **orig.COEF)


def direct(w, q, *, omit_latest=True):
    # No production lookup, pooling, rate or marginal functions. Recompute independent rational
    # counts at each frontier rather than updating the streaming state. Include the open last row.
    n = len(q)
    rates = np.zeros((n, 256))
    features = np.zeros((n, 256, 2))
    tv = np.zeros(n)
    for t, a in enumerate(q):
        occurrences = []
        for d in (8, 4, 2, 1):
            if t + 1 >= d:
                key = w[t + 1 - d : t + 1]
                occurrences = [j for j in range(d, t - 15) if w[j - d : j] == key]
            if occurrences:
                break
        if len(occurrences) < 2:
            continue
        records = [(q[j - 1], w[j]) for j in occurrences[:-1]]
        wrong = [(g, b) for g, b in records if b != g]
        hit = Counter(b for g, b in wrong if g == a)
        h = sum(hit.values())
        if not h:
            continue
        A = Fraction(h, 1 + h + records.count((a, a)))
        local = Counter(b for g, b in wrong if b != a)
        # Re-enumerate every permitted arrival in the whole prefix. Unlike the chosen reader,
        # the test can materialize the old positions. `omit_latest=False` is a forbidden foil.
        edges = Counter(
            (q[j - 1], w[j])
            for j in range(1, t - 15)
            if (not omit_latest or j != occurrences[-1]) and q[j - 1] != w[j]
        )
        cols = Counter()
        for (g, b), v in edges.items():
            cols[b] += v
        z = {b: Fraction(m * edges[a, b], cols[b]) for b, m in local.items()}
        den = sum(z.values())
        assert den > 0
        for b, m in local.items():
            r = z[b] / den
            v = Fraction(m, sum(local.values()))
            rates[t, b] = float(r)
            features[t, b, 0] = float(A * v)
            features[t, b, 1] = float(A * (r - v))
            tv[t] += 0.5 * abs(float(r - Fraction(hit[b], h)))
        features[t, a, 0] = -float(A)
    return rates, features, tv


def planted():
    # Different keys, the same guessed bytes and possible corrections. The newest eligible
    # record can change the rates, so forgetting to discount it is observable.
    bag = [
        (0, 80, 20),
        (0, 80, 30),
        (0, 81, 20),
        (1, 81, 30),
        (1, 80, 20),
        (0, 80, 30),
        (0, 81, 30),
        (1, 81, 20),
        (0, 80, 20),
        (1, 80, 30),
    ] * 2
    w = b"".join(
        (b"ABC\0D\xffEk", b"pqr\x1esuVk")[k] + bytes([b]) + b"!@#$%&;"
        for k, a, b in bag
    )
    q = bytearray(w[1:] + b"\xff")  # all other artificial guesses correct
    for i, (_, a, _) in enumerate(bag):
        q[16 * i + 7] = a
    return w, bytes(q)


def rates(w, q):
    n = len(w)
    tab = pc.old.true_counts(w, bytes(n), bytes(n), b"\0" + q[:-1])
    m = pc._marginal(tab, q)
    r, work = pt.broad.distribution(w, q, tab, m)
    out = np.zeros((n, 256))
    out[m.row, m.symbol] = r
    return out, work


def test_latest_discount_once_per_arrival_support_and_reset_against_direct_enumeration():
    w, q = planted()
    expect, _, tv = direct(w, q)
    got, work = rates(w, q)
    np.testing.assert_allclose(got, expect, rtol=0, atol=3e-16)
    assert (
        tv.max() > 0.1
        and np.abs(expect - direct(w, q, omit_latest=False)[0]).max() > 0.02
    )
    c = Counter((q[j - 1], w[j]) for j in range(1, len(w) - 16) if q[j - 1] != w[j])
    assert (
        work["peak_pair_cells"] == len(c) == 4
        and work["peak_guess_rows"] == 2
        and work["peak_nonempty_actual_columns"] == 2
    )
    assert (
        work["admitted_records"] == len(w) - 17
        and work["mistake_writes"] == sum(c.values())
        and 0
        < work["discounted_edge_reads"]
        <= work["discounted_column_reads"]
        <= work["latest_mistake_discounts"]
    )
    for n in (1, 16, 17, 18, 40, 64, 65, len(w) - 1):
        np.testing.assert_array_equal(rates(w[:n], q[:n])[0], got[:n])
    w = b"\x00" * 180
    q = b"\xff" * len(w)
    r, work = rates(w, q)
    np.testing.assert_array_equal(r, direct(w, q)[0])
    assert (
        not r[:19].any()
        and r[19, 0] == 1
        and not r[24].any()
        and r[-1, 0] == 1
        and not r[:, 1:].any()
        and work["peak_pair_cells"] == 1
    )
    r, work = rates(w, w)
    assert not r.any() and work["peak_pair_cells"] == work["active_rows"] == 0


@pytest.mark.parametrize("two_keys", (False, True), ids=("same-law", "different-laws"))
def test_dense_prefix_scores_tv_and_all_old_answers_native_writer(binary, two_keys):
    # The original PC motif exercises the pair exchange but its local and prefix
    # correction laws coincide. Keep that null; mix literal keys for a contrast.
    w = planted()[0] if two_keys else orig.word()
    wr = binary[-1]
    host, head, _, _, p = orig.fixture(w, wr)
    q = orig.queries(len(w))
    _, x, tv = direct(w, q)
    expected = dict(
        p.anchors,
        **{
            a: pc.response(p, a, orig.COEF["PAIR" if a == "SWAP" else a])
            for a in (*pc.DIMS, "SWAP")
        },
    )
    raw = p.base
    tilted = raw + torch.from_numpy(x) @ torch.tensor(
        orig.COEF["PAIR"], dtype=torch.float64
    )
    expected["PREFIX"] = tilted - (tilted.logsumexp(-1) - raw.logsumexp(-1))[:, None]
    ans, _, row = pt.answers(host, head, w, BETA, 4902, 4, wr, width=8)
    assert (
        set(ans) == set(expected) == set(pt.ARMS)
        and row["work"]["matched_query_rows"] > 0
        and row["work"]["prefix_different_law_rows"] == sum(tv > pt.TV_TOL)
        and bool(np.any(tv > pt.TV_TOL)) == two_keys
        and isclose(
            row["conditional_tv_sum"],
            sum(tv),
            rel_tol=1e-14,
            abs_tol=np.finfo(np.float64).eps * row["work"]["marginal_entries"],
        )
    )
    assert row["work"]["changed_conditional_rows"] == p.work[
        "changed_conditional_rows"
    ] and torch.equal(ans["PREFIX"][~x.any((1, 2))], p.base[~x.any((1, 2))])
    if two_keys:
        assert not torch.allclose(ans["PREFIX"], ans["PAIR"], rtol=0, atol=1e-12)
    else:
        torch.testing.assert_close(ans["PREFIX"], ans["PAIR"], rtol=0, atol=2e-14)
    for a in ans:
        torch.testing.assert_close(ans[a], expected[a], rtol=0, atol=2e-14)
    torch.testing.assert_close(
        ans["PREFIX"].logsumexp(-1), raw.logsumexp(-1), rtol=0, atol=2e-14
    )
    assert (
        row["seconds"]["shared_ERROR_s"]
        >= row["seconds"]["prefill_s"] + row["seconds"]["shared_posthost_ERROR_s"]
        and row["seconds"]["comparison_only_s"] > 0
    )
    for a in pt.BRANCHES:
        clocks = {k: v for k, v in row["seconds"].items() if k.startswith(a + "_")}
        assert clocks.pop(a + "_branch_s") >= sum(clocks.values()) > 0
    y = w[1:] + b"\x00"
    r = pt.document(host, head, w + b"\x00", BETA, 4902, 4, wr, width=8)
    for a in ans:
        assert isclose(
            r["nll_sums"][a],
            -float(expected[a][torch.arange(len(w)), list(y)].sum()),
            rel_tol=1e-13,
            abs_tol=1e-12,
        )


@pytest.mark.parametrize("a", pt.BRANCHES)
def test_only_chosen_index_and_one_prefill_no_forbidden_work(binary, monkeypatch, a):
    w = orig.word()
    wr = binary[-1]
    host, head, _, _, _ = orig.fixture(w, wr)
    want = pt.answers(host, head, w, BETA, 4901, 2, wr, width=8)[0][a]
    events = []

    def blocked(*args, **kw):
        raise AssertionError("unneeded index, answer, fit or anchor")

    def mark(name, f):
        def call(*args, **kw):
            events.append(name)
            return f(*args, **kw)

        return call

    for obj, k in [
        (pt, "_controls"),
        (pc, "_anchors"),
        (pc.ix, "switched_pairs"),
        (pc.ix, "_switch"),
        (pc.u, "response"),
        (pc.old, "displaced_counts"),
    ]:
        monkeypatch.setattr(obj, k, blocked)
    for obj, k, key in [
        (pc.ix, "true_pairs", "PAIR"),
        (pt.broad, "distribution", "PREFIX"),
    ]:
        monkeypatch.setattr(obj, k, mark(key, getattr(obj, k)) if key == a else blocked)
    monkeypatch.setattr(pt, "frozen_prefill", mark("prefill", pt.frozen_prefill))
    b = BETA["PAIR" if a == "PREFIX" else a]
    out = pt.predict(host, head, w, BETA["J"], BETA["ERROR"], a, b, wr, width=8)
    assert events == (["prefill"] if a == "MARG" else ["prefill", a])
    torch.testing.assert_close(out, want, rtol=0, atol=2e-14)
    for n in (1, 16, 64, 65, len(w) - 1):
        torch.testing.assert_close(
            pt.predict(host, head, w[:n], BETA["J"], BETA["ERROR"], a, b, wr, width=8),
            out[:n],
            rtol=0,
            atol=3e-6,
        )


def test_all_branch_orders_only_input_until_all_scores_finished(binary, monkeypatch):
    w = orig.word()
    wr = binary[-1]
    host, head, _, _, _ = orig.fixture(w, wr)
    s = 4903
    events = []
    expect = list(permutations(("MARG", "PAIR", "PREFIX")))

    def mark(kind, f):
        def go(*args, **kw):
            if kind == "prefill":
                assert args[1] == w[:-1]
            if kind == "labels":
                assert args[1] == w[1:]
            events.append(args[1] if kind == "branch" else kind)
            return f(*args, **kw)

        return go

    for k, event in [
        ("frozen_prefill", "prefill"),
        ("_chosen", "branch"),
        ("_controls", "control"),
        ("_labels", "labels"),
    ]:
        monkeypatch.setattr(pt, k, mark(event, getattr(pt, k)))
    ref = None
    for i in range(6):
        r = pt.document(host, head, w, BETA, s, i, wr, width=8)
        assert (
            events == ["prefill", *expect[(2 + i) % 6], "control", "labels"]
            and r["host_prefills"] == 1
            and r["targets"] == len(w) - 1
        )
        events.clear()
        if ref is None:
            ref = r["nll_sums"]
        assert ref == r["nll_sums"]
    for s, i in ((True, 0), (4917, 0), (4901, False), (4901, -1)):
        with pytest.raises(ValueError):
            pt.order(s, i)
    for w2, bad in (
        (b"", BETA),
        (bytes(4096), BETA),
        (bytes(1), {**BETA, "PREFIX": [0, 0]}),
    ):
        with pytest.raises(ValueError):
            pt.answers(host, head, w2, bad, 4901, 0, wr, width=8)


@pytest.mark.parametrize("n", (1, 16, 64, 65, 4095))
def test_open_short_shapes_match_locked_pc_without_padding(binary, n):
    w = (orig.word() * 17)[:n]
    writer = binary[-1]
    host, head, _, _, ref = orig.fixture(w, writer)
    lp, geom, row = pt.answers(host, head, w, BETA, 4901, 0, writer, width=8)
    assert (
        row["targets"] == len(w) == n
        and row["work"]["prefix_admitted_records"] == max(0, n - 17)
        and row["work"]["guesses_written"] == n - 1
    )
    torch.testing.assert_close(geom.query, ref.query, rtol=0, atol=0)
    for k, v in ref.anchors.items():
        torch.testing.assert_close(lp[k], v, rtol=0, atol=2e-14)
    for a in (*pc.DIMS, "SWAP"):
        torch.testing.assert_close(
            lp[a],
            pc.response(ref, a, BETA["PAIR" if a == "SWAP" else a]),
            rtol=0,
            atol=2e-14,
        )
    assert lp["PREFIX"].shape == (n, 256) and (
        row["work"]["matched_query_rows"] or torch.equal(lp["PREFIX"], lp["ERROR"])
    )


def test_reader_binds_prior_fit_before_probe_rejects_later_substitutes(monkeypatch):
    k = "4901"
    ck = "weight-1"
    b = deepcopy(BETA)
    base = {a: b[a] for a in reader.panel.FIXED}
    beta = {a: b[a] for a in pc.DIMS}
    meta = dict(fit_identity_sha="fit-only")
    r = dict(
        checkpoints={k: ck},
        ref="qualified-source",
        freeze_sha="before-fit",
        data=meta,
        fit_dimensions=dict(pc.DIMS),
        swap_fit="PAIR",
        u_beta={k: b["U"]},
        j_beta={k: b["J"]},
        beta={k: beta},
    )
    e = dict(checkpoint_sha=ck, beta=deepcopy(beta))
    seal = dict(
        spec=reader.source.SPEC,
        fit_units=768,
        plan_sha="before-fit",
        fit_identity_sha="fit-only",
        seeds={k: e},
    )
    z = dict(
        spec=reader.source.SPEC,
        ref=r["ref"],
        data=meta,
        endpoints={k: dict(sha256=ck)},
        fit_dimensions=r["fit_dimensions"],
        swap_fit="PAIR",
        **{f: {k: base[a]} for a, f in reader.panel.FIXED.items()},
    )

    def bound(path, h):
        if path.name == "fit-frozen.json":
            assert h == reader.FIT
            return seal
        assert path.name == "frozen.json" and h == r["freeze_sha"]
        return z

    monkeypatch.setattr(reader, "parent", lambda: r)
    monkeypatch.setattr(reader.source, "bound", bound)
    monkeypatch.setattr(reader.panel, "fixed", lambda s: (ck, base))
    assert reader.contract(4901) == (ck, b)
    for mapping, key, v in [
        (e, "checkpoint_sha", "weight-2"),
        (e["beta"], "PAIR", [0.42, 0.31]),
        (seal, "fit_identity_sha", "later"),
        (seal, "fit_units", 769),
        (z["error_beta"], k, [0.0] * 6),
    ]:
        with monkeypatch.context() as m:
            m.setitem(mapping, key, v)
            with pytest.raises(ValueError, match="foreign PT"):
                reader.contract(4901)
    with pytest.raises(ValueError, match="unknown PT"):
        reader.contract(k)
