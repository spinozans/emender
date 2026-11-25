"""ET first gate: existing eight predictions on invented short streams; isolated and actually timed branches."""

from copy import deepcopy
from math import isclose
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    error_history as eh,
    error_history_index as ix,
    error_transfer_score as score,
    error_transfer_reader as reader,
    joint_reader as joint,
    local_u,
    test_joint_reader as native_fixture,
)
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import frozen_prefill
from experiments.expressivity_tasks.x49_live_trace_memory.test_error_history import (
    Host,
    trained_base,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_error_history_index import (
    oracle,
    word,
    dense,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import SEEDS

binary = native_fixture.binary


def beta(s):
    return dict(
        U=list(local_u.contract(s)[1]),
        J=list(joint.contract(s)[1]),
        **{
            a: [(-1) ** i * (i + 1) * (0.05 + 0.01 * j) for i in range(d)]
            for j, (a, d) in enumerate(eh.DIMS.items())
        },
    )


def setup(s):
    torch.manual_seed(4916)
    return Host().eval(), trained_base().eval(), beta(s)


def test_history_only_counts_real_followers_never_wrong_only_symbols():
    w = word()
    lt = b"\0" + bytes((241 + i % 12) for i in range(len(w) - 1))
    nt = b"\0" + bytes(i % 3 for i in range(len(w) - 1))
    g = b"\0" + bytes(c if i % 3 == 0 else 199 + i % 7 for i, c in enumerate(w[1:]))
    flags = b"\0" + bytes(a != c for a, c in zip(g[1:], w[1:], strict=True))
    h = ix.history_counts(w, lt, nt, flags)
    t = ix.true_counts(w, lt, nt, g)
    ref = oracle(w, lt, nt, g)[0]
    np.testing.assert_array_equal(dense(h), ref[:, :, :4])
    np.testing.assert_array_equal(h.position, t.position[t.counts[:, 0] > 0])
    assert (
        len(h.position) < len(t.position)
        and h.work["peak_cells"] < t.work["peak_cells"]
        and h.work["packed_bytes"] == 5 * len(w) + 20 * len(h.position)
    )
    with pytest.raises(ValueError, match="flags"):
        ix.history_counts(w, lt, nt, g)


@pytest.mark.parametrize("n", (1, 16, 64, 65, 4095))
def test_short_full_scores_vs_registered_panel_and_cost_support(binary, n):
    s = 4902
    host, head, b = setup(s)
    w = (word() * 32)[:n]
    y = w[1:] + b"\x19"
    writer = binary[-1]
    h, z, _ = frozen_prefill(host, w, 8)
    p = eh.fields(head, w, h, z, b["J"], writer, ub=b["U"])
    orig = dict(
        p.anchors,
        **{
            a: eh.response(p, a, b["ERROR" if a == "SWAP" else a])
            for a in (*eh.DIMS, "SWAP")
        },
    )
    lp, q, r = score.answers(host, head, w, b, s, 4, writer, width=8)
    assert (
        set(orig) == set(lp) == set(eh.ARMS)
        and r["targets"] == n
        and r["first_branch"] == "ERROR"
        and r["host_prefills"] == 1
        and r["work"]["changed_histogram_rows"] == p.work["changed_histogram_rows"]
    )
    torch.testing.assert_close(q.query, p.query, atol=0, rtol=0)
    for a in eh.ARMS:
        torch.testing.assert_close(lp[a], orig[a], rtol=0, atol=2e-14)
    for a in score.BRANCHES:
        cw = r["chosen_work"][a]
        cs = r["seconds"]
        assert (
            cw["top1_written"] == n - 1
            and cw["top1_bytes"] == n
            and cw["miss_flag_bytes"] == (n if a == "HISTORY" else 0)
            and cw["active_rows"] == p.work["pool_active_rows"]
            and cw["fp64_feature_bytes"]
            == cw["packed_feature_entries"] * 8 * eh.DIMS[a]
        )
        assert (
            cs[a + "_branch_s"]
            >= sum(
                cs[a + "_" + k + "_s"] for k in ("prep", "pool", "feature", "response")
            )
            > 0
        )
    assert (
        r["seconds"]["shared_J_s"] >= r["seconds"]["prefill_s"] > 0
        and r["seconds"]["comparison_only_s"] > 0
    )
    a = score.document(host, head, w + y[-1:], b, s, 4, writer, width=8)
    idx = torch.arange(n)
    assert all(
        isclose(
            a["nll_sums"][k],
            -float(lp[k][idx, list(y)].sum()),
            abs_tol=1e-12,
            rel_tol=1e-14,
        )
        for k in lp
    )


@pytest.mark.parametrize("arm", score.BRANCHES)
def test_chosen_standalone_cannot_use_control_or_other_index(binary, monkeypatch, arm):
    host, head, b = setup(4901)
    w = word()
    writer = binary[-1]
    all_lp = score.answers(host, head, w, b, 4901, 1, writer, width=8)[0]
    events = []

    def forbidden(*a, **k):
        raise AssertionError("comparison work on chosen path")

    def mark(name, f):
        def wrap(*args, **kw):
            events.append(name)
            return f(*args, **kw)

        return wrap

    ok = "history_counts" if arm == "HISTORY" else "true_counts"
    bad = "true_counts" if arm == "HISTORY" else "history_counts"
    monkeypatch.setattr(ix, bad, forbidden)
    monkeypatch.setattr(ix, ok, mark(ok, getattr(ix, ok)))
    monkeypatch.setattr(ix, "guesses", mark("guess", ix.guesses))
    monkeypatch.setattr(ix, "_donors", forbidden)
    monkeypatch.setattr(ix, "displaced_counts", forbidden)
    monkeypatch.setattr(score, "_controls", forbidden)
    monkeypatch.setattr(score.u, "response", forbidden)
    ref = score.predict(host, head, w, b["J"], arm, b[arm], writer, width=8)
    assert events == ["guess", ok]
    torch.testing.assert_close(ref, all_lp[arm], rtol=0, atol=2e-14)
    for end in (1, 18, len(w) - 1):
        torch.testing.assert_close(
            score.predict(host, head, w[:end], b["J"], arm, b[arm], writer, width=8),
            ref[:end],
            rtol=0,
            atol=3e-6,
        )


def test_alternate_order_labels_last_and_no_extra_prefill(binary, monkeypatch):
    s = 4903
    host, head, b = setup(s)
    w = word()
    events = []
    chosen = score._chosen
    prefill = score.frozen_prefill
    controls = score._controls
    labels = score._labels

    def fmt(*args):
        events.append(args[1])
        return chosen(*args)

    def pre(h, obs, width):
        assert obs == w[:-1]
        events.append("prefill")
        return prefill(h, obs, width)

    def diag(*a):
        events.append("controls")
        return controls(*a)

    def lab(*a):
        events.append("labels")
        return labels(*a)

    for name, f in [
        ("_chosen", fmt),
        ("frozen_prefill", pre),
        ("_controls", diag),
        ("_labels", lab),
    ]:
        monkeypatch.setattr(score, name, f)
    r = score.document(host, head, w, b, s, 0, binary[-1], width=8)
    assert events == ["prefill", "HISTORY", "ERROR", "controls", "labels"]
    events.clear()
    r2 = score.document(host, head, w, b, s, 1, binary[-1], width=8)
    assert (
        events == ["prefill", "ERROR", "HISTORY", "controls", "labels"]
        and r["nll_sums"] == r2["nll_sums"]
    )
    assert (
        r["work"]
        == score.document(host, head, w[:-1] + b"\xbb", b, s, 1, binary[-1], width=8)[
            "work"
        ]
    )
    for n in (0, 4096):
        with pytest.raises(ValueError, match="observed prefix"):
            score.answers(host, head, bytes(n), b, s, 0, binary[-1], width=8)
    for k in ("J", "SWAP"):
        bad = deepcopy(b)
        bad.pop(k) if k == "J" else bad.update(SWAP=b["ERROR"])
        with pytest.raises(ValueError, match="six frozen"):
            score.answers(host, head, w, bad, s, 0, binary[-1], width=8)
    for s, i in ((4916, 0), (4901, -1), (4901, False)):
        with pytest.raises(ValueError, match="seed or reserved rank"):
            score.first(s, i)


def test_no_fit_reader_rejects_seed_swapped_or_changed_frozen_coefficients(monkeypatch):
    s = 4901
    k = str(s)
    ck = local_u.contract(s)[0]
    b = beta(s)
    coef = {a: b[a] for a in eh.DIMS}
    meta = dict(fit_identity_sha="fit")
    e = dict(checkpoint_sha=ck, beta=coef)
    r = dict(
        freeze_sha="plan",
        checkpoints={k: ck},
        u_beta={k: b["U"]},
        j_beta={k: b["J"]},
        beta={k: coef},
        data=meta,
        fit_dimensions=eh.DIMS,
        swap_fit="ERROR",
    )
    z = dict(
        spec=reader.source.SPEC,
        data=meta,
        endpoints={k: dict(sha256=ck)},
        fit_dimensions=eh.DIMS,
        swap_fit="ERROR",
        u_beta=r["u_beta"],
        j_beta=r["j_beta"],
    )
    seal = dict(
        spec=reader.source.SPEC,
        plan_sha="plan",
        fit_identity_sha="fit",
        fit_units=768,
        seeds={k: e},
    )
    monkeypatch.setattr(reader, "parent", lambda: r)

    def bound(path, h):
        if path.name == "fit-frozen.json":
            assert h == reader.FIT
            return seal
        assert path.name == "frozen.json" and h == r["freeze_sha"]
        return z

    monkeypatch.setattr(reader.source, "bound", bound)
    assert reader.contract(s) == (ck, b)
    for field, value in [
        ("checkpoint_sha", local_u.contract(SEEDS[1])[0]),
        ("beta", {**coef, "SHIFT": coef["ERROR"]}),
    ]:
        with monkeypatch.context() as mt:
            mt.setitem(e, field, value)
            with pytest.raises(ValueError, match="foreign ET weight"):
                reader.contract(s)
    with pytest.raises(ValueError, match="unknown ET checkpoint"):
        reader.contract("4901")
