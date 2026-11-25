"""PC all-byte, independent dense Fisher and standalone no-comparison-work checks on invented streams."""

from math import isclose
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    error_history as eh,
    paired_correction as pc,
    paired_correction_fit as fit,
    test_joint_reader as native_fixture,
)
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import frozen_prefill
from experiments.expressivity_tasks.x49_live_trace_memory.test_correction_index import (
    word,
    queries,
    oracle,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_error_history import (
    Host,
    trained_base,
)
from experiments.expressivity_tasks.x49_live_trace_memory.literal_mark import text_tape

binary = native_fixture.binary
JB = [-0.17, 0.13, 0.07, 0.1, -0.24]
UB = [0.1, -0.2, 0.16, 0.17]
HB = [0.14, -0.13, 0.12, 0.11, 0.18]
EB = [0.15, -0.13, 0.11, 0.12, 0.18, 0.35]
COEF = {"MARG": [0.62], "PAIR": [0.42, 0.77], "SHUF": [0.41, 0.31]}


class QueryHost(Host):
    def forward(self, x):
        h = super().forward(x)[0, :, :8]
        n = len(h)
        t = torch.arange(n)
        z = torch.zeros(n, 256)
        z[t, 255] = 3.0
        z[t, list(queries(n))] = 3.0  # tie, lower byte wins
        return torch.cat((h, z), dim=1)[None]


def fixture(w, writer):
    torch.manual_seed(4917)
    host = QueryHost().eval()
    head = trained_base().eval()
    h, z, _ = frozen_prefill(host, w, 8)
    p = pc.fields(head, w, h, z, JB, EB, writer, mode="panel", anchors=(UB, HB))
    return host, head, h, z, p


def for_arm(x, a):
    return x[:, :, :1] if a == "MARG" else x[:, :, [0, 1 if a == "PAIR" else 2]]


def test_sparse_columns_all_256_predictions_and_control_support_against_prefix_enumeration(
    binary,
):
    w = word()
    _, head, h, z, p = fixture(w, binary[-1])
    q = queries(len(w))
    assert pc._queries(z) == (b"\0" + q[:-1], q)
    ref = eh.fields(head, w, h, z, JB, text_tape, ub=UB)
    expect = dict(
        ref.anchors,
        HISTORY=eh.response(ref, "HISTORY", HB),
        ERROR=eh.response(ref, "ERROR", EB),
    )
    for a, b in expect.items():
        torch.testing.assert_close(b, p.anchors[a], rtol=0, atol=3e-14)
    _, _, _, dr, ds, x = oracle(w, q)
    x = torch.from_numpy(x)
    raw = p.base
    changed = (dr != ds).any(1)
    hpos = dr.sum(1) > 0
    dense = torch.zeros_like(x)
    dense[p.row, p.symbol] = p.x
    torch.testing.assert_close(dense, x, atol=2e-15, rtol=0)
    assert torch.count_nonzero(x[torch.arange(len(w)), list(q), 1:]) == 0
    assert (
        p.work["changed_conditional_rows"] == changed.sum() > 0
        and p.work["changed_conditional_uses"] == np.maximum(dr - ds, 0).sum()
        and p.work["matched_query_rows"] == hpos.sum()
        and p.work["true_nonmarginal_rows"]
        == torch.count_nonzero(x[:, :, 1].ne(0).any(1))
    )
    for a in (*pc.DIMS, "SWAP"):
        b = COEF["PAIR" if a == "SWAP" else a]
        delta = for_arm(x, a) @ torch.tensor(b, dtype=torch.float64)
        logits = raw + delta
        expect[a] = logits - (logits.logsumexp(-1) - raw.logsumexp(-1))[:, None]
        ans = pc.response(p, a, b)
        torch.testing.assert_close(ans, expect[a], rtol=0, atol=2e-14)
        assert torch.equal(raw[~hpos], ans[~hpos])
        torch.testing.assert_close(
            ans.logsumexp(-1), raw.logsumexp(-1), rtol=0, atol=2e-14
        )
    assert not torch.allclose(
        expect["PAIR"][changed], expect["SWAP"][changed], rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        expect["PAIR"][~changed], expect["SWAP"][~changed], atol=2e-14, rtol=0
    )
    y = w[1:] + b"\0"
    r, losses, _ = fit.score(p, COEF, y)
    assert r == int((torch.tensor(list(y))[p.query] == p.candidate).sum()) and set(
        losses
    ) == set(pc.ARMS)
    for a, loss in losses.items():
        assert isclose(
            loss,
            -float(expect[a][torch.arange(len(w)), list(y)].sum()),
            abs_tol=2e-11,
            rel_tol=0,
        )
    for bad in [
        dict(COEF, SWAP=COEF["PAIR"]),
        dict(PAIR=COEF["PAIR"]),
        dict(COEF, SHUF=[0]),
    ]:
        with pytest.raises(ValueError):
            fit.score(p, bad, y)


def test_all_target_fisher_matches_dense_loglikelihood_autograd_including_unchanged_rows(
    binary,
):
    w = word()[:180]
    _, _, _, _, p = fixture(w, binary[-1])
    y = w[1:] + b"\xff"
    m = fit.Moments()
    m.add(p, y)
    f = m.freeze()
    x = torch.from_numpy(oracle(w, queries(len(w)))[-1])
    lab = list(y)
    n = len(w)
    assert (
        m.documents == 1
        and 0 < len(p.active) < m.targets == n
        and set(f["beta"]) == set(pc.DIMS)
    )
    for a, d in pc.DIMS.items():
        col = for_arm(x, a)
        v = torch.zeros(d, dtype=torch.float64, requires_grad=True)

        def obj(b):
            return -(p.base + col @ b).log_softmax(-1)[torch.arange(n), lab].sum()

        rhs = -torch.autograd.grad(obj(v), v)[0]
        gram = torch.autograd.functional.hessian(obj, v)
        np.testing.assert_allclose(m.rhs[a], rhs, rtol=1e-12, atol=2e-13)
        np.testing.assert_allclose(m.gram[a], gram, rtol=1e-12, atol=2e-13)
        np.testing.assert_allclose(
            f["beta"][a],
            torch.linalg.solve(
                gram / n + torch.eye(d, dtype=torch.float64) / 4096, rhs / n
            ),
            rtol=1e-11,
            atol=2e-12,
        )
    m.gram["PAIR"][0, 0] += 1
    with pytest.raises(ValueError, match="mismatched current features"):
        m.freeze()


@pytest.mark.parametrize("arm", ["MARG", "PAIR"])
def test_single_chosen_predictor_uses_no_sham_other_answers_or_future(
    binary, monkeypatch, arm
):
    w = word()
    writer = binary[-1]
    host, head, _, _, panel = fixture(w, writer)
    ans = pc.response(panel, arm, COEF[arm])
    events = []

    def forbidden(*args, **kwargs):
        raise AssertionError("unneeded pair, sham or anchor on standalone path")

    def mark(name, fn):
        def use(*args, **kwargs):
            events.append(name)
            return fn(*args, **kwargs)

        return use

    monkeypatch.setattr(pc, "frozen_prefill", mark("prefill", pc.frozen_prefill))
    monkeypatch.setattr(pc.ix, "switched_pairs", forbidden)
    monkeypatch.setattr(pc.ix, "_switch", forbidden)
    monkeypatch.setattr(pc.old, "displaced_counts", forbidden)
    monkeypatch.setattr(pc.u, "response", forbidden)
    monkeypatch.setattr(
        pc.ix,
        "true_pairs",
        mark("pair", pc.ix.true_pairs) if arm == "PAIR" else forbidden,
    )
    a = pc.predict(host, head, w, JB, EB, arm, COEF[arm], writer, width=8)
    assert events == (["prefill", "pair"] if arm == "PAIR" else ["prefill"])
    torch.testing.assert_close(a, ans, rtol=0, atol=2e-14)
    for end in [1, 17, 64, 65, 145, len(w) - 1]:
        v = pc.predict(host, head, w[:end], JB, EB, arm, COEF[arm], writer, width=8)
        torch.testing.assert_close(
            v, a[:end], atol=3e-6, rtol=0
        )  # short FP32 head kernels can differ
    with pytest.raises(ValueError, match="choose one"):
        pc.predict(host, head, w, JB, EB, "SHUF", COEF["SHUF"], writer, width=8)


def test_fit_probe_exclusive_one_prefill_and_all_answers_before_gather(
    binary, monkeypatch
):
    w = (word() * 17)[:4097]
    writer = binary[-1]
    host = QueryHost().eval()
    head = trained_base().eval()
    events = []
    label = w[1:]
    old_pre = fit.frozen_prefill
    old_res = fit.response
    old_lab = fit._labels

    def pre(host, obs, width):
        assert obs == w[:-1]
        events.append("prefill")
        return old_pre(host, obs, width)

    def resp(p, a, b):
        events.append(a)
        return old_res(p, a, b)

    def labels(p, y):
        assert y == label
        events.append("label")
        return old_lab(p, y)

    monkeypatch.setattr(fit, "frozen_prefill", pre)
    monkeypatch.setattr(fit, "response", resp)
    monkeypatch.setattr(fit, "_labels", labels)
    kw = dict(ub=UB, jb=JB, hb=HB, eb=EB, writer=writer, width=8)
    r = fit.document(host, head, w, **kw, beta=COEF)
    assert (
        events == ["prefill", "MARG", "PAIR", "SHUF", "SWAP", "label"]
        and r["host_prefills"] == 1
        and r["targets"] == 4096
        and set(r["nll_sums"]) == set(pc.ARMS)
    )
    events.clear()
    m = fit.Moments()
    rf = fit.document(host, head, w, **kw, fit=m)
    assert (
        events == ["prefill"]
        and "nll_sums" not in rf
        and "other_anchors_s" not in rf["seconds"]
        and m.targets == 4096
        and rf["work"] == r["work"]
    )
    for f, b in [(m, COEF), (None, None), (object(), None)]:
        with pytest.raises(ValueError, match="exactly one"):
            fit.document(host, head, w, **kw, fit=f, beta=b)
    with pytest.raises(ValueError, match="fully shifted"):
        fit.document(host, head, w[:-1], **kw, beta=COEF)
