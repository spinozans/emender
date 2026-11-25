"""LS literal estimator, mark bindings, Fisher step and frozen comparator on invented data only."""

from math import log
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    literal_mark as ls,
    literal_mark_fit as fit,
    surprise as u,
    local_u,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_surprise import (
    example,
    scalar_code,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import oracle
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny


def dense_forecast(prefix):
    # Independently enumerate all earlier followers at every depth for every possible next byte.
    j = len(prefix)
    p = (np.bincount(np.frombuffer(prefix, dtype=np.uint8), minlength=256) + 0.5) / (
        j + 128
    )
    for d in (1, 2, 4, 8):
        if j < d:
            continue
        followers = [prefix[i] for i in range(d, j) if prefix[i - d : i] == prefix[-d:]]
        if followers:
            count = np.bincount(followers, minlength=256)
            K = np.count_nonzero(count)
            p = (count + K * p) / (len(followers) + K)
    return p


def test_every_dense_prefix_distribution_and_actual_arrival_code_before_update():
    w = b"a" * 12 + b"abababa\xff\0\x1e abc cab a\xff\0\x1e" * 3
    state = ls.LiteralHistory()
    expected = []
    for j, b in enumerate(w):
        p = dense_forecast(w[:j])
        lookups = state.lookups
        actual = np.array([state.probability(c) for c in range(256)])
        np.testing.assert_allclose(actual, p, atol=2e-16, rtol=1e-15)
        assert (
            actual.sum() == pytest.approx(1.0, abs=2e-15)
            and state.lookups == lookups
            and sum(state.order_zero) == state.n == j
        )
        code = scalar_code(-log(p[b])) if j else 0
        expected.append(code)
        assert state.observe(b) == code
    tape, work = ls.text_tape(w)
    assert (
        tape.dtype == torch.uint8
        and tape.tolist() == expected
        and tape.nbytes == len(w)
    )
    contexts = set()
    pairs = set()
    for j, b in enumerate(w):
        for d in (1, 2, 4, 8):
            if j >= d:
                contexts.add(w[j - d : j])
                pairs.add((w[j - d : j], b))
    visited = sum(max(0, len(w) - d) for d in (1, 2, 4, 8))
    assert (
        work["text_context_updates"] == work["text_context_lookups"] == visited
        and work["text_distinct_contexts"]
        == len(contexts)
        == visited - work["text_seen_context_lookups"]
        and work["text_distinct_pairs"] == len(pairs)
    )
    assert (
        work["text_order_zero_updates"] == len(w)
        and work["text_arrivals_encoded"] == len(w) - 1
    )


def test_extreme_surprise_overlaps_no_future_and_window_reset():
    w = b"a" * 4095 + b"\0"
    t, work = ls.text_tape(w)
    assert t[-1] == 255 and t[0] == 0 and work["text_marks_saturated"] == 1
    assert torch.equal(t[:-1], ls.text_tape(w[:-1])[0]) and torch.equal(
        t[:1500], ls.text_tape(w[:1500] + b"\xff" * 2596)[0][:1500]
    )
    assert ls.text_tape(b"\0")[0].tolist() == [0]
    for bad in (b"", bytearray(b"!"), w + b"x"):
        with pytest.raises(ValueError):
            ls.text_tape(bad)
    s = ls.LiteralHistory()
    for bad in (-1, 256, True, 2.5):
        with pytest.raises(ValueError):
            s.observe(bad)
    assert s.n == 0 and not s.tables


def test_same_b_vector_and_correct_text_mark_with_only_the_neural_mark_displaced():
    head, w, h, z = example()
    p = ls.fields(head, w, h, z)
    ref = u.fields(head, w, h, z)
    assert torch.equal(p.base, ref.base) and torch.equal(p.phi["H"], ref.phi["U"])
    ub = local_u.contract(4901)[1]
    assert torch.equal(ls.response(p, "U", ub), u.response(ref, "U", ub))
    cols = []
    for i, t in enumerate(p.query.tolist()):
        j, _, _, k = oracle(w, t, 4096)
        c = w[j]
        tc = scalar_code(-log(dense_forecast(w[:j])[c]))
        hc = scalar_code(-z[j - 1].double().log_softmax(-1)[c])
        sc = scalar_code(-z[k - 1].double().log_softmax(-1)[c])
        n = ref.phi["N"][i].tolist()
        assert w[k] == c == p.candidate[i] and p.latest[i] == j
        assert all(
            p.phi[a][i].tolist() == v
            for a, v in {
                "N": n,
                "T": [*n, tc / 64],
                "H": [*n, hc / 64],
                "J": [*n, tc / 64, hc / 64],
                "S": [*n, tc / 64, sc / 64],
            }.items()
        )
        cols.append((tc, hc, sc))
    t, hc, s = np.array(cols).T
    v = p.work
    assert (
        v["text_mark_reads"] == v["U_mark_reads"] == v["S_mark_reads"] == len(cols)
        and v["S_codes_changed"] == int((hc != s).sum()) > 0
        and v["text_vs_neural_changed"] == int((t != hc).sum()) > 0
    )
    assert (
        v["text_code_sum"],
        v["neural_code_sum"],
        v["text_neural_code_product_sum"],
        v["text_code_square_sum"],
        v["neural_code_square_sum"],
    ) == (t.sum(), hc.sum(), t @ hc, t @ t, hc @ hc)
    # With identical readout weights, the correct-vs-sham difference has exactly the changed-neural-code support.
    b = [-0.5, 0.1, 0.2, -0.25, 1.0]
    ja, sa = (ls.response(p, a, b) for a in "JS")
    assert torch.equal(
        (ja != sa).any(-1).nonzero().flatten(),
        p.query[p.phi["J"][:, -1] != p.phi["S"][:, -1]],
    )


def test_matched_ls_systems_and_all_256_scores_against_dense_gradient_and_curvature():
    head, w, h, z = example()
    p = ls.fields(head, w, h, z)
    m = fit.Moments()
    ys = w[1:] + b"\xff"
    m.add(p, ys)
    miss = ls.fields(head, b"\xff", h[:1], z[:1])
    m.add(miss, b"\0")
    r = m.freeze()
    q = p.query.clone()
    c = p.candidate.clone()
    y = torch.tensor(list(ys))
    idx = torch.arange(len(ys))
    base = p.base.clone().double()
    n = len(y) + 1
    assert r["targets"] == n and r["documents"] == 2 and set(r["beta"]) == set("NTHJS")
    for arm, d in ls.DIMS.items():
        F = p.phi[arm].clone()
        theta = torch.zeros(d, dtype=torch.double, requires_grad=True)

        def dense(v):
            moved = base.index_put((q, c), base[q, c] + F @ v)
            return moved - (moved.logsumexp(-1) - base.logsumexp(-1))[:, None]

        def obj(v):
            return (-dense(v)[idx, y].sum() - miss.base[0, 0]) / n + (v @ v) * r[
                "ridge"
            ] / 2

        g = torch.autograd.grad(obj(theta), theta)[0]
        H = torch.autograd.functional.hessian(obj, theta)
        np.testing.assert_allclose(m.rhs[arm] / n, -g.numpy(), atol=2e-13, rtol=1e-12)
        np.testing.assert_allclose(
            m.gram[arm] / n + np.eye(d) * r["ridge"], H.numpy(), atol=2e-13, rtol=1e-12
        )
        np.testing.assert_allclose(
            r["beta"][arm],
            np.linalg.solve(H.numpy(), -g.numpy()),
            atol=1e-10,
            rtol=1e-12,
        )
        b = torch.arange(d, dtype=torch.double) * 0.5 - 1
        torch.testing.assert_close(ls.response(p, arm, b), dense(b), rtol=0, atol=3e-14)
        assert torch.equal(ls.response(p, arm, theta), base) and torch.equal(
            ls.response(miss, arm, b), miss.base.double()
        )
    ub = local_u.contract(4902)[1]
    out = fit.score(p, r["beta"], ub, ys)
    assert set(out["nll_sums"]) == set("NTHJSU")
    for arm, s in out["nll_sums"].items():
        assert s == float(
            -ls.response(p, arm, ub if arm == "U" else r["beta"][arm])[idx, y].sum()
        )
    m.gram["J"][3, 3] += 0.1
    with pytest.raises(ValueError, match="literal field"):
        m.freeze()


def test_full_window_one_prefill_five_fits_and_separate_original_u():
    head, w, _, _ = example()
    host = Tiny().eval()
    word = (w * 30)[:4097]
    ub = local_u.contract(4903)[1]
    m = fit.Moments()
    calls = []
    hook = host.register_forward_pre_hook(lambda _mod, x: calls.append(x[0].clone()))
    try:
        a = fit.document(host, head, word, fit=m, u_beta=ub, width=8)
        f = m.freeze()
        b = fit.document(host, head, word, beta=f["beta"], u_beta=ub, width=8)
        g = fit.document(
            host, head, word[:-1] + b"!", beta=f["beta"], u_beta=ub, width=8
        )
    finally:
        hook.remove()
    assert (
        len(calls) == 3
        and all(torch.equal(x, torch.tensor([list(word[:-1])])) for x in calls)
        and a["host_prefills"] == b["host_prefills"] == 1
        and a["work"] == b["work"] == g["work"]
        and b["targets"] == f["targets"] == 4096
        and set(b["nll_sums"]) == set("NTHJSU")
        and "nll_sums" not in a
        and f == m.freeze()
    )
    with pytest.raises(ValueError):
        fit.document(host, head, word, fit=m, beta=f["beta"], u_beta=ub, width=8)
    with pytest.raises(ValueError):
        fit.document(host, head, word[:-1], fit=m, u_beta=ub, width=8)
