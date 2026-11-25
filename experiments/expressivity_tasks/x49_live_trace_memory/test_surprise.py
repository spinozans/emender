"""X49-U synthetic marks, prefix closure, one-read B identity and exact derivatives at zero."""

import math
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    surprise as u,
    surprise_fit as fit,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    oracle,
    trained_base,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny


def example():
    torch.manual_seed(4907)
    w = (b"\x00ab\xffabc\x1e abb:  ba a\xe9" * 10)[:185]
    head = trained_base().eval()
    h = torch.randn(len(w), 8)
    z = torch.randn(len(w), 256)
    return head, w, h, z


def scalar_code(x):
    return min(255, math.floor(16 * max(0.0, float(x)) + 0.5))


def test_quantized_arrivals_observe_only_the_preceding_forecast_and_reset():
    s = torch.tensor(
        [-2, 0, 0.03124, 0.03125, 0.09375, 254.49 / 16, 254.5 / 16, 1e200],
        dtype=torch.double,
    )
    assert u.quantize(s).tolist() == [0, 0, 0, 1, 2, 254, 255, 255]
    with pytest.raises(ValueError):
        u.quantize(torch.tensor([float("nan")], dtype=torch.double))
    with pytest.raises(ValueError):
        u.quantize(s.float())
    _, w, _, z = example()
    ids = torch.tensor(list(w))
    q = torch.arange(0, len(w), 3)
    t, reused, missing = u._arrival_tape(z, ids, q, z[q].double().logsumexp(-1))
    assert (
        t.dtype == torch.uint8
        and t.nbytes == len(w)
        and t[0] == 0
        and reused == len(q[q < len(w) - 1])
        and reused + missing == len(w) - 1
    )
    expected = [
        scalar_code(-z[j - 1].double().log_softmax(-1)[x]) for j, x in enumerate(w) if j
    ]
    assert t.tolist() == [0, *expected]
    z2 = z.clone()
    z2[15, w[16]] = 75
    t2, _, _ = u._arrival_tape(z2, ids, q, z2[q].double().logsumexp(-1))
    assert torch.where(t != t2)[0].tolist() == [16] and t2[16] == 0
    t3, _, _ = u._arrival_tape(
        z[-1:],
        ids[-1:],
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.double),
    )
    assert t3.tolist() == [0]


def test_actual_scalar_addresses_current_features_and_wrong_field_only():
    head, w, h, z = example()
    r = u.fields(head, w, h, z)
    jwork = {}
    lp = head(w, h, z, mode="B", work=jwork)
    assert torch.equal(r.base, lp) and all(r.work[k] == v for k, v in jwork.items())
    matches = [(t, m) for t in range(len(w)) if (m := oracle(w, t, 4096)) is not None]
    assert r.query.tolist() == [t for t, _ in matches]

    def codes(i, c):
        return scalar_code(-z[i].double().log_softmax(-1)[c]) / 64

    for i, (t, (j, _, _, k)) in enumerate(matches):
        c = w[j]
        assert w[k] == c == r.candidate[i]
        common = [1, scalar_code(-lp[t].double().log_softmax(-1)[c]) / 64, codes(t, c)]
        assert (
            r.phi["N"][i].tolist() == common
            and r.phi["U"][i].tolist() == [*common, codes(j - 1, c)]
            and r.phi["S"][i].tolist() == [*common, codes(k - 1, c)]
        )
    assert r.work["U_mark_reads"] == r.work["S_mark_reads"] == r.work[
        "literal_value_reads"
    ] == len(matches) and 0 < r.work["S_codes_changed"] <= r.work[
        "S_slots_displaced"
    ] == sum(j != k for _, (j, _, _, k) in matches)
    with pytest.raises(ValueError, match="empty B hit output"):
        head(w, h, z, mode="B", hit_out={"latest": torch.zeros(len(w))})
    v = [0.0, 0.0, 0.0, 0.75]
    a = u.response(r, "U", v)
    b = u.response(r, "S", v)
    different = r.query[r.phi["U"][:, -1] != r.phi["S"][:, -1]]
    assert torch.equal((a != b).any(-1).nonzero().flatten(), different)


def test_no_future_bytes_or_features_in_frozen_predictors_even_with_nonzero_readouts():
    head, w, h, z = example()
    r = u.fields(head, w, h, z)
    t = int(r.query[28])
    n = t + 1
    h2 = h.clone()
    h2[n:] += 32
    z2 = z.clone()
    z2[n:] *= -10
    p = u.fields(head, w[:n] + bytes([255]) * (len(w) - n), h2, z2)
    short = u.fields(head, w[:n], h[:n], z[:n])
    qmask = r.query <= t
    assert (
        r.query[qmask].tolist()
        == short.query.tolist()
        == p.query[p.query <= t].tolist()
    )
    for a, d in u.DIMS.items():
        beta = np.linspace(-0.25, 0.75, d)
        x = u.response(r, a, beta)
        assert torch.equal(x[:n], u.response(p, a, beta)[:n])
        torch.testing.assert_close(x[:n], u.response(short, a, beta), rtol=0, atol=3e-6)
        assert torch.equal(r.phi[a][qmask], short.phi[a])
        zero = u.response(r, a, np.zeros(d))
        assert torch.equal(zero, r.base.double())
    miss = u.fields(head, b"\xff", h[:1], z[:1])
    assert miss.work["arrivals_encoded"] == miss.work["S_mark_reads"] == 0
    assert torch.equal(miss.base.double(), u.response(miss, "U", [2.0, 1.0, 1.0, 3.0]))


def test_mass_preservation_and_dense_softmax_oracle_including_confident_candidate():
    head, w, h, z = example()
    p = u.fields(head, w, h, z)
    base = p.base.double()
    assert torch.equal(u.response(p, "N", [0] * 3), base)
    for a, d in u.DIMS.items():
        b = torch.arange(d, dtype=torch.double) - 1.5
        delta = p.phi[a] @ b
        dense = base.clone()
        dense[p.query, p.candidate] += delta
        ref = dense - (dense.logsumexp(-1) - base.logsumexp(-1))[:, None]
        out = u.response(p, a, b)
        torch.testing.assert_close(out, ref, rtol=0, atol=8e-14)
        torch.testing.assert_close(
            out.logsumexp(-1), base.logsumexp(-1), rtol=0, atol=8e-14
        )
    z = torch.full((2, 256), -150.0, dtype=torch.float32)
    z[0, 33] = 0
    z[1, 33] = -450
    v = u._candidate_margin(z, torch.tensor([0, 1]), torch.tensor([33, 33]))
    np.testing.assert_allclose(
        v.numpy(), [150 - math.log(255), -300 - math.log(255)], atol=2e-13, rtol=0
    )


def test_fisher_fit_is_dense_gradient_and_hessian_on_all_real_bytes():
    head, w, h, z = example()
    p = u.fields(head, w, h, z)
    ys = w[1:] + b"!"
    y = torch.tensor(list(ys))
    m = fit.Moments()
    r = m.add(p, ys)
    assert r["candidate_correct"] == sum(
        w[oracle(w, t, 4096)[0]] == ys[t] for t in p.query.tolist()
    )
    miss = u.fields(head, b"\0", h[:1], z[:1])
    m.add(miss, b"a")
    result = m.freeze()
    n = len(w) + 1
    assert result["documents"] == 2 and result["targets"] == n
    for a, d in u.DIMS.items():
        phi = p.phi[a].clone()
        lp = p.base.double().clone()
        q = p.query.clone()
        c = p.candidate.clone()
        b = torch.zeros(d, dtype=torch.float64, requires_grad=True)

        def loss(b):
            moved = lp.index_put((q, c), lp[q, c] + phi @ b)
            return (
                -moved.log_softmax(-1)[torch.arange(len(w)), y].sum()
                - miss.base[0, 97].double()
            ) / n + fit.RIDGE * b.square().sum() / 2

        grad = torch.autograd.grad(loss(b), b)[0]
        hess = torch.autograd.functional.hessian(loss, b)
        np.testing.assert_allclose(m.rhs[a] / n, -grad.numpy(), atol=2e-13, rtol=0)
        np.testing.assert_allclose(
            m.gram[a] / n + fit.RIDGE * np.eye(d), hess.numpy(), atol=2e-13, rtol=0
        )
        np.testing.assert_allclose(
            result["beta"][a],
            np.linalg.solve(hess.numpy(), -grad.numpy()),
            atol=1e-10,
            rtol=1e-12,
        )
    out = fit.score(p, result["beta"], ys)
    assert out["candidate_correct"] == r["candidate_correct"]
    for a, v in out["nll_sums"].items():
        pred = p.base.double() if a == "B" else u.response(p, a, result["beta"][a])
        assert v == float(-pred[torch.arange(len(y)), y].sum())
    with pytest.raises(ValueError):
        fit.Moments().freeze()
    with pytest.raises(ValueError):
        fit.score(p, {"S": [0] * 4}, ys)
    with pytest.raises(ValueError):
        m.add(p, ys[:-1])


def test_document_fit_and_probe_are_separate_one_prefill_zero_host_updates():
    head, w, _, _ = example()
    m = Tiny().eval()
    word = (w * 23)[:4097]
    m0 = {k: v.clone() for k, v in m.state_dict().items()}
    h0 = {k: v.clone() for k, v in head.state_dict().items()}
    opt = fit.Moments()
    report = fit.document(m, head, word, fit=opt, width=8)
    z = opt.freeze()
    a = fit.document(m, head, word, beta=z["beta"], width=8)
    b = fit.document(
        m, head, word[:-1] + bytes([word[-1] ^ 255]), beta=z["beta"], width=8
    )
    assert (
        report["host_prefills"] == a["host_prefills"] == b["host_prefills"] == 1
        and report["targets"] == 4096
        and "nll_sums" not in report
        and "fit_statistics_s" not in a["seconds"]
    )
    assert (
        report["work"] == a["work"] == b["work"]
        and z["targets"] == 4096
        and z["documents"] == 1
        and all(v == opt.freeze()["beta"][k] for k, v in z["beta"].items())
    )
    assert all(torch.equal(m.state_dict()[k], v) for k, v in m0.items()) and all(
        torch.equal(head.state_dict()[k], v) for k, v in h0.items()
    )
    assert a["work"]["mark_storage_bytes"] == 4096
    with pytest.raises(ValueError):
        fit.document(m, head, word, fit=opt, beta=z["beta"], width=8)
    with pytest.raises(ValueError):
        fit.document(m, head, word[:-1], fit=opt, width=8)
