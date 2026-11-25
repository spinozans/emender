"""Independent EH full-score and dense likelihood-derivative oracles on invented text."""

from math import isclose
import numpy as np
import pytest
import torch
from torch import nn
from experiments.expressivity_tasks.x49_live_trace_memory import (
    error_history as eh,
    error_history_fit as fit,
    literal_mark as ls,
    joint_reader as joint,
    local_u,
    test_joint_reader as native_fixture,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_error_history_index import (
    oracle,
    word,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
)
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import frozen_prefill

binary = native_fixture.binary


class Host(nn.Module):
    """Position-dependent wrong guesses, including binary and correct arrivals. Prefix-only."""

    def __init__(self):
        super().__init__()
        self.freq = nn.Parameter(torch.linspace(0.11, 0.83, 8))

    def forward(self, x):
        t = torch.arange(x.shape[1])
        c = x.flatten()
        h = (self.freq * (t[:, None] + c[:, None])).sin()
        z = torch.zeros(len(t), 256)
        a = torch.where(t % 4 == 0, torch.tensor(32), 200 + t % 5)
        z[t, a] = 3.0
        return torch.cat((h, z), dim=-1)[None, :, :]


def fixture(w, writer):
    torch.manual_seed(4915)
    host = Host().eval()
    head = trained_base().eval()
    h, z, _ = frozen_prefill(host, w, 8)
    ub = local_u.contract(4902)[1]
    jb = joint.contract(4902)[1]
    p = eh.fields(head, w, h, z, jb, writer, ub=ub)
    return host, head, h, z, ub, jb, p


def oracle_features(w, z, base):
    literal = bytes(ls.text_tape(w)[0].tolist())
    ids = list(w)
    lp = z.double().log_softmax(-1)

    def quant(s):
        return torch.floor(16 * s.clamp_min(0) + 0.5).clamp_max(255)

    n = bytes([0, *quant(-lp[torch.arange(len(w) - 1), ids[1:]]).int().tolist()])
    g = bytes([0, *z[:-1].max(1).indices.tolist()])
    counts = torch.from_numpy(oracle(w, literal, n, g)[0]).double()
    total = counts[:, :, 0].sum(1).clamp_min(1)[:, None]
    freq = counts[:, :, 0] / total
    return torch.stack(
        (
            freq,
            freq * (quant(-base.log_softmax(-1)) / 64),
            counts[:, :, 1] / (64 * total),
            counts[:, :, 2] / (64 * total),
            counts[:, :, 3] / total,
            -counts[:, :, 4] / total,
            -counts[:, :, 5] / total,
        ),
        dim=2,
    )


def arm(x, a):
    if a in ("SHIFT", "SWAP"):
        return x[:, :, [0, 1, 2, 3, 4, 6]]
    return x[:, :, : eh.DIMS[a]]


def test_dense_all_256_named_answers_support_and_conditional_sham(binary):
    w = word()
    _, head, h, z, ub, jb, p = fixture(w, binary[-1])
    ref = ls.fields(head, w, h, z)
    base = ls.response(ref, "J", jb)
    assert (
        torch.equal(p.base, base)
        and torch.equal(p.anchors["U"], ls.response(ref, "U", ub))
        and torch.equal(p.anchors["B"], ref.base.double())
    )
    x = oracle_features(w, z, base)
    np.testing.assert_array_equal(p.x, x[p.row, p.symbol])
    active = x[:, :, 0].sum(1) > 0
    changed = (x[:, :, 5] != x[:, :, 6]).any(1)
    assert (
        active.any()
        and changed.any()
        and p.work["changed_histogram_rows"] == int(changed.sum())
    )
    coeff = {
        a: torch.linspace(-0.4, 0.7, d, dtype=torch.float64).tolist()
        for a, d in eh.DIMS.items()
    }
    values = dict(p.anchors)
    for a in (*eh.DIMS, "SWAP"):
        v = coeff["ERROR" if a == "SWAP" else a]
        values[a] = eh.response(p, a, v)
        delta = arm(x, a) @ torch.tensor(v, dtype=torch.float64)
        expected = (
            base + delta - ((base + delta).logsumexp(-1) - base.logsumexp(-1))[:, None]
        )
        torch.testing.assert_close(values[a], expected, rtol=0, atol=2e-14)
        assert torch.equal(values[a][~active], base[~active])
        torch.testing.assert_close(
            base.exp().sum(1), values[a].exp().sum(1), rtol=0, atol=2e-14
        )
    torch.testing.assert_close(
        values["ERROR"][~changed], values["SWAP"][~changed], rtol=0, atol=2e-14
    )
    assert not torch.allclose(
        values["ERROR"][changed], values["SWAP"][changed], rtol=0, atol=1e-12
    )
    y = w[1:] + b"\xff"
    right, loss, _ = fit.score(p, coeff, y)
    assert set(loss) == set(eh.ARMS) and right == int(
        (torch.tensor(list(y))[ref.query] == ref.candidate).sum()
    )
    for a, r in loss.items():
        assert isclose(
            r,
            -float(values[a][torch.arange(len(w)), list(y)].sum()),
            abs_tol=1e-12,
            rel_tol=0,
        )
    with pytest.raises(ValueError):
        fit.score(p, dict(coeff, SWAP=coeff["ERROR"]), y)


def test_sparse_rhs_and_covariance_match_dense_crossentropy_autograd(binary):
    w = word()[:145]
    _, _, _, z, _, _, p = fixture(w, binary[-1])
    y = w[1:] + b"\x00"
    x = oracle_features(w, z, p.base)
    fitm = fit.Moments()
    fitm.add(p, y)
    sol = fitm.freeze()
    lab = torch.tensor(list(y))
    idx = torch.arange(len(w))
    assert (sol["documents"], sol["targets"]) == (1, len(y))
    for a, d in eh.DIMS.items():
        xf = arm(x, a)
        v = torch.zeros(d, dtype=torch.float64, requires_grad=True)

        def objective(b):
            return -(p.base + xf @ b).log_softmax(-1)[idx, lab].sum()

        rhs = -torch.autograd.grad(objective(v), v)[0]
        cv = torch.autograd.functional.hessian(objective, v)
        np.testing.assert_allclose(fitm.rhs[a], rhs, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(fitm.gram[a], cv, rtol=1e-12, atol=1e-13)
        dense = torch.linalg.solve(
            cv / len(y) + torch.eye(d, dtype=torch.float64) / 4096, rhs / len(y)
        )
        np.testing.assert_allclose(
            sol["beta"][a], dense.detach(), rtol=1e-10, atol=1e-10
        )


def test_standalone_error_has_no_sham_or_extra_responses_and_obeys_prefix(
    binary, monkeypatch
):
    writer = binary[-1]
    w = word()
    host, head, _, _, _, jb, p = fixture(w, writer)
    beta = [0.19, -0.21, 0.16, 0.35, -0.11, 0.26]
    want = eh.response(p, "ERROR", beta)

    def forbid(*a, **kw):
        raise AssertionError("standalone correct read used a comparison-only path")

    monkeypatch.setattr(eh.eh, "displaced_counts", forbid)
    monkeypatch.setattr(eh.eh, "_donors", forbid)
    monkeypatch.setattr(eh.u, "response", forbid)
    assert torch.allclose(
        eh.predict(host, head, w, jb, beta, writer, width=8), want, rtol=0, atol=2e-14
    )
    for end in [1, 17, 28, 81, len(w) - 1]:
        torch.testing.assert_close(
            eh.predict(host, head, w[:end], jb, beta, writer, width=8),
            want[:end],
            rtol=0,
            atol=3e-6,
        )
    w = bytes(range(100))
    base = eh.predict(host, head, w, jb, beta, writer, width=8)
    assert torch.equal(base, eh.predict(host, head, w, jb, [0] * 6, writer, width=8))
