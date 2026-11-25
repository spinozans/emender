"""X49-V synthetic oracles. No natural text, updates, leased model execution or source reservations."""

import pytest
import numpy as np
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    alternative as a,
    alternative_fit as fit,
    local_u,
    surprise as u,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_surprise import (
    example,
    scalar_code,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import oracle
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import frozen_prefill


def fixture():
    head, w, h, z = example()
    z[::2, 23] = 13.0
    z[1::2, 241] = 11.0
    z[4::9, 47] = 19.0
    for j in range(1, len(w), 7):
        z[j - 1, w[j]] = 22.0  # old top sometimes equals the observed byte
    beta = local_u.contract(4902)[1]
    return head, w, h, z, beta


def test_local_u_is_identical_to_confirmed_readout_without_rotation_or_other_responses(
    monkeypatch, tmp_path
):
    head, w, _, _, b = fixture()
    host = Tiny().eval()
    h, z, _ = frozen_prefill(host, w, 8)
    ref = u.response(u.fields(head, w, h, z), "U", b)

    def forbidden(*args):
        raise AssertionError("U alone must not search for a wrong record")

    monkeypatch.setattr(u, "_rotation", forbidden)
    p = u.bound_fields(head, w, h, z)
    assert set(p.phi) == {"U"} and "S_mark_reads" not in p.work
    assert torch.equal(local_u._predict(host, head, b, w, 8), ref)
    once = local_u._predict(host, head, b, b"\xff", 8)
    assert once.shape == (1, 256) and not once.requires_grad
    for bad in (b"", bytearray(w), b"z" * 4097):
        with pytest.raises(ValueError):
            local_u._predict(host, head, b, bad, 8)
    assert len(set(local_u.contract(s) for s in (4901, 4902, 4903))) == 3
    with pytest.raises(ValueError, match="unconfirmed"):
        local_u.contract(1)
    f = tmp_path / "changed"
    f.write_bytes(local_u.RECEIPT.read_bytes() + b" ")
    monkeypatch.setattr(local_u, "RECEIPT", f)
    with pytest.raises(ValueError, match="changed U"):
        local_u.contract(4901)


def test_previous_forecast_runner_up_ties_binary_bytes_and_separate_tape():
    z = torch.zeros(6, 256)
    z[0, 255] = 100
    z[0, 30] = z[0, 3] = 99
    z[1, 1] = z[1, 0] = 5
    z[2, 0] = 10
    z[3, 30] = z[3, 2] = 4
    ids = torch.tensor([1, 255, 0, 30, 2, 255])
    t = a._write_alternatives(z, ids)
    assert (
        t.dtype == torch.uint8 and t.nbytes == 6 and t.tolist() == [0, 3, 1, 0, 30, 0]
    )
    z[2, 100] = 11
    assert (t != a._write_alternatives(z, ids)).nonzero().flatten().tolist() == [3]
    assert a._write_alternatives(z[-1:], ids[:1]).tolist() == [0]
    head, w, h, z, b = fixture()
    p = a.fields(head, w, h, z, b)
    ref = u.fields(head, w, h, z)
    assert torch.equal(p.base, u.response(ref, "U", b)) and "S_mark_reads" not in p.work
    disp = change = 0
    for i, t in enumerate(p.query.tolist()):
        j, _, _, k = oracle(w, t, 4096)
        c = w[j]
        assert w[k] == c == p.copy[i].item()
        non = [v for v in range(256) if v != c]
        expected = [
            max(non, key=lambda v: float(row[v]))
            for row in [p.base[t], z[j - 1], z[k - 1]]
        ]
        for arm, s in zip("DVK", expected, strict=True):
            block = p.blocks[arm][-1]
            assert block.byte[i] == s
            lp = p.base[t, s] - p.base[t, non].logsumexp(0)
            assert (
                block.feature[i].tolist() == [1, scalar_code(-lp) / 64]
                and abs(float(block.probability[i]) - float(lp.exp())) < 1e-13
            )
        disp += j != k
        change += expected[1] != expected[2]
    n = len(p.query)
    assert (
        p.work["K_slots_displaced"] == disp >= change == p.work["K_codes_changed"] > 0
        and 0 < p.work["V_equals_current"] < n
    )
    assert p.work["alternatives_encoded"] == len(w) - 1 and p.work[
        "alternative_storage_bytes"
    ] == p.work["mark_storage_bytes"] == len(w)
    assert (
        p.work["V_alternative_reads"]
        == p.work["K_alternative_reads"]
        == p.work["U_mark_reads"]
        == p.work["literal_value_reads"]
        == n
    )


def dense_design(p, arm):
    t = p.query.clone()
    row = torch.arange(len(t))
    F = torch.zeros(len(t), 256, a.DIMS[arm], dtype=torch.double)
    for j, x in enumerate(p.blocks[arm]):
        F[row, x.byte, 2 * j : 2 * j + 2] = x.feature.clone()
    mask = torch.ones(len(t), 256, dtype=torch.bool)
    mask[row, p.copy] = False
    return F, mask


def test_dense_conditional_score_gradient_curvature_including_proposal_collisions_and_misses():
    head, w, h, z, b = fixture()
    p = a.fields(head, w, h, z, b)
    y = bytearray(w[1:] + b"!")
    q = p.query.tolist()
    v = p.blocks["V"][-1].byte.tolist()
    for i, t in enumerate(q):
        y[t] = w[t] if i % 3 == 0 else (int(p.copy[i]) if i % 3 == 1 else v[i])
    ys = bytes(y)
    label = torch.tensor(y, dtype=torch.long)
    baseline = p.base.clone()
    m = fit.Moments()
    cnt, _ = m.add(p, ys)
    miss = a.fields(head, b"\0", h[:1], z[:1], b)
    cnt0, _ = m.add(miss, b"\xff")
    assert (
        not any(cnt0.values())
        and 0 < cnt["copy_correct"] < len(q)
        and cnt["hit_noncopy"] == len(q) - cnt["copy_correct"]
    )
    end = m.freeze()
    n = len(ys) + 1
    for arm in a.DIMS:
        F, mask = dense_design(p, arm)
        t = p.query.clone()
        stored = baseline[t]
        masked = stored.masked_fill(~mask, -torch.inf)
        missing = ~torch.isin(torch.arange(len(ys)), t)

        def oracle_lp(theta):
            bumped = stored + F @ theta
            shift = bumped.masked_fill(~mask, -torch.inf).logsumexp(
                -1
            ) - masked.logsumexp(-1)
            return baseline.index_copy(
                0, t, torch.where(mask, bumped - shift[:, None], stored)
            )

        def objective(theta):
            return (
                -oracle_lp(theta)[torch.arange(len(ys)), label].sum()
                - miss.base[0, 255]
            ) / n + (theta @ theta) * end["ridge"] / 2

        theta = torch.zeros(a.DIMS[arm], dtype=torch.double, requires_grad=True)
        g = torch.autograd.grad(objective(theta), theta)[0]
        H = torch.autograd.functional.hessian(objective, theta)
        np.testing.assert_allclose(m.rhs[arm] / n, -g.numpy(), atol=1e-13, rtol=1e-12)
        np.testing.assert_allclose(
            m.gram[arm] / n + end["ridge"] * np.eye(len(theta)),
            H.numpy(),
            atol=1e-13,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            end["beta"][arm],
            np.linalg.solve(H.numpy(), -g.numpy()),
            atol=1e-10,
            rtol=1e-12,
        )
        assert torch.equal(a.response(p, arm, theta), baseline)
        v = torch.arange(len(theta), dtype=torch.double) * -0.75 + 1.0
        out = a.response(p, arm, v)
        torch.testing.assert_close(out, oracle_lp(v), rtol=0, atol=3e-14)
        assert torch.equal(out[t, p.copy], baseline[t, p.copy]) and torch.equal(
            out[missing], baseline[missing]
        )
        torch.testing.assert_close(
            out[t].masked_fill(~mask, -torch.inf).logsumexp(-1),
            masked.logsumexp(-1),
            atol=3e-14,
            rtol=0,
        )
    # Saturating non-copy feature and extreme (but finite) logits; never subtract the confident copy's exp from 1.
    other = torch.full((1, 256), -1000.0, dtype=torch.double)
    other[0, 255] = -torch.inf
    other[0, 0] = -100
    prop = a._proposal(other, other.logsumexp(-1), torch.tensor([1]))
    assert prop.feature.tolist() == [[1, 255 / 64]]
    counts, sums, _ = fit.score(p, end["beta"], ys)
    assert counts == cnt
    for arm, score in sums.items():
        lp = baseline if arm == "U" else a.response(p, arm, end["beta"][arm])
        assert score == float(-lp[torch.arange(len(ys)), label].sum())
    with pytest.raises(ValueError):
        m.add(p, ys[:-1])


def test_prefix_and_tail_mutation_leave_answers_closed():
    head, w, h, z, b = fixture()
    p = a.fields(head, w, h, z, b)
    n = 111
    z2 = z.clone()
    z2[n:] = z[n:].flip(-1)
    h2 = h.clone()
    h2[n:] += 5
    f = a.fields(head, w[:n] + bytes([255]) * (len(w) - n), h2, z2, b)
    short = a.fields(head, w[:n], h[:n], z[:n], b)
    for arm, d in a.DIMS.items():
        vec = np.linspace(-1.25, 0.75, d)
        full = a.response(p, arm, vec)
        assert torch.equal(full[:n], a.response(f, arm, vec)[:n])
        torch.testing.assert_close(
            full[:n], a.response(short, arm, vec), rtol=0, atol=4e-6
        )
        assert torch.equal(
            short.blocks[arm][-1].byte, p.blocks[arm][-1].byte[p.query < n]
        )


def test_full_window_one_prefill_fit_then_frozen_scoring():
    head, w, _, _, b = fixture()
    host = Tiny().eval()
    word = (w * 30)[:4097]
    seen = []
    hook = host.register_forward_pre_hook(lambda _m, x: seen.append(x[0].clone()))
    m = fit.Moments()
    try:
        x = fit.document(host, head, word, u_beta=b, fit=m, width=8)
        sealed = m.freeze()
        r = fit.document(host, head, word, u_beta=b, beta=sealed["beta"], width=8)
        r2 = fit.document(
            host, head, word[:-1] + b"\xe9", u_beta=b, beta=sealed["beta"], width=8
        )
    finally:
        hook.remove()
    assert len(seen) == 3 and all(
        torch.equal(row, torch.tensor([list(word[:-1])])) for row in seen
    )
    assert (
        x["host_prefills"] == r["host_prefills"] == 1
        and x["targets"] == r["targets"] == sealed["targets"] == 4096
        and sealed["documents"] == 1
    )
    assert (
        x["work"] == r["work"] == r2["work"]
        and set(r["nll_sums"]) == set("UDVK")
        and "nll_sums" not in x
        and m.freeze() == sealed
    )
    for bad in ({}, {"fit": m, "beta": sealed["beta"]}):
        with pytest.raises(ValueError):
            fit.document(host, head, word, u_beta=b, width=8, **bad)
    with pytest.raises(ValueError):
        fit.document(host, head, word[:-1], u_beta=b, fit=m, width=8)
