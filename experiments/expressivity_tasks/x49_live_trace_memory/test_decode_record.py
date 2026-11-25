"""DR fits and all-256 response oracles on invented binary streams; no corpus or GPU."""

from math import sqrt
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    decode_fit as fit,
    decode_record as dr,
    joint_reader as joint,
    local_u,
    literal_mark as ls,
    test_joint_reader as native_fixture,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
    oracle,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_surprise import (
    scalar_code,
)
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import frozen_prefill
from experiments.expressivity_tasks.x49_live_trace_memory.compact import evidence

binary = native_fixture.binary


def test_all_write_centered_fit_vs_augmented_least_squares_and_fallback():
    rng = np.random.default_rng(4913)
    mom = fit.Moments()
    with pytest.raises(ValueError, match="empty"):
        mom.solve()
    aa, vv, cc = [], [], []
    for d in range(5):
        # 37 appears only as the unreadable first slot; 128 occurs exactly once.
        ids = np.r_[
            37, rng.choice([1, 0, 255, 17], size=83), [128] if d == 0 else []
        ].astype(np.uint8)
        marks = rng.integers(0, 256, (len(ids), 2), dtype=np.uint8)
        marks[ids == 1] = [17, 255]  # one exactly constant cell
        marks[ids == 17, 1] = 239  # constant in only one column
        marks[ids == 255, 0] = marks[ids == 255, 1]  # perfect collinearity
        marks[0] = 0
        v = np.tanh(rng.normal(size=(len(ids), 32))).astype(np.float32)
        n = mom.add(ids.tobytes(), *torch.from_numpy(marks.T), torch.from_numpy(v))
        assert n == len(ids) - 1
        aa.extend(marks[1:])
        vv.extend(v[1:])
        cc.extend(ids[1:])
    a, v, c = (
        np.array(aa).astype(np.float64) / 64,
        np.array(vv).astype(np.float64),
        np.asarray(cc),
    )
    dec = mom.solve()
    assert (
        np.array_equal(mom.count, np.bincount(c, minlength=256))
        and mom.count[128] == 1
        and mom.count[37] == 0
    )
    assert sum(x.nbytes for x in dec.arrays().values()) == 100352
    for i in range(256):
        mask = c == i
        m = v[mask].mean(0) if mask.any() else v.mean(0)
        b = a[mask].mean(0) if mask.any() else np.zeros(2)
        if mask.any():
            # Independent dense augmented system, not normal equations or per-cell raw sums.
            X = np.concatenate((a[mask] - b, np.eye(2) * sqrt(mask.sum() / 4096)))
            Y = np.concatenate((v[mask] - m, np.zeros((2, 32))))
            K = np.linalg.lstsq(X, Y, rcond=None)[0]
        else:
            K = np.zeros((2, 32))
        np.testing.assert_array_equal(dec.m[i].numpy(), m.astype(np.float32))
        np.testing.assert_array_equal(dec.b[i].numpy(), b.astype(np.float32))
        np.testing.assert_allclose(
            dec.k[i].numpy(), K.astype(np.float32), rtol=2e-7, atol=3e-13
        )
    assert (
        torch.count_nonzero(dec.k[[1, 128, 37]])
        == torch.count_nonzero(dec.k[17, 1])
        == 0
    )
    assert torch.count_nonzero(dec.k[17, 0]) > 0
    with pytest.raises(ValueError, match="unaligned"):
        mom.add(
            b"x",
            torch.tensor([1], dtype=torch.uint8),
            torch.zeros(1, dtype=torch.uint8),
            torch.zeros(1, 32),
        )
    with pytest.raises(ValueError, match="invalid frozen"):
        fit.Decoder(dec.m.double(), dec.b, dec.k)


def setup():
    torch.manual_seed(4913)
    host = Tiny().eval()
    head = trained_base().eval()
    # A deliberate strong response to payload differences (even candidate confidence can change).
    with torch.no_grad():
        head.delta.weight.mul_(5)
        head.mix.weight[:, 8:40].mul_(4)
    dec = fit.Decoder(
        torch.rand(256, 32) * 1.9 - 0.95,
        torch.rand(256, 2) * 3.9,
        torch.randn(256, 2, 32) * 0.8,
    )
    ub = local_u.contract(4901)[1]
    jb = joint.contract(4901)[1]
    return host, head, dec, ub, jb


@torch.inference_mode()
def test_full_and_compact_all_256_vs_independent_dense_response(binary):
    host, head, dec, ub, jb = setup()
    writer = binary[-1]
    w = (b"color: \xff\0 ababa\x1e!" + b"color: blue " * 4) * 3
    panel = dr.panel_answers(host, head, w, dec, ub, jb, writer=writer, width=8)
    h, z, _ = frozen_prefill(host, w, 8)
    p = ls.fields(head, w, h, z)
    assert torch.equal(panel.scores["B"], p.base.double())
    for arm, beta in [("U", ub), ("J", jb)]:
        assert torch.equal(panel.scores[arm], ls.response(p, arm, beta))
    matches = [(t, r) for t in range(len(w)) if (r := oracle(w, t, 4096)) is not None]
    q = torch.tensor([t for t, _ in matches])
    j = torch.tensor([r[0] for _, r in matches])
    k = torch.tensor([r[3] for _, r in matches])
    ids = torch.tensor(list(w))
    c = ids[j]
    assert torch.equal(panel.query, q) and torch.equal(panel.candidate, c)
    # All old marks independently, then prototype / clipped decoded vectors for all input rows.
    text = ls.text_tape(w)[0]
    neural = torch.tensor(
        [0]
        + [
            scalar_code(-z[i - 1].double().log_softmax(0)[w[i]])
            for i in range(1, len(w))
        ],
        dtype=torch.uint8,
    )
    marks = torch.stack((text, neural), dim=-1).float() / 64
    decoded = (
        dec.m[ids]
        + ((marks - dec.b[ids])[:, :1] * dec.k[ids, 0])
        + ((marks - dec.b[ids])[:, 1:] * dec.k[ids, 1])
    ).clamp(-1, 1)
    assert (decoded.abs() == 1).any() and not torch.equal(decoded[j], decoded[k])
    truth = head.writer(torch.cat((h, head.byte(ids)), -1)).tanh()[j]
    f, *_ = evidence(w, 4096, device=h.device, rotate=False)
    bare, _ = head.direct(h, z, f)
    wstats = panel.work
    assert (
        wstats["mse_components"] == len(q) * 32
        and wstats["full_value_writes"] == len(w)
        and wstats["full_value_reads"] == len(q)
    )
    assert (
        wstats["sham_positions_displaced"] == int((j != k).sum())
        and wstats["sham_pairs_changed"] == int((marks[j] != marks[k]).any(1).sum()) > 0
    )
    for arm, payload in (("P", dec.m[c]), ("D", decoded[j]), ("S", decoded[k])):
        raw = head.answer(bare, h, f, q, head.byte(c), payload).double()
        # Direct dense logit offset + preserve original log mass. Neither B's raw scores
        # nor the wrong-occurrence trust note enters this oracle, even for S.
        dl = []
        for i, (t, _) in enumerate(matches):
            phi = [
                1,
                scalar_code(-raw[t].log_softmax(0)[c[i]]) / 64,
                scalar_code(-z[t].double().log_softmax(0)[c[i]]) / 64,
                *marks[j[i]].double().tolist(),
            ]
            dl.append(sum(a * b for a, b in zip(phi, jb, strict=True)))
        adjusted = raw.clone()
        adjusted[q, c] += torch.tensor(dl, dtype=torch.double)
        adjusted[q] += (raw[q].logsumexp(1) - adjusted[q].logsumexp(1))[:, None]
        torch.testing.assert_close(panel.scores[arm], adjusted, atol=2e-14, rtol=0)
        assert (
            abs(
                panel.squared_error_sums[arm]
                - float((truth.double() - payload.double()).square().sum())
            )
            < 1e-12
        )
    assert not torch.equal(panel.scores["D"], panel.scores["S"])
    y = w[1:] + b"\x00"
    row = dr.score(panel, y)
    assert row["candidate_correct"] == sum(y[t] == w[r[0]] for t, r in matches)
    for arm, x in panel.scores.items():
        assert np.isclose(
            row["nll_sums"][arm],
            sum(-x[t, b].item() for t, b in enumerate(y)),
            rtol=0,
            atol=5e-12,
        )


@torch.inference_mode()
def test_standalone_D_no_teacher_sham_or_sibling_and_prefix_reset(binary, monkeypatch):
    host, head, dec, ub, jb = setup()
    writer = binary[-1]
    w = (b"\x00aaaaababa\xff\x1e baba" * 18)[:241]
    ref = dr.panel_answers(host, head, w, dec, ub, jb, writer=writer, width=8)
    assert ref.work["literal_hits"] > 100

    def forbidden(*a, **kw):
        raise AssertionError("standalone D tried to use the teacher or sham")

    monkeypatch.setattr(dr.u, "_rotation", forbidden)
    monkeypatch.setattr(head.writer, "forward", forbidden)
    monkeypatch.setattr(head, "forward", forbidden)
    calls = []
    hook = host.register_forward_pre_hook(lambda m, x: calls.append(x[0].clone()))
    try:
        for v in (w, w[:122], w[:122] + b"z" * (len(w) - 122), b"\xff", w):
            o = dr.compact_predict(host, head, v, dec, jb, writer=writer, width=8)
            if v == w:
                assert torch.equal(o, ref.scores["D"])
            if v[:122] == w[:122]:
                torch.testing.assert_close(
                    o[:122], ref.scores["D"][:122], rtol=0, atol=3e-6
                )
            assert not o.requires_grad and torch.equal(
                calls[-1], torch.tensor([list(v)])
            )
    finally:
        hook.remove()
    assert len(calls) == 5


def test_fit_uses_all_real_writes_not_next_label_or_later_hit(binary, monkeypatch):
    host, head, _, ub, jb = setup()
    w = (bytes(range(256)) * 17)[:4096]
    writer = binary[-1]
    mom = fit.Moments()
    expected = fit.Moments()

    def forbidden(*a, **kw):
        raise AssertionError(
            "FIT tried to score a target, look up an address or fit a readout on later hits"
        )

    for name in ("evidence", "score", "_trust"):
        monkeypatch.setattr(dr, name, forbidden)
    kw = dict(ub=ub, jb=jb, writer=writer, width=8)
    one = dr.document(host, head, w + b"\xff", fit=mom, **kw)
    two = dr.document(host, head, w + b"\0", fit=expected, **kw)
    for key, val in mom.arrays().items():
        np.testing.assert_array_equal(val, expected.arrays()[key])
    assert (
        "nll_sums" not in one
        and one["host_prefills"] == two["host_prefills"] == 1
        and one["work"] == two["work"]
    )
    assert one["work"]["fit_writes"] == int(mom.count.sum()) == 4095 and np.array_equal(
        mom.count, np.bincount(list(w[1:]), minlength=256)
    )
    with torch.no_grad():
        h, z, _ = frozen_prefill(host, w, 8)
        allv = head.writer(torch.cat((h, head.byte(torch.tensor(list(w)))), -1)).tanh()
    np.testing.assert_allclose(
        mom.v.sum(axis=0), allv[1:].double().sum(0).numpy(), rtol=1e-14, atol=1e-13
    )
    with pytest.raises(ValueError, match="one full"):
        dr.document(host, head, w, fit=mom, **kw)


def test_misses_unchanged_and_no_label_until_all_six_answers(binary, monkeypatch):
    host, head, dec, ub, jb = setup()
    writer = binary[-1]
    word = (b"\xff\0abac\x1e" * 600)[:4097]
    events = []
    fn = dr.panel_answers
    sc = dr.score

    def panel(*args, **kw):
        assert args[2] == word[:-1]
        a = fn(*args, **kw)
        events.append(set(a.scores))
        return a

    def checked(a, y):
        assert events == [set(dr.ARMS)] and y == word[1:]
        events.append("score")
        return sc(a, y)

    monkeypatch.setattr(dr, "panel_answers", panel)
    monkeypatch.setattr(dr, "score", checked)
    row = dr.document(
        host, head, word, decoder=dec, ub=ub, jb=jb, writer=writer, width=8
    )
    assert (
        events[-1] == "score"
        and row["targets"] == 4096
        and sum(row["work"][k] for k in ("literal_hits", "literal_misses"))
        == row["targets"]
    )
    a = fn(host, head, bytes(range(64)), dec, ub, jb, writer=writer, width=8)
    assert (
        a.work["literal_hits"] == 0
        and all(torch.equal(a.scores["B"], v) for v in a.scores.values())
        and a.squared_error_sums == dict(P=0, D=0, S=0)
    )
