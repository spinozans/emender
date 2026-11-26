"""Invented data; independent one-frontier tape, gradient, membership and cost witnesses."""

import hashlib
import numpy as np
import pytest
import torch

from experiments.expressivity_tasks.x50_read_tape import program as old
from experiments.expressivity_tasks.x50_read_tape.test_read_tape import Fixture
from . import data as d, study as s
from .reader import ARMS, T, acquire, predict, unigram_counts, unigram_log


@pytest.fixture
def packet():
    torch.set_num_threads(1)
    torch.manual_seed(413)
    m = old.Tape().double()
    with torch.no_grad():
        m.emit.weight.normal_(std=0.12)
    x = torch.tensor(
        [[65, 255, 0, 66, 30, 66], [65, 66, 66, 65, 67, 90]], dtype=torch.long
    )
    p = torch.rand(*x.shape, 256, dtype=torch.float64).softmax(-1)
    u = torch.arange(1, 257, dtype=torch.float64).log_softmax(-1)
    return m, x, p, u


@pytest.mark.parametrize("arm", ARMS)
def test_explicit_every_prefix_logit_matches_single_actual_tape(packet, arm):
    m, x, p, u = packet
    z, _ = predict(m, x, None if arm == "NONE" else p, arm, u)
    for t in range(x.shape[1]):
        rows = []
        for j in range(t + 1):
            obs = x[:, j]
            pos = torch.tensor(j)
            rows.append(m.row(0, obs, pos))
            visible = arm == "KEEP" or (arm == "DISCARD" and t == j)
            rows.append(m.row(1, obs, pos, p[:, j] if visible else None))
        expected = m.emit(m.attend(torch.stack(rows, dim=1))[:, -1]) + (
            u if arm == "NONE" else old.log_prior(p[:, t])
        )
        torch.testing.assert_close(z[:, t], expected, atol=2e-12, rtol=3e-12)
        pref, _ = predict(
            m, x[:, : t + 1], None if arm == "NONE" else p[:, : t + 1], arm, u
        )
        torch.testing.assert_close(z[:, t], pref[:, -1], atol=2e-12, rtol=3e-12)


def test_keep_is_x50_one_function_and_gradients(packet):
    m, x, p, u = packet
    p.requires_grad_()
    z, _ = predict(m, x, p, "KEEP", u)
    ref, _ = old.predict(m, old.Trace("ONE", x, p, None, None, None))
    torch.testing.assert_close(z, ref, rtol=3e-12, atol=2e-12)
    vec = torch.randn_like(z)
    args = (*m.parameters(), p)
    a = torch.autograd.grad((vec * z).sum(), args, allow_unused=True)
    b = torch.autograd.grad((vec * ref).sum(), args, allow_unused=True)
    for i, j in zip(a, b, strict=True):
        if i is None:
            assert j is None
        else:
            torch.testing.assert_close(i, j, rtol=3e-10, atol=2e-10)


def test_past_content_intervention_and_gradient_but_not_current_or_first(packet):
    m, x, p, u = packet
    p.requires_grad_()
    other = p.detach().clone()
    other[:, :-1] = other[:, :-1].roll(11, -1)
    k, _ = predict(m, x, p, "KEEP", u)
    c, _ = predict(m, x, p, "DISCARD", u)
    k2, _ = predict(m, x, other, "KEEP", u)
    c2, _ = predict(m, x, other, "DISCARD", u)
    assert (
        torch.equal(k[:, 0], c[:, 0])
        and torch.equal(c[:, -1], c2[:, -1])
        and (k[:, -1] - k2[:, -1]).abs().max() > 1e-6
    )
    for z, retained in [(k, True), (c, False)]:
        (g,) = torch.autograd.grad(z[:, -1, 23].sum(), p)
        assert (
            bool(g[:, :-1].abs().max() > 1e-8) == retained and g[:, -1].abs().max() > 0
        )


def test_no_read_capability_refusal_and_prior_is_only_shifted_fit_targets(packet):
    m, x, p, u = packet
    inputs = x.numpy().astype(np.uint8)
    x0, p0 = acquire(None, inputs, "NONE")
    assert p0 is None
    with pytest.raises(ValueError):
        acquire(object(), inputs, "NONE")
    with pytest.raises(ValueError):
        predict(m, x, p, "NONE", u)
    with pytest.raises(ValueError):
        acquire(None, inputs, "DISCARD")
    _, a = predict(m, x0, None, "NONE", u)
    torch.testing.assert_close(a, u.expand_as(a))
    fit = np.zeros((2, T + 1), np.uint8)
    fit[:, 0] = 255
    fit[1, -1] = 30
    counts = unigram_counts(fit)
    assert counts[0] == 2 * T - 1 and counts[255] == 0 and counts[30] == 1
    np.testing.assert_allclose(
        unigram_log(counts).exp(), (counts + 0.5) / (2 * T + 128), rtol=1e-6
    )


def test_matched_fits_current_digests_drop_cost_and_no_extra_reads():
    torch.set_num_threads(1)
    rng = np.random.default_rng(41)
    win = rng.choice(np.array([0, 30, 65, 66, 255], np.uint8), size=(16, T + 1))
    u = unigram_log(unigram_counts(win))
    ids = []
    for arm in ARMS:
        m = s.model_at_birth(d.SEEDS[0])
        tick = []
        pages = Fixture() if arm != "NONE" else None
        f, _ = s.phase(m, pages, win, u, d.SEEDS[0], arm, True, tick.append)
        p, score = s.phase(m, pages, win, u, d.SEEDS[0], arm, False, tick.append)
        assert (
            tick == [16 * T] * 4
            and f["archive"]["parent_reads"] == (0 if arm == "NONE" else 3 * 16 * T)
            and f["archive"]["child_reads"] == p["archive"]["child_reads"] == 0
        )
        assert (
            p["archive"]["parent_reads"] == (0 if arm == "NONE" else 16 * T)
            and p["cost"]["chosen_wall_s"] > 0
        )
        assert (
            m.emit.weight.abs().sum() > 0
            and ("drop" in score) == (arm == "KEEP")
            and (p["cost"]["drop_wall_s"] > 0) == (arm == "KEEP")
        )
        ids.append(
            (f["current_answer_and_prior_sha"], p["current_answer_and_prior_sha"])
        )
    assert ids[0] == (None, None) and ids[1] == ids[2] and ids[1][0] != ids[1][1]


def test_rank_contiguity_and_foreign_predecessor_rejected():
    from experiments.expressivity_tasks.x49_live_trace_memory.panels import select

    rec = []
    n = 0
    while len(rec) < 12:
        h = hashlib.sha256(str(n).encode()).hexdigest()
        n += 1
        if int(h, 16) % 100 < 2:
            continue
        off = d.MIN_OFFSET + 10000 * len(rec)
        rec.append(
            dict(
                sha256=h, document=off, byte_offset=off, byte_length=5000, split="train"
            )
        )
    size = rec[-1]["document"] + 10000
    ex = {rec[3]["sha256"]}
    allrows, _ = select(rec, ex, {"train": 9}, size)
    a = allrows["train"]
    z, exnew, _ = d.next_block(rec, ex, a[:4], 5, size)
    assert z == a[4:] and len(exnew) == 5
    with pytest.raises(ValueError, match="contiguous"):
        d.next_block(rec, ex, a[1:5], 5, size)
    with pytest.raises(ValueError, match="blocked"):
        d.next_block(rec, ex | {a[2]["sha256"]}, a[:4], 5, size)


def test_probe_cannot_open_before_all_states_and_independent_ack(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "RUN", tmp_path)
    monkeypatch.setattr(d, "qualified", lambda: {})
    with pytest.raises(FileNotFoundError, match="frozen.json"):
        d.extract("probe")
    monkeypatch.setattr(d, "frozen", lambda: {"fit_targets": 123})
    with pytest.raises(FileNotFoundError, match="freeze-ack.json"):
        d.extract("probe")
    assert not list(tmp_path.iterdir())


def test_paired_intervals_independent_multiplicity_and_mixed_seed():
    rng = np.random.default_rng(55)
    v = rng.random((3, 9)) * T
    curves = dict(
        KEEP=v.copy(),
        DISCARD=v + 1,
        NONE=v + 2,
        DROP=v - 0.3,
        ARCHIVE_PRIOR=v,
        UNIGRAM=v,
    )
    curves["DROP"][1] += 1
    pol, c = s.compare(curves, draws=501)
    _, b = s.compare(curves, draws=501, multiplicity=True)
    for k in c:
        np.testing.assert_allclose(c[k]["ci95"], b[k]["ci95"], atol=1e-15, rtol=1e-12)
    assert (
        c["KEEP:DISCARD"]["decision"] == c["DISCARD:NONE"]["decision"] == "win"
        and c["KEEP:DROP"]["decision"] == "unresolved"
    )
    curves["KEEP"] += 2
    assert s.compare(curves, draws=3)[1]["KEEP:DISCARD"]["decision"] == "loss"
    assert np.isclose(pol["DISCARD"]["mean"] - pol["KEEP"]["mean"], 1 / T)
