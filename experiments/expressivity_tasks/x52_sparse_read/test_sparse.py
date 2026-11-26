"""Invented windows only; examine what is requested, what is visible, and what is scored."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks.x50_read_tape.kernel import Kernel
from experiments.expressivity_tasks.x50_read_tape.program import Tape, log_prior
from experiments.expressivity_tasks.x50_read_tape.test_read_tape import (
    Fixture,
    reference,
)
from experiments.expressivity_tasks.x51_read_retention import reader as dense
from . import data as d, study as s
from .reader import ARMS, Packet, acquire, predict, schedule


class Requests(Fixture):
    def __init__(self):
        super().__init__(floor=2)
        self.keys = []

    def suffix(self, raw):
        self.keys.append(raw)
        return super().suffix(raw)


def test_only_scheduled_suffixes_requested_no_dense_scan_or_fetching_gap_answers():
    pages = Requests()
    bank = Kernel(pages)
    bank.scan = lambda *_: pytest.fail("dense scan in sparse path")
    raw = np.array(
        [list(b"ABAYAYAYAYABACZZ"), list(b"\xff\x00\x1e\xffABYZ\x00QQQABQQ")], np.uint8
    )
    p = acquire(bank, raw, "KEEP4")
    at = (0, 4, 8, 12)
    assert p.at == at
    assert pages.keys == [
        row[max(0, t - 7) : t + 1].tobytes() for row in raw for t in at
    ]
    assert (
        len(pages.keys) == bank.counts["parent_reads"] == 8
        and bank.counts["child_reads"] == 0
    )
    expected = [
        [reference(row[: t + 1].tobytes(), pages.ids) for t in at] for row in raw
    ]
    np.testing.assert_array_equal(p.pdf, expected)
    every = acquire(Kernel(Requests()), raw, "EVERY")
    np.testing.assert_array_equal(p.pdf, every.pdf[:, ::4])
    assert bank.counts["prefix_frames"] == 8


def test_capability_and_invalid_packet_fail_closed():
    x = np.arange(9, dtype=np.uint8)[None, :]
    p = acquire(None, x, "NONE")
    assert p.at == () and p.pdf is None
    for bank, arm in [(None, "KEEP4"), (object(), "NONE"), (None, "bad")]:
        with pytest.raises(ValueError):
            acquire(bank, x, arm)
    with pytest.raises(ValueError):
        acquire(None, np.zeros((2, 0), np.uint8), "NONE")
    with pytest.raises(ValueError):
        predict(
            Tape(),
            replace(p, at=(4,), pdf=torch.zeros(1, 1, 256)),
            "KEEP4",
            torch.zeros(256),
        )


@pytest.fixture
def sample():
    torch.set_num_threads(1)
    torch.manual_seed(825)
    m = Tape().double()
    x = torch.randint(0, 256, (2, 11))
    p = torch.randn(*x.shape, 256, dtype=torch.double).softmax(-1)
    u = torch.randn(256, dtype=torch.double).log_softmax(0)
    with torch.no_grad():
        m.emit.weight.normal_(std=0.1)
    return m, x, p, u


@pytest.mark.parametrize("arm", ARMS)
def test_prefixes_equal_explicit_tape_including_empty_gap_slots(sample, arm):
    m, x, p, u = sample
    at = tuple(schedule(arm, x.shape[1]))
    z, base = predict(m, Packet(x, at, p[:, at] if at else None), arm, u)
    for t in range(x.shape[1]):
        rows = []
        for j in range(t + 1):
            pos = torch.tensor(j)
            rows.append(m.row(0, x[:, j], pos))
            visible = j in at and (j == t or arm in ("KEEP4", "EVERY"))
            rows.append(m.row(1, x[:, j], pos, p[:, j] if visible else None))
        direct = log_prior(p[:, t]) if t in at else u.expand(len(x), 256)
        torch.testing.assert_close(base[:, t], direct, rtol=0, atol=0)
        expected = m.emit(m.attend(torch.stack(rows, 1))[:, -1]) + direct
        torch.testing.assert_close(z[:, t], expected, rtol=1e-11, atol=1e-12)
        seen = tuple(j for j in at if j <= t)
        pref, _ = predict(
            m, Packet(x[:, : t + 1], seen, p[:, seen] if seen else None), arm, u
        )
        torch.testing.assert_close(pref[:, -1], z[:, t], rtol=1e-11, atol=1e-12)


def test_every_equals_x51_keep_forward_and_backward(sample):
    m, x, p, u = sample
    p.requires_grad_()
    q = Packet(x, tuple(range(x.shape[1])), p)
    z, _ = predict(m, q, "EVERY", u)
    ref, _ = dense.predict(m, x, p, "KEEP", u)
    torch.testing.assert_close(z, ref, rtol=1e-12, atol=1e-12)
    v = torch.randn_like(z)
    args = (*m.parameters(), p)
    g1 = torch.autograd.grad((v * z).sum(), args, allow_unused=True)
    g2 = torch.autograd.grad((v * ref).sum(), args, allow_unused=True)
    for a, b in zip(g1, g2, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_gap_dependence_only_via_past_not_future_or_direct_prior(sample):
    m, x, p, u = sample
    at = (0, 4, 8)
    pdf = p[:, at].requires_grad_()
    q = Packet(x, at, pdf)
    k, prior = predict(m, q, "KEEP4", u)
    e, alt = predict(m, q, "DISCARD4", u)
    assert torch.equal(k[:, 0], e[:, 0]) and torch.equal(prior, alt)
    for z, live in [(k, True), (e, False)]:
        g = torch.autograd.grad(z[:, 6, 20].sum(), pdf)[0]
        assert (g[:, :2].abs().max() > 1e-8).item() == live and torch.count_nonzero(
            g[:, 2]
        ) == 0
    p2 = pdf.detach().clone()
    p2[:, 0] = p2[:, 0].roll(7, -1)
    k2, _ = predict(m, replace(q, pdf=p2), "KEEP4", u)
    e2, _ = predict(m, replace(q, pdf=p2), "DISCARD4", u)
    assert torch.equal(e[:, 1:], e2[:, 1:]) and not torch.equal(k[:, 1:4], k2[:, 1:4])


def test_fit_then_probe_cost_phase_support_and_paired_priors():
    rng = np.random.default_rng(823)
    w = rng.choice(np.array([0, 30, 65, 66, 255], np.uint8), (s.BATCH, d.T + 1))
    u = dense.unigram_log(dense.unigram_counts(w))
    stamps = []
    priors = {}
    for arm in ARMS:
        m = s.model_at_birth(d.SEEDS[0])
        facts = []
        tick = []
        for fit in (True, False):
            pages = None if arm == "NONE" else Requests()
            f, v = s.phase(m, pages, w, u, d.SEEDS[0], arm, fit, tick.append)
            facts.append(f)
            reads = {"NONE": 0, "EVERY": 1024, "KEEP4": 256, "DISCARD4": 256}[arm] * (
                3 if fit else 1
            )
            assert (
                f["archive"]["parent_reads"] == reads
                and f["archive"]["child_reads"] == 0
                and (f["cost"]["erase_s"] > 0)
                == ("erase" in v)
                == (arm == "KEEP4" and not fit)
                and f["cost"]["chosen_s"] >= f["cost"]["acquire_s"] > 0
            )
            if not fit:
                priors[arm] = v["prior"]
        assert tick == [1024] * 4 and m.emit.weight.abs().sum() > 0
        stamps.append(tuple(f["refresh_hash"] for f in facts))
    assert stamps[0] == (None, None) and stamps[1] == stamps[2] == stamps[3]
    np.testing.assert_array_equal(priors["EVERY"][:, 0], priors["KEEP4"][:, 0])
    np.testing.assert_array_equal(priors["KEEP4"][:, 1:], priors["NONE"][:, 1:])
    np.testing.assert_array_equal(priors["KEEP4"], priors["DISCARD4"])


def test_unseen_targets_only_and_window_reset():
    w = np.zeros((2, d.T + 1), np.uint8)
    w[:, 0] = 255
    w[1, -1] = 48
    counts = dense.unigram_counts(w)
    assert (
        counts.sum() == 128
        and counts[0] == 127
        and counts[48] == 1
        and counts[255] == 0
    )
    p = acquire(Kernel(Fixture()), w[:, :-1], "DISCARD4")
    w[1, -1] = 113
    p2 = acquire(Kernel(Fixture()), w[:, :-1], "KEEP4")
    assert torch.equal(p.pdf, p2.pdf)
    w[0] = 44
    p3 = acquire(Kernel(Fixture()), w[:, :-1], "KEEP4")
    assert torch.equal(p.pdf[1], p3.pdf[1])


def test_next_block_after_two_spent_studies_with_no_skip_or_overlap():
    from experiments.expressivity_tasks.x49_live_trace_memory.panels import select

    records = []
    for i in range(18):
        h = hashlib.sha256(str(i).encode()).hexdigest()
        if int(h, 16) % 100 < 2:
            continue
        off = d.MIN_OFFSET + 10000 * i
        records.append(
            dict(
                sha256=h, document=off, byte_offset=off, byte_length=5000, split="train"
            )
        )
    ex = {records[-1]["sha256"]}
    size = records[-1]["document"] + 6000
    rank = select(records, ex, {"train": 12}, size)[0]["train"]
    r, blocked, _ = d.last.next_block(records, ex, rank[:2] + rank[2:7], 5, size)
    assert r == rank[7:] and len(blocked) == 8
    for spent in [rank[:2] + rank[3:8], rank[:2] + rank[1:6]]:
        with pytest.raises(ValueError):
            d.last.next_block(records, ex, spent, 5, size)


def test_seal_all_twelve_and_witness_before_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "RUN", tmp_path)
    monkeypatch.setattr(d, "qualified", lambda: {})
    assert len(d.state_keys()) == 12
    with pytest.raises(FileNotFoundError, match="frozen.json"):
        d.extract("probe")
    monkeypatch.setattr(d, "frozen", lambda: {"fit_targets": 12})
    with pytest.raises(FileNotFoundError, match="freeze-ack.json"):
        d.extract("probe")
    assert not list(tmp_path.iterdir())


def test_gap_denominators_and_paired_bootstrap_mixed_seed():
    rng = np.random.default_rng(10)
    v = rng.random((3, 13, 4)) * 48
    curves = {k: v.copy() for k in s.POLICIES}
    curves["DISCARD4"][:, :, 1:] += 16
    curves["ERASE4"][:, :, 1:] += 4
    curves["ERASE4"][0, :, 1:] -= 8
    a, cs = s.summarize(curves, draws=301)
    _, m = s.summarize(curves, draws=301, multiplicity=True)
    g = cs["KEEP4:DISCARD4"]
    assert (
        g["gaps"]["decision"] == "win"
        and g["all"]["gain"] == pytest.approx(0.75)
        and g["gaps"]["gain"] == pytest.approx(1)
        and cs["KEEP4:ERASE4"]["gaps"]["decision"] == "unresolved"
    )
    assert a["KEEP4"]["gaps"]["mean"] == pytest.approx(np.mean(v[:, :, 1:]) / 16)
    assert a["KEEP4"]["read"]["mean"] == pytest.approx(np.mean(v[:, :, 0]) / 16)
    for key in cs:
        for band in cs[key]:
            np.testing.assert_allclose(
                cs[key][band]["ci95"], m[key][band]["ci95"], atol=1e-14
            )
