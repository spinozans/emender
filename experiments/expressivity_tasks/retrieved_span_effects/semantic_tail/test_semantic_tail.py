from collections import Counter
from itertools import combinations
import math

import pytest
import torch

from .editions import (
    ALIAS,
    ARMS,
    K,
    KEY,
    SEEDS,
    TOKENIZER,
    VALUE,
    canonical,
    digest,
    docs,
    fit_editions,
    order,
    packet,
    probe_pairs,
    transpose,
)
from .measure import Means, View, compare, view
from .run import REGISTRATION, conclusions, make_model, probe, save, sha


# Algebra fixture, not a training schedule.
P = (1, 4, 3, 5, 6, 0, 2, 7)
Q = transpose(P, 2, 6)


def test_disjoint_balanced_permutation_orbits_and_transpositions():
    fit = fit_editions()
    pairs = tuple(probe_pairs())
    books = [x for a, b, _ in pairs for x in (a, b)]
    assert len(set(fit)) == len(fit) == 128 and len(set(books)) == 448
    assert len({canonical(x) for x in fit}) == 16
    assert len({canonical(x) for x in books}) == 56
    assert not {canonical(x) for x in books}.intersection(map(canonical, fit))
    assert Counter(pair for _, _, pair in pairs) == {
        pair: K for pair in combinations(range(K), 2)
    }
    for a, b, pair in pairs:
        assert transpose(a, *pair) == b and sorted(a) == sorted(b) == list(range(K))
    for eds in (fit, [a for a, _, _ in pairs], [b for _, b, _ in pairs]):
        for i in range(K):
            assert Counter(ed[i] for ed in eds) == {
                val: len(eds) // K for val in range(K)
            }
    assert len({digest(order(s)) for s in SEEDS}) == 3
    for s in SEEDS:
        seq = order(s)
        assert len(seq) == 1024 and all(
            sorted(seq[i : i + 128]) == list(range(128))
            for i in range(0, len(seq), 128)
        )


def test_only_two_successors_change_no_exact_match_at_any_frontier():
    m = make_model(2)
    w, v = [packet(m, p, corpus=True) for p in (P, Q)]
    arc, live = docs(P)
    assert not {t for _, a in arc for t in a}.intersection(
        t for _, p in live for t in p[:-1]
    )
    assert w.archive.index.version != v.archive.index.version
    for u in (w, v):
        assert u.inputs.ids.tolist() == [t for a in ALIAS for t in (0, a)]
        assert u.archive.inputs.ids.tolist() == list(KEY)
        assert u.local_rows.tolist() == list(range(0, 2 * K, 2))
        assert [(b.start, b.stop) for b in u.blocks] == [
            (0, 5),
            (5, 10),
            (10, 15),
            (15, 16),
        ]
        for b in u.blocks:
            assert (
                not b.plan.exact
                and not b.plan.edge
                and len(b.plan.nodes) == b.stop - b.start
            )
        assert u.targets[::2].tolist() == list(ALIAS)
    assert (
        w.archive.successors != v.archive.successors
    ).nonzero().flatten().tolist() == [2, 6]
    assert torch.equal(w.inputs.ids, v.inputs.ids) and torch.equal(
        w.archive.inputs.ids, v.archive.inputs.ids
    )
    with pytest.raises(ValueError):
        docs([0] * K)


@pytest.mark.parametrize("arm", ARMS)
def test_literal_row_read_source_normalizers_scored_logits_and_target_isolation(arm):
    m = make_model(7).double()
    corpus = arm != "local"
    law = arm if corpus else "dense"
    w = packet(m, P, corpus=corpus)
    p = view(w, m, law)
    native = w.predict(m, mode=law)
    torch.testing.assert_close(p.logp.exp(), native.probabilities)
    torch.testing.assert_close(p.shares, native.attention.shares)
    # Hand-enumerate the eight actual keys, one causal local row and the zero hint.
    # Use ordinary positive features, independent of the log/tree implementation.
    i = 2
    ids = torch.tensor([0, ALIAS[i]])
    live = m._states(ids)
    q = m.query(live[-1])
    k = m.key(m._states(torch.tensor(KEY)[:, None])[:, 0])
    lk = m.key(live[0])
    lv = m.value(m.embed(ids[1]))
    v = m.value(m.embed(torch.tensor([VALUE[a] for a in P])))
    null = (m.skip(live[-1]).squeeze() + math.log(8 if corpus else 1)).exp()
    local = (q @ lk / m.dim**0.5).exp()
    if law == "dense":
        weights = (k @ q / m.dim**0.5).exp()
    else:

        def feature(x):
            a = x * m.dim**-0.25
            return (a @ m.omega.T - a.square().sum(-1, keepdim=True) / 2).exp() / len(
                m.omega
            ) ** 0.5

        weights = feature(k) @ feature(q)
    if not corpus:
        weights = torch.zeros_like(weights)
    mass = torch.stack((weights.sum(), local, null))
    r = (weights @ v + local * lv) / mass.sum()
    logp = m.lm(live[-1] + m.out(r)).log_softmax(-1)
    torch.testing.assert_close(logp, p.logp[i * 2 + 1])
    torch.testing.assert_close(mass / mass.sum(), p.shares[i * 2 + 1])
    assert (p.rows is None) == (not corpus)
    if corpus:
        torch.testing.assert_close(weights / weights.sum(), p.rows[i])
    w.targets.fill_(0)
    torch.testing.assert_close(view(w, m, law).logp, p.logp, atol=0, rtol=0)


@pytest.mark.parametrize("law", ("dense", "indexed"))
def test_streamed_gradient_connects_query_key_vector_and_live_lm(law):
    m = make_model(4)
    w = packet(m, P, corpus=True)
    z = w.backward_nll(m, mode=law)
    assert z.isfinite() and not z.requires_grad
    for _, p in m.named_parameters():
        assert (
            p.grad is not None
            and p.grad.isfinite().all()
            and p.grad.count_nonzero() > 0
        )


def test_counts_stale_direction_and_joint_not_conditional_accuracy():
    y0, y1 = [torch.tensor([t for _, p in docs(ed)[1] for t in p[1:]]) for ed in (P, Q)]

    def predicts(y):
        return View(
            torch.zeros(len(y), 25, dtype=torch.float64)
            .scatter_(1, y[:, None], 8.0)
            .log_softmax(-1),
            torch.ones(len(y), 3) / 3,
            torch.eye(K),
        )

    total = Means()
    a, b = predicts(y0), predicts(y1)
    compare(total, a, y0, b, y1, (2, 6), cross=a)
    z = total.result()
    assert (
        total.counts["changed.old_and_new_correct"] == 2
        and total.counts["unchanged.edited_nll"] == 6
        and total.counts["edited.all_nll"] == 16
    )
    assert z["changed.old_and_new_correct"] == z["unchanged.old_and_new_correct"] == 1
    assert (
        z["changed.gain_vs_stale"] == pytest.approx(8.0)
        and z["unchanged.gain_vs_stale"] == 0
    )
    assert z["other_law.row_tv"] == 0 and z["other_law.token_tv"] > 0
    # One old prediction already gave the future answer; predicting it again is NOT following an edit.
    ybad = y0.clone()
    ybad[5] = y1[5]
    total = Means()
    compare(total, predicts(ybad), y0, b, y1, (2, 6))
    assert total.result()["changed.old_and_new_correct"] == 0.5
    with pytest.raises(ValueError):
        compare(total, a, y0, b, y1, (0, 7))


def test_rule_keeps_a_seed_nonwin_and_requires_separate_dense_and_indexed_passes():
    values = {
        f"{s}.{a}": {
            "edited.answer_nll": 1.0 if a == "local" else 0.5,
            "changed.gain_vs_stale": 1.0,
            "changed.old_and_new_correct": 0.6,
        }
        for s in SEEDS
        for a in ARMS
    }
    values[f"{SEEDS[-1]}.indexed"]["changed.old_and_new_correct"] = 0.5
    result = conclusions(values)
    assert result["dense"]["passes"] and not result["indexed"]["passes"]


def test_incomplete_freeze_cannot_open_probe(tmp_path):
    save(
        tmp_path / "frozen.json",
        dict(
            schema=1,
            registered=REGISTRATION,
            qualified="abc",
            tokenizer=TOKENIZER,
            fit_editions_sha256=digest(fit_editions()),
            fits={},
        ),
    )
    with pytest.raises(ValueError, match="unwitnessed"):
        probe(tmp_path, "abc", "bad")
    with pytest.raises(ValueError, match="incomplete"):
        probe(tmp_path, "abc", sha(tmp_path / "frozen.json"))
    assert not (tmp_path / "probe-opened.json").exists()
