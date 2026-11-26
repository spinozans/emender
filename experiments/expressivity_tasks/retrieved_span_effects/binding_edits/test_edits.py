from collections import Counter
from itertools import product

import pytest
import torch

from ..semantic_tail.editions import ALIAS, K, SEEDS, VALUE, canonical
from ..semantic_tail.measure import view
from ..semantic_tail.run import make_model
from .decompose import Stats, accumulate, polarity
from .run import probe, verdict
from .worlds import fits, order, packet, panels


def test_disjoint_nonpermutation_books_independence_and_order():
    train = fits()
    assert len(set(train)) == len(train) == 1024
    assert all(
        Counter(p[k] for p in train) == {a: 128 for a in range(K)} for k in range(K)
    )
    for i in range(K):
        for j in range(0, 128, K):
            group = train[i * 128 + j : i * 128 + j + K]
            assert {p[i] for p in group} == set(range(K)) and len(
                {p[:i] + p[i + 1 :] for p in group}
            ) == 1
    used = {canonical(p) for p in train}
    assert len(panels()) == 28
    for (i, j), rows in panels():
        fixed = {p[:i] + p[i + 1 : j] + p[j + 1 :] for p in rows}
        assert len(fixed) == 1 and 2 <= len(set(next(iter(fixed)))) < 6
        assert tuple((p[i], p[j]) for p in rows) == tuple(product(range(K), repeat=2))
        orbit = {canonical(p) for p in rows}
        assert len(orbit) == len(rows) == 64 and orbit.isdisjoint(used)
        used.update(orbit)
    assert all(len(set(p)) < K for p in train) and all(
        sorted(order(s)) == list(range(1024)) for s in SEEDS
    )


@pytest.mark.parametrize("law", ("dense", "indexed"))
def test_decoder_contrast_matches_native_single_swap_and_labels_do_not_reach_inputs(
    law,
):
    m = make_model(19).double()
    i, j = 2, 5
    before = [0, 2, 4, 2, 1, 7, 4, 1]
    a, b = before[i], before[j]
    single = before.copy()
    single[i] = b
    swap = single.copy()
    swap[j] = a
    packets = [packet(m, tuple(p)) for p in (before, single, swap)]
    assert all(
        w.inputs.ids.tolist() == [t for alias in ALIAS for t in (0, alias)]
        for w in packets
    )
    z = [view(w, m, law) for w in packets]
    k = polarity(m)
    g = z[0].rows[i] * z[0].shares[2 * i + 1, 0]
    for v, coeff in zip(z[1:], (g[i], g[i] - g[j]), strict=True):
        torch.testing.assert_close(v.rows, z[0].rows, atol=0, rtol=0)
        torch.testing.assert_close(v.shares, z[0].shares, atol=0, rtol=0)
        dif = v.logp[2 * i + 1] - z[0].logp[2 * i + 1]
        torch.testing.assert_close(
            dif[VALUE[b]] - dif[VALUE[a]], coeff * k[a, b], atol=1e-12, rtol=1e-10
        )
    with pytest.raises(ValueError):
        packet(m, tuple(range(K)))


def linear_panel(g, sign):
    # Deliberately no model helper for this independent contrast/count oracle.
    keys = (1, 4)
    logits = torch.zeros(K, K, 2, 25, dtype=torch.float64) - 20
    for a, b in product(range(K), repeat=2):
        book = [0, 0, 2, 3, 7, 1, 2, 2]
        book[keys[0]] = a
        book[keys[1]] = b
        for s, i in enumerate(keys):
            logits[a, b, s, list(VALUE)] = 0
            for j, v in enumerate(book):
                logits[a, b, s, VALUE[v]] += sign * 8 * g[i, j]
    k = torch.full((K, K), 16.0 * sign, dtype=torch.float64)
    k.fill_diagonal_(0)
    t = Stats()
    accumulate(t, keys, logits.log_softmax(-1), g, k, True)
    return t


def test_direct_read_has_unit_joint_following_and_chance_clamp():
    t = linear_panel(torch.eye(K, dtype=torch.float64), 1)
    v = t.result()
    assert (
        t.counts["answer_nll"] == 128
        and t.counts["single.both_correct"] == 896
        and t.counts["swap.both_correct"] == 112
    )
    assert (
        v["answer_acc"] == v["single.both_correct"] == v["swap.both_correct"] == 1
        and v["clamped_acc"] == 1 / 8
    )
    assert v["single.gain_vs_stale"] == pytest.approx(8.0) and v[
        "single.delta_odds"
    ] == pytest.approx(16.0)
    assert t.max_error < 1e-12


def test_complement_can_follow_permutation_swaps_while_single_edit_moves_odds_backwards():
    g = (1 - torch.eye(K, dtype=torch.float64)) / 7 * 0.99 + torch.eye(
        K, dtype=torch.float64
    ) * 0.01
    for i in range(K):
        for a, b in ((1, 3), (6, 7)):
            p = list(range(K))
            p[a], p[b] = p[b], p[a]
            scores = torch.zeros(K, dtype=torch.float64)
            scores.scatter_add_(0, torch.tensor(p), -g[i])
            assert scores.argmax() == p[i]
    t = linear_panel(g, -1)
    v = t.result()
    assert (
        v["single.both_correct"] == 0
        and v["single.delta_odds"] < 0 < v["swap.delta_odds"]
        and t.max_error < 1e-12
    )


def test_primary_requires_address_specificity_and_keeps_seed_failures():
    z = {
        f"{c}.{s}.{arm}": dict(
            answer_nll=1 if arm == "local" else 0.2,
            **{
                "single.both_correct": 0.8,
                "single.gain_vs_stale": 0.5,
                "swap.both_correct": 0.8,
            },
        )
        for c in ("permutation", "replacement")
        for s in SEEDS
        for arm in ("local", "dense", "indexed")
    }
    z["permutation.5702.indexed"]["swap.both_correct"] = 0.5
    z["permutation.5701.dense"]["single.gain_vs_stale"] = 0
    v = verdict(z)
    assert (
        v["permutation.indexed"]["per_seed"] == [True, False, True]
        and not v["permutation.dense"]["passes"]
        and v["replacement.indexed"]["passes"]
    )


def test_probe_without_new_freeze_cannot_open(tmp_path):
    (tmp_path / "frozen.json").write_text("{}")
    assert not (tmp_path / "probe-opened.json").exists()
    with pytest.raises(ValueError, match="unwitnessed"):
        probe(tmp_path, tmp_path, "foreign", "no")
    assert not (tmp_path / "probe-opened.json").exists()
