import itertools

import numpy as np
import pytest
import torch

from ..corpus_kernel.plan import Layout, Plan, tiles
from .routes import Reference, compact
from .lm import ARMS, LanguageLM, Pack
from .settings import SEEDS, N
from .report import summarize


def test_regular_planner_matches_disjoint_oracle_without_any_budget_cutoff():
    rows = 64
    leaf = 8
    rank = 17
    bounds = np.array([(a, b) for a in range(rows + 1) for b in range(a, rows + 1)])
    a, b, c = compact(bounds, rows, leaf, rank)
    p = Plan.build(Layout("x", rows, leaf), [[r] for r in bounds])

    def canon(x):
        return sorted(map(tuple, x))

    assert canon(a) == canon(tiles(p.exact, 16))
    assert canon(b) == canon(tiles(p.edge, 16))
    assert canon(c) == canon(
        tiles([(i, j * rank, (j + 1) * rank) for i, j in p.nodes], 16)
    )
    with pytest.raises(ValueError):
        compact(bounds, 63, leaf, rank)


def test_counts_and_all_suffix_intervals_against_document_local_literal():
    raw = np.array(
        [
            [2, 3, 5, 9, 6, 2, 3, 5, 7],
            [3, 5, 9, 6, 2, 3, 5, 9, 3],
            [8, 7, 6, 2, 3, 1, 9, 4, 1],
        ],
        np.uint8,
    )
    x = np.array(
        [[2, 3, 5, 9, 8, 7, 6], [5, 7, 3, 5, 9, 6, 2], [9, 9, 1, 1, 1, 1, 1]], np.uint8
    )
    ref = Reference(raw)
    ci, iv = ref.lookup(x)
    root = (np.bincount(raw[:, 1:].ravel(), minlength=256) + 0.5) / (
        raw[:, 1:].size + 128
    )
    for i, p in enumerate(x):
        for j in range(len(p)):
            pre = p[: j + 1].tolist()
            true = ref.index.frontier(pre).hit_range(3)
            interval = iv[i * len(p) + j]
            assert tuple(interval) == true
            expected = root
            for d in range(1, min(4, len(pre)) + 1):
                seq = [
                    doc[k + 1]
                    for doc in raw
                    for k in range(d - 1, len(doc) - 1)
                    if doc[k + 1 - d : k + 1].tolist() == pre[-d:]
                ]
                if seq:
                    expected = (np.bincount(seq, minlength=256) + 8 * root) / (
                        len(seq) + 8
                    )
            np.testing.assert_allclose(ref.probs[ci[i, j]], expected, rtol=1e-6)


def test_state_and_routes_exclude_target_and_other_live_windows():
    rng = np.random.default_rng(5763)
    raw = rng.integers(0, 256, (4, 9), dtype=np.uint8)
    m = LanguageLM(5763, d=8, rank=5, leaf=4, time=4)
    pack = Pack(raw, m, 2)
    x = raw[:2, :5].copy()
    p = pack.bind(x)
    y = x.copy()
    y[0, 3:] = 0
    y[1] = 31
    q = pack.bind(y)
    for arm in ARMS:
        v = m.logits(p, arm, "torch")[0]
        z = m.logits(q, arm, "torch")[0]
        torch.testing.assert_close(v[:3], z[:3], atol=0, rtol=0)
    assert torch.equal(p.count_ids[:3], q.count_ids[:3])
    ws = pack.dataset(raw, shuffle=False)
    assert np.array_equal(
        torch.cat([p.targets for p in ws]).numpy(), raw[:, 1:].flatten()
    )


def test_byte_prices_and_paired_seed_reversal():
    docs = N["probe"]
    v = {
        f"{s}.{a}": np.full(docs, 4096 * np.log(2))
        for s, a in itertools.product(SEEDS, ARMS)
    }
    v[f"{SEEDS[0]}.indexed"] *= 0.5
    v[f"{SEEDS[1]}.indexed"] *= 1.1
    q = summarize(v)
    assert q["mean_bpb"]["local"] == pytest.approx((4096 + 8) / 4097)
    assert q["contrasts"]["indexed:local"]["gain"] > 0 and not q["primary_positive"]
