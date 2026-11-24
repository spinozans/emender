"""Synthetic tests only. No model checkpoint, corpus, accelerators, or private survey panels."""

from collections import Counter
import random

import pytest
import torch
from ndm.memory_program.direct import Branch, DirectTrust, DirectView, direct_features
from experiments.expressivity_tasks.x49_live_trace_memory.literal import (
    DEPTHS,
    HORIZONS,
    MIN_AGE,
    MISS,
    Match,
    literal_plan,
    rotated_payload_index,
)
from experiments.expressivity_tasks.x49_live_trace_memory.values import LiveValueHead


def scan(prefix, h):
    t = len(prefix) - 1
    lo = max(0, len(prefix) - h)
    history = prefix[lo : max(lo, t - MIN_AGE + 1)]
    for depth in DEPTHS:
        key = prefix[-depth:]
        followers = [
            i + depth
            for i in range(len(history) - depth)
            if history[i : i + depth] == key
        ]
        if followers:
            counts = Counter(history[i] for i in followers)
            return Match(depth, lo + followers[-1], tuple(sorted(counts.items())))
    return MISS


@pytest.mark.parametrize("horizon", HORIZONS)
def test_every_frontier_matches_independent_scan_and_prefix(horizon):
    rng = random.Random(491)
    word = (
        b"aaa" * 23 + b"\0\x1e\xffq" * 11 + bytes(rng.randrange(4) for _ in range(193))
    )
    rows = literal_plan(word, horizon)
    assert rows == tuple(scan(word[: i + 1], horizon) for i in range(len(word)))
    assert all(r == MISS for r in rows[:17])
    for cut in (3, 17, 66, 127, 192, len(word) - 1):
        assert literal_plan(word[:cut], horizon) == rows[:cut]
        assert literal_plan(word[:cut] + b"\xff" * 33, horizon)[:cut] == rows[:cut]
    # Independent invocation implies independent address scope: no cross-window continuation.
    assert literal_plan(word[-8:], horizon) == (MISS,) * 8


def test_age_floor_horizon_key_start_and_long_hit():
    k = b"\0\x1e\xffabcde"
    for age in (15, 16, 17, 120, 121, 1025, 3000):
        j = len(k)
        word = k + b"\xff" + b"_" * (age - len(k)) + k
        assert len(word) - 1 - j == age
        for h in HORIZONS:
            hit = literal_plan(word, h)[-1]
            assert hit == scan(word, h)
            assert (hit.depth == len(k)) == (age >= 16 and age + len(k) < h)
    with pytest.raises(ValueError, match="4096"):
        literal_plan(bytes(4097), 4096)
    with pytest.raises(ValueError, match="horizon"):
        literal_plan(b"a", 256)


def test_sham_is_a_prefix_only_rotation_and_preserves_actual_follower_and_age_bin():
    word = b"ababaaaaaaaa" * 45
    h = 128
    changed = 0
    for t, row in enumerate(literal_plan(word, h)):
        j = row.latest
        if j is None:
            continue
        eligible = [
            i
            for i in range(max(0, t + 1 - h) + row.depth, t - MIN_AGE + 1)
            if word[i] == word[j] and (t - i).bit_length() == (t - j).bit_length()
        ]
        expected = eligible[(eligible.index(j) + 1) % len(eligible)]
        assert (
            rotated_payload_index(word, t, row, h)
            == rotated_payload_index(word[: t + 1], t, row, h)
            == expected
        )
        changed += expected != j
    assert changed > 250


@pytest.fixture
def host_head():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4901)
            word = (b"abcdefghQabcdefff" * 11) + b"abcdefgh"
            h = torch.randn(len(word), 8, requires_grad=True)
            base = torch.randn(len(word), 256, requires_grad=True)
            yield LiveValueHead(DirectTrust(8)), word, h, base
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def test_all_256_birth_scores_match_incumbent_on_identical_bytes(host_head):
    head, word, h, base = host_head
    views = []
    branch = Branch("none", "no-archive", (), None, None)
    for t in range(len(word)):
        prefix = word[: t + 1]
        row = scan(prefix, 4096)
        j = row.latest
        c = dict(row.counts)
        views.append(
            DirectView(
                0,
                0,
                t + 1,
                prefix[-8:],
                prefix[-row.depth :] if row.depth else b"",
                tuple(c.get(i, 0) for i in range(256)),
                word[j] if j is not None else None,
                t - j if j is not None else None,
                None,
            )
        )
    ref, _ = head.direct(h, base, direct_features(views, [branch] * len(views)))
    for mode in ("A", "B", "rotated"):
        actual = head(word, h, base, mode=mode)
        torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        torch.testing.assert_close(
            actual.logsumexp(-1), torch.zeros(len(word)), atol=5e-7, rtol=0
        )


def test_delayed_gradient_reaches_real_old_write_not_current_recompute(host_head):
    head, word, h, base = host_head
    j = scan(word, 4096).latest
    assert j is not None and j < len(word) - MIN_AGE
    opt = torch.optim.SGD(
        head.parameters(), lr=0.1
    )  # fixture only, not the natural optimizer
    loss = -head(word, h, base, mode="B")[-1, 42]
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert head.delta.weight.abs().max() > 0
    for mode in ("A", "B"):
        x = h.detach().clone().requires_grad_(True)
        loss = -head(word, x, base.detach(), mode=mode)[-1, 42]
        dx, dw = torch.autograd.grad(loss, (x, head.writer.weight))
        active = dx.abs().sum(-1).nonzero().flatten().tolist()
        assert active == ([len(word) - 1] if mode == "A" else [j, len(word) - 1])
        assert dw.norm() > 0
    # The sham transports the gradient to the donor, not to the correct literal address.
    changed = [
        (t, r, rotated_payload_index(word, t, r, 4096))
        for t, r in enumerate(literal_plan(word, 4096))
        if r.latest is not None and rotated_payload_index(word, t, r, 4096) != r.latest
    ]
    t, r, donor = changed[-1]
    for mode, old in [("B", r.latest), ("rotated", donor)]:
        x = h.detach().clone().requires_grad_(True)
        (dx,) = torch.autograd.grad(
            -head(word, x, base.detach(), mode=mode)[t, 42], (x,)
        )
        assert dx.abs().sum(-1).nonzero().flatten().tolist() == [old, t]
    # No host information from a future frontier, even after the readout has moved off zero.
    with torch.no_grad():
        for mode in ("A", "B", "rotated"):
            cut = 90
            y = head(word, h, base, mode=mode)
            shorter = head(word[:cut], h[:cut], base[:cut], mode=mode)
            torch.testing.assert_close(y[:cut], shorter, atol=2e-6, rtol=0)
