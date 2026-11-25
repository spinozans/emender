"""Independent scalar scans and frozen B readouts on invented bytes only."""

import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory.query_depth import (
    Hit,
    addresses,
    score,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
)


def oracle(w, t, H, policy):
    lo = max(0, t + 1 - H)
    for d in (8, 4, 2, 1):
        js = [k for k in range(lo + d, t - 15) if w[k - d : k] == w[t + 1 - d : t + 1]]
        if not js:
            continue
        j = js[-1]
        compatible = [k for k in js if w[k] == w[j]]

        def ell(k):
            n = 0
            while n < 32 and k - 1 - n >= lo and w[k - 1 - n] == w[t - n]:
                n += 1
            return n

        e = max(compatible, key=lambda k: (ell(k), k))
        lj = ell(j)
        alt = (
            [
                k
                for k in compatible
                if k not in (j, e)
                and ell(k) == lj
                and (t - k).bit_length() == (t - e).bit_length()
            ]
            if e != j and policy == "M"
            else []
        )
        s = min(alt, key=lambda k: (abs(k - e), -k)) if alt else e
        return Hit(t, d, j, e, s, lj, ell(e), bool(alt))
    return None


def example(*, both=True):
    t = 350
    w = bytearray([254]) * 384
    key = b"q\x00r\xffstu\x1e"
    prefix = b"A"
    follower = b"\0"
    for k, x in [(150, prefix), (290, b"C")] + (
        [(126, b"D"), (174, b"E")] if both else []
    ):
        w[k - 9 : k + 1] = x + key + follower
    w[t - 8 : t + 1] = prefix + key
    return bytes(w), t


@pytest.mark.parametrize("H", [128, 1024, 4096])
def test_address_plan_is_independent_binary_overlapping_prefix_oracle(H):
    w, t = example()
    v = bytes(range(256)) + w[:235] + b"\x00\x1e\xff" * 16
    for mode in ("E", "M"):
        expected = {}
        for n in range(len(v)):
            r = oracle(v, n, H, mode)
            if r:
                expected[n] = r
        plan = addresses(v, horizon=H, policy=mode)
        assert {r.t: r for r in plan} == expected
        assert plan == addresses(v, horizon=H, policy=mode)
        for end in (1, 18, 137, 389):
            assert tuple(r for r in plan if r.t < end) == addresses(
                v[:end], horizon=H, policy=mode
            )
        assert not addresses(w[t - 8 : t + 1], horizon=H, policy=mode)


def test_longer_context_ties_matched_tie_break_and_no_control_fallback():
    for both in (True, False):
        w, t = example(both=both)
        for mode in ("E", "M"):
            r = addresses(w[: t + 1], policy=mode)[-1]
            assert r == oracle(w, t, 4096, mode)
            assert (r.depth, r.latest, r.extended, r.latest_length) == (
                8,
                290,
                150,
                8,
            ) and r.longest > 8
            assert r.selected == (
                174 if both and mode == "M" else 150
            ) and r.matched == (both and mode == "M")
    w = bytearray([255]) * 570
    t = 540
    key = bytes(range(32))
    for k in (150, 290):
        w[k - 32 : k + 1] = key + b"B"
    w[392:401] = key[-8:] + b"B"
    w[t - 31 : t + 1] = key
    word = bytes(w)
    r = addresses(word[: t + 1], policy="E")[-1]
    assert r == next(row for row in addresses(word, policy="E") if row.t == t)
    assert r == oracle(word, t, 4096, "E")
    assert (r.t, r.latest, r.extended, r.longest) == (540, 400, 290, 32)
    with pytest.raises(ValueError):
        addresses(bytes(w), policy="rotate")


def test_rebinding_moves_only_the_chosen_old_row_gradient_not_the_evidence():
    torch.manual_seed(4906)
    w, t = example()
    b = trained_base()
    h = torch.randn(len(w), 8, requires_grad=True)
    z = torch.randn(len(w), 256)
    answers = {}
    reference = b(w, h, z, mode="B")
    base_work = {}
    b(w, h, z, mode="B", work=base_work)
    for a, source in [("J", 290), ("E", 150), ("M", 174)]:
        info = {}
        work = {}
        y = score(b, w, h, z, policy=a, work=work, diagnostics=info)
        answers[a] = y.detach()
        assert all(work[k] == v for k, v in base_work.items())
        grad = torch.autograd.grad(-y[t, 87], h)[0]
        assert set(torch.where(grad.abs().sum(-1) > 0)[0].tolist()) == {source, t}
        if a == "J":
            assert torch.equal(reference, y) and not info
        else:
            plan = info["hits"]
            assert plan == list(addresses(w, policy=a))
            moved = {r.t for r in plan if r.selected != r.latest}
            same = [i for i in range(len(w)) if i not in moved]
            assert torch.equal(y[same], reference[same])
            assert torch.equal(
                y[t], score(b, w[: t + 1] + bytes(len(w) - t - 1), h, z, policy=a)[t]
            )
            assert work["literal_value_reads"] == len(plan) and work[
                "matched_controls"
            ] == sum(r.matched for r in plan)
    assert not torch.equal(answers["E"][t], answers["M"][t])
    assert not torch.equal(answers["J"][t], answers["E"][t])
    no_alt, _ = example(both=False)
    assert torch.equal(
        score(b, no_alt, h, z, policy="M")[t], score(b, no_alt, h, z, policy="E")[t]
    )


def test_full_length_singleton_misses_and_capped_repeats():
    for w in (b"\0", bytes([0x1E]) * 4096, bytes(range(256)) * 16):
        for mode in ("E", "M"):
            rs = addresses(w, policy=mode)
            assert all(
                r.latest == r.selected == r.extended and not r.matched for r in rs
            )
            assert all(1 <= r.depth <= r.longest <= 32 for r in rs)
            assert not rs or rs[-1].longest == 32
    with pytest.raises(ValueError):
        addresses(b"bad", policy="E", horizon=32)
