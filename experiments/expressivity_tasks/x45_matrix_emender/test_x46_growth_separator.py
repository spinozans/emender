"""X46 tests: the derived ceilings' mechanical halves, machine-checked.

Every claim the DESIGN doc and the PREREGISTRATION pre-state about the fixed
machines' ceilings is either proven here from the machinery's own behavior
(the registered oracle scan, the registered trigger statistic) or covered by
the refutation clause.  All CPU, all exact, no GPU needed.
"""
from __future__ import annotations

import math

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender import (
    x46_growth_separator as X)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy)


# ---------------------------------------------------------------- the ceiling

def test_linear_channel_ceiling_numbers():
    """The pre-stated ceiling arithmetic, pinned to the PREREGISTRATION's
    table: ours-fixed rank 128, gdn2-fixed 2x128, chance 1/128, band 0.05."""
    # ours-fixed: 128/320 + 192/320 * 1/128
    ours_320 = X.linear_channel_ceiling(128, 320)
    assert ours_320 == pytest.approx(128 / 320 + (192 / 320) / 128)
    assert ours_320 == pytest.approx(0.4046875)
    # gdn2-fixed: 256/320 + 64/320 * 1/128
    gdn2_320 = X.linear_channel_ceiling(256, 320)
    assert gdn2_320 == pytest.approx(0.8015625)
    # n=640: ours 128/640 + 512/640/128, gdn2 256/640 + 384/640/128
    assert X.linear_channel_ceiling(128, 640) == pytest.approx(0.20625)
    assert X.linear_channel_ceiling(256, 640) == pytest.approx(0.4046875)
    # n=64: capacity >= n on both fixed machines -> no binding ceiling
    assert X.linear_channel_ceiling(128, 64) == 1.0
    assert X.linear_channel_ceiling(256, 64) == 1.0
    # the grown machines escape at n=320 (512 >= 320) but are under pressure
    # at n=640 (512 < 640): the matched-grown-state cell
    assert X.linear_channel_ceiling(512, 320) == 1.0
    assert X.linear_channel_ceiling(512, 640) == pytest.approx(0.8015625)


def test_ceiling_record_honest_arithmetic():
    record = X.ceiling_record("ours-fixed", 320, rank=128)
    assert record["capacity"] == 128
    assert record["ceiling"] == pytest.approx(0.4046875)
    assert record["ceiling_plus_band"] == pytest.approx(0.4546875)
    record = X.ceiling_record("gdn2-growth", 320, heads=4)
    assert record["capacity"] == 512
    assert record["escape"] == "capacity >= n: no binding ceiling (the control regime)"
    with pytest.raises(ValueError):
        X.ceiling_record("unknown", 320)


# ------------------------------------------------------- the slot-count lemma

def test_slot_count_lemma():
    """The hard half of the ours ceiling: at most R live rows at every token;
    a post-fill binding enters only by destroying a live row (every
    post-fill APPEND is paid for by a POP); at most R distinct payload
    directions in the state."""
    checks = X.check_slot_lemma(rank=8, n_writes=24)
    assert checks["passed"], checks
    assert checks["live_rows_le_rank"]
    assert checks["every_post_fill_append_paid_by_a_pop"]
    assert checks["distinct_directions_le_rank"]


def test_slot_count_lemma_scales():
    """The same lemma at a larger rank, more bindings."""
    checks = X.check_slot_lemma(rank=16, n_writes=64)
    assert checks["passed"], checks


def test_post_fill_binding_costs_a_live_row():
    """The registered five-op chain at a FULL store with an unmatched write:
    POP (worst <= tau_pop) or NONE -- the write itself never lands in a
    fresh slot; the (R+1)-th distinct binding enters only on the next write,
    into the slot the POP freed.  The overflow arithmetic: POPs >=
    post-fill APPENDs."""
    from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
        import sequential_oracle_scan
    rank, n_writes = 8, 20
    value_dim = n_writes
    payload = torch.zeros(1, n_writes, 1, value_dim)
    payload[0, torch.arange(n_writes), 0, torch.arange(n_writes)] = 1.0
    result = sequential_oracle_scan(torch.zeros_like(payload), payload,
                                    torch.full((1, n_writes, 1), 0.999),
                                    policy=RoutingPolicy(), rank=rank,
                                    rank_max=rank, collect=True)
    ops = result["schedule"].op[0, :, 0, 0]
    assert (ops[:rank] == X.APPEND).all()   # APPEND into the free slots
    # the FIRST post-fill write is itself a POP or a NONE (never an APPEND:
    # the store was full at its entry)
    assert int(ops[rank]) in (X.NONE, X.POP)
    assert int((ops[rank:] == X.APPEND).sum()) <= int((ops[rank:] == X.POP).sum())


# ------------------------------------------------------ the aliasing swap lemma

def test_aliasing_swap_lemma():
    """Two aliased bindings in one row: the read outputs at the two probes
    are the SAME direction up to a positive scalar, so no fixed linear head
    answers both probes of a window."""
    checks = X.check_aliasing_swap_lemma(trials=200)
    assert checks["passed"], checks
    assert checks["read_outputs_same_direction"]
    assert checks["heads_answering_both_probes"] == 0


# --------------------------------------------------------- the gdn2 key-space

def test_gdn2_key_space_bound():
    """The Welch/frame-potential bound at the task's numbers, the exact
    orthonormal regime, and the past-capacity crosstalk regime."""
    checks = X.check_gdn2_key_space(n=320, d=128)
    assert checks["passed"], checks
    assert checks["welch_holds"]
    assert checks["orthonormal_exact"]
    assert checks["crosstalk_exceeds_signal"]
    # the per-head exact bound: N = d retrieves exactly; the Welch-floor
    # crosstalk/signal has the closed form sqrt((n-d)/d) -- EXACTLY 1 at
    # n = 2d (the crossing), sqrt(1.5) at the headline n=320
    assert X.welch_bound(128, 128) == 0.0
    assert X.welch_bound(129, 128) == pytest.approx(
        math.sqrt(1 / (128 * 128)))
    assert X.crosstalk_to_signal(256, 128) == pytest.approx(1.0)
    assert X.crosstalk_to_signal(320, 128) == pytest.approx(math.sqrt(1.5))
    assert X.crosstalk_to_signal(320, 128) > 1.0


def test_gdn2_frame_potential_holds_for_every_draw():
    """The bound is configuration-free: ten random draws all sit at or above
    the Welch line (it is a theorem, not a sampling statement)."""
    for seed in range(10):
        keys = torch.nn.functional.normalize(
            torch.randn(320, 128, generator=torch.Generator().manual_seed(seed)),
            dim=-1)
        gram = keys @ keys.T
        off = gram - torch.eye(320)
        rms = float((off ** 2).sum().div(320 * 319).sqrt())
        assert rms >= X.welch_bound(320, 128) * 0.999


# ---------------------------------------------------------- the growth trigger

def test_registered_trigger_fires_on_overflow():
    """The ours-growable arm's REGISTERED trigger fires mechanically on the
    constructed overflow stream, at the first boundary past the fill."""
    checks = X.check_growth_trigger_on_overflow(rank=64, n_writes=200)
    assert checks["passed"], checks
    assert checks["fired"]
    assert checks["growth_target"] == 128
    assert checks["fire_boundary"] == 128   # the first boundary past the fill


# ------------------------------------------------------------ the construction

def test_construction_invariants():
    assert X.check_construction()["passed"]


def test_both_encodings_decode_the_same_stream():
    """The two encodings carry the same symbolic bindings: the matrix targets
    and the byte targets name the same (key -> value) assignment."""
    window = X.sample_window(torch.Generator().manual_seed(7), 16, 96)
    m = X.make_matrix_batch(window)
    b = X.make_byte_batch(window)
    assert torch.equal(m["order"], b["order"])
    assert torch.equal(
        m["targets"] - X.VAL_BASE,
        b["targets"] - X.VAL_BYTE_BASE)
    # window lengths: 3n+1 tokens; the byte window padded to the registered
    # 64-token statistic boundary (the committed growth wrapper's own
    # segment arithmetic; the padding is never a target)
    assert m["ids"].shape[1] == 3 * 96 + 1
    assert b["ids"].shape[1] == X.byte_window_length(96)
    assert b["ids"].shape[1] % 64 == 0
    assert b["ids"].shape[1] >= 6 * 96 + 1


def test_byte_alphabet_is_disjoint_and_causal():
    """Every byte is in exactly one role range, and the probe's predict
    position sees exactly the two key bytes before the target."""
    window = X.sample_window(torch.Generator().manual_seed(11), 4, 40)
    b = X.make_byte_batch(window)
    ids = b["ids"]
    assert ids.min() >= 0 and ids.max() < 256
    for row in ids:
        seen = set()
        for byte in row.tolist():
            if byte in (X.BYTE_PAD, X.BYTE_BOS):
                seen.add("pad")
            elif X.VAL_BYTE_BASE <= byte < X.VAL_BYTE_BASE + X.N_VALUES:
                seen.add("value")
            elif X.KEY_HI_BASE <= byte < X.KEY_HI_BASE + 20:
                seen.add("key_hi")
            elif X.KEY_LO_BASE <= byte < X.KEY_LO_BASE + 32:
                seen.add("key_lo")
            else:
                raise AssertionError(f"unassigned byte {byte}")
    predict = b["probe_positions"][0] - 1
    key = ids[0, predict].item()
    assert X.KEY_LO_BASE <= key < X.KEY_LO_BASE + 32


def test_matrix_pair_tokens_carry_the_binding():
    """The combined pair token is a fixed function of (key, value): the
    depth-1 writability construction."""
    window = X.sample_window(torch.Generator().manual_seed(13), 3, 8)
    m = X.make_matrix_batch(window)
    for i in range(8):
        token = int(m["ids"][0, X.matrix_layout(8)[0][i]])
        k, v = int(window.keys[0, i]), int(window.values[0, i])
        assert token == X.PAIR_BASE + k * X.N_VALUES + v


def test_keys_are_distinct_and_values_addressable():
    """Each window's keys are distinct (the address requirement) and each
    key has exactly one binding (no ambiguity from value repeats)."""
    window = X.sample_window(torch.Generator().manual_seed(17), 32, 100)
    for row in window.keys:
        assert len(set(row.tolist())) == 100


# ------------------------------------------------------------- fail-closed

def test_arm_refuses_without_operator_signoff(tmp_path, monkeypatch):
    """The headline gate: an unsigned (or missing) PREREGISTRATION refuses
    by name -- and the gate is STATUS-LINE-only: the registration's own
    prose that DESCRIBES the gate contains the mark and must never satisfy
    it (the real registration does)."""
    from experiments.expressivity_tasks.x45_matrix_emender import (
        x46_growth_separator as mod)
    real = mod.PREREGISTRATION
    # the real, current registration is a DRAFT: it must refuse
    with pytest.raises(ValueError, match="fail-closed"):
        mod.require_operator_signoff()
    unsigned = tmp_path / "PREREGISTRATION_X46_GROWTH_SEPARATOR.md"
    unsigned.write_text(
        "Status: DRAFT -- PENDING OPERATOR SIGN-OFF\n"
        "...the arm runner is FAIL-CLOSED until this file carries the\n"
        "string `OPERATOR-SIGNED` with a date...\n")
    mod.PREREGISTRATION = unsigned
    try:
        with pytest.raises(ValueError, match="fail-closed"):
            mod.require_operator_signoff()
    finally:
        mod.PREREGISTRATION = real


def test_arm_accepts_operator_signoff(tmp_path):
    from experiments.expressivity_tasks.x45_matrix_emender import (
        x46_growth_separator as mod)
    real = mod.PREREGISTRATION
    signed = tmp_path / "PREREGISTRATION_X46_GROWTH_SEPARATOR.md"
    signed.write_text(
        "**Status: OPERATOR-SIGNED (2026-12-01)** -- the seven rulings in §8\n"
        "are amended in; the sweep may run.\n")
    mod.PREREGISTRATION = signed
    try:
        mod.require_operator_signoff()
    finally:
        mod.PREREGISTRATION = real


def test_lease_required():
    import os
    from experiments.expressivity_tasks.x45_matrix_emender import (
        x46_growth_separator as mod)
    saved = {k: os.environ.pop(k, None) for k in
             ("GPU_LEASE_HELD", "CUDA_VISIBLE_DEVICES")}
    try:
        with pytest.raises(ValueError, match="no GPU lease"):
            mod.require_lease()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_calibrated_tau_is_fail_closed(tmp_path, monkeypatch):
    from experiments.expressivity_tasks.x45_matrix_emender import (
        x46_growth_separator as mod)
    monkeypatch.setattr(mod, "RECEIPTS", tmp_path)
    with pytest.raises(ValueError, match="CALIBRATED tau"):
        mod.calibrated_tau()
    (tmp_path / "x46_gdn2_calibration.json").write_text('{"tau_cal": 23.5}')
    assert mod.calibrated_tau() == 23.5


# --------------------------------------------------------------- the selftest

def test_selftest_passes():
    receipt = X.selftest(out_path=None)
    assert receipt["passed"], receipt
