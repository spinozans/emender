"""Only invented accounting fields and tmp_path. Never query a broker or live campaign."""

import json

import pytest

from .device_gate import capacity, charge


def test_sequence_gate_debits_four_jobs_and_both_limits():
    end = 1795712280
    w = dict(
        deadline_epoch=end,
        aggregate_card_seconds_upper=43200,
        max_cards_at_once=1,
        visible_allowlist="0,1",
        remote=False,
        auto_retry=False,
    )
    c = {"meter": 29, "natural-meter": 39, "panel": 40000, "corpus-kernel-meter": 100}
    assert capacity(w, end - 210, c) == 150
    assert capacity(w, end - 127, c) == 127
    assert capacity(w, end - 600, dict(c, panel=42903)) == 129
    assert capacity(w, end - 600, {**c, "corpus-kernel-meter": 700}) == 132
    for cs, now in (
        ({k: v for k, v in c.items() if k != "corpus-kernel-meter"}, end - 300),
        (dict(c, panel=43000), end - 500),
        ({**c, "corpus-kernel-meter": 731}, end - 600),
        (c, end - 119),
    ):
        with pytest.raises(RuntimeError):
            capacity(w, now, cs)


def test_charge_checks_actual_creation_release_and_identity(tmp_path):
    p = tmp_path
    for name, z in [
        ("admission", dict(cap=150, deadline=250)),
        (
            "charge",
            dict(rc=0, release_verified=True, renewed_progress=True, upper_card_s=81),
        ),
        ("complete", dict(units=12)),
    ]:
        (p / (name + ".json")).write_text(json.dumps(z))

    def record(name, **kv):
        (p / name).write_text("".join(f"{k}={v}\n" for k, v in kv.items()))

    common = dict(pid=10, gpu=0, starttime=42, created=130)
    record("initial", **common, heartbeat=130)
    record("last", **common, heartbeat=207)
    record("released", released_epoch=210, rc=0)
    with pytest.raises(RuntimeError):
        charge(p, 250)
    (p / "watch-ack").write_text("12\n")
    assert charge(p, 250) == 81
    record("released", released_epoch=209, rc=0)
    with pytest.raises(RuntimeError):
        charge(p, 250)
    record("released", released_epoch=210, rc=0)
    record("last", **dict(common, starttime=43), heartbeat=207)
    with pytest.raises(RuntimeError):
        charge(p, 250)
