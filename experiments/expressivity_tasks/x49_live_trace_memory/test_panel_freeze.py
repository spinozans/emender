"""Exercise complete and corrupt frozen endpoint maps using tiny synthetic files, no real approvals/leases/text."""

import json
import pytest
from experiments.expressivity_tasks.x49_live_trace_memory import panel as p
from experiments.expressivity_tasks.x49_live_trace_memory.panel_freeze import freeze
from experiments.expressivity_tasks.x49_live_trace_memory.panel_spec import (
    ARMS,
    SEEDS,
    JOBS,
    STEPS,
    T,
    binding,
    training,
)
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    sha,
    load,
    write_json,
)


def test_wall_endpoints_are_deduplicated_and_hashed_before_final(tmp_path):
    ref = "e" * 40
    root = tmp_path
    rows = "".join(
        json.dumps(dict(step=i, targets=T, preupdate_nll=2.0)) + "\n"
        for i in range(1, STEPS + 1)
    )
    for arm in ARMS:
        for seed in SEEDS:
            d = root / training(arm, seed)
            d.mkdir()
            b = binding(ref, arm, seed)
            write_json(
                d / "start-state.json",
                dict(
                    binding=b,
                    birth=dict(
                        host="H",
                        literal=str(seed),
                        head=("AB" if arm in "AB" else "FC") + str(seed),
                    ),
                ),
            )
            (d / "steps.jsonl").write_text(rows)
            ms = []
            for i in range(32, STEPS + 1, 32):
                f = d / f"weights-{i:04d}.pt"
                f.write_bytes(f"synthetic:{arm}:{seed}:{i}".encode())
                n = dict(
                    file=f.name,
                    step=i,
                    binding=b,
                    bytes=f.stat().st_size,
                    sha256=sha(f),
                )
                write_json(f.with_suffix(".json"), n)
                ms.append(
                    dict(
                        step=i,
                        sha256=n["sha256"],
                        bytes=n["bytes"],
                        wall_s=i * (1.25 if arm == "C" else 1),
                    )
                )
            (d / "clock.jsonl").write_text("".join(json.dumps(m) + "\n" for m in ms))
            rc = d / "step-1024.pt"
            rc.write_bytes(b"synthetic full")
            write_json(
                d / "latest.json",
                dict(
                    step=STEPS,
                    binding=b,
                    file=rc.name,
                    bytes=rc.stat().st_size,
                    sha256=sha(rc),
                ),
            )
            write_json(
                d / "complete.json",
                dict(
                    binding=b,
                    kind="train",
                    units=STEPS,
                    targets=STEPS * T,
                    steps_sha256=sha(d / "steps.jsonl"),
                    clock_sha256=sha(d / "clock.jsonl"),
                    final_weights=ms[-1]["sha256"],
                    final_recovery=sha(rc),
                ),
            )
            write_json(
                d / "charge.json",
                dict(
                    renewed=True,
                    rc=0,
                    witnessed_units=STEPS,
                    upper_card_s=1100,
                    inclusive_wall_s=ms[-1]["wall_s"] + 1,
                    complete_sha=sha(d / "complete.json"),
                ),
            )
    for name in ["cal-AB", "cal-FC"]:
        d = root / name
        d.mkdir()
        freeze(JOBS[name], ref, root)
        assert len(load(d / "frozen.json")["endpoints"]) == 6
        write_json(d / "complete.json", dict(kind="calibration", units=660))
        write_json(
            d / "charge.json",
            dict(rc=0, witnessed_units=660, complete_sha=sha(d / "complete.json")),
        )
    f = root / "final"
    f.mkdir()
    wrong = root / "C-4901/weights-0800.pt"
    original = wrong.read_bytes()
    wrong.write_bytes(bytes(len(original)))
    with pytest.raises(ValueError, match="weights not bound"):
        freeze(JOBS["final"], ref, root)
    assert not (f / "frozen.json").exists()
    wrong.write_bytes(original)
    dig = freeze(JOBS["final"], ref, root)
    m = load(f / "frozen.json")
    assert (
        dig == sha(f / "frozen.json")
        and m["units"] == 128 * 15
        and m["split"] == "final_test"
    )
    assert (
        all(f"C-{s}@800" in m["endpoints"] for s in SEEDS)
        and len(m["calibration_charges"]) == 2
    )
    with pytest.raises(ValueError, match="already frozen"):
        freeze(JOBS["final"], ref, root)


def test_next_job_cannot_ignore_a_charged_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    monkeypatch.setattr(p, "OUT", tmp_path)
    d = tmp_path / "A-4901"
    d.mkdir()
    write_json(d / "charge.json", dict(job="A-4901", rc=124, upper_card_s=1000))
    (d / "released").write_text("rc=124\nreleased_epoch=1001\n")
    (d / "initial").write_text("created=2\n")
    (d / "after").write_text("heartbeat=4\n")
    write_json(d / "complete.json", {})
    with pytest.raises(ValueError, match="predecessor partial"):
        p.prior(JOBS["B-4901"])
    (d / "steps.jsonl").write_text('{"step":1}\n{"step":2}\n{"st')
    a = p.partial(JOBS["A-4901"])
    assert (
        a["upper_card_s"] == 1000
        and a["complete_recovery_sha"] is None
        and a["known_uncheckpointed_updates"] == 2
    )
