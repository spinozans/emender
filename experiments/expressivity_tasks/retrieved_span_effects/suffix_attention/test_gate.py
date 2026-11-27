"""Gate oracles: campaign ledger, stage admission, freeze witness."""

import json

import pytest

import experiments.expressivity_tasks.retrieved_span_effects.suffix_attention.gate as gate


@pytest.fixture
def env(tmp_path, monkeypatch):
    base = tmp_path / "sal"
    data = base / "data"
    job = base / "device"
    data.mkdir(parents=True)
    job.mkdir()
    for name in ("rows.json", "build.npy", "fit.npy", "fit.json", "build.json"):
        (data / name).write_bytes(b"fixture")
    (data / "ready.json").write_text(
        json.dumps(
            dict(
                rows=gate.sha(data / "rows.json"),
                build=gate.sha(data / "build.json"),
                fit=gate.sha(data / "fit.json"),
            )
        )
    )
    monkeypatch.setattr(gate, "BASE", base)
    monkeypatch.setattr(gate, "DATA", data)
    monkeypatch.setattr(gate, "JOB", job)
    monkeypatch.setattr(
        gate,
        "cpu",
        lambda: None,
    )
    monkeypatch.setattr(
        gate,
        "qualified",
        lambda: "pubref",
    )
    return base, data, job


def test_admission_enforces_campaign_ledger_and_single_use(env):
    base, data, job = env
    monkeypatch_time = {"now": gate.time.time() + 0}
    a = gate.admission("device_check", 400)
    assert a["ledger"] == 0 and a["cap"] == 400
    with (base / "campaign.jsonl").open("a") as f:
        f.write(json.dumps(dict(stage="device_check", upper_card_s=100, rc=0)) + "\n")
    gate.put(
        job / "charge-device_check.json",
        dict(stage="device_check", upper_card_s=100, rc=0),
    )
    gate.time.time = lambda: monkeypatch_time["now"]
    a2 = gate.admission("run", 14000)
    assert a2["ledger"] == 100
    with pytest.raises(RuntimeError):
        gate.admission("run", 15000)  # 100 + 15000 > campaign cap
    with pytest.raises(RuntimeError):
        gate.admission("device_check", 400)  # charge file exists -> no retry
    (job / "charge-device_check.json").unlink()
    monkeypatch_time["now"] = gate.CAMPAIGN_END - 10
    with pytest.raises(RuntimeError):
        gate.admission("device_check", 400)  # too late in the campaign window


def test_freeze_witness_requires_six_paired_fits(env):
    base, data, job = env
    h = gate.sha(data / "rows.json")
    gate.put(job / "admission.json", dict(qualified="pubref", rows_sha=h))
    f = dict(
        qualified="pubref",
        rows_sha=h,
        fit_sha=gate.sha(data / "fit.npy"),
        build_sha=gate.sha(data / "build.npy"),
        fits={},
    )
    for s in gate.SEEDS:
        for arm in gate.ARMS:
            k = f"{s}.{arm}"
            torch_weights = b"weights"
            (job / (k + ".pt")).write_bytes(torch_weights)
            gate.put(
                job / (k + ".json"),
                dict(
                    seed=s,
                    arm=arm,
                    updates=2048,
                    targets=gate.N["fit"] * 4096,
                    order_sha=gate.order_sha(s),
                    initial=str(s),
                    weights_sha=gate.sha(job / (k + ".pt")),
                ),
            )
            f["fits"][k] = gate.sha(job / (k + ".json"))
    gate.put(job / "frozen.json", f)
    gate.witness_freeze()
    assert gate.get(job / "freeze-ack.json")["fits"] == 6
    # Tamper with one fit: witness must refuse.
    k = "5751.local"
    gate.put(
        job / (k + ".json"),
        dict(
            seed=5751,
            arm="local",
            updates=2047,
            targets=gate.N["fit"] * 4096,
            order_sha=gate.order_sha(5751),
            initial="5751",
            weights_sha=gate.sha(job / (k + ".pt")),
        ),
    )
    with pytest.raises(RuntimeError):
        gate.witness_freeze()


def test_monitor_stops_only_after_a_successful_run_stage(env, monkeypatch):
    base, data, job = env
    gate.put(
        job / "charge-device_check.json",
        dict(
            stage="device_check",
            upper_card_s=100,
            rc=0,
            release_verified=True,
            renewed_progress=True,
            within_stage=True,
        ),
    )
    calls = {"n": 0}

    def fast_sleep(_):
        calls["n"] += 1
        if calls["n"] == 2:
            gate.put(
                job / "charge-run.json",
                dict(
                    stage="run",
                    upper_card_s=200,
                    rc=0,
                    release_verified=True,
                    renewed_progress=True,
                    within_stage=True,
                ),
            )
        if calls["n"] > 4:
            raise SystemExit

    monkeypatch.setattr(gate.time, "sleep", fast_sleep)
    gate.monitor("run")
    assert calls["n"] >= 2
    gate.put(
        job / "charge-run.json",
        dict(
            stage="run",
            upper_card_s=200,
            rc=1,
            release_verified=True,
            renewed_progress=True,
            within_stage=True,
        ),
    )
    with pytest.raises(RuntimeError):
        gate.monitor("run")
