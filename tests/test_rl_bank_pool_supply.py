"""E6/E7 regressions for the bank refresh and standing-supply repairs."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "rl-loop-v1"


def load_script(name, monkeypatch):
    # standing_supply adjusts sys.path for its deployed dependencies.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pool(tmp_path, monkeypatch):
    bank = load_script("rl_bank", monkeypatch)
    paths = bank.bank_paths(tmp_path / "bank")
    bank.ensure_bank_layout(paths)
    program = tmp_path / "validator.py"
    program.write_text("sealed validator fixture")
    bundles = [
        {"task": {"identity": str(i) * 64, "family_id": "read"}, "split": split}
        for i, split in enumerate(("train", "train", "development", "development"), 1)
    ]
    bodies_built = []

    def body(bundle, lake, **kwargs):
        bodies_built.append(bundle["task"]["identity"])
        return {"prompt": bundle["task"]["identity"], "files": {"token": "123"},
                "split": bundle["split"], "receipt_eligible": bundle["split"] == "train"}

    monkeypatch.setitem(sys.modules, "rl_gym_tasks", SimpleNamespace(
        gym_task_body=body, pull_validated_tasks=lambda lake: (bundles, {"tasks": 4})))
    return bank, paths, program, bundles, bodies_built


def refresh(pool, number):
    bank, paths, program, *_ = pool
    return bank.refresh_pool(paths, lake=program.parent, round_number=number,
                             program_path=program)


def test_refresh_pool_twice_has_zero_duplicate_content(pool, capsys):
    bank, paths, _, _, built = pool
    first = refresh(pool, 1)
    second = refresh(pool, 2)
    assert len(first["frozen"]) == 2
    assert second["frozen"] == []
    assert len(list(paths["pool_pending"].glob("*.json"))) == 2
    assert len({t["task_sha256"] for t in first["frozen"]}) == 2
    assert second["max_task_replays"] == bank.MAX_TASK_REPLAYS == 0
    assert all(t["replay_count"] == 0 for t in first["frozen"])
    assert [s["reason"] for s in second["skipped"]].count("replay-limit") == 2
    assert len(built) == 4  # only the two training bodies per refresh
    assert "reason=replay-limit" in capsys.readouterr().out
    assert json.loads((paths["state"] / "pool-pull-round-0002.json").read_text()) == second


@pytest.mark.parametrize("destination,suffix", [("pool_done", ".json"),
                                               ("pool_claims", ".claim")])
def test_refresh_pool_skips_legacy_content_in_done_or_claims(pool, destination, suffix):
    _, paths, *_ = pool
    first = refresh(pool, 1)
    for row in first["frozen"]:
        source = paths["pool_pending"] / (row["task_id"] + ".json")
        task = json.loads(source.read_text())
        task.pop("replay_count")  # legacy bank records have no replay counter
        task["task_id"] = "legacy-" + row["task_id"]
        source.unlink()
        (paths[destination] / (task["task_id"] + suffix)).write_text(json.dumps(task))
    second = refresh(pool, 2)
    assert second["frozen"] == []
    assert list(paths["pool_pending"].glob("*.json")) == []
    assert len([s for s in second["skipped"] if s["reason"] == "replay-limit"]) == 2


def test_refresh_pool_development_never_enters_pending(pool, capsys):
    _, paths, _, bundles, built = pool
    development = {b["task"]["identity"] for b in bundles if b["split"] == "development"}
    record = refresh(pool, 1)
    assert not development.intersection(built)
    assert {s["identity"] for s in record["skipped"]} == development
    assert all(json.loads(p.read_text())["body"]["split"] == "train"
               for p in paths["pool_pending"].glob("*.json"))
    assert "split=development reason=non-training-split" in capsys.readouterr().out


def test_freeze_pool_task_refuses_development(pool):
    bank, paths, *_ = pool
    with pytest.raises(ValueError, match="receipt-ineligible"):
        bank.freeze_pool_task(paths, {"task_id": "dev", "body": {
            "split": "development", "receipt_eligible": False}})
    assert list(paths["pool_pending"].glob("*.json")) == []


def test_refresh_pool_deduplicates_aliases_within_one_pull(pool, monkeypatch):
    _, paths, _, bundles, *_ = pool
    monkeypatch.setattr(sys.modules["rl_gym_tasks"], "gym_task_body",
                        lambda *args, **kwargs: {"split": "train", "receipt_eligible": True})
    record = refresh(pool, 1)
    assert len(record["frozen"]) == 1
    assert len(list(paths["pool_pending"].glob("*.json"))) == 1


@pytest.mark.parametrize("pending,live,stale,expected", [
    (0, 0, 0, True), (1, 0, 0, False), (0, 1, 0, False), (0, 0, 1, False)])
def test_coordinator_refresh_waits_for_empty_pending_and_claims(
        monkeypatch, pending, live, stale, expected):
    load_script("rl_bank", monkeypatch)
    coord = load_script("rl_bank_coordinator", monkeypatch)
    assert coord.pool_refresh_needed({"pending": pending + 155,
                                      "pending_training": pending, "claims_live": live,
                                      "claims_stale": stale}) is expected


def test_supply_severe_backoff_is_monotonic_and_recovers_on_clean_streaks(monkeypatch):
    supply = load_script("standing_supply", monkeypatch)
    monkeypatch.setattr(supply, "control_probe", lambda: {"ok": False, "latency_s": 61})
    monkeypatch.setattr(supply, "scan_bank_failures", lambda state: [])
    monkeypatch.setattr(supply, "append_metrics", lambda record: None)
    state = {"lanes": supply.LANE_MIN, "clean_streak": 4, "backoff_events": []}
    history = []
    for _ in range(5):
        supply.health_adjust(state)
        history.append(state["lanes"])
        assert state["clean_streak"] == 0
    assert history == [6, 4, 2, 2, 2]
    assert supply.backoff_lanes(1, 2) == 1  # never increases an already-low state
    assert supply.recovery_lanes(2, 0, 0) == 2
    assert supply.recovery_lanes(2, 1, 0) == 2
    assert supply.recovery_lanes(2, 2, 0) == 4
    assert supply.recovery_lanes(4, 2, 0) == 6
    assert supply.recovery_lanes(6, 2, 0) == 8
    assert supply.recovery_lanes(15, 2, 0) == supply.LANE_SOFT_CAP
    assert supply.recovery_lanes(16, 3, 0) == 16
    assert supply.recovery_lanes(16, 4, supply.LOW_WATER) == 16
    assert supply.recovery_lanes(16, 4, 0) == 18
    assert supply.recovery_lanes(23, 4, 0) == supply.LANE_HARD_CAP
    assert supply.recovery_lanes(supply.LANE_HARD_CAP, 10, 0) == supply.LANE_HARD_CAP


def test_supply_degraded_and_failed_tranche_backoff_stay_below_normal_min(monkeypatch):
    supply = load_script("standing_supply", monkeypatch)
    monkeypatch.setattr(supply, "control_probe", lambda: {"ok": True, "latency_s": 31})
    monkeypatch.setattr(supply, "scan_bank_failures", lambda state: [])
    monkeypatch.setattr(supply, "append_metrics", lambda record: None)
    state = {"lanes": 4, "clean_streak": 4, "backoff_events": []}
    supply.health_adjust(state)
    assert state["lanes"] == 3
    assert state["clean_streak"] == 0
    monkeypatch.setattr(supply, "heartbeat", lambda state: None)
    monkeypatch.setattr(supply, "pool_counts", lambda: {"pending": 0})
    monkeypatch.setattr(supply, "health_adjust", lambda state: {"ok": True, "latency_s": 1})
    monkeypatch.setattr(supply, "save_state", lambda state: None)
    monkeypatch.setattr(supply, "next_tranche", lambda: 1)
    monkeypatch.setattr(supply, "run_tranche", lambda *args: {
        "status": "failed", "authoring": {"transport_error": True}})
    supply.daemon_iteration(state)
    assert state["lanes"] == supply.BACKOFF_FLOOR
    supply.daemon_iteration(state)
    assert state["lanes"] == supply.BACKOFF_FLOOR


def test_supply_bank_failure_resets_clean_streak_without_increasing_lanes(monkeypatch):
    supply = load_script("standing_supply", monkeypatch)
    monkeypatch.setattr(supply, "control_probe", lambda: {"ok": True, "latency_s": 1})
    monkeypatch.setattr(supply, "scan_bank_failures", lambda state: ["LANE_FAILURE"])
    monkeypatch.setattr(supply, "append_metrics", lambda record: None)
    state = {"lanes": 2, "clean_streak": 4, "backoff_events": []}
    supply.health_adjust(state)
    assert state["lanes"] == 2
    assert state["clean_streak"] == 0


def test_refresh_pool_declared_replay_limit_has_its_own_counter(pool, monkeypatch):
    bank, paths, *_ = pool
    monkeypatch.setattr(bank, "MAX_TASK_REPLAYS", 1)
    first = refresh(pool, 1)
    for row in first["frozen"]:
        name = row["task_id"] + ".json"
        (paths["pool_pending"] / name).rename(paths["pool_done"] / name)
    replay = refresh(pool, 2)
    assert len(replay["frozen"]) == 2
    assert all(row["replay_count"] == 1 for row in replay["frozen"])
    assert refresh(pool, 3)["frozen"] == []


@pytest.mark.parametrize("latency,expected", [(1, 4), (61, 2)])
def test_supply_daemon_recovers_only_after_healthy_clean_tranches(monkeypatch, latency, expected):
    supply = load_script("standing_supply", monkeypatch)
    monkeypatch.setattr(supply, "heartbeat", lambda state: None)
    monkeypatch.setattr(supply, "pool_counts", lambda: {"pending": 0})
    monkeypatch.setattr(supply, "control_probe", lambda: {"ok": True, "latency_s": latency})
    monkeypatch.setattr(supply, "scan_bank_failures", lambda state: [])
    monkeypatch.setattr(supply, "append_metrics", lambda record: None)
    monkeypatch.setattr(supply, "save_state", lambda state: None)
    monkeypatch.setattr(supply, "next_tranche", lambda: 1)
    monkeypatch.setattr(supply, "run_tranche", lambda *args: {
        "status": "admitted", "authoring": {"transport_error": False},
        "pool_after": {"pending": 0}})
    state = {"lanes": 2, "clean_streak": 0, "backoff_events": [], "scaleup_events": []}
    supply.daemon_iteration(state)
    assert state["lanes"] == 2
    supply.daemon_iteration(state)
    assert state["lanes"] == expected
    assert state["clean_streak"] == (2 if latency == 1 else 0)


def test_claim_pool_task_preserves_and_skips_existing_development(pool, capsys):
    bank, paths, *_ = pool
    legacy = {"task_id": "000-development", "task_sha256": "dev-hash", "body": {
        "split": "development", "receipt_eligible": False}}
    pending = paths["pool_pending"] / "000-development.json"
    pending.write_text(json.dumps(legacy))  # simulate a pre-repair frozen task
    before = pending.read_bytes()
    refresh(pool, 1)
    status = bank.pool_status(paths)
    assert status["pending"] == 3
    assert status["pending_training"] == 2
    for _ in range(2):
        task = bank.claim_pool_task(paths, 0, ttl_seconds=3600)
        assert task["body"]["split"] == "train"
        bank.retire_task(paths, task, outcome={})
    assert bank.claim_pool_task(paths, 0, ttl_seconds=3600) is None
    assert pending.read_bytes() == before
    assert bank.pool_status(paths)["pending_training"] == 0
    assert "000-development reason=receipt-ineligible" in capsys.readouterr().out


def test_claim_pool_task_blocks_already_retired_round_scoped_straggler(pool, capsys):
    bank, paths, *_ = pool
    first = refresh(pool, 1)
    row = first["frozen"][0]
    source = paths["pool_pending"] / (row["task_id"] + ".json")
    task = json.loads(source.read_text())
    source.rename(paths["pool_done"] / source.name)
    task["task_id"] = "000-straggler-new-round"
    task.pop("replay_count")
    straggler = paths["pool_pending"] / (task["task_id"] + ".json")
    straggler.write_text(json.dumps(task))
    before = straggler.read_bytes()
    status = bank.pool_status(paths)
    assert status["pending"] == 2
    assert status["pending_training"] == 1
    claimed = bank.claim_pool_task(paths, 0, ttl_seconds=3600)
    assert claimed["task_sha256"] != task["task_sha256"]
    assert straggler.read_bytes() == before
    assert "reason=replay-limit" in capsys.readouterr().out
