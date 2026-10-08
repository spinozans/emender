"""Source-named F7 accounting tests; failures cannot inflate accepted throughput."""
import importlib.util
import json
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location("rl_teacher_probe_report",
    Path(__file__).resolve().parents[1] / "scripts/rl-loop-v1/rl_teacher_probe_report.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_measure_uses_full_wall_and_retains_failed_worker_api_attempts(tmp_path):
    good = {"index": 0, "receipt": True, "grade_passed": True, "status": "finished",
            "close_verified": True, "targets": 100, "elapsed_s": 20}
    failed = {"index": 1, "receipt": False, "grade_passed": False, "targets": 0,
              "elapsed_s": 280, "error": "worker killed"}
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps({"rows": [good, failed], "n": 2, "wall_s": 300,
        "model": "model", "concurrency": 2, "completed_per_hour": 999999}))
    for index in range(2):
        (tmp_path / f"case-{index:04d}").mkdir()
    (tmp_path / "case-0000/api.jsonl").write_text(json.dumps({"latency_s": 10, "attempt": 0,
        "usage": {"completion_tokens": 150, "prompt_tokens": 900}})+"\n")
    (tmp_path / "case-0001/api.jsonl").write_text(json.dumps({"latency_s": 240, "attempt": 1,
        "error": "HTTPError: HTTP Error 429: Too Many Requests"})+"\n")
    row = module.measure(path)
    assert row["finalized_outcomes_per_hour"] == 24
    assert row["completed_closed_per_hour"] == row["receipts_per_hour"] == 12
    assert row["accepted_targets_per_hour"] == 1200
    assert row["api_attempts"] == 2 and row["api_errors"] == row["http_429"] == 1
    assert row["queue_timeout_240s_errors"] == row["transport_retry_attempts"] == 1
    assert row["api_latency_s"]["p50"] == 125
    assert row["retry_outcomes"]["no_receipt_after_api_error"] == 1
    assert row["api_completion_tokens_per_receipt"] == 150
    good["close_verified"] = False
    data = json.loads(path.read_text()); data["rows"][0] = good
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        module.measure(path)


def test_distribution_empty_single_and_quantiles():
    assert module.distribution([])["p95"] is None
    assert module.distribution([7])["p95"] == 7
    assert module.distribution([0, 10])["p90"] == 9
