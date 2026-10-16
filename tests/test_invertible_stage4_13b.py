"""X4 source-named CPU regression tests; no released-weight load or GPU use."""
import json
import math
from pathlib import Path

import pytest

from experiments.expressivity_tasks import run_invertible_stage2 as runner
from experiments.expressivity_tasks import invertible_stage4_13b as stage4


def history(final=9.995):
    return [{"step": s, "best_accuracy": .2, "train_loss_window_first250": 10.,
             "train_loss_window_final250": final} for s in range(0, 10250, 250)]


def test_amendment5_calibrated_plateau_is_strict_and_keeps_old_rule():
    rows = history()
    assert not runner.plateau(rows[:-1], .001)  # Not eligible before10K.
    assert runner.plateau(rows, .001)
    assert not runner.plateau(history(9.99), .001)  # Exactly0.1% is movement.
    assert not runner.plateau(history(9.95), .001)
    assert runner.plateau(history(9.95))  # Old receipts retain1% rule.
    rows[-1]["best_accuracy"] = .201
    assert not runner.plateau(rows, .001)  # Exactly0.1pp is movement.
    assert runner.plateau(history(10.01), .001)  # Worsening is stagnant.


def test_amendment5_named_marginal_optimal_reference_is_computed_not_shorthand():
    assert runner.MARGINAL_OPTIMAL_LOSS == math.log(3721)
    assert runner.MARGINAL_REFERENCE["display"] == "marginal-optimal loss ln(3721) = 8.221748 nats"
    counts = {"count": 2, "correct": 1, "x_correct": 1, "y_correct": 1,
              "terminal_correct": 1, "terminal_count": 2, "post_suffix7_count": 0}
    assert runner.finish_metrics(counts, 4.)["marginal_optimal_reference"] == runner.MARGINAL_REFERENCE


def args(*extra):
    return runner.parse_args(["--arm", "e97", "--seed", "0", "--drivers", "128", "--out", "unused",
                              "--protocol-amendment", "5", "--lr", "0.001", *extra])


def test_amendment5_accepts_only_the_registered_six_run_recipe():
    assert args().lr == .001
    for extra in (["--lr", "0.0002"], ["--seed", "3"], ["--arm", "e97_identity"],
                  ["--drivers", "1"], ["--pilot"]):
        with pytest.raises(SystemExit):
            args(*extra)
    assert args("--checkpoint-dir", "/weights/run").checkpoint_dir == "/weights/run"


def test_amendment5_pruning_checks_actual_weights_filesystem_and_keeps_required(tmp_path, monkeypatch):
    from collections import namedtuple
    out, weights = tmp_path / "receipts" / "run", tmp_path / "weights"
    out.mkdir(parents=True)
    weights.mkdir()
    other = out.parent / "other"
    other.mkdir()
    (other / "selection_frozen.json").write_text('{"stop_step": 10750}')
    cps = {}
    for step in [0, 250, 500, 1000, 4000, 10000, 10750, 11000]:
        path = weights / f"step_{step:06d}.pt"
        path.write_bytes(str(step).encode())
        cps[str(step)] = {"path": str(path.resolve()), "sha256": runner.digest(path)}
    usage = namedtuple("Usage", "total used free")
    checked = []
    def disk(path):
        checked.append(Path(path))
        return usage(1000, 900, 100)
    monkeypatch.setattr(runner.shutil, "disk_usage", disk)
    runner.prune_checkpoints(out, cps, {0, 500}, 11000, weights)
    assert checked == [weights]
    assert not Path(cps["250"]["path"]).exists()
    assert all(Path(cp["path"]).exists() for s, cp in cps.items() if s != "250")
    event = json.loads((out / "pruning.jsonl").read_text())
    assert event["pruned_steps"] == [250]
    assert event["known_stop_steps"] == [10750]


def test_amendment5_missing_runs_fail_closed_and_print_named_references(tmp_path):
    (tmp_path / "amendment5.json").write_text('{"amendment": 5}')
    report = tmp_path / "report.md"
    assert not stage4.build(tmp_path, report)
    summary = json.loads((tmp_path / "results_summary.json").read_text())
    assert summary["status"] == "incomplete_fail_closed"
    assert len(summary["runs"]) == 6
    assert not summary["aggregate"]
    assert runner.MARGINAL_REFERENCE["display"] in report.read_text()
    assert "0.026874%" in report.read_text()
    assert "no final architecture conclusion" in report.read_text()


def test_amendment5_aggregator_replays_calibrated_not_legacy_plateau(tmp_path):
    """A run that continues past legacy1% but not calibrated0.1% must be accepted."""
    from experiments.expressivity_tasks.aggregate_invertible_stage2 import read_result
    run = tmp_path / "headline_e97_seed0"
    run.mkdir()
    rows = [{"step": s, "accuracy": .2, "loss": 9., "best_accuracy": .2,
             "train_loss_window_first250": None if s < 2000 else 10.,
             "train_loss_window_final250": None if s < 2000 else (9.95 if s == 10000 else 9.995)}
            for s in range(0, 10500, 250)]
    # Only stop check assertions are reached before absent checkpoint/data artifacts.
    (run / "validation.json").write_text(json.dumps(rows))
    (run / "config.json").write_text(json.dumps({"protocol_amendment": 5, "min_relative_train_loss_improvement": .001}))
    (run / "result.json").write_text(json.dumps({"step": 0, "stop_step": 10250, "stop_reason": "plateau"}))
    with pytest.raises(FileNotFoundError, match="selection_frozen"):
        read_result(tmp_path, run.name)
