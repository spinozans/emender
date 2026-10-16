"""Amendment #3 registration and fail-closed, receipt-replayed common-LR gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from experiments.expressivity_tasks.run_invertible_stage2 import digest, pilot_gate, write_json

ROOT = Path("experiments/expressivity_tasks/results_invertible_stage2c")
FAIRNESS = "whichever lr rescues E97 is the lr BOTH arms run; no per-arm tuning; if 1e-3 is needed to move E97, GDN2 runs 1e-3 too."


def now():
    return datetime.now(timezone.utc).isoformat()


def register(root):
    path = root / "amendment3.json"
    if path.exists():
        raise FileExistsError("amendment #3 already registered; never silently overwrite")
    prior = Path("experiments/expressivity_tasks/results_invertible_stage2b")
    write_json(path, {
        "amendment": 3, "registered_at": now(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "authority": "/home/erikg/emender-scratch/fix-fanout/taskX2c-eq-lr.md and live supervisor gate/recipe ruling",
        "recipe": "X2b unchanged except LR: fresh AdamW (0.9,0.95), batch32, wd0.01, clip0.5, bf16, 300 warmup then constant LR; Schedule-Free y-mode released-weight recovery, no optimizer state loaded",
        "pilot": "Both released arms seed0, drivers128, 3000 fresh updates at2e-4; E97 gate: >1% first250/final250 improvement AND negative OLS slope, both over trailing2000 updates. If fail, one fresh paired repeat at1e-3. If both fail, no full runs.",
        "full_runs": "Seven headline runs: E97/GDN2 seeds0,1,2; E97 identity seed0. Three seed0 one-step diagnostics, preserving X2b contract. All start fresh from released weights at gated LR.",
        "fairness_rule": FAIRNESS,
        "evaluation": "X2/X2b frozen val/test pools unchanged, same seeds/input hashes; validation-only selection, joint-class full-vocab argmax; causal right-padding to16; x4/x8; matched and union-of-stop panels",
        "plateau": "Eligibility>=10000; trailing2000 first250-vs-final250 mean train-loss improvement<1% AND best-val gain<0.001; cap60000; all arms identical",
        "capacity": json.loads((prior / "amendment2.json").read_text())["capacity"],
        "x2b_status": "X2b's 5e-5 results are retained as the lr-sensitivity data point, superseded as the primary comparison by the gated-lr runs. Active X2b GDN2 runs finish naturally and are archived, never cancelled or modified by X2c.",
        "basis": {"X2b_report_sha256": digest("X2b-report.md"),
                  "X2b_amendment_sha256": digest(prior / "amendment2.json"),
                  "task_sha256": digest("/home/erikg/emender-scratch/fix-fanout/taskX2c-eq-lr.md")},
        "resilience_requirements": "Not applicable: standalone single-GPU expressivity fine-tunes, no DiLoCo behavior changed; local lambda01, no Slurm or WorksGood service"})


def audit_pilot(root, arm, tag, lr):
    run = root / f"pilot_{tag}_{arm}_seed0"
    result = json.loads((run / "pilot_result.json").read_text())
    config = json.loads((run / "config.json").read_text())
    lines = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    if len(lines) != 3000 or [r["step"] for r in lines] != list(range(1, 3001)):
        raise RuntimeError(f"pilot updates invalid: {run}")
    gate = pilot_gate([r["loss"] for r in lines])
    if gate != result["gate"] or result["status"] != "complete" or result["completed_updates"] != 3000:
        raise RuntimeError(f"pilot gate replay mismatch: {run}")
    if not (config["arm"] == arm and config["seed"] == 0 and config["drivers"] == 128
            and config["pilot"] and config["lr"] == lr and config["protocol_amendment"] == 3):
        raise RuntimeError(f"pilot config invalid: {run}")
    if digest(result["checkpoint"]["path"]) != result["checkpoint"]["sha256"]:
        raise RuntimeError(f"pilot checkpoint mismatch: {run}")
    data = json.loads((run / "data_receipt.json").read_text())
    if digest(run / "frozen_pools.npz") != data["sha256"]:
        raise RuntimeError(f"pilot pool hash mismatch: {run}")
    prior = root.with_name("results_invertible_stage2b") / f"headline_{arm}_seed0"
    if data != json.loads((prior / "data_receipt.json").read_text()):
        raise RuntimeError(f"pilot pools differ from X2b: {run}")
    prior_config = json.loads((prior / "config.json").read_text())
    for key in ("source_checkpoint_sha256", "source_args_sha256", "trainable_parameters", "recurrent_states",
                "padding", "emission", "batch", "betas", "weight_decay", "grad_clip", "warmup", "dtype"):
        if config[key] != prior_config[key]:
            raise RuntimeError(f"pilot recipe differs from X2b: {run}/{key}")
    if config["source_sha256"] != digest(root / "receipts/runner_at_launch.py"):
        raise RuntimeError(f"pilot execution source mismatch: {run}")
    exit_receipt = (root / "receipts" / run.name / "exit.txt").read_text()
    if "exit_code=0\n" not in exit_receipt:
        raise RuntimeError(f"pilot failed exit: {run}")
    return {"name": run.name, "lr": lr, "gate": gate, "data_receipt": data,
            "training_sha256": digest(run / "training.jsonl"),
            "pilot_result_sha256": digest(run / "pilot_result.json"),
            "checkpoint": result["checkpoint"]}


def gate(root, tag):
    if tag not in ("2e-4", "1e-3"):
        raise ValueError("only the registered two LR candidates are allowed")
    decision_path = root / "lr_decision.json"
    if decision_path.exists():
        raise FileExistsError("common-LR decision is immutable")
    trials = []
    for t, lr in [("2e-4", 2e-4), ("1e-3", 1e-3)]:
        pair = {arm: audit_pilot(root, arm, t, lr) for arm in ("e97", "gdn2")}
        if pair["e97"]["data_receipt"] != pair["gdn2"]["data_receipt"]:
            raise RuntimeError("pilot frozen pools are not paired")
        trials.append(pair)
        passed = pair["e97"]["gate"]["passed"]
        gate_path = root / f"pilot_gate_{t}.json"
        if gate_path.exists():
            previous = json.loads(gate_path.read_text())
            if previous["trials"] != trials or previous["e97_passed"] != passed:
                raise RuntimeError("published pilot gate receipts changed")
        else:
            write_json(gate_path, {"audited_at": now(), "trials": trials, "e97_passed": passed})
        if passed:
            write_json(decision_path, {"decided_at": now(), "status": "passed", "lr": lr,
                                      "fairness_rule": FAIRNESS, "trials": trials})
            return True
        if t == tag:
            if t == "1e-3":
                write_json(decision_path, {"decided_at": now(), "status": "failed_both_rates", "lr": None,
                                          "fairness_rule": FAIRNESS, "trials": trials})
            return False
    return False


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("action", choices=["register", "gate"])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--tag", choices=["2e-4", "1e-3"])
    a = p.parse_args()
    if a.action == "register":
        register(a.root)
    elif not gate(a.root, a.tag):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
