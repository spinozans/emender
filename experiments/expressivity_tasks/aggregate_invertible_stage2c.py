"""Receipt-derived X2c report; validates the common gated LR and frozen-data fairness."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from experiments.expressivity_tasks.aggregate_invertible_stage2 import ARMS, aggregate, number, read_result
from experiments.expressivity_tasks.invertible_stage2c import FAIRNESS, ROOT, audit_pilot, now
from experiments.expressivity_tasks.run_invertible_stage2 import FLOOR, MATCHED_STEPS, digest, plateau, write_json


def audit_run(root, name, result, lr):
    run = root / name
    cfg = result["config"]
    if not (cfg["protocol_amendment"] == 3 and cfg["lr"] == lr and not cfg["pilot"]
            and cfg["batch"] == 32 and cfg["weight_decay"] == .01 and cfg["warmup"] == 300
            and cfg["betas"] == [.9, .95] and cfg["grad_clip"] == .5
            and cfg["train_sequences"] == "fresh procedural per batch"):
        raise RuntimeError(f"common recipe/LR mismatch: {name}")
    if cfg["source_sha256"] != digest(root / "receipts/runner_at_launch.py"):
        raise RuntimeError(f"execution source mismatch: {name}")
    prior = root.with_name("results_invertible_stage2b") / name
    if result["data_receipt"] != json.loads((prior / "data_receipt.json").read_text()):
        raise RuntimeError(f"frozen pools differ from X2b: {name}")
    prior_cfg = json.loads((prior / "config.json").read_text())
    for key in ["source_checkpoint_sha256", "source_args_sha256", "trainable_parameters",
                "recurrent_states", "padding", "emission", "physical_train_tokens_per_sequence"]:
        if cfg[key] != prior_cfg[key]:
            raise RuntimeError(f"released model/task recipe mismatch: {name}/{key}")
    if result["stop_step"] < 10000 or not (plateau(json.loads((run / "validation.json").read_text()))
                                          or result["stop_step"] == 60000):
        raise RuntimeError(f"ineligible stop: {name}")
    losses = [json.loads(line)["loss"] for line in (run / "training.jsonl").read_text().splitlines()]
    for row in json.loads((run / "validation.json").read_text()):
        s = row["step"]
        if s >= 2000:
            for field, expected in [("train_loss_window_first250", statistics.mean(losses[s-2000:s-1750])),
                                    ("train_loss_window_final250", statistics.mean(losses[s-250:s]))]:
                if not math.isclose(row[field], expected, rel_tol=1e-12):
                    raise RuntimeError(f"loss-window replay mismatch: {name}/{s}")
    if (root / "receipts" / name / "exit.txt").read_text().splitlines()[0] != "exit_code=0":
        raise RuntimeError(f"nonzero exit: {name}")
    if (run / "selection_frozen.json").stat().st_mtime_ns >= (run / "best_test.json").stat().st_mtime_ns:
        raise RuntimeError(f"test selection chronology invalid: {name}")
    cp = json.loads((run / "checkpoints.json").read_text())
    for s in set(result["matched_test"]) | {str(result["step"])}:
        if digest(cp[s]["path"]) != cp[s]["sha256"]:
            raise RuntimeError(f"checkpoint mismatch: {name}/{s}")
    if not {str(s) for s in MATCHED_STEPS if s <= result["stop_step"]} <= set(result["matched_test"]):
        raise RuntimeError(f"mandatory matched panel missing: {name}")
    panels = [result["test"], *result["matched_test"].values()]
    for row in result.get("cross_plateau_test", {}).values():
        if row["status"] == "evaluated":
            if digest(row["checkpoint"]["path"]) != row["checkpoint"]["sha256"]:
                raise RuntimeError(f"cross-stop checkpoint mismatch: {name}")
            panels.append(row["panels"])
        elif row["reason"] not in ("arm already stopped; no extra training", "checkpoint pruned before stop step became required"):
            raise RuntimeError(f"unknown unavailable reason: {name}")
    for panel in panels:
        expected = {"test_1x"} if cfg["drivers"] == 1 else {"test_1x", "test_4x", "test_8x"}
        if set(panel) != expected:
            raise RuntimeError(f"missing depth panel: {name}")
        for key, m in panel.items():
            scale = int(key.split("_")[1][:-1])
            if not (m["count"] == 512 * cfg["drivers"] * scale and m["terminal_count"] == 512
                    and m["accuracy"] == m["correct"] / m["count"]
                    and m["x_accuracy"] == m["x_correct"] / m["count"]
                    and m["y_accuracy"] == m["y_correct"] / m["count"]):
                raise RuntimeError(f"metric counts invalid: {name}/{key}")
    return {"status": "passed", "common_lr": lr, "frozen_pools_identical_to_X2b": True,
            "loss_windows_replayed": True, "selected_matched_cross_checkpoints_rehashed": True,
            "selection_precedes_test": True, "metric_counts_verified": True}


def build(root, report):
    amendment = json.loads((root / "amendment3.json").read_text())
    decision = json.loads((root / "lr_decision.json").read_text()) if (root / "lr_decision.json").exists() else None
    trials = []
    for tag, lr in [("2e-4", 2e-4), ("1e-3", 1e-3)]:
        if (root / f"pilot_gate_{tag}.json").exists():
            pair = {a: audit_pilot(root, a, tag, lr) for a in ("e97", "gdn2")}
            if pair["e97"]["data_receipt"] != pair["gdn2"]["data_receipt"]:
                raise RuntimeError("pilot pools unpaired")
            trials.append(pair)
    if decision:
        if trials != decision["trials"]:
            raise RuntimeError("LR decision trial receipts changed")
        passes = [p for p in trials if p["e97"]["gate"]["passed"]]
        if decision["status"] == "passed" and (len(passes) != 1 or decision["lr"] != passes[0]["e97"]["lr"]):
            raise RuntimeError("common LR was not selected by E97 gate")
        if len(trials) == 2 and trials[0]["e97"]["gate"]["passed"]:
            raise RuntimeError("unauthorized escalation after passing gate")
        if decision["status"] == "failed_both_rates":
            if len(trials) != 2 or passes or decision["lr"] is not None:
                raise RuntimeError("invalid failed-both-rates decision")
        elif decision["status"] != "passed":
            raise RuntimeError("unknown LR decision status")
    gate_failed = bool(decision and decision["status"] == "failed_both_rates")
    runs, audits = {}, {}
    for name in ARMS:
        run = root / name
        if (run / "result.json").exists() and name.startswith("headline") and not (run / "cross_plateau_test.json").exists():
            runs[name] = {"status": "awaiting_cross_stop_panels"}
            continue
        result = read_result(root, name)
        if gate_failed and result["status"] == "not_completed":
            if run.exists() or (root / "receipts" / name).exists():
                raise RuntimeError("a full run was launched despite failed gate")
            result = {"status": "not_launched_gate_failed", "reason": "no authorized common LR"}
        if result["status"] == "complete":
            if not decision or decision["status"] != "passed":
                raise RuntimeError("full run without passing gate")
            audits[name] = audit_run(root, name, result, decision["lr"])
        runs[name] = result
    complete = all(r["status"] == "complete" for r in runs.values())
    aggregates, contrasts = {}, []
    for arm in ("e97", "gdn2"):
        group = [runs[f"headline_{arm}_seed{s}"] for s in range(3)]
        if all(r["status"] == "complete" for r in group):
            aggregates[arm] = {"best_validation": aggregate([r["best_validation"]["accuracy"] for r in group]),
                               "steps_to_plateau": aggregate([r["stop_step"] for r in group]),
                               **{k: aggregate([r["test"][k]["accuracy"] for r in group])
                                  for k in ("test_1x", "test_4x", "test_8x")}}
    if complete:
        for seed in range(3):
            e, g = [runs[f"headline_{a}_seed{seed}"] for a in ("e97", "gdn2")]
            if e["data_receipt"] != g["data_receipt"]:
                raise RuntimeError("unpaired full-run evaluation pools")
            contrasts.append({"seed": seed, **{k: e["test"][k]["accuracy"] - g["test"][k]["accuracy"]
                                             for k in ("test_1x", "test_4x", "test_8x")}})
        e, identity = runs["headline_e97_seed0"], runs["headline_e97_identity_seed0"]
        if e["data_receipt"] != identity["data_receipt"] or identity["config"]["identity_layers"] != 11:
            raise RuntimeError("identity ablation differs from paired contract")
        write_json(root / "receipts/final_audit.json", {"status": "passed", "audited_at": now(), "runs": audits,
                   "pilot_trials_replayed": True, "common_lr_gate_replayed": True,
                   "all_ten_exit_codes_zero": True, "all_frozen_pools_identical_to_X2b": True})
    if gate_failed:
        if any(r["status"] != "not_launched_gate_failed" for r in runs.values()):
            raise RuntimeError("full-run artifacts exist after failed gate")
        write_json(root / "receipts/final_audit.json", {"status": "passed", "audited_at": now(),
                   "pilot_trials_replayed": True, "common_lr_gate_replayed": True,
                   "all_four_pilot_exit_codes_zero": True, "all_frozen_pools_identical_to_X2b": True,
                   "full_runs_launched": 0, "reason": "both registered E97 LR gates failed; stop per supervisor ruling"})
    status = "complete" if complete else ("gate_failed_no_full_runs" if gate_failed else "incomplete_fail_closed")
    summary = {"status": status, "amendment3": amendment, "lr_decision": decision, "pilot_trials": trials,
               "suffix_floor": FLOOR, "runs": runs, "aggregate": aggregates, "paired_e97_minus_gdn2": contrasts,
               "uncertainty": "three training seeds; sample SD and SEM, not token-binomial independent-observation errors"}
    write_json(root / "results_summary.json", summary)
    text = ["# X2c — fresh both-arm fine-tunes at an E97-gated common LR", "",
            f"Status: **{status}**. Exact suffix floor **{number(FLOOR)}** (printed rounded: **0.0269%**, 1/3721).", "",
            ("Neither registered LR rescued E97 by the approved gate. No gated LR exists: the seven full runs and three one-step diagnostics were NOT launched. Three-seed means, steps-to-plateau, x4/x8 and ablation effects are unavailable, not zero or imputed. Stopping after both gates fail is the binding supervisor ruling."
             if gate_failed else "Primary comparison is pending unless all ten full-run receipts pass audit."), "",
            "## Amendment #3 registration", "", "```json", json.dumps(amendment, indent=2), "```", "",
            "## Fairness rule (verbatim)", "", FAIRNESS, "",
            amendment["x2b_status"], "",
            ("That is the registration's intended primary-comparison designation, not a claim of completed replacement evidence: the gate failed, so no gated-LR primary comparison exists and no X2b measurement has been replaced."
             if gate_failed else "X2b remains the archived 5e-5 sensitivity point."), "", "## Pilot stall/rescue evidence", "",
            "Gate preregistered before launch: 3000 fresh updates, trailing2000 first250/final250 means, >1% improvement AND negative OLS slope. Both arms use fresh released weights/cleared AdamW states for every attempt. No test-pool selection.", "",
            "| Pilot | LR | First250 loss | Final250 loss | Improvement | OLS nats/update | Gate | Floor |",
            "|---|---:|---:|---:|---:|---:|---|---:|"]
    for pair in trials:
        for p in pair.values():
            g = p["gate"]
            text.append(f"| {p['name']} | {p['lr']:g} | {g['first250_mean']:.6f} | {g['final250_mean']:.6f} | {100*g['relative_improvement']:.4f}% | {g['ols_slope_nats_per_update']:.9g} | {g['passed']} | {number(FLOOR)} |")
    text += ["", f"Common LR decision: `{decision['lr'] if decision else 'pending'}`; decision timestamp: `{decision['decided_at'] if decision else 'pending'}`.", "",
             "## Honest per-seed comparison", "",
             "Full-vocabulary argmax exact joint class `61*x+y`; invalid classes are wrong. Best validation accuracy, then CE, then earlier update selects every arm. Test panels are evaluated only after selection_frozen.json. Fresh procedural batch32; unchanged frozen512 val/test pools; right-pad to16 and slice before loss/scoring.", "",
             "| Run | Best val | Selected update | Stop update/reason | Test1x | Test4x | Test8x | Floor |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for name, r in runs.items():
        if r["status"] != "complete":
            text.append(f"| {name} | {r['status']} | — | — | — | — | — | {number(FLOOR)} |")
        else:
            panel = [number(r["test"][f"test_{x}x"]["accuracy"]) if f"test_{x}x" in r["test"] else "not applicable" for x in (1, 4, 8)]
            text.append(f"| {name} | {number(r['best_validation']['accuracy'])} | {r['step']} | {r['stop_step']}/{r['stop_reason']} | {' | '.join(panel)} | {number(FLOOR)} |")
    text += ["", "## Three-seed means ± sample SD (SEM in JSON)", "",
             "| Arm | Best val | Steps to plateau/cap | Test1x | Test4x | Test8x | Floor |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, a in aggregates.items():
        cells = [f"{number(a[k]['mean'])} ± {number(a[k]['sample_sd'])}" for k in ("best_validation", "test_1x", "test_4x", "test_8x")]
        s = a["steps_to_plateau"]
        text.append(f"| {arm} | {cells[0]} | {s['mean']:.1f} ± {s['sample_sd']:.1f} | {' | '.join(cells[1:])} | {number(FLOOR)} |")
    if gate_failed:
        for arm in ("e97", "gdn2"):
            text.append(f"| {arm} | unavailable: gate failed | unavailable | unavailable | unavailable | unavailable | {number(FLOOR)} |")
    text += ["", "## Coordinates and cost", "",
             "| Run | x1x | y1x | Terminal1x | Parameters | Matrix state elements | Real train tokens | Tokens/s | Wall seconds | Floor |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in runs.items():
        if r["status"] == "complete":
            m = r["test"]["test_1x"]
            state = sum(s["elements"] for s in r["config"]["recurrent_states"])
            text.append(f"| {name} | {number(m['x_accuracy'])} | {number(m['y_accuracy'])} | {number(m['terminal_accuracy'])} | {r['trainable_parameters']} | {state} | {r['total_training_tokens']} | {r['training_tokens_per_second']:.1f} | {r['wall_seconds']:.1f} | {number(FLOOR)} |")
    text += ["", "Real train tokens include initialization, exclude pad overhead. State counts exclude short-conv caches; per-layer shapes/flags in configs. Released full-vocabulary heads and all parameters are fine-tuned; not scratch-model or parameter-matched arms.", "",
             "## Matched-step and union-of-stop panels", "",
             "No extra training beyond stop; pruned-before-required cells are unavailable, never imputed.", "",
             "| Run | Update | Test1x | Test4x | Test8x | Floor |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in runs.items():
        if r["status"] != "complete" or not name.startswith("headline"):
            continue
        panels = {s: {"status": "evaluated", "panels": p} for s, p in r["matched_test"].items()}
        panels.update(r["cross_plateau_test"])
        for s, row in sorted(panels.items(), key=lambda item: int(item[0])):
            cells = [number(row["panels"][f"test_{x}x"]["accuracy"]) for x in (1, 4, 8)] if row["status"] == "evaluated" else [f"unavailable: {row['reason']}"] * 3
            text.append(f"| {name} | {s} | {' | '.join(cells)} | {number(FLOOR)} |")
    text += ["", "## Interpretation and receipts", "",
             ("The observed pilot loss slopes are negative but the required >1% window movement is absent at both registered LRs. This is a failed rescue gate, not a seven-run architecture comparison, arithmetic diagnostic, extrapolation result, or impossibility claim. The requested full comparison remains unperformed; any further training requires a new operator decision."
              if gate_failed else "No architecture conclusion is claimed from an incomplete experiment."), "",
             "This comparison tests learning efficiency of this released pair under one common LR, not universal representability or CMA causality. A one-step arithmetic failure blocks a depth-specific interpretation; a single-seed tanh-removal contrast is exploratory. Near-floor results are not learned map computation. All seeds and incomplete/failing arms remain visible.", "",
             ("Pilot receipts bind source checkpoint/args/runner hashes, exact commands, leased GPU UUID/index snapshots, training/validation logs, fresh optimizer declaration, exit0 for all four pilots, unchanged frozen-pool hashes, replayed gate windows/slopes and final checkpoint hashes. No full-run selected/test/matched/cross-stop measurements exist."
              if gate_failed else "Receipts bind source checkpoint/args/runner hashes, exact commands, leased GPU UUID/index snapshots, logs, optimizer declaration, exit codes, frozen pool hashes, loss-window replay, selected/matched/cross-stop checkpoint hashes and metric counts."), "",
             "Binary checkpoints are local ignored artifacts, identified by hashes, not git payloads. See receipts/cpu_tests.txt, receipts/environment.txt, receipts/validation_commands.txt and receipts/final_audit.json. Independent acceptance review remains required.", "",
             "X2 and X2b evidence is preserved. Local lambda01, exclusive scripts/gpu_lease.sh leases; no Slurm or WorksGood service. Resilient DiLoCo gap-matrix IDs: not applicable (no resilient behavior changed).", ""]
    report.write_text("\n".join(text) + "\n")
    return status in ("complete", "gate_failed_no_full_runs")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--report", type=Path, default=Path("X2c-report.md"))
    a = p.parse_args()
    if not build(a.root, a.report):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
