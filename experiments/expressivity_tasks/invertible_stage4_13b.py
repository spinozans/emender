"""Amendment #5 registration and receipt-derived matched-LR 1.3B comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from experiments.expressivity_tasks.aggregate_invertible_stage2 import aggregate, read_result
from experiments.expressivity_tasks.run_invertible_stage2 import (
    FLOOR, MARGINAL_REFERENCE, digest, write_json,
)

ROOT = Path("experiments/expressivity_tasks/results_invertible_stage4_13b")
ARMS = [f"headline_{arm}_seed{s}" for arm in ("e97", "gdn2") for s in range(3)]
REFERENCE = MARGINAL_REFERENCE["display"] + f"; suffix floor {100 * FLOOR:.6f}%"


def register(root):
    path = root / "amendment5.json"
    if path.exists():
        raise FileExistsError("amendment #5 already registered")
    prior = root.with_name("results_invertible_stage2")
    data = {}
    for seed in range(3):
        receipts = [json.loads((prior / f"headline_{arm}_seed{seed}" / "data_receipt.json").read_text())
                    for arm in ("e97", "gdn2")]
        if receipts[0] != receipts[1]:
            raise RuntimeError("X2 frozen pools are not paired")
        data[str(seed)] = receipts[0]
    old_capacity = json.loads((root.with_name("results_invertible_stage2b") / "amendment2.json").read_text())["capacity"]
    write_json(path, {
        "amendment": 5, "registered_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "authority": "/home/erikg/emender-scratch/fix-fanout/taskX4-13b-confirm.md; operator matched1e-3 ruling and live storage/concurrency/reference correction approvals",
        "matrix": [{"name": name, "arm": name.split("_")[1], "seed": int(name[-1]),
                    "lr": 0.001, "drivers": 128} for name in ARMS],
        "released_parameters": {"e97": 1286589072, "gdn2": 1286713448},
        "recipe": "scripts/expressivity_1p3b.py released strict loaders with Schedule-Free train/y weight recovery; full bf16 fine-tune, fresh AdamW(0.9,0.95), batch32, wd0.01, clip0.5, 300-update warmup then constant matched1e-3; no prior optimizer state loaded",
        "training": "Fresh procedural generate_batch on every update with seed0/1/2 streams paired across arms; archived128 pool unused; no pilot continuation",
        "evaluation": "Same X-series frozen512val/512test at128/512/1024 drivers plus2 initialization tokens; joint class61*x+y over full pretrained vocab, causal right-pad-to16 sliced before loss/scoring",
        "frozen_data_receipts": data,
        "selection": "Max exact joint-state validation accuracy, then lower validation CE, then earlier update; selection_frozen persisted before test/x4/x8; all3 seeds, never best-of-seed",
        "plateau": "Eligibility>=10000; every250 completed updates; trailing2000 best-val gain<0.001 (0.1pp) AND first250/final250 relative train-loss mean improvement<0.001 (0.1%); strict thresholds; cap60000; calibrated rule from amendment#4, not a proof of convergence",
        "marginal_optimal_reference": MARGINAL_REFERENCE,
        "reference_correction": "Task shorthand8.224 corrected before launch by live supervisor: ln(3721)=8.221747728346623, displayed8.221748 nats; 8.224 is only the task-spec shorthand, not an equality. Below-reference loss indicates learned predictive signal, not by itself deep computation.",
        "supersession": "Matched1e-3 for BOTH arms supersedes pilot per-architecture E97@2e-4/GDN2@1e-3 reading, not archived evidence. X2b5e-5, X2c failed pilots, X3 symmetric8M grid/difficulty/one-step and ablation results all stand unchanged. No extra ablations/diagnostics launched by X4.",
        "capacity": old_capacity,
        "storage": "Binary weights only off-repo at /mnt/nvme1n1/erikg/invertible_stage4_13b/<run>/; receipts bind absolute paths, bytes and SHA256. Capacity checks target actual weights filesystem. JSON/logs/pools remain this results root. No older roots moved or pruned; X3 retention unchanged.",
        "resources": "Six concurrent single-GPU processes on six exclusive scripts/gpu_lease.sh leases explicitly approved; single-writer implementation/debugging; local lambda01 only",
        "basis": {"task_sha256": digest("/home/erikg/emender-scratch/fix-fanout/taskX4-13b-confirm.md"),
                  **{f"{r}_sha256": digest(r) for r in ("X2b-report.md", "X2c-report.md", "X3-report.md")}},
        "resilience_requirements": "Not applicable: no DiLoCo behavior changed, no Frontier/Slurm or WorksGood service"})


def percent(value):
    return f"{100 * value:.6f}%"


def panel(metrics):
    return f"{percent(metrics['accuracy'])} / {metrics['loss']:.6f}"


def build(root, report):
    amendment = json.loads((root / "amendment5.json").read_text())
    runs = {name: read_result(root, name) for name in ARMS}
    complete = all(r["status"] == "complete" for r in runs.values())
    aggregates = {}
    for arm in ("e97", "gdn2"):
        group = [runs[f"headline_{arm}_seed{s}"] for s in range(3)]
        if all(r["status"] == "complete" for r in group):
            aggregates[arm] = {"best_validation": aggregate([r["best_validation"]["accuracy"] for r in group]),
                "validation_loss": aggregate([r["best_validation"]["loss"] for r in group]),
                "steps_to_stop": aggregate([r["stop_step"] for r in group]),
                **{f"{length}_{key}": aggregate([r["test"][length][key] for r in group])
                   for length in ("test_1x", "test_4x", "test_8x")
                   for key in ("accuracy", "loss", "post_suffix7_accuracy", "x_accuracy", "y_accuracy", "terminal_accuracy")}}
    contrasts = []
    for seed in range(3):
        e, g = runs[f"headline_e97_seed{seed}"], runs[f"headline_gdn2_seed{seed}"]
        if e["status"] == g["status"] == "complete":
            if e["data_receipt"] != g["data_receipt"] or e["data_receipt"] != amendment["frozen_data_receipts"][str(seed)]:
                raise RuntimeError("frozen data differ from preregistered X-series pools")
            contrasts.append({"seed": seed, **{f"{length}_{key}": e["test"][length][key] - g["test"][length][key]
                for length in ("test_1x", "test_4x", "test_8x") for key in ("accuracy", "loss")}})
    summary = {"status": "complete" if complete else "incomplete_fail_closed", "amendment5": amendment,
        "suffix_floor": FLOOR, "marginal_optimal_reference": MARGINAL_REFERENCE,
        "runs": runs, "aggregate": aggregates, "paired_e97_minus_gdn2": contrasts,
        "paired_aggregate": {k: aggregate([r[k] for r in contrasts]) for k in contrasts[0] if k != "seed"} if len(contrasts) == 3 else {},
        "uncertainty": "All3 training seeds; sample SD and SEM, not token-binomial intervals; running positions correlated"}
    write_json(root / "results_summary.json", summary)
    text = ["# X4 — released 1.3B pair, matched lr 1e-3", "", f"Status: **{summary['status']}**.", "",
        f"Named reference throughout: **{REFERENCE}**. Task-spec shorthand `8.224` was corrected before launch; it is not ln(3721).",
        "Loss below the marginal-optimal reference indicates predictive signal, not alone learned deep computation. Full-vocabulary argmax exact joint class61*x+y; invalid classes fail both coordinates. All table accuracy units are percent; CE is nats.", "",
        "## Amendment #5 registration (before launch)", "", "```json", json.dumps(amendment, indent=2), "```", "",
        "Fresh released weights and cleared AdamW states, not pilot continuation. Both arms matched1e-3, warmup300 then constant, batch32, wd.01, clip.5. Fresh procedural batches; unchanged frozen evaluation. Causal right-pad-to16; pad logits never scored. Validation-only selection frozen before test. Calibrated plateau eligibility10K, trailing2000 val<0.1pp AND train-window<0.1%, cap60K. No new arithmetic controls or ablations in this stage.", "",
        "## Three-seed final comparison (mean ± sample SD; SEM in JSON)", "",
        "| Arm / LR | Best val % | Val CE | Test1x % / CE | x4 % / CE | x8 % / CE | Stop updates | Named references |",
        "|---|---:|---:|---:|---:|---:|---|---|"]
    for arm, a in aggregates.items():
        values = []
        for length in ("test_1x", "test_4x", "test_8x"):
            acc, loss = a[f"{length}_accuracy"], a[f"{length}_loss"]
            values.append(f"{percent(acc['mean'])} ± {percent(acc['sample_sd'])} / {loss['mean']:.6f} ± {loss['sample_sd']:.6f}")
        stops = [runs[f"headline_{arm}_seed{s}"]["stop_step"] for s in range(3)]
        text.append(f"| {arm} / 0.001 | {percent(a['best_validation']['mean'])} ± {percent(a['best_validation']['sample_sd'])} | {a['validation_loss']['mean']:.6f} ± {a['validation_loss']['sample_sd']:.6f} | {' | '.join(values)} | {stops} | {REFERENCE} |")
    text += ["", "## Per-seed selected results", "",
        "| Run | Best val % / CE | Selected | Stop / reason | Test1x % / CE | x4 % / CE | x8 % / CE | Named references |",
        "|---|---:|---:|---|---:|---:|---:|---|"]
    for name, r in runs.items():
        if r["status"] != "complete":
            text.append(f"| {name} | {r['status']} | — | — | — | — | — | {REFERENCE} |")
            continue
        cells = [panel(r["test"][f"test_{x}x"]) for x in (1, 4, 8)]
        text.append(f"| {name} | {panel(r['best_validation'])} | {r['step']} | {r['stop_step']} / {r['stop_reason']} | {' | '.join(cells)} | {REFERENCE} |")
    text += ["", "## Matched-step panels and union of stopping steps", "",
        "Never train beyond a stop to fill a cell; pruned-before-required cells are unavailable, never imputed. Selection panels above are not equal-step comparisons.", "",
        "| Run | Step | Test1x % / CE | x4 % / CE | x8 % / CE | Named references |", "|---|---:|---:|---:|---:|---|"]
    for name, r in runs.items():
        if r["status"] != "complete":
            continue
        panels = dict(r["matched_test"])
        for step, cross in r["cross_plateau_test"].items():
            panels[step] = cross["panels"] if cross["status"] == "evaluated" else cross["reason"]
        for step, p in sorted(panels.items(), key=lambda kv: int(kv[0])):
            cells = [panel(p[f"test_{x}x"]) for x in (1, 4, 8)] if isinstance(p, dict) else ["unavailable: " + p] * 3
            text.append(f"| {name} | {step} | {' | '.join(cells)} | {REFERENCE} |")
    text += ["", "## Coordinates, depth-relevant post-suffix7, and cost", "",
        "| Run | x1x % | y1x % | Terminal1x % | Post7 1x % | Post7 x4 % | Post7 x8 % | Parameters | Real tokens | Train tokens/s | Wall seconds | Named references |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, r in runs.items():
        if r["status"] != "complete":
            continue
        t = r["test"]["test_1x"]
        cells = [percent(t[k]) for k in ("x_accuracy", "y_accuracy", "terminal_accuracy")]
        cells += [percent(r["test"][f"test_{x}x"]["post_suffix7_accuracy"]) for x in (1, 4, 8)]
        text.append(f"| {name} | {' | '.join(cells)} | {r['trainable_parameters']} | {r['total_training_tokens']} | {r['training_tokens_per_second']:.1f} | {r['wall_seconds']:.1f} | {REFERENCE} |")
    text += ["", "Real tokens include initialization, exclude pad overhead. Per-layer matrix-state geometry and flags are in configs. Seed-paired contrasts/counts and all full-vocabulary CE are retained in JSON.", "",
        "## Interpretation across the X-series", "",
        "Incomplete/fail-closed: no final architecture conclusion is supported." if not complete else
        "The released-pair matched1e-3 measurements above must be read alongside X2b's matched5e-5 fine-tunes, not as an LR-invariant architecture verdict. X2's frozen128 training pool memorization is superseded as primary evidence by fresh-data X2b; X2c's failed3K rescue gates remain failed gates, not completed full runs. X3's symmetric nominal8M (measured9.90M/9.887M) tuned grid selects1e-3 for both architectures on p61; both best-grid and matched-LR readings remain near the suffix reference on post7/x4/x8. Its p13/p31 ladder shows easier-task signal without establishing deep invertible-map computation. Late one-step arithmetic successes do not certify arithmetic mastery at an earlier long-task plateau, and failed arithmetic controls block a depth-specific interpretation. X2b's single-seed tanh-removal contrast is exploratory, not a causal recurrent-nonlinearity result. Full pretrained vocabulary/AdamW fine-tunes and task-sized-vocabulary/ScheduleFree scratch runs are different experiment classes; raw scale comparisons are not architecture-only effects. Differences in recurrence state size, pretraining and optimization remain confounds. Loss departures below the named marginal-optimal reference show predictive signal but do not by themselves establish robust deep computation, universal expressivity, impossibility, or CMA causality. The calibrated task-observed stopping rule is not proof of eventual convergence.", "",
        "## Receipts, storage, and validation", "",
        "Each run binds command, GPU lease/UUID, source checkpoint/args/runner hashes, optimizer freshness, paired frozen NPZ/input identities, every actual update/validation boundary, selected-before-test receipt, binary SHA256 and required-set pruning ledgers. NVMe weights are local off-repo artifacts; textual evidence and pools stay in the tracked results root. No old result roots modified. See receipts/cpu_tests.txt, launch_commit.json, environment.txt, final_audit.json and validation_commands.txt. Independent acceptance review remains required.", "",
        "Local lambda01; six exclusive GPU leases explicitly approved; no Slurm, Frontier, or WorksGood service. Resilient DiLoCo requirement IDs: not applicable (no resilient behavior changed)."]
    report.write_text("\n".join(text) + "\n")
    return complete


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("action", choices=["register", "report"])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--report", type=Path, default=Path("X4-report.md"))
    a = p.parse_args()
    if a.action == "register":
        register(a.root)
    else:
        build(a.root, a.report)


if __name__ == "__main__":
    main()
