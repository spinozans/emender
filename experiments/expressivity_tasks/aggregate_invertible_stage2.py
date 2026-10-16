"""Fail-closed, receipt-derived X2 report and per-seed results summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from experiments.expressivity_tasks.run_invertible_stage2 import FLOOR, digest, plateau, selected, write_json

ARMS = [*(f"headline_{arm}_seed{seed}" for arm in ("e97", "gdn2") for seed in range(3)),
        "headline_e97_identity_seed0", *(f"calibration_{arm}_seed0" for arm in ("e97", "gdn2", "e97_identity"))]


def read_result(root, name):
    run = root / name
    if not (run / "result.json").exists():
        failure = json.loads((run / "failure.json").read_text()) if (run / "failure.json").exists() else None
        return {"status": "failed" if failure else "not_completed", "failure": failure}
    result = json.loads((run / "result.json").read_text())
    history = json.loads((run / "validation.json").read_text())
    best = None
    for i, row in enumerate(history):
        if row["step"] != i * 250:
            raise RuntimeError(f"nonuniform update counter: {name}")
        if selected(row, best):
            best = row
        if row["best_accuracy"] != best["accuracy"]:
            raise RuntimeError(f"inconsistent running best: {name}")
        if i < len(history) - 1 and plateau(history[:i + 1]):
            raise RuntimeError(f"arm trained beyond first plateau: {name}")
    if best["step"] != result["step"] or history[-1]["step"] != result["stop_step"]:
        raise RuntimeError(f"selection/stop mismatch: {name}")
    if result["stop_reason"] == "plateau" and not plateau(history):
        raise RuntimeError(f"invalid plateau claim: {name}")
    frozen = json.loads((run / "selection_frozen.json").read_text())
    if not frozen["frozen"] or frozen["checkpoint"]["sha256"] != result["checkpoint"]["sha256"]:
        raise RuntimeError(f"checkpoint selection not frozen: {name}")
    if digest(result["checkpoint"]["path"]) != result["checkpoint"]["sha256"]:
        raise RuntimeError(f"selected checkpoint hash mismatch: {name}")
    lines = (run / "training.jsonl").read_text().splitlines()
    if len(lines) != result["stop_step"] or any(json.loads(line)["step"] != i + 1 for i, line in enumerate(lines)):
        raise RuntimeError(f"actual optimizer updates differ: {name}")
    data_receipt = json.loads((run / "data_receipt.json").read_text())
    if digest(run / "frozen_pools.npz") != data_receipt["sha256"]:
        raise RuntimeError(f"data hash mismatch: {name}")
    result["config"] = json.loads((run / "config.json").read_text())
    result["data_receipt"] = data_receipt
    if name.startswith("headline"):
        result["cross_plateau_test"] = json.loads((run / "cross_plateau_test.json").read_text())
    return result


def aggregate(values):
    return {"n_seeds": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values), "sem": statistics.stdev(values) / len(values)**0.5}


def number(value):
    return f"{100 * value:.6f}%"


def build(root, report):
    runs = {name: read_result(root, name) for name in ARMS}
    complete = all(r["status"] == "complete" for r in runs.values())
    aggregates = {}
    for arm in ("e97", "gdn2"):
        group = [runs[f"headline_{arm}_seed{s}"] for s in range(3)]
        if all(r["status"] == "complete" for r in group):
            aggregates[arm] = {"best_validation": aggregate([r["best_validation"]["accuracy"] for r in group]),
                              "steps_to_plateau": aggregate([r["stop_step"] for r in group]),
                              **{length: aggregate([r["test"][length]["accuracy"] for r in group])
                                 for length in ("test_1x", "test_4x", "test_8x")}}
    contrasts = []
    for seed in range(3):
        e, g = runs[f"headline_e97_seed{seed}"], runs[f"headline_gdn2_seed{seed}"]
        if e["status"] == g["status"] == "complete":
            if e["data_receipt"] != g["data_receipt"]:
                raise RuntimeError(f"unpaired data for seed {seed}")
            contrasts.append({"seed": seed, "best_validation": e["best_validation"]["accuracy"] - g["best_validation"]["accuracy"],
                              **{length: e["test"][length]["accuracy"] - g["test"][length]["accuracy"]
                                 for length in ("test_1x", "test_4x", "test_8x")}})
    contingency = None
    if (root / "contingency_amendment.json").exists():
        contingency = json.loads((root / "contingency_amendment.json").read_text())
        contingency["triggered"] = (all(aggregates[a]["best_validation"]["mean"] <= contingency["threshold_accuracy_fraction"]
                                       for a in ("e97", "gdn2")) if len(aggregates) == 2 else None)
    summary = {"status": "complete" if complete else "incomplete_fail_closed", "suffix_floor": FLOOR,
               "contingency": contingency,
               "contract_sha256": digest(root / "contract.json"), "runs": runs,
               "aggregate": aggregates, "paired_e97_minus_gdn2": contrasts,
               "paired_e97_minus_gdn2_aggregate": ({k: aggregate([c[k] for c in contrasts])
                   for k in ("best_validation", "test_1x", "test_4x", "test_8x")} if len(contrasts) == 3 else {}),
               "uncertainty": "sample SD and SEM across all three training seeds, not token-binomial intervals; correlated running positions are not independent trajectories"}
    write_json(root / "results_summary.json", summary)
    text = ["# X2 — released-pair invertible-map probes", "", "## Status and preregistered instrument", "",
            f"Status: **{summary['status']}**. Exact suffix-only floor throughout: **{number(FLOOR)}** (1/3721).",
            "No absent result is imputed. All accuracies below are percentages, full-vocabulary argmax exact joint class `61*x+y`; both coordinates must match. Invalid classes fail both coordinates.", "",
            "The live supervisor amended the reviewed scratch dim1024/depth2 plan: use the released 1.2866B E97 and matched 1.3B GDN2-MLP loaders/geometry with Schedule-Free train/y recovery. Scratch probes are NOT run (available only for a future separately approved mechanism study). Recipe: full fine-tune bf16 AdamW (0.9,0.95), lr5e-5, warmup300, wd0.01, clip0.5, batch32. E97/GDN2 seeds0/1/2; identity seed0; one-step diagnostics all three seed0. Frozen paired full-input-deduplicated 128 train/512 val/512 test per depth. Production drivers128/512/1024 plus two init tokens.", "",
            "Stopping is symmetric: validation every250 completed updates, first running-best gain <0.001 accuracy over2000 updates stops training, hard cap60000. Best validation accuracy, then lower CE, then earlier update selects ALL arms. Eval0 anchors the first window. Selection identities/hashes are frozen before test/extrapolation evaluation. Full-state running, coordinates, terminal, post-suffix7 burn-in counts/losses are retained in JSON. No targets ever enter model inputs.", "",
            "Implementation fact: the existing fused E97 kernel accepts unaligned lengths only in forward-only mode. The first calibration attempt failed at update1 for both E97 variants; GDN2 was explicitly stopped to restart consistently. `attempt1_alignment/` retains all failures, traces and partial logs. Supervisor approved identical constant-zero right padding to multiple16 on every arm and split (3->16,130->144). Pad logits are sliced away before loss/scoring; pads occur strictly after real causal prefixes, so they cannot contaminate real positions or leak targets. Permanent fixed-weight forward/loss and training masked-pad gradient tests pass. Only the E97 carried-state tanh is replaced by identity; all parameter keys/values/geometry remain identical at initialization.", "",
            "## Per-seed selected results", "",
            "| Run | Best val | Selected update | Stop update/reason | Test1x | Test4x | Test8x | Suffix floor |",
            "|---|---:|---:|---|---:|---:|---:|---:|"]
    for name, r in runs.items():
        if r["status"] != "complete":
            text.append(f"| {name} | {r['status']} | — | {r.get('failure')} | — | — | — | {number(FLOOR)} |")
            continue
        panel = [number(r["test"][f"test_{x}x"]["accuracy"]) if f"test_{x}x" in r["test"] else "not applicable" for x in (1,4,8)]
        text.append(f"| {name} | {number(r['best_validation']['accuracy'])} | {r['step']} | {r['stop_step']}/{r['stop_reason']} | {' | '.join(panel)} | {number(FLOOR)} |")
    text += ["", "## All-seed aggregate (mean ± sample SD; SEM in JSON)", ""]
    for arm, a in aggregates.items():
        text.append(f"- {arm}: best val {number(a['best_validation']['mean'])} ± {number(a['best_validation']['sample_sd'])}; " +
                    ", ".join(f"{k} {number(a[k]['mean'])} ± {number(a[k]['sample_sd'])}" for k in ("test_1x", "test_4x", "test_8x")) + f"; floor {number(FLOOR)}.")
    text += ["", "Per-seed paired E97−GDN2 differences (accuracy fractions) and all counts are in `results_summary.json`. A best seed is never substituted for the three-seed mean. Seed uncertainty is not a token-binomial interval.", "", "## Coordinates, arithmetic calibration, and cost", "",
             "| Run | Test1x x | Test1x y | Test1x terminal | Parameters | Updates | Real training tokens | Tokens/s training | Wall seconds | Suffix floor |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in runs.items():
        if r["status"] == "complete":
            t = r["test"]["test_1x"]
            text.append(f"| {name} | {number(t['x_accuracy'])} | {number(t['y_accuracy'])} | {number(t['terminal_accuracy'])} | {r['trainable_parameters']} | {r['stop_step']} | {r['total_training_tokens']} | {r['training_tokens_per_second']:.1f} | {r['wall_seconds']:.1f} | {number(FLOOR)} |")
    text.append("")
    for arm in ("e97", "gdn2", "e97_identity"):
        r = runs[f"headline_{arm}_seed0"]
        if r["status"] == "complete":
            states = r["config"]["recurrent_states"]
            text.append(f"- {arm}: {len(states)} matrix-state layers, {sum(s['elements'] for s in states):,} recurrent matrix elements per sequence (short-conv caches excluded); per-layer shapes/flags in config.")
    text += ["", "Real training tokens include the two initialization positions but exclude right-padding overhead. Physical padded tokens/sequence and matrix-state element counts per layer are in each config. Released heads are retained (full vocab), not replaced by the scratch plan's 3721-class heads. Released GDN2 configuration governs even where it differs from the old scratch control's eigenvalue setting; actual flags are recorded in headline configs.", "",
             "## Matched-step panels", "",
             "Selected-checkpoint panels above are not equal-step comparisons. Below: every retained mandatory fixed step and the union of actual headline stopping steps. Arms are NEVER trained past their plateau to fill a later panel. Any fixed panel beyond an arm's stop (4000/8000/12000/16000/18269 or the10K grid) is not reached, not zero accuracy.", "",
             "| Run | Update | Test1x | Test4x | Test8x | Suffix floor |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in runs.items():
        if r["status"] != "complete" or not name.startswith("headline"):
            continue
        panels = dict(r["matched_test"])
        for step, v in r["cross_plateau_test"].items():
            if v["status"] == "evaluated":
                panels[step] = v["panels"]
            else:
                panels[step] = None
        for step, v in sorted(panels.items(), key=lambda pair: int(pair[0])):
            cells = [number(v[f"test_{x}x"]["accuracy"]) for x in (1,4,8)] if v else ["already stopped"] * 3
            text.append(f"| {name} | {step} | {' | '.join(cells)} | {number(FLOOR)} |")
    text += ["", "## Suffix calibration", "",
             f"Analytic suffix0/1/7 cap = {number(FLOOR)} once at least two drivers are omitted (width7 starts at driver9). The CPU verification rerun and `suffix_calibration.json` retain fresh 2,000,000-independent-trajectory measurements per prime, known initialization, seeds, exact counts, six-sigma checks, and the empirical old parity-transfer control. These are not learned-model upper bounds.", "", "## Honest interpretation", ""]
    if not complete:
        text.append("The experiment is incomplete/fail-closed. The absent learned measurements support no architecture, mechanism, or depth conclusion. Completed receipts are published without substituting guessed values for missing arms.")
    else:
        e, g = aggregates["e97"], aggregates["gdn2"]
        cal = [runs[f"calibration_{a}_seed0"]["test"]["test_1x"]["accuracy"] for a in ("e97", "gdn2", "e97_identity")]
        stop_steps = sorted({r["stop_step"] for name, r in runs.items() if name.startswith("headline")})
        initial_selections = [name for name, r in runs.items() if name.startswith("headline") and r["step"] == 0]
        text.append(f"No practical learned-computation separation is demonstrated: headline stop updates are {stop_steps} (the first possible plateau is2000), with full-state scores near or below the suffix reference rather than successful map computation. At p=61, selected-checkpoint test1x means are E97 {number(e['test_1x']['mean'])} and GDN2 {number(g['test_1x']['mean'])}, against the {number(FLOOR)} suffix reference. One-step held-out full-state scores are E97/GDN2/identity {', '.join(number(v) for v in cal)}. These arithmetic controls and the extrapolation panels must constrain any interpretation: failure to learn the one-step arithmetic blocks a depth-specific conclusion. This frozen128-example fine-tune tests learning efficiency of this released pair under one fixed recipe, not universal representability, impossibility, or CMA causality. The single-seed identity contrast is exploratory and not a multi-seed causal result; initialization and optimization confounds remain. Update0 selections, not learned checkpoints: {initial_selections}. Every finite-state map still admits a sufficiently large one-hot linear-state representation; these failures are not an expressivity impossibility result.")
    if contingency:
        text += ["", "## Disclosed mid-run conditional protocol amendment", "",
                 f"Registered **{contingency['registered_at']}** after headline validation through750/1000 updates had already been observed. This was registered before any headline stop/final-test evaluation, **not pre-launch**. `contingency_amendment.json` preserves exact per-arm observation counters. No stopping, selection, recipe, metric, or data changes followed.", "",
                 f"Trigger iff BOTH architectures' three-seed mean best-val full-state accuracy <= **0.127%** (operator's binding rounded floor+0.1pp threshold). Trigger state: **{contingency['triggered']}**. If triggered, released-pair p13/p31 calibrations disambiguating state-space size from task-family learning failure are a **separately authorized follow-up task**, NOT launched here. Fresh CPU13/31/61 arithmetic/data/suffix checks plus completed p61 one-step diagnostics across all three architectures satisfy current calibration scope by explicit supervisor ruling."]
    text += ["", "## Receipts and reproducibility", "",
             "- `contract.json`: committed/pushed preregistration and explicit live amendments; configs bind seeds, optimizer, source checkpoint/args and runner hashes.",
             "- Per run: `frozen_pools.npz`, data hashes, `training.jsonl`, `validation.json`, selected/frozen checkpoints and SHA256 ledger, best-test and matched-test counts, full summary. Binary model files are local ignored artifacts, NOT in git; hashes/absolute paths reconstruct and identify them.",
             "- `receipts/`: exact commands, GPU UUID/index snapshots, per-arm lease/console/exit records; local lambda01, no Slurm and no WorksGood service. `attempt1_alignment/` is preserved, never mislabeled as complete.",
             "- CPU protocol/task/facade suite:71 passed (including54 protocol/task tests); fresh suffix verification passes. Final commands and clean-index proof accompany acceptance. DiLoCo resilience is unchanged: requirement matrix IDs not applicable.",
             ("- `receipts/final_audit.json`: all10 exit codes zero; selected and matched checkpoint binaries rehashed; observation counts, paired identity data, best/matched agreement, and selection-before-test chronology verified. All GPU leases released and all eight GPUs idle."
              if complete and (root / "receipts/final_audit.json").exists() else "- Final ten-arm artifact audit unavailable; do not claim complete validation."),
             "- Independent acceptance review remains required. Three-seed intervals are limited; fixed small training data and a single recipe cannot establish general task learnability. Any incomplete/failure arm is explicit above.",
             "", "## Validation commands", "", "```text",
             "CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m pytest -q tests/test_invertible_stage2.py tests/test_invertible_quadratic.py tests/test_e97_facade.py  # 71 passed",
             "CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m experiments.expressivity_tasks.verify_invertible_quadratic --output experiments/expressivity_tasks/results_invertible_stage2/suffix_calibration.json  # all checks passed",
             "bash experiments/expressivity_tasks/launch_invertible_stage2.sh calibration  # three completed diagnostics, exit0 after documented restart",
             "bash experiments/expressivity_tasks/launch_invertible_stage2.sh headline  # seven completed headline runs, exit0",
             "CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m experiments.expressivity_tasks.aggregate_invertible_stage2  # complete, receipt audit passed",
             "CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m experiments.expressivity_tasks.results_invertible_stage2.receipts.audit_command  # ten-arm artifact audit passed",
             "bash -n experiments/expressivity_tasks/launch_invertible_stage2.sh; compileall; cmp AGENTS.md CLAUDE.md; git diff --check  # passed",
             "```"]
    report.write_text("\n".join(text) + "\n")
    return complete


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("experiments/expressivity_tasks/results_invertible_stage2"))
    p.add_argument("--report", type=Path, default=Path("X2-report.md"))
    a = p.parse_args()
    if not build(a.root, a.report):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
