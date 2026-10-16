"""Register, replay-audit, and report the X3 symmetric scratch architecture sweep."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.expressivity_tasks.run_invertible_stage3_8m import (
    FAIRNESS, LRS, ROOT, build_model, digest, frozen_pools, matrix, plateau, selected, write_json,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def aggregate(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()),
            "sample_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
            "sem": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None}


def register(root):
    path = root / "amendment4.json"
    if path.exists():
        raise FileExistsError("amendment #4 is immutable")
    calibration = {}
    prior = Path("experiments/expressivity_tasks/results_invertible_stage2b")
    for run in sorted(prior.glob("headline_*/validation.json")):
        h = json.loads(run.read_text())
        last = h[-1]
        if last.get("train_loss_window_first250"):
            first, final = last["train_loss_window_first250"], last["train_loss_window_final250"]
            calibration[str(run.parent.name)] = {
                "step": last["step"], "first250": first, "final250": final,
                "relative_loss_improvement": (first-final)/first,
                "best_val_gain_2000": last["best_accuracy"] - h[-9]["best_accuracy"],
                "validation_sha256": digest(run)}
    x2c = Path("experiments/expressivity_tasks/results_invertible_stage2c")
    for run in sorted(x2c.glob("pilot_*/pilot_result.json")):
        calibration[run.parent.name] = {**json.loads(run.read_text())["gate"], "receipt_sha256": digest(run)}
    models = {}
    for arm in ("e97", "gdn2"):
        model = build_model(arm, 61)
        layer = model.layers[0] if arm == "e97" else model.layers[0].gdn
        heads = layer.n_heads if arm == "e97" else layer.num_v_heads
        kd = layer.n_state if arm == "e97" else layer.head_k_dim
        vd = layer.head_v_dim
        models[arm] = {"parameters": sum(p.numel() for p in model.parameters()),
                       "mixer_class": type(model.layers[0]).__name__, "dim": 256, "depth": 4,
                       "mlp_ratio": 2, "expansion": 1 if arm == "e97" else 8.625,
                       "state_per_layer": [heads, kd, vd], "total_state_elements": 4*heads*kd*vd,
                       "allow_neg_eigval": layer.allow_neg_eigval if arm == "gdn2" else None,
                       "state_activation": "tanh" if arm == "e97" else "linear"}
    write_json(path, {
        "amendment": 4, "registered_at": now(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "authority": "/home/erikg/emender-scratch/fix-fanout/taskX3-8m-sweep.md; live supervisor geometry/recipe/matrix rulings",
        "task_sha256": digest("/home/erikg/emender-scratch/fix-fanout/taskX3-8m-sweep.md"),
        "matrix": matrix(), "models": models,
        "geometry_correction": "Prelaunch inspection rejected legacy gdn-8m: it wraps fla.DeltaNet and silently ignores allow_neg_eigval. Approved replacement is actual FLAGatedDeltaNetLayer/fla.GatedDeltaNet, expansion8.625 chosen by instantiated parameter counts at dim256/depth4; wrapper default2 heads, key128, value1104. No legacy-gdn training performed. E97 retains approved H32/N32/expansion1. Parameter matching is not state-size matching or a one-factor mechanism ablation.",
        "parameter_gap_gdn2_relative_e97": models["gdn2"]["parameters"] / models["e97"]["parameters"] - 1,
        "recipe": "Prior scratch8M suite: AdamWScheduleFree betas(.9,.95), wd.01, clip1, bf16 autocast/fp32 parameters, batch32, no warmup; every evaluation and checkpoint uses ScheduleFree eval/x-mode, training resumes y-mode; all trainable tied task-sized heads; no pretrained weights or optimizer state loaded",
        "instrument_difference": "Dedicated trainable3721-class scratch vocab vs full pretrained vocab in X2 1.3B fine-tunes. Separate experiment classes, not directly numerically comparable; 1/p^2 remains reference. Small-prime diagnostics use p^2 vocab with unchanged backbone geometry.",
        "plateau": "Eligibility>=10000; every250 boundaries; over trailing2000 best-val gain<0.001 (0.1pp) AND first250/final250 relative train-loss mean improvement<0.001 (0.1%); hard cap60000. Strict thresholds; no change to archived X2 stops.",
        "calibration_basis": calibration,
        "calibration_rationale": "X2c E97 loss-window movement0.1491% at2e-4 and0.0757% at1e-3; GDN2 0.1189% and0.3210%. Prior1% threshold called all stagnant despite negative OLS slopes; calibrated0.1% resolves movement of this order. X2b terminal windows retained above. Task-observed calibration on released fine-tunes is disclosed, not an assertion of scratch convergence.",
        "data": "Fresh procedural batch/update, seed-paired training streams across architectures/LRs; frozen128 archived train(unused)/512val/512test1x/512test4x/512test8x; p61 exact X2 input and NPZ hashes. One-step has only test1x; difficulty seed0 allLRs.",
        "selection": "Per run maximum validation exact-state accuracy, lower validation CE, earlier update. Per architecture LR chosen by mean three-seed best validation accuracy, lower mean valCE, then smaller LR. No test-based LR/step/seed selection. Full matched-LR grid and best-LR views share the same data.",
        "fairness_verbatim": FAIRNESS,
        "capacity": "Every1000 plus best-val chain and stop; if free<400GiB prune to step0/current/current best/every10000, retaining checkpoint identities/hashes in ledger; never fabricate pruned cells. Fail closed below8GiB. Three co-tenants/card under exclusive leases, no preemption.",
        "resilience_requirements": "Not applicable: standalone local lambda01 single-GPU training runs, no DiLoCo behavior, Frontier, Slurm, or WorksGood service used."})
    write_json(root / "receipts/prelaunch.json", {"recorded_at": now(), "models": models,
        "amendment_sha256": digest(path), "matrix_counts": {k: sum(r["kind"] == k for r in matrix()) for k in ["primary", "difficulty", "one_step"]}})


def audit_run(root, row):
    out, receipt = root / row["name"], root / "receipts" / row["name"]
    result = json.loads((out / "result.json").read_text())
    config = json.loads((out / "config.json").read_text())
    data = json.loads((out / "data_receipt.json").read_text())
    selection = json.loads((out / "selection_frozen.json").read_text())
    history = json.loads((out / "validation.json").read_text())
    training = [json.loads(line) for line in (out / "training.jsonl").read_text().splitlines()]
    stop = result["stop_step"]
    assert result["status"] == "complete" and len(training) == stop
    assert [r["step"] for r in training] == list(range(1, stop+1))
    assert [r["step"] for r in history] == list(range(0, stop+1, 250))
    assert all(config[k] == row[k] for k in ("arm", "p", "drivers", "seed", "lr"))
    assert config["hard_cap"] == 60000 and config["protocol_amendment"] == 4
    assert config["source_sha256"] == digest(root / "receipts/runner_at_launch.py")
    assert config["optimizer"] == "AdamWScheduleFree" and config["warmup"] == 0
    assert config["val_gain_threshold"] == config["relative_train_loss_threshold"] == .001
    assert "exit_code=0\n" in (receipt / "exit.txt").read_text()
    losses = np.array([r["loss"] for r in training])
    assert np.isfinite(losses).all()
    best = None
    for i, boundary in enumerate(history):
        step = boundary["step"]
        if selected(boundary, best):
            best = boundary
        assert boundary["best_accuracy"] == best["accuracy"]
        if step >= 2000:
            assert boundary["train_loss_window_first250"] == float(losses[step-2000:step-1750].mean())
            assert boundary["train_loss_window_final250"] == float(losses[step-250:step].mean())
        if i != len(history)-1:
            assert not plateau(history[:i+1]), "continued past registered first plateau"
    assert selection["step"] == best["step"] == result["best_validation"]["step"]
    assert result["best_validation"]["accuracy"] == best["accuracy"]
    assert result["best_validation"]["loss"] == best["loss"]
    assert selection["frozen"] and selection["frozen_at"] < result["test_completed_at"]
    assert all(selection[k] == result[k] for k in selection)
    assert (plateau(history) and selection["stop_reason"] == "plateau") or (stop == 60000 and selection["stop_reason"] == "hard_cap")
    assert digest(selection["checkpoint"]["path"]) == selection["checkpoint"]["sha256"]
    assert digest(out / "frozen_pools.npz") == data["sha256"]
    pools = frozen_pools(row["seed"], row["drivers"], row["p"])
    for k, pool in pools.items():
        assert data["pools"][k]["input_sha256"] == hashlib.sha256(pool[0].tobytes()).hexdigest()
    if row["p"] == 61:
        prior = Path("experiments/expressivity_tasks/results_invertible_stage2b") / (
            f"headline_e97_seed{row['seed']}" if row["drivers"] == 128 else "calibration_e97_seed0")
        assert data == json.loads((prior / "data_receipt.json").read_text()), "X2 hash mismatch"
    for k, metrics in result["test"].items():
        assert metrics["count"] == int(pools[k][2].sum())
        assert metrics["accuracy"] == metrics["correct"] / metrics["count"]
        assert metrics["suffix_floor"] == 1 / row["p"]**2
        assert metrics["out_of_range"] == 0
    return {**row, "status": "complete", "result": result,
            "receipt_hashes": {name: digest(out / name) for name in ["config.json", "data_receipt.json", "training.jsonl", "validation.json", "selection_frozen.json", "result.json"]}}


def pct(value):
    return "N/A" if value is None else f"{100*value:.4f}%"


def mean_sd(stats, percent=True):
    factor = 100 if percent else 1
    return f"{factor*stats['mean']:.4f} ± {factor*stats['sample_sd']:.4f}" if stats['sample_sd'] is not None else f"{factor*stats['mean']:.4f}"


def build(root, report, audit=True):
    rows, errors = {}, []
    for row in matrix():
        try:
            if audit:
                rows[row["name"]] = audit_run(root, row)
            else:
                rows[row["name"]] = {**row, "status": "complete", "result": json.loads((root / row["name"] / "result.json").read_text())}
        except Exception as exc:
            rows[row["name"]] = {**row, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"{row['name']}: {type(exc).__name__}: {exc}")
    grid, best_lrs = {}, {}
    for arm in ("e97", "gdn2"):
        for lr in LRS:
            group = [r for r in rows.values() if r["kind"] == "primary" and r["arm"] == arm and r["lr"] == lr]
            if not all(r["status"] == "complete" for r in group):
                continue
            results = [r["result"] for r in group]
            stats = {"validation": aggregate([r["best_validation"]["accuracy"] for r in results]),
                     "validation_loss": aggregate([r["best_validation"]["loss"] for r in results]),
                     "stop_step": aggregate([r["stop_step"] for r in results])}
            for panel in ("test_1x", "test_4x", "test_8x"):
                stats[panel] = {k: aggregate([r["test"][panel][k] for r in results]) for k in ["accuracy", "x_accuracy", "y_accuracy", "post_suffix7_accuracy", "loss"]}
            grid[f"{arm}_{lr:g}"] = {"arm": arm, "lr": lr, **stats}
        if all(f"{arm}_{lr:g}" in grid for lr in LRS):
            best_lrs[arm] = min(LRS, key=lambda lr: (-grid[f"{arm}_{lr:g}"]["validation"]["mean"], grid[f"{arm}_{lr:g}"]["validation_loss"]["mean"], lr))
    complete = not errors and len(best_lrs) == 2
    matched = {}
    for lr in LRS:
        if all(f"{arm}_{lr:g}" in grid for arm in ("e97", "gdn2")):
            pairs = []
            for seed in range(3):
                pair = {r["arm"]: r["result"] for r in rows.values() if r["kind"] == "primary" and r["lr"] == lr and r["seed"] == seed and r["status"] == "complete"}
                pairs.append(pair["e97"]["test"]["test_1x"]["accuracy"] - pair["gdn2"]["test"]["test_1x"]["accuracy"])
            matched[f"{lr:g}"] = {"e97_minus_gdn2_paired_test1x": aggregate(pairs), "per_seed": pairs}
    amendment = json.loads((root / "amendment4.json").read_text()) if (root / "amendment4.json").exists() else {}
    summary = {"status": "complete_audited" if complete and audit else "complete_unaudited" if complete else "incomplete_fail_closed",
               "generated_at": now(), "amendment4": amendment, "runs": rows, "grid": grid,
               "best_lrs_validation_only": best_lrs, "matched_lr": matched, "errors": errors,
               "floor": 1/61**2, "fairness": FAIRNESS}
    write_json(root / "results_summary.json", summary)
    lines = ["# X3 — from-scratch architecture sweep on the invertible map", "",
             f"Status: **{summary['status']}**; {sum(r['status']=='complete' for r in rows.values())}/36 runs available. Exact p61 floor **{pct(1/61**2)}** (1/3721).", "",
             "## Fairness (verbatim)", FAIRNESS, "", "## Amendment #4 / prelaunch registration", "```json", json.dumps(amendment, indent=2), "```", "",
             "## Symmetric primary LR grid — three seeds, mean ± sample SD", "Validation-only LR choice; all test panels use each run's best-validation checkpoint. SEM and paired differences are in JSON. Percent units below; no best-of-seed claims.", "",
             "| Arm | LR | Best val % | Test1x % | Test4x % | Test8x % | Post-suffix7 1x % | Stop updates |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("e97", "gdn2"):
        for lr in LRS:
            g = grid.get(f"{arm}_{lr:g}")
            if g:
                lines.append(f"| {arm} | {lr:g} | {mean_sd(g['validation'])} | {mean_sd(g['test_1x']['accuracy'])} | {mean_sd(g['test_4x']['accuracy'])} | {mean_sd(g['test_8x']['accuracy'])} | {mean_sd(g['test_1x']['post_suffix7_accuracy'])} | {mean_sd(g['stop_step'], False)} |")
            else:
                lines.append(f"| {arm} | {lr:g} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")
    lines += ["", "## Both grid readings", ""]
    for arm, lr in best_lrs.items():
        g = grid[f"{arm}_{lr:g}"]
        lines.append(f"- **{arm} best validation LR={lr:g}**: test1x {mean_sd(g['test_1x']['accuracy'])}%; x4 {mean_sd(g['test_4x']['accuracy'])}%; x8 {mean_sd(g['test_8x']['accuracy'])}%. The full matched-LR reading follows, not an alternative experiment.")
    for lr, contrast in matched.items():
        lines.append(f"- Matched LR={lr}: paired E97−GDN2 test1x {mean_sd(contrast['e97_minus_gdn2_paired_test1x'])} pp (all3 paired seeds; positive favors E97); both arms' absolute values are in the full grid above.")
    lines += ["", "## All primary seeds / selected steps / extrapolation / coordinates / suffix calibration", "",
              "| Run | Val % | Selected | Stop/reason | Test1x % | x4 % | x8 % | x1x % | y1x % | Post7 1x % | Suffix7-only 1x % |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows.values():
        if row["kind"] != "primary":
            continue
        if row["status"] != "complete":
            lines.append(f"| {row['name']} | unavailable | — | {row['error']} | — | — | — | — | — | — | — |")
            continue
        r = row["result"]
        m = r["test"]["test_1x"]
        lines.append(f"| {row['name']} | {pct(r['best_validation']['accuracy'])} | {r['step']} | {r['stop_step']}/{r['stop_reason']} | {pct(m['accuracy'])} | {pct(r['test']['test_4x']['accuracy'])} | {pct(r['test']['test_8x']['accuracy'])} | {pct(m['x_accuracy'])} | {pct(m['y_accuracy'])} | {pct(m['post_suffix7_accuracy'])} | {pct(r['suffix7']['test_1x']['accuracy'])} |")
    lines += ["", "## Difficulty ladder and one-step arithmetic diagnostics (all LRs, seed0)", "",
              "One-step has no depth or suffix7 panel. Small-prime arms retain backbone geometry but train task-sized p² embeddings/heads. Floors: p13="+pct(1/13**2)+", p31="+pct(1/31**2)+", p61="+pct(1/61**2)+".", "",
              "| Run | Best val % | Selected | Stop/reason | Test1x % | x4 % | x8 % | x1x % | y1x % | Post7 1x % | Suffix7-only % |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows.values():
        if row["kind"] == "primary":
            continue
        if row["status"] != "complete":
            lines.append(f"| {row['name']} | unavailable | — | {row['error']} | — | — | — | — | — | — | — |")
            continue
        r = row["result"]
        m = r["test"]["test_1x"]
        extrap = [pct(r["test"][p]["accuracy"]) if p in r["test"] else "N/A" for p in ["test_4x", "test_8x"]]
        lines.append(f"| {row['name']} | {pct(r['best_validation']['accuracy'])} | {r['step']} | {r['stop_step']}/{r['stop_reason']} | {pct(m['accuracy'])} | {extrap[0]} | {extrap[1]} | {pct(m['x_accuracy'])} | {pct(m['y_accuracy'])} | {pct(m['post_suffix7_accuracy'])} | {pct(r['suffix7']['test_1x']['accuracy'])} |")
    lines += ["", "## Suffix7-only calibration at every reported depth", "",
              "Nontrained canonical(0,0) propagated using only the last7 drivers; scored after two or more omitted drivers (driver>=9). Identical frozen inputs give identical calibrations across architectures/LRs. Sequence-level SEM is in JSON; overlapping positions are not independent trials. One-step is N/A. Fresh procedural training is not filtered against frozen inputs: at one step the finite61³ input domain can be revisited, so this is an arithmetic/encoding control, not a novel-input generalization certificate.", "",
              "| Field / seed | Floor % | Suffix7 1x % | Suffix7 x4 % | Suffix7 x8 % |", "|---|---:|---:|---:|---:|"]
    baseline_seen = set()
    for row in rows.values():
        key = (row['p'], row['seed'])
        if row['status'] != 'complete' or row['drivers'] != 128 or key in baseline_seen:
            continue
        baseline_seen.add(key)
        b = row['result']['suffix7']
        lines.append(f"| p{key[0]} / seed{key[1]} | {pct(1/key[0]**2)} | {pct(b['test_1x']['accuracy'])} | {pct(b['test_4x']['accuracy'])} | {pct(b['test_8x']['accuracy'])} |")
    lines += ["", "## Cost and evidence", "", "| Run | Parameters | Real train tokens | Train tokens/s | Wall seconds |", "|---|---:|---:|---:|---:|"]
    for row in rows.values():
        if row["status"] == "complete":
            r = row["result"]
            lines.append(f"| {row['name']} | {r['trainable_parameters']} | {r['total_training_tokens']} | {r['training_tokens_per_second']:.1f} | {r['wall_seconds']:.1f} |")
    lines += ["", "Full source, environment, lease/GPU UUID snapshots, exact commands, exit codes, fresh training losses, frozen pools/hash receipts, all validation boundaries, selected-before-test receipts, checkpoint hashes, and strict stopping-rule replay accompany the JSON. Binary checkpoints remain local ignored artifacts, not git payloads. Every1000 retention plus best-val chain/stop; pruning activates below400GiB and records hashes/removals. Smoke runs are instrument checks, excluded from the36-run experiment.", "",
              "Resilient DiLoCo requirement IDs: not applicable (standalone local lambda01, no resilient behavior or Frontier/Slurm submission). No WorksGood service, workers, or dispatcher used. Independent acceptance review remains required.", "",
              "## Architecture question and single-1.3B recommendation", ""]
    if not complete:
        lines += ["Incomplete/fail-closed: these data support no final architecture conclusion or validated 1.3B LR recommendation. Missing/audit-failing arms remain visible; no values are zero-filled or selected from partial seeds."]
    else:
        lines += ["This is a learning-efficiency comparison of two approximately parameter-matched scratch backbones with fixed-depth MLPs, not a representability impossibility result or a controlled state-nonlinearity ablation: recurrent state sizes and mixer plumbing differ. The full LR grid and validation-selected best LR must be read together. Full-sequence accuracy can benefit from early positions; the post-suffix7 panel is the depth-relevant check against1/p². A failed one-step arithmetic control blocks a depth-specific interpretation of the corresponding long-task failure. The calibrated plateau rule is a disclosed task-observed stopping criterion, not proof of eventual convergence.",
                  "The scratch task-sized output vocabulary and ScheduleFree recipe differ from X2's full-pretrained-vocabulary fine-tunes and AdamW warmup. Do not compare raw numeric accuracy across those experiment classes as an architecture-only effect. A single1.3B follow-up would be a separately authorized scale-transfer diagnostic, not a two-arm architecture comparison. Numerical diagnostic qualification and the conditional recommendation follow; no1.3B job is launched by X3."]
    if complete:
        lines += ["", "### Observed diagnostic qualification (best-LR AND matched-LR views)", "",
                  "The primary best-LR and matched-LR results above do not demonstrate learned deep invertible-map computation: inspect their post7/extrapolation values beside the printed floor, not merely the elevated full-sequence score that includes early drivers. No controlled causal advantage of recurrent nonlinearity is established at this nominal8M budget (measured totals9.90M/9.887M). A null under these geometry/optimizer/stopping rules is not a proof of equal ultimate capabilities. In particular, one-step training may require far more updates than a primary run's plateau; a later arithmetic success cannot certify that the primary model already mastered arithmetic at its earlier stop.",
                  "**Single1.3B recommendation:** the receipt/selection/stopping plumbing is validated, but this sweep has not validated a successful architecture-discriminating protocol. Do not label one further large run an architecture confirmation of a scratch separation that was not observed. If a separately approved scale-transfer calibration is pursued, the validation-best LR below is the symmetric-grid candidate, not a proven transferable optimum. First require arithmetic calibration at the target instrument/precision, and preregister an eligibility/budget decision that addresses late one-step learning before spending the single deep-task run; changing amendment#4 or launching that run is outside X3. Retain fresh training, frozen pools, explicit vocabulary/optimizer class, validation-only selection, both grid readings, x4/x8 and suffix calibration. No1.3B job was launched.", ""]
        for arm, lr in best_lrs.items():
            g = grid[f'{arm}_{lr:g}']
            diagnostic = next(r['result'] for r in rows.values() if r['kind']=='one_step' and r['arm']==arm and r['lr']==lr)
            lines.append(f"- {arm}, validation-best LR={lr:g}: one-step test {pct(diagnostic['test']['test_1x']['accuracy'])}; long-task post7 joint1x {mean_sd(g['test_1x']['post_suffix7_accuracy'])}%, x4 {mean_sd(g['test_4x']['post_suffix7_accuracy'])}%, x8 {mean_sd(g['test_8x']['post_suffix7_accuracy'])}%, against floor {pct(1/61**2)}. Best-LR headline and all three matched-LR contrasts are reported above; neither reading can be omitted.")
        lines += ["", "| Matched LR | E97 one-step % | GDN2 one-step % | E97 post7 1x % | GDN2 post7 1x % |", "|---|---:|---:|---:|---:|"]
        for lr in LRS:
            d = {r['arm']: r['result'] for r in rows.values() if r['kind']=='one_step' and r['lr']==lr}
            lines.append(f"| {lr:g} | {pct(d['e97']['test']['test_1x']['accuracy'])} | {pct(d['gdn2']['test']['test_1x']['accuracy'])} | {mean_sd(grid[f'e97_{lr:g}']['test_1x']['post_suffix7_accuracy'])} | {mean_sd(grid[f'gdn2_{lr:g}']['test_1x']['post_suffix7_accuracy'])} |")
    report.write_text("\n".join(lines)+"\n")
    write_json(root / "receipts/final_audit.json", {"audited_at": now(), "status": summary["status"], "n_complete": sum(r["status"]=="complete" for r in rows.values()), "errors": errors, "report_sha256": digest(report), "summary_sha256": digest(root / "results_summary.json")})
    return complete


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("action", choices=["register", "report"])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--report", type=Path, default=Path("X3-report.md"))
    a = p.parse_args()
    if a.action == "register":
        register(a.root)
    elif not build(a.root, a.report):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
