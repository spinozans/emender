#!/usr/bin/env python3
"""Analyze SF-DiLoCo P5 island-count scaling logs.

This analyzer is deliberately eligibility-first: planned runs with missing logs,
missing fused-guard coverage, missing held-out metrics, or failed process output
remain represented in the JSON outputs as incomplete/ineligible rather than
silently disappearing from the paired analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev


STEP_RE = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+([0-9.eE+-]+)")
MERGE_RE = re.compile(r"\[DiLoCo\]\s+(?:FINAL\s+)?merge #(\d+) at step (\d+)")
FINAL_RE = re.compile(r"^(FINAL_HELDOUT_[A-Z_]+):\s+(.+)$")
GUARD_RE = re.compile(r"\[fused-guard\].*NO eager fallback")
RANK_RE = re.compile(r"\[fused-guard\] rank (\d+)/(\d+):")
SYNC_TOTAL_RE = re.compile(r"^DILOCO_SYNC_TOTAL_S:\s+([0-9.eE+-]+)")
SYNC_AVG_RE = re.compile(r"^DILOCO_SYNC_AVG_MS:\s+([0-9.eE+-]+)")
DILOCO_MERGES_RE = re.compile(r"^DILOCO_MERGES:\s+(\d+)")
DILOCO_K_RE = re.compile(r"^DILOCO_K:\s+(\d+)")
ACQUIRED_RE = re.compile(r"\[p5\] acquired CUDA_VISIBLE_DEVICES=(.+)")
PLANNED_RE = re.compile(r"\[p5\] task=.* label=(\S+) W=(\d+) seed=(\d+) arm=(\S+)")
OUTER_AVG_RE = re.compile(r"\[DiLoCo\] outer optimizer: avg")
OUTER_SFSGD_Y_RE = re.compile(r"\[DiLoCo\] outer optimizer: sfsgd .*export_basis=y")
NONFINITE_RE = re.compile(r"Non-finite|nonfinite|nan|inf", re.IGNORECASE)
EAGER_BAD_RE = re.compile(r"eager fallback", re.IGNORECASE)

T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
}
EPSILON_BPB = 0.005


def parse_value(value: str):
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def planned_runs(root: Path) -> list[dict]:
    plan_path = root / "planned_matrix.json"
    if plan_path.exists():
        payload = json.loads(plan_path.read_text())
        return payload["runs"]
    runs = []
    for w in (2, 4, 8):
        for seed in range(7000, 7006):
            arms = ["avg", "sfsgd_y"] if seed % 2 == 0 else ["sfsgd_y", "avg"]
            for order, arm in enumerate(arms, start=1):
                label = f"W{w:02d}_seed{seed}_{arm}"
                runs.append({
                    "world_size": w,
                    "seed": seed,
                    "arm": arm,
                    "order_within_seed": order,
                    "label": label,
                    "log": f"logs/{label}.log",
                    "curve": f"curves/{label}_heldout_curve.csv",
                    "run_dir": f"runs/{label}",
                })
    return runs


def load_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                parsed[key] = parse_value(value)
            rows.append(parsed)
    return rows


def compute_shock(losses: list[dict], merges: list[dict]) -> list[dict]:
    by_step = {item["step"]: item["loss"] for item in losses}
    steps = sorted(by_step)
    out = []
    for merge in merges:
        step = merge["step"]
        pre_steps = [s for s in steps if s < step]
        post_steps = [s for s in steps if s >= step]
        if not pre_steps or not post_steps:
            continue
        pre_step = pre_steps[-1]
        post_step = post_steps[0]
        pre_loss = by_step[pre_step]
        post_loss = by_step[post_step]
        recovery_step = None
        for s in steps:
            if s > post_step and by_step[s] <= pre_loss:
                recovery_step = s
                break
        out.append({
            "merge": merge["merge"],
            "step": step,
            "pre_step": pre_step,
            "pre_loss": pre_loss,
            "post_step": post_step,
            "post_loss": post_loss,
            "jump": post_loss - pre_loss,
            "recovery_step": recovery_step,
            "recovery_steps": None if recovery_step is None else recovery_step - step,
        })
    return out


def parse_log(path: Path) -> dict:
    losses = []
    merges = []
    finals = {}
    guard_lines = []
    ranks = set()
    world_size = None
    acquired_gpus = None
    declared = {}
    sync_total_s = None
    sync_avg_ms = None
    diloco_merges = None
    diloco_k = None
    saw_outer_avg = False
    saw_outer_sfsgd_y = False
    nonfinite = False
    bad_eager = False
    return_code_known_failed = False

    if not path.exists():
        return {"log_exists": False}

    for line in path.read_text(errors="replace").splitlines():
        m_planned = PLANNED_RE.search(line)
        if m_planned:
            declared = {
                "label": m_planned.group(1),
                "world_size": int(m_planned.group(2)),
                "seed": int(m_planned.group(3)),
                "arm": m_planned.group(4),
            }
        m_acquired = ACQUIRED_RE.search(line)
        if m_acquired:
            acquired_gpus = [x.strip() for x in m_acquired.group(1).split(",") if x.strip()]
        if GUARD_RE.search(line):
            guard_lines.append(line)
            m_rank = RANK_RE.search(line)
            if m_rank:
                ranks.add(int(m_rank.group(1)))
                world_size = int(m_rank.group(2))
        m_step = STEP_RE.search(line)
        if m_step:
            losses.append({"step": int(m_step.group(1)), "loss": float(m_step.group(2))})
        m_merge = MERGE_RE.search(line)
        if m_merge:
            merges.append({"merge": int(m_merge.group(1)), "step": int(m_merge.group(2))})
        m_final = FINAL_RE.search(line)
        if m_final:
            finals[m_final.group(1)] = parse_value(m_final.group(2))
        m_sync_total = SYNC_TOTAL_RE.search(line)
        if m_sync_total:
            sync_total_s = float(m_sync_total.group(1))
        m_sync_avg = SYNC_AVG_RE.search(line)
        if m_sync_avg:
            sync_avg_ms = float(m_sync_avg.group(1))
        m_merges = DILOCO_MERGES_RE.search(line)
        if m_merges:
            diloco_merges = int(m_merges.group(1))
        m_k = DILOCO_K_RE.search(line)
        if m_k:
            diloco_k = int(m_k.group(1))
        saw_outer_avg = saw_outer_avg or bool(OUTER_AVG_RE.search(line))
        saw_outer_sfsgd_y = saw_outer_sfsgd_y or bool(OUTER_SFSGD_Y_RE.search(line))
        nonfinite = nonfinite or bool(NONFINITE_RE.search(line))
        if EAGER_BAD_RE.search(line) and "NO eager fallback" not in line:
            bad_eager = True
        if "ChildFailedError" in line or "RuntimeError:" in line or "Traceback (most recent call last)" in line:
            return_code_known_failed = True

    return {
        "log_exists": True,
        "declared": declared,
        "losses": losses,
        "merges": merges,
        "final": finals,
        "fused_guard_count": len(guard_lines),
        "fused_guard_ranks": sorted(ranks),
        "fused_guard_world_size": world_size,
        "acquired_gpus": acquired_gpus,
        "diloco_merges_reported": diloco_merges,
        "diloco_k_reported": diloco_k,
        "sync_total_s": sync_total_s,
        "sync_avg_ms": sync_avg_ms,
        "saw_outer_avg": saw_outer_avg,
        "saw_outer_sfsgd_y": saw_outer_sfsgd_y,
        "nonfinite_seen": nonfinite,
        "bad_eager_seen": bad_eager,
        "failure_text_seen": return_code_known_failed,
    }


def summarize_run(root: Path, planned: dict) -> dict:
    log_path = root / planned["log"]
    curve_path = root / planned["curve"]
    parsed = parse_log(log_path)
    curve = load_curve(curve_path)
    shocks = compute_shock(parsed.get("losses", []), parsed.get("merges", []))
    jumps = [s["jump"] for s in shocks]
    positive = [j for j in jumps if j > 0]
    recoveries = [s["recovery_steps"] for s in shocks if s["recovery_steps"] is not None]
    unrecovered_positive = [
        s for s in shocks if s["jump"] > 0 and s["recovery_step"] is None
    ]

    final = parsed.get("final", {})
    expected_w = int(planned["world_size"])
    expected_arm = planned["arm"]
    reasons = []
    if not parsed.get("log_exists"):
        reasons.append("missing log")
    if parsed.get("fused_guard_world_size") != expected_w:
        reasons.append("fused guard world size missing or mismatched")
    if len(parsed.get("fused_guard_ranks", [])) != expected_w:
        reasons.append("fused guard missing for one or more ranks")
    if "FINAL_HELDOUT_BPB" not in final:
        reasons.append("missing final held-out BPB")
    if final.get("FINAL_HELDOUT_MODE") != "x":
        reasons.append("held-out eval mode is not x")
    if not curve:
        reasons.append("missing held-out curve")
    elif not any(row.get("step") == 1500 for row in curve):
        reasons.append("missing held-out curve row at step 1500")
    if (parsed.get("diloco_merges_reported") or len(parsed.get("merges", []))) < 6:
        reasons.append("fewer than 6 DiLoCo merges")
    if parsed.get("nonfinite_seen"):
        reasons.append("nonfinite text seen in log")
    if parsed.get("bad_eager_seen"):
        reasons.append("eager fallback text seen outside guard")
    if parsed.get("failure_text_seen"):
        reasons.append("failure traceback/runtime text seen")
    if expected_arm == "avg" and not parsed.get("saw_outer_avg"):
        reasons.append("avg outer optimizer line missing")
    if expected_arm == "sfsgd_y" and not parsed.get("saw_outer_sfsgd_y"):
        reasons.append("sfsgd export_basis=y optimizer line missing")
    if parsed.get("acquired_gpus") is not None and len(parsed["acquired_gpus"]) != expected_w:
        reasons.append("acquired GPU count does not match W")

    curve_1500 = [row for row in curve if row.get("step") == 1500]
    avg_sync_s = None
    if parsed.get("sync_avg_ms") is not None:
        avg_sync_s = parsed["sync_avg_ms"] / 1000.0
    elif parsed.get("sync_total_s") is not None and parsed.get("diloco_merges_reported"):
        avg_sync_s = parsed["sync_total_s"] / parsed["diloco_merges_reported"]
    sync_fraction_per_k = None
    if avg_sync_s is not None and parsed.get("diloco_k_reported"):
        sync_fraction_per_k = avg_sync_s / max(float(parsed["diloco_k_reported"]), 1.0)

    return {
        **planned,
        "log_abs": str(log_path),
        "curve_abs": str(curve_path),
        "run_dir_abs": str(root / planned["run_dir"]),
        "parsed": parsed,
        "eligible": not reasons,
        "ineligible_reasons": reasons,
        "final_heldout_bpb": final.get("FINAL_HELDOUT_BPB"),
        "final_heldout_ce": final.get("FINAL_HELDOUT_CE"),
        "final_heldout_tokens": final.get("FINAL_HELDOUT_TOKENS"),
        "final_heldout_mode": final.get("FINAL_HELDOUT_MODE"),
        "final_curve_bpb": curve_1500[-1]["heldout_bpb"] if curve_1500 else (curve[-1]["heldout_bpb"] if curve else None),
        "curve_rows": len(curve),
        "shock": shocks,
        "mean_jump": mean(jumps) if jumps else None,
        "max_jump": max(jumps) if jumps else None,
        "mean_positive_jump": mean(positive) if positive else 0.0,
        "max_recovery_steps": max(recoveries) if recoveries else None,
        "unrecovered_positive_jumps": len(unrecovered_positive),
        "sync_total_s": parsed.get("sync_total_s"),
        "sync_avg_ms": parsed.get("sync_avg_ms"),
        "sync_fraction_per_k_step_window": sync_fraction_per_k,
    }


def ci_95(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return None, None, None, None
    m = mean(values)
    if len(values) == 1:
        return m, None, None, None
    sd = stdev(values)
    se = sd / math.sqrt(len(values))
    t = T_CRIT_95.get(len(values) - 1, 1.96)
    return m, sd, m - t * se, m + t * se


def per_w_decision(deltas: list[dict]) -> dict:
    vals = [d["delta_bpb"] for d in deltas]
    m, sd, lo, hi = ci_95(vals)
    n = len(vals)
    signs = {
        "negative": sum(1 for v in vals if v < 0),
        "positive": sum(1 for v in vals if v > 0),
        "zero": sum(1 for v in vals if v == 0),
    }
    if n < 5:
        decision = "inconclusive"
        reason = "fewer than 5 complete eligible pairs"
    elif lo is not None and hi is not None and hi < -EPSILON_BPB and signs["negative"] >= 4:
        decision = "sfsgd_y_win"
        reason = "95% CI upper bound below -0.005 BPB and at least 4/6 negative deltas"
    elif lo is not None and hi is not None and lo > EPSILON_BPB and signs["positive"] >= 4:
        decision = "avg_win"
        reason = "95% CI lower bound above +0.005 BPB and at least 4/6 positive deltas"
    elif lo is not None and hi is not None and lo >= -EPSILON_BPB and hi <= EPSILON_BPB:
        decision = "practical_tie"
        reason = "entire 95% CI lies within [-0.005, +0.005] BPB"
    else:
        decision = "inconclusive"
        reason = "confidence interval or signs do not satisfy a win/tie rule"
    return {
        "n_pairs": n,
        "mean_delta_bpb": m,
        "sd_delta_bpb": sd,
        "ci95_low": lo,
        "ci95_high": hi,
        "paired_cohen_dz": None if not sd else m / sd,
        "sign_count": signs,
        "decision": decision,
        "reason": reason,
    }


def build_pairs(runs: list[dict]) -> dict[str, dict]:
    by_key = {(r["world_size"], r["seed"], r["arm"]): r for r in runs}
    result = {}
    for w in sorted({r["world_size"] for r in runs}):
        pairs = []
        deltas = []
        for seed in sorted({r["seed"] for r in runs if r["world_size"] == w}):
            avg_run = by_key.get((w, seed, "avg"))
            sf_run = by_key.get((w, seed, "sfsgd_y"))
            eligible = bool(avg_run and sf_run and avg_run["eligible"] and sf_run["eligible"])
            pair = {
                "world_size": w,
                "seed": seed,
                "eligible_pair": eligible,
                "avg": avg_run,
                "sfsgd_y": sf_run,
            }
            if eligible:
                delta = {
                    "world_size": w,
                    "seed": seed,
                    "delta_bpb": sf_run["final_heldout_bpb"] - avg_run["final_heldout_bpb"],
                    "delta_curve_bpb": sf_run["final_curve_bpb"] - avg_run["final_curve_bpb"],
                    "delta_max_jump": (
                        None
                        if sf_run["max_jump"] is None or avg_run["max_jump"] is None
                        else sf_run["max_jump"] - avg_run["max_jump"]
                    ),
                }
                pair["delta"] = delta
                deltas.append(delta)
            else:
                pair["missing_or_ineligible"] = [
                    arm for arm, run in (("avg", avg_run), ("sfsgd_y", sf_run))
                    if not run or not run["eligible"]
                ]
            pairs.append(pair)
        result[str(w)] = {
            "pairs": pairs,
            "deltas": deltas,
            "decision": per_w_decision(deltas),
        }
    return result


def slope_by_log2_w(paired_by_w: dict[str, dict]) -> float | None:
    points = []
    for w_s, payload in paired_by_w.items():
        mean_delta = payload["decision"]["mean_delta_bpb"]
        if mean_delta is not None:
            points.append((math.log2(float(w_s)), mean_delta))
    if len(points) < 2:
        return None
    xbar = mean([p[0] for p in points])
    ybar = mean([p[1] for p in points])
    denom = sum((x - xbar) ** 2 for x, _ in points)
    if denom == 0:
        return None
    return sum((x - xbar) * (y - ybar) for x, y in points) / denom


def avg_shock_worse_with_w(runs: list[dict]) -> bool:
    avg_by_w = []
    for w in sorted({r["world_size"] for r in runs}):
        eligible_avg = [
            r for r in runs
            if r["world_size"] == w and r["arm"] == "avg" and r["eligible"] and r["max_jump"] is not None
        ]
        if eligible_avg:
            avg_by_w.append((w, mean([r["max_jump"] for r in eligible_avg])))
    return len(avg_by_w) >= 2 and avg_by_w[-1][1] > avg_by_w[0][1]


def scaling_decision(paired_by_w: dict[str, dict], runs: list[dict]) -> dict:
    slope = slope_by_log2_w(paired_by_w)
    decisions = {int(w): payload["decision"]["decision"] for w, payload in paired_by_w.items()}
    means = {int(w): payload["decision"]["mean_delta_bpb"] for w, payload in paired_by_w.items()}
    complete_ws = [
        int(w) for w, payload in paired_by_w.items()
        if payload["decision"]["n_pairs"] >= 5
    ]
    missing_or_inconclusive = (
        set(complete_ws) != {2, 4, 8}
        or any(decisions.get(w) == "inconclusive" for w in (2, 4, 8))
    )
    shock_worse = avg_shock_worse_with_w(runs)

    if (
        decisions.get(8) in {"practical_tie", "avg_win"}
        and means.get(2) is not None
        and means.get(8) is not None
        and means[8] >= means[2] - EPSILON_BPB
        and not shock_worse
    ):
        conclusion = "avg_remains_safe_across_local_w_scaling"
        risk = (
            "No local evidence that avg's small-W advantage is a parameterization artifact; "
            "Frontier-scale canary should still monitor merge shock and held-out drift."
        )
    elif (
        decisions.get(8) == "sfsgd_y_win"
        or (decisions.get(8) == "inconclusive" and means.get(8) is not None and means[8] <= -EPSILON_BPB)
    ) and slope is not None and slope <= -EPSILON_BPB and not shock_worse:
        conclusion = "sfsgd_y_more_robust_to_increasing_island_count"
        risk = (
            "Local W trend supports outer ScheduleFree absorbing larger endpoint variance; "
            "use sfsgd_y as scale-out canary candidate or require a 16+ island follow-up."
        )
    elif all(
        decisions.get(w) == "practical_tie"
        or (means.get(w) is not None and -EPSILON_BPB <= means[w] <= EPSILON_BPB)
        for w in (2, 4, 8)
    ) and slope is not None and abs(slope) < EPSILON_BPB:
        conclusion = "practical_tie_across_local_w_scaling"
        risk = "All complete W points are ties or inside the practical tie band with no material W trend."
    else:
        conclusion = "local_evidence_insufficient_for_thousands_of_islands"
        risk = (
            "Missing or inconclusive W point, contradictory signs, or trend below the frozen "
            "0.005 BPB-per-doubling threshold; keep avg as conservative default and create a "
            "larger true-island follow-up if scale risk remains important."
        )

    return {
        "epsilon_bpb": EPSILON_BPB,
        "per_w_decisions": decisions,
        "mean_delta_by_w": means,
        "trend_slope_bpb_per_log2_w": slope,
        "avg_shock_worse_with_w": shock_worse,
        "has_missing_or_inconclusive_w": missing_or_inconclusive,
        "system_level_conclusion": conclusion,
        "thousands_of_islands_risk_statement": risk,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="P5/P6 run root")
    args = ap.parse_args()

    root = args.root
    runs = [summarize_run(root, item) for item in planned_runs(root)]
    paired = build_pairs(runs)
    decision = scaling_decision(paired, runs)
    summary = {
        "task": "sf-diloco-p6",
        "root": str(root),
        "run_count_planned": len(runs),
        "run_count_eligible": sum(1 for r in runs if r["eligible"]),
        "runs": runs,
        "output_paths": {
            "logs": str(root / "logs"),
            "curves": str(root / "curves"),
            "runs": str(root / "runs"),
            "summary": str(root / "summary.json"),
            "paired_by_w": str(root / "paired_by_w.json"),
            "scaling_decision": str(root / "scaling_decision.json"),
        },
    }

    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "paired_by_w.json").write_text(json.dumps(paired, indent=2) + "\n")
    (root / "scaling_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "summary": str(root / "summary.json"),
        "paired_by_w": str(root / "paired_by_w.json"),
        "scaling_decision": str(root / "scaling_decision.json"),
        "eligible": summary["run_count_eligible"],
        "planned": summary["run_count_planned"],
        "decision": decision["system_level_conclusion"],
    }, indent=2))


if __name__ == "__main__":
    main()
