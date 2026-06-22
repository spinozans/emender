#!/usr/bin/env python3
"""Analyze SF-DiLoCo P4 outer-regime logs.

Parses the real training logs for:
  * fused-guard NO-eager coverage per rank
  * matched-token final held-out BPB/CE
  * post-sync shock and recovery from logged train-loss windows
  * held-out curve rows written from the fixed lb-compare tensor
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


STEP_RE = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+([0-9.]+)")
MERGE_RE = re.compile(r"\[DiLoCo\]\s+(?:FINAL\s+)?merge #(\d+) at step (\d+)")
FINAL_RE = re.compile(r"^(FINAL_HELDOUT_[A-Z_]+):\s+(.+)$")
GUARD_RE = re.compile(r"\[fused-guard\].*NO eager fallback")
RANK_RE = re.compile(r"\[fused-guard\] rank (\d+)/(\d+):")


def parse_float(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def parse_log(path: Path) -> dict:
    losses: list[dict] = []
    merges: list[dict] = []
    finals: dict = {}
    fused_guard_lines: list[str] = []
    ranks: set[int] = set()
    world_size = None

    for line in path.read_text(errors="replace").splitlines():
        if GUARD_RE.search(line):
            fused_guard_lines.append(line)
            m_rank = RANK_RE.search(line)
            if m_rank:
                ranks.add(int(m_rank.group(1)))
                world_size = int(m_rank.group(2))
        m_step = STEP_RE.search(line)
        if m_step:
            losses.append({"step": int(m_step.group(1)), "loss": float(m_step.group(2))})
            continue
        m_merge = MERGE_RE.search(line)
        if m_merge:
            merges.append({"merge": int(m_merge.group(1)), "step": int(m_merge.group(2))})
            continue
        m_final = FINAL_RE.search(line)
        if m_final:
            finals[m_final.group(1)] = parse_float(m_final.group(2))

    return {
        "log": str(path),
        "label": path.stem,
        "losses": losses,
        "merges": merges,
        "final": finals,
        "fused_guard_count": len(fused_guard_lines),
        "fused_guard_ranks": sorted(ranks),
        "world_size": world_size,
        "fused_guard_all_ranks": world_size is not None and len(ranks) == world_size,
    }


def load_curve(log_path: Path) -> list[dict]:
    stem = log_path.stem
    root = log_path.parent.parent
    curve = root / f"{stem}_heldout_curve.csv"
    rows: list[dict] = []
    if not curve.exists():
        return rows
    with curve.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if key in {"step", "tokens", "heldout_tokens"}:
                    parsed[key] = int(value)
                elif key in {"train_loss", "heldout_ce", "heldout_bpb", "heldout_bytes_per_token"}:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def compute_shock(run: dict) -> list[dict]:
    by_step = {item["step"]: item["loss"] for item in run["losses"]}
    steps = sorted(by_step)
    out: list[dict] = []
    for merge in run["merges"]:
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


def summarize(run: dict) -> dict:
    shocks = run["shock"]
    jumps = [s["jump"] for s in shocks]
    positive = [j for j in jumps if j > 0]
    recoveries = [s["recovery_steps"] for s in shocks if s["recovery_steps"] is not None]
    return {
        "label": run["label"],
        "final_heldout_bpb": run["final"].get("FINAL_HELDOUT_BPB"),
        "final_heldout_ce": run["final"].get("FINAL_HELDOUT_CE"),
        "final_heldout_tokens": run["final"].get("FINAL_HELDOUT_TOKENS"),
        "final_heldout_mode": run["final"].get("FINAL_HELDOUT_MODE"),
        "fused_guard_all_ranks": run["fused_guard_all_ranks"],
        "merges": len(shocks),
        "mean_jump": mean(jumps) if jumps else None,
        "max_jump": max(jumps) if jumps else None,
        "mean_positive_jump": mean(positive) if positive else 0.0,
        "max_recovery_steps": max(recoveries) if recoveries else None,
        "final_curve_bpb": run["heldout_curve"][-1]["heldout_bpb"] if run["heldout_curve"] else None,
    }


def verdict(rows: list[dict]) -> str:
    eligible = [r for r in rows if r["final_heldout_bpb"] is not None and r["fused_guard_all_ranks"]]
    if not eligible:
        return "No eligible run: missing final held-out BPB or fused guard."
    best = min(eligible, key=lambda r: (r["final_heldout_bpb"], r["max_jump"] if r["max_jump"] is not None else 999.0))
    return (
        f"{best['label']} has the lowest matched-token final held-out BPB "
        f"({best['final_heldout_bpb']:.4f}) among fused-guard-clean runs."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = []
    for log in sorted(args.log_dir.glob("*.log")):
        run = parse_log(log)
        run["heldout_curve"] = load_curve(log)
        run["shock"] = compute_shock(run)
        runs.append(run)

    summaries = [summarize(r) for r in runs]
    payload = {
        "runs": runs,
        "summary": summaries,
        "verdict": verdict(summaries),
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
