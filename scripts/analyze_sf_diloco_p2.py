#!/usr/bin/env python3
"""Summarize SF-DiLoCo P2 matched-gain runs from training logs.

Parses:
  * train loss lines at log_every cadence
  * DiLoCo merge lines
  * P2 [DiLoCo-geom] per-merge ratios
  * heldout curve CSV / FINAL_HELDOUT_BPB lines when present

Writes a JSON summary and a compact markdown report for downstream WG tasks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


STEP_RE = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+([0-9.eE+-]+)")
MERGE_RE = re.compile(r"\[DiLoCo\]\s+(?:FINAL\s+)?merge #(\d+) at step (\d+)")
CONFIG_RE = re.compile(r"periodic model-weight averaging: K=\d+ outer_lr=([0-9.eE+-]+) outer_beta=([0-9.eE+-]+)")
GEOM_RE = re.compile(
    r"\[DiLoCo-geom\].*?land_frac\([^)]*\)=([0-9.eE+-]+)\s+"
    r"disp_mag\([^)]*\)=([0-9.eE+-]+)\s+"
    r"gap_health\([^)]*\)=([0-9.eE+-]+)\s+"
    r"\(outer_lr=([0-9.eE+-]+) outer_beta=([0-9.eE+-]+)\)"
)
FINAL_BPB_RE = re.compile(r"FINAL_HELDOUT_BPB:\s+([0-9.eE+-]+)")
FUSED_RE = re.compile(r"\[fused-guard\].*NO eager fallback")
NONFINITE_RE = re.compile(r"non[- ]?finite|nan|inf", re.IGNORECASE)


def _fnum(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_log(path: Path) -> dict:
    losses: dict[int, float] = {}
    merges: list[dict] = []
    geom: list[dict] = []
    fused_guard_ranks: set[str] = set()
    final_bpb = None
    nonfinite_hits: list[str] = []
    outer_lr = None
    outer_beta = None

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        m = CONFIG_RE.search(line)
        if m:
            outer_lr = float(m.group(1))
            outer_beta = float(m.group(2))
            continue
        m = STEP_RE.match(line)
        if m:
            losses[int(m.group(1))] = float(m.group(2))
            continue
        m = MERGE_RE.search(line)
        if m:
            merges.append({"merge": int(m.group(1)), "step": int(m.group(2))})
            continue
        m = GEOM_RE.search(line)
        if m:
            geom.append({
                "land_frac": float(m.group(1)),
                "disp_mag": float(m.group(2)),
                "gap_health": float(m.group(3)),
                "outer_lr": float(m.group(4)),
                "outer_beta": float(m.group(5)),
            })
            continue
        m = FINAL_BPB_RE.search(line)
        if m:
            final_bpb = float(m.group(1))
        if FUSED_RE.search(line):
            rank_match = re.search(r"rank\s+(\d+)/(\d+)", line)
            fused_guard_ranks.add(rank_match.group(0) if rank_match else line)
        if NONFINITE_RE.search(line) and "non-finite score" not in line:
            nonfinite_hits.append(line)

    shocks = []
    for merge in merges:
        step = merge["step"]
        # The training loop logs every 25 steps in these runs; use nearest
        # available prior/current records so the parser remains tolerant.
        prior_steps = [s for s in losses if s < step]
        post_steps = [s for s in losses if s >= step]
        if not prior_steps or not post_steps:
            continue
        pre_step = max(prior_steps)
        post_step = min(post_steps)
        pre_loss = losses[pre_step]
        post_loss = losses[post_step]
        recovery_step = None
        for s in sorted(k for k in losses if k > post_step):
            if losses[s] <= pre_loss:
                recovery_step = s
                break
        shocks.append({
            "merge": merge["merge"],
            "step": step,
            "pre_step": pre_step,
            "pre_loss": pre_loss,
            "post_step": post_step,
            "post_loss": post_loss,
            "jump": post_loss - pre_loss,
            "recovery_step": recovery_step,
            "recovery_steps": None if recovery_step is None else recovery_step - post_step,
        })

    return {
        "log": str(path),
        "losses": [{"step": s, "loss": losses[s]} for s in sorted(losses)],
        "merges": merges,
        "geom": geom,
        "shock": shocks,
        "final_heldout_bpb": final_bpb,
        "fused_guard_count": len(fused_guard_ranks),
        "nonfinite_hits": nonfinite_hits,
        "outer_lr": outer_lr,
        "outer_beta": outer_beta,
    }


def parse_curve(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "step": int(row["step"]),
                "tokens": int(row["tokens"]),
                "train_loss": _fnum(row.get("train_loss")),
                "heldout_bpb": _fnum(row.get("heldout_bpb")),
            })
    return {"path": str(path), "rows": rows, "last_bpb": rows[-1]["heldout_bpb"] if rows else None}


def summarize_run(name: str, log: Path, curve: Path | None) -> dict:
    data = parse_log(log)
    data["name"] = name
    data["heldout_curve"] = parse_curve(curve)
    if data["final_heldout_bpb"] is None and data["heldout_curve"]:
        data["final_heldout_bpb"] = data["heldout_curve"]["last_bpb"]
    data["stable"] = (
        not data["nonfinite_hits"]
        and bool(data["losses"])
        and all(math.isfinite(x["loss"]) for x in data["losses"])
    )
    return data


def write_markdown(summary: dict, out: Path) -> None:
    lines = [
        "# SF-DiLoCo P2 matched-gain summary",
        "",
        "All momentum configs are matched effective gain (`outer_lr ~= 1 - outer_beta`); `beta=0.9, outer_lr=1.0` is not run.",
        "",
        "| run | outer_beta | outer_lr | final heldout BPB | final train loss | max shock | max recovery steps | geom land_frac range | gap_health range | fused guard ranks | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for run in summary["runs"]:
        geom = run["geom"]
        beta = geom[0]["outer_beta"] if geom else run.get("outer_beta")
        lr = geom[0]["outer_lr"] if geom else run.get("outer_lr")
        losses = run["losses"]
        final_loss = losses[-1]["loss"] if losses else None
        shocks = run["shock"]
        max_jump = max((s["jump"] for s in shocks), default=None)
        rec_vals = [s["recovery_steps"] for s in shocks if s["recovery_steps"] is not None]
        max_rec = max(rec_vals) if rec_vals else None
        land = [g["land_frac"] for g in geom]
        gap = [g["gap_health"] for g in geom]
        verdict = "stable" if run["stable"] else "unstable"
        land_range = f"{min(land):.4f}-{max(land):.4f}" if land else "n/a"
        gap_range = f"{min(gap):.3e}-{max(gap):.3e}" if gap else "n/a"
        lines.append(
            f"| {run['name']} | {beta if beta is not None else ''} | "
            f"{lr if lr is not None else ''} | "
            f"{run['final_heldout_bpb'] if run['final_heldout_bpb'] is not None else ''} | "
            f"{final_loss if final_loss is not None else ''} | "
            f"{max_jump if max_jump is not None else ''} | "
            f"{max_rec if max_rec is not None else ''} | "
            f"{land_range} | {gap_range} | {run['fused_guard_count']} | {verdict} |"
        )
    lines += ["", "## Per-merge shock and geometry", ""]
    for run in summary["runs"]:
        lines += [f"### {run['name']}", ""]
        lines.append("| merge | step | loss pre -> post | jump | recovery steps | land_frac | disp_mag | gap_health |")
        lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for idx, shock in enumerate(run["shock"]):
            geom = run["geom"][idx] if idx < len(run["geom"]) else {}
            land_frac = geom.get("land_frac")
            disp_mag = geom.get("disp_mag")
            gap_health = geom.get("gap_health")
            lines.append(
                f"| {shock['merge']} | {shock['step']} | "
                f"{shock['pre_loss']:.4f} -> {shock['post_loss']:.4f} | "
                f"{shock['jump']:+.4f} | "
                f"{'' if shock['recovery_steps'] is None else shock['recovery_steps']} | "
                f"{land_frac:.4f}" if land_frac is not None else
                f"| {shock['merge']} | {shock['step']} | "
                f"{shock['pre_loss']:.4f} -> {shock['post_loss']:.4f} | "
                f"{shock['jump']:+.4f} | "
                f"{'' if shock['recovery_steps'] is None else shock['recovery_steps']} | n/a"
            )
            if land_frac is not None:
                lines[-1] += f" | {disp_mag:.3e} | {gap_health:.3e} |"
            else:
                lines[-1] += " | n/a | n/a |"
        lines.append("")
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", nargs=3, metavar=("NAME", "LOG", "CURVE"),
                    required=True, help="Run name, log path, heldout curve CSV or '-'")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    runs = []
    for name, log, curve in args.run:
        curve_path = None if curve == "-" else Path(curve)
        runs.append(summarize_run(name, Path(log), curve_path))
    summary = {"runs": runs}
    Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, Path(args.out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
