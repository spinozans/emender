#!/usr/bin/env python3
"""Aggregate the 1.3B CMA-ES redo runs into a ranked leaderboard.

This closes the reproduction gap for the e97-raw 1.3B LM leaderboard: it turns
the per-model run directories (each containing ``results.json`` and/or
``eval_*/<id>.done`` files) into the ranked table reported in
``docs/E97_RAW_1P3B_LEADERBOARD.md``.

The ranking key is the CMA-ES fitness ``loss`` (mean loss over the candidate's
15-minute training trajectory), NOT ``final_loss`` (the last-window loss).

The raw run directories live under ``experiments/local/`` which is git-ignored,
so this script is the committed code path; point ``--root`` at the local redo
root to regenerate the table.

Usage:
    python scripts/aggregate_cmaes_leaderboard.py \
        --root experiments/local/cmaes_redo_1300m_20260529
    python scripts/aggregate_cmaes_leaderboard.py --root <root> --json   # machine-readable
"""
import argparse
import json
from pathlib import Path

# Models that participated in the 1.3B ctx2k CMA-ES redo, in arbitrary order
# (the output is sorted by fitness). Override with --models to restrict.
DEFAULT_MODELS = [
    "e97-raw", "gdn2-mlp", "e97", "e88-linear", "e88",
    "e88-raw", "e97-linear", "fla-gdn", "m2rnn", "gdn2", "transformer",
]


def collect_model(arch_dir: Path):
    """Return (best_eval, eval_count, status) for one model directory.

    Mirrors the aggregation used to build the published leaderboard: union the
    finalized ``results.json`` all_results with any per-eval ``.done`` files,
    de-duplicate by ``eval_id`` (falling back to (params, loss)), then take the
    lowest-``loss`` successful candidate.
    """
    best = None
    eval_count = 0
    status = "missing"
    if not arch_dir.exists():
        return best, eval_count, status

    for run in sorted(p for p in arch_dir.iterdir() if p.is_dir()):
        evals = []
        res = run / "results.json"
        if res.exists():
            try:
                evals += json.loads(res.read_text()).get("all_results", [])
                status = "complete"
            except Exception:
                pass
        for done in run.glob("eval_*/*.done"):
            try:
                evals.append(json.loads(done.read_text()))
            except Exception:
                pass

        seen = set()
        dedup = []
        for e in evals:
            k = e.get("eval_id")
            if k is None:
                k = (json.dumps(e.get("params", {}), sort_keys=True), e.get("loss"))
            if k not in seen:
                seen.add(k)
                dedup.append(e)
        eval_count = max(eval_count, len(dedup))
        if not res.exists() and dedup and status == "missing":
            status = "recoverable"
        for e in dedup:
            if e.get("success", True) and isinstance(e.get("loss"), (int, float)):
                if best is None or e["loss"] < best["loss"]:
                    best = e
    return best, eval_count, status


def build_rows(root: Path, models):
    rows = []
    for name in models:
        best, eval_count, status = collect_model(root / name)
        if best is not None:
            rows.append({
                "model": name,
                "evals": eval_count,
                "status": status,
                "loss": best["loss"],
                "final_loss": best.get("final_loss"),
                "actual_params": best.get("actual_params"),
                "eval_id": best.get("eval_id"),
                "config": best.get("params", {}),
            })
    rows.sort(key=lambda r: r["loss"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="CMA-ES redo root (e.g. experiments/local/cmaes_redo_1300m_20260529)")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                    help="model subdirectories to aggregate")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    rows = build_rows(Path(args.root), args.models)

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"{'rk':>2} {'model':14s} {'evals':>5} {'status':11s} "
          f"{'avg_loss':>9} {'final':>8} {'params_M':>9} {'eval':>4}")
    for rank, r in enumerate(rows, 1):
        pm = (r["actual_params"] or 0) / 1e6
        fl = r["final_loss"]
        fl = f"{fl:.4f}" if isinstance(fl, (int, float)) else "NA"
        print(f"{rank:2d} {r['model']:14s} {r['evals']:5d} {r['status']:11s} "
              f"{r['loss']:9.4f} {fl:>8} {pm:9.1f} {str(r['eval_id']):>4}")
        print(f"     config={json.dumps(r['config'])}")


if __name__ == "__main__":
    main()
