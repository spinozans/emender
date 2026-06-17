#!/usr/bin/env python3
"""Promote the FUSED re-score to the canonical offline-eval CSVs.

task: re-run-offline.

  1. Archive the committed EAGER CSVs as archive_eager_*_heldout_bpb.csv
     (provenance: the original curve was produced on the eager emender path).
  2. Compute the row-matched fused-vs-eager BPB delta and write
     fused_vs_eager_delta.csv.
  3. Overwrite the canonical *_heldout_bpb.csv with the fused numbers.

Run analyze.py afterwards to rebuild matched_token_bpb_curve.csv,
who_leads_at_matched_tokens.csv, and the plot from the fused CSVs.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path: Path):
    rows = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["step"])] = {
                "step": int(r["step"]),
                "tokens": int(r["tokens"]),
                "ce": float(r["ce"]),
                "bpb": float(r["bpb"]),
                "split": r.get("split", "primary"),
                "checkpoint": r["checkpoint"],
                "ce_str": r["ce"],
                "bpb_str": r["bpb"],
            }
    return rows


def main():
    delta_rows = []
    for name in ("emender", "gdn2"):
        canonical_path = HERE / f"{name}_heldout_bpb.csv"
        fused_path = HERE / f"{name}_heldout_bpb.fused.csv"  # new = fused
        archive_path = HERE / f"archive_eager_{name}_heldout_bpb.csv"

        # 1. Archive the eager CSV the FIRST time (before any promotion). On a
        # re-run the canonical is already fused, so the archive is the only true
        # eager baseline -- always read eager from the archive when present.
        if not archive_path.exists():
            shutil.copy2(canonical_path, archive_path)
            print(f"archived eager -> {archive_path.name}")
        else:
            print(f"archive already exists: {archive_path.name} (eager baseline)")

        eager = load(archive_path)
        fused = load(fused_path)
        eager_path = canonical_path  # promotion target below

        # 2. Delta on the shared step set.
        for step in sorted(set(eager) & set(fused)):
            e, f = eager[step], fused[step]
            dbpb = f["bpb"] - e["bpb"]
            dce = f["ce"] - e["ce"]
            delta_rows.append({
                "model": name,
                "step": step,
                "tokens": f["tokens"],
                "eager_bpb": f"{e['bpb']:.8f}",
                "fused_bpb": f"{f['bpb']:.8f}",
                "fused_minus_eager_bpb": f"{dbpb:+.8f}",
                "eager_ce": f"{e['ce']:.8f}",
                "fused_ce": f"{f['ce']:.8f}",
                "fused_minus_eager_ce": f"{dce:+.8f}",
            })

        # 3. Overwrite canonical CSV with the fused rows (preserve column order).
        with fused_path.open(newline="") as src, eager_path.open("w", newline="") as dst:
            dst.write(src.read())
        print(f"promoted fused -> {eager_path.name}")

    # Write the delta CSV.
    delta_csv = HERE / "fused_vs_eager_delta.csv"
    with delta_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model", "step", "tokens",
            "eager_bpb", "fused_bpb", "fused_minus_eager_bpb",
            "eager_ce", "fused_ce", "fused_minus_eager_ce",
        ])
        w.writeheader()
        for r in delta_rows:
            w.writerow(r)
    print(f"wrote {delta_csv.name}")

    # Print a human summary.
    print("\n=== fused - eager BPB delta (row-matched) ===")
    print(f"{'model':>8} {'step':>8} {'eager_bpb':>11} {'fused_bpb':>11} {'Δbpb':>12}")
    maxabs = 0.0
    for r in delta_rows:
        d = float(r["fused_minus_eager_bpb"])
        maxabs = max(maxabs, abs(d))
        print(f"{r['model']:>8} {r['step']:>8} {r['eager_bpb']:>11} {r['fused_bpb']:>11} {d:>+12.8f}")
    print(f"\nmax |Δbpb| across all rows = {maxabs:.8f}")


if __name__ == "__main__":
    main()
