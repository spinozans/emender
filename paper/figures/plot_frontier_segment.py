#!/usr/bin/env python3
"""Figure B: measured communication over the first 256-node Frontier segment.

Data source: reports/frontier/e97-final-seed-production-256n-5134243-verdict.json
(committed machine verdict for Slurm job 5134243). Plots the recorded
`collective_duration_ms` (73 hierarchical K40 merges) and `k40_cadence_seconds`
(72 between-merge intervals). These are measurements from one production
segment; no cells are interpolated.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
VERDICT = ROOT / "reports/frontier/e97-final-seed-production-256n-5134243-verdict.json"
OUT = pathlib.Path(__file__).resolve().parent / "frontier_segment_5134243.png"

with VERDICT.open() as fh:
    v = json.load(fh)

merges = v["collective_duration_ms"]          # 73 values, seconds*1000
cadence = v["k40_cadence_seconds"]            # 72 intervals, seconds
merge_steps = v["merge_steps"]                # 73 worker steps

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=False)

idx = range(1, len(merges) + 1)
ax1.bar(list(idx), [m / 1000.0 for m in merges], color="#2c5fa8", width=0.8)
ax1.axhline(v["median_collective_duration_ms"] / 1000.0, color="#c0392b",
            linestyle="--", linewidth=1.0,
            label=f"median {v['median_collective_duration_ms']/1000.0:.3f} s")
ax1.set_ylabel("merge duration (s)", fontsize=9)
ax1.set_title("73 hierarchical K40 merges, job 5134243", fontsize=10)
ax1.legend(fontsize=8, frameon=False)
ax1.tick_params(labelsize=8)

ax2.bar(range(1, len(cadence) + 1), cadence, color="#1f7a3f", width=0.8)
ax2.axhline(v["median_k40_cadence_seconds"], color="#c0392b", linestyle="--",
            linewidth=1.0, label=f"median {v['median_k40_cadence_seconds']:.0f} s")
ax2.set_xlabel("merge index (resumed 150.794 B-token seed \u2192 199.615 B-token checkpoint)",
               fontsize=9)
ax2.set_ylabel("inter-merge interval (s)", fontsize=9)
ax2.legend(fontsize=8, frameon=False)
ax2.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}  ({len(merges)} merges, {len(cadence)} intervals, "
      f"final merge step {merge_steps[-1]})")
