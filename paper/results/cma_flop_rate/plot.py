#!/usr/bin/env python3
"""Render the convergence plot for the CMA-tuned FLOPs-per-bit finding.

Produces two panels (PNG + PDF):
  Panel A — loss (bits/token, smoothed) vs cumulative training FLOPs (log scale).
            When the curves overlap, the slope is identical across model families.
  Panel B — the running FLOPs-per-bit-of-compression rate (FLOPs / (bits saved
            vs uniform baseline)) along training. As training proceeds this rate
            climbs (each marginal bit is more expensive), but the four curves
            track one another to within ~10–50% across two orders of magnitude
            of FLOPs.
"""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent

LABELS = {
    "e88":     "NDM (E88) — nonlinear delta, matrix state",
    "fla-gdn": "FLA-GDN — linear gated delta-net",
    "mamba2":  "Mamba2 — linear selective SSM",
    "e1":      "E1 — vanilla nonlinear Elman, dense W_h",
}
COLORS = {
    "e88":     "#d62728",  # NDM red
    "fla-gdn": "#1f77b4",  # blue
    "mamba2":  "#2ca02c",  # green
    "e1":      "#9467bd",  # purple
}
ORDER = ["e88", "fla-gdn", "mamba2", "e1"]


def load_overlay():
    series = defaultdict(list)
    with open(OUT / "overlay.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            series[row["model"]].append({
                "step": int(row["step"]),
                "flops": float(row["flops"]),
                "bits": float(row["bits_per_token"]),
                "fpb": float(row["flops_per_bit_reduction"]),
            })
    return series


def main():
    series = load_overlay()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axA, axB = axes

    for model in ORDER:
        if model not in series:
            continue
        rows = series[model]
        # Skip the very early warm-up where bits/token is meaningless.
        rows = [r for r in rows if r["bits"] < 6.0 and r["step"] >= 20]
        flops = [r["flops"] for r in rows]
        bits = [r["bits"] for r in rows]
        fpb = [r["fpb"] for r in rows]
        axA.plot(flops, bits, label=LABELS[model], color=COLORS[model], lw=1.6)
        axB.plot(flops, fpb, label=LABELS[model], color=COLORS[model], lw=1.6)

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("Cumulative training FLOPs (= 6 · N · tokens)")

    axA.set_ylabel("Validation loss (bits / token, smoothed)")
    axA.set_title("A. Loss vs FLOPs — CMA-tuned best config per model family")
    axA.set_ylim(1.4, 4.0)

    axB.set_yscale("log")
    axB.set_ylabel("FLOPs per bit of compression\n(cumulative FLOPs / bits saved vs uniform baseline)")
    axB.set_title("B. FLOPs-per-bit rate — four families collapse to one curve")

    # Single combined legend
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)
    fig.suptitle(
        "CMA-tuned recurrent baselines converge to a common FLOPs-per-bit rate "
        "(N=4 model families, ~480M parameters each)", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    for ext in ("png", "pdf"):
        path = OUT / f"convergence.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
