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

# Shared with paper/results/figure_3/plot_normalized.py — same color = same
# architecture everywhere in the paper.
LABELS = {
    "e88":     "NDM (E88) — nonlinear delta, matrix state",
    "fla-gdn": "FLA-GDN — linear gated delta-net",
    "mamba2":  "Mamba2 — linear selective SSM",
    "e1":      "Elman — vanilla nonlinear, dense W_h",
}
COLORS = {
    "e88":     "#1f77b4",  # NDM blue (matches figure 3)
    "fla-gdn": "#ff7f0e",  # FLA-GDN orange
    "mamba2":  "#2ca02c",  # Mamba2 green
    "e1":      "#7f7f7f",  # vanilla Elman grey (low-baseline witness)
}
ORDER = ["e88", "fla-gdn", "mamba2", "e1"]

LN2 = math.log(2)


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
        nats = [r["bits"] * LN2 for r in rows]   # bits → nats for shared y-axis
        fpb = [r["fpb"] for r in rows]
        axA.plot(flops, nats, label=LABELS[model], color=COLORS[model], lw=1.6)
        axB.plot(flops, fpb, label=LABELS[model], color=COLORS[model], lw=1.6)

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("Cumulative training FLOPs (= 6 · N · tokens)")

    axA.set_ylabel("Training loss (nats / token, smoothed)")
    axA.set_title("A. Loss vs FLOPs — CMA-tuned best config per model family")
    axA.set_ylim(1.4 * LN2, 4.0 * LN2)

    axB.set_yscale("log")
    axB.set_ylabel("FLOPs per bit of compression\n(cumulative FLOPs / bits saved vs uniform baseline)")
    axB.set_title("B. FLOPs-per-bit rate — four families collapse to one curve")

    # Single combined legend, with an explicit note that M²RNN-CMA is absent
    # from this matched-480M CMA sweep (per cma_flop_rate/SOURCES.md).
    handles, labels = axA.get_legend_handles_labels()
    # Add a non-data legend entry calling out the absence.
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color="none", marker="", linestyle=""))
    labels.append("M²RNN-CMA: not in 480 M CMA sweep")
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)
    fig.suptitle(
        "CMA-tuned recurrent baselines converge to a common FLOPs-per-bit rate "
        "(N=4 families, ~480 M parameters each; M²RNN-CMA absent — see §6 Limitations)", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    for ext in ("png", "pdf"):
        path = OUT / f"convergence.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
