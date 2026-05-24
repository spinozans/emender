#!/usr/bin/env python3
"""Render the normalized Figure 3 from the existing CSVs.

Reads E88_NDM.csv, FLA_GDN.csv, Mamba2.csv, M2RNN_CMA.csv (produced by smooth.py
when run on the host with /tmp/pile_convergence_* logs present), and writes
figure_3_draft.png.

Normalized presentation conventions (shared with cma_flop_rate/plot.py):
  x-axis : wall-clock training hours (log scale)
  y-axis : training loss in nats per token (10K-step centred moving average)
  colors : NDM = #1f77b4 (blue)
           FLA-GDN = #ff7f0e (orange)
           Mamba2 = #2ca02c (green)
           M2RNN-CMA = #d62728 (red) — emphasises strict-above (worse) position
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

# Shared palette across all loss-vs-X panels in the paper.
COLORS = {
    "NDM":       "#1f77b4",
    "FLA-GDN":   "#ff7f0e",
    "Mamba2":    "#2ca02c",
    "M2RNN-CMA": "#d62728",
}

# Plot in an order that draws M2RNN-CMA on top so its strict-above position
# is unambiguous when curves get close near the data-entropy floor.
ORDER = ["FLA-GDN", "NDM", "Mamba2", "M2RNN-CMA"]

FILES = {
    "NDM":       OUT / "E88_NDM.csv",
    "FLA-GDN":   OUT / "FLA_GDN.csv",
    "Mamba2":    OUT / "Mamba2.csv",
    "M2RNN-CMA": OUT / "M2RNN_CMA.csv",
}

PARAMS = {
    "NDM":       "1.27 B",
    "FLA-GDN":   "1.35 B",
    "Mamba2":    "0.93 B",
    "M2RNN-CMA": "1.31 B",
}


def load(path: Path):
    xs, ys = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            h = float(r["wallclock_h"])
            if h <= 0:
                continue
            xs.append(h)
            ys.append(float(r["smooth_10k"]))
    return np.array(xs), np.array(ys)


def main():
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    data = {name: load(FILES[name]) for name in ORDER}

    # Panel A — full log-x view
    for name in ORDER:
        xs, ys = data[name]
        axA.plot(
            xs, ys,
            label=f"{name} ({PARAMS[name]})",
            color=COLORS[name],
            lw=2.2 if name == "M2RNN-CMA" else 1.7,
            zorder=5 if name == "M2RNN-CMA" else 3,
        )
        axA.annotate(
            f"{ys[-1]:.2f}",
            (xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=COLORS[name],
            va="center",
        )
    axA.set_xscale("log")
    axA.set_xlim(1.0, 500.0)
    axA.set_ylim(2.55, 4.7)
    axA.set_xlabel("Wall-clock training hours (log scale)", fontsize=11)
    axA.set_ylabel("Training loss (nats / token, 10K-step smoothed)", fontsize=11)
    axA.set_title("A. Full curve (log-x)", fontsize=11)
    axA.legend(fontsize=9, loc="upper right", framealpha=0.95)
    axA.grid(True, which="both", alpha=0.3)

    # Panel B — tail zoom (linear-x) where the strict order is the headline
    tail_lo = 40.0
    for name in ORDER:
        xs, ys = data[name]
        mask = xs >= tail_lo
        axB.plot(
            xs[mask], ys[mask],
            color=COLORS[name],
            lw=2.4 if name == "M2RNN-CMA" else 1.8,
            zorder=5 if name == "M2RNN-CMA" else 3,
        )
        # End-of-curve label
        xs_t, ys_t = xs[mask], ys[mask]
        if len(xs_t):
            axB.annotate(
                f"{name}: {ys_t[-1]:.3f}",
                (xs_t[-1], ys_t[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color=COLORS[name],
                va="center",
            )
    axB.set_xlim(tail_lo, 460.0)
    axB.set_ylim(2.63, 3.05)
    axB.set_xlabel("Wall-clock training hours", fontsize=11)
    axB.set_ylabel("Training loss (nats / token)", fontsize=11)
    axB.set_title("B. Tail (h ≥ 40) — strict wall-clock order is M²RNN-CMA > Mamba2 > NDM", fontsize=10)
    axB.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Figure 3 — 1.27 B language-model loss vs wall-clock  "
        "(Pile, ctx=2048, schedule-free AdamW, bf16, as of 2026-05-24)",
        fontsize=11, y=1.00,
    )
    out = OUT / "figure_3_draft.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
