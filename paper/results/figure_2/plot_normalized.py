#!/usr/bin/env python3
"""Render the normalized Figure 2 from the existing CSVs.

Reads E88_NDM.csv, FLA_GDN.csv, M2RNN_CMA.csv (produced by smooth.py
when run on the host with /tmp/pile_convergence_* logs present), and writes
figure_2_draft.png.

Normalized presentation conventions (shared with cma_flop_rate/plot.py):
  x-axis : wall-clock training hours (log scale)
  y-axis : training loss in bits per byte (100K-step trailing moving average)
           BPB = nats/token × log2(e) / bytes_per_token
           bytes/token = 3.918625 (canonical 2000-sample sweep on Pile,
           p50k_base, chunk_tokens=2048; see
           scripts/estimate_tokenizer_bytes_per_token.json)
  colors : Emender = #1f77b4 (blue)
           GDN = #ff7f0e (orange)
           M²RNN-CMA = #d62728 (red) — emphasises strict-above (worse) position
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
REPO = OUT.parents[2]

# Canonical bytes-per-token (Pile, p50k_base, chunk_tokens=2048, 2000-sample sweep).
# Pinned by v22-bpb-fix; sourced from scripts/estimate_tokenizer_bytes_per_token.json.
_TOKENIZER_JSON = REPO / "scripts" / "estimate_tokenizer_bytes_per_token.json"
with open(_TOKENIZER_JSON) as _f:
    _CANON = json.load(_f)
BYTES_PER_TOKEN = float(_CANON["mean_bytes_per_token"])  # 3.918625
NATS_TO_BPB = math.log2(math.e) / BYTES_PER_TOKEN         # 0.368164
assert abs(NATS_TO_BPB - _CANON["bits_per_byte_per_nat_per_token"]) < 1e-9

# Shared palette across all loss-vs-X panels in the paper.
COLORS = {
    "Emender":    "#1f77b4",
    "GDN":        "#ff7f0e",
    "M²RNN-CMA":  "#d62728",
}

# Plot in an order that draws M²RNN-CMA on top so its strict-above position
# is unambiguous when curves get close near the data-entropy floor.
ORDER = ["GDN", "Emender", "M²RNN-CMA"]
SMOOTH_COLUMN = "trail_100k"
SMOOTH_LABEL = "100K-step trailing average"

FILES = {
    "Emender":    OUT / "E88_NDM.csv",
    "GDN":        OUT / "FLA_GDN.csv",
    "M²RNN-CMA":  OUT / "M2RNN_CMA.csv",
}

PARAMS = {
    "Emender":    "1.27 B",
    "GDN":        "1.35 B",
    "M²RNN-CMA":  "1.31 B",
}

FULL_LABEL_OFFSETS = {
    "GDN": (6, -9),
    "Emender": (6, 0),
    "M²RNN-CMA": (6, 9),
}

TAIL_LABEL_OFFSETS = {
    "GDN": (-6, -10),
    "Emender": (-6, -8),
    "M²RNN-CMA": (-6, 10),
}


def load(path: Path):
    """Return (wallclock_h, bpb) where bpb is the BPB conversion of SMOOTH_COLUMN.

    Underlying CSVs remain in native units (nats/token); the BPB conversion
    happens here at the display step so the training-log artefacts are not
    rewritten.
    """
    xs, ys = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            h = float(r["wallclock_h"])
            if h <= 0:
                continue
            xs.append(h)
            ys.append(float(r[SMOOTH_COLUMN]) * NATS_TO_BPB)
    return np.array(xs), np.array(ys)


def main():
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [1.0, 1.22]}
    )

    data = {name: load(FILES[name]) for name in ORDER}
    max_x = max(float(xs[-1]) for xs, _ in data.values() if len(xs))
    visible_ys = np.concatenate([ys[xs >= 1.0] for xs, ys in data.values() if len(ys)])

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
            f"{ys[-1]:.3f}",
            (xs[-1], ys[-1]),
            xytext=FULL_LABEL_OFFSETS[name],
            textcoords="offset points",
            fontsize=8,
            color=COLORS[name],
            va="center",
        )
    axA.set_xscale("log")
    axA.set_xlim(1.0, max(520.0, max_x * 1.08))
    axA.set_ylim(min(0.94, float(visible_ys.min()) - 0.015), max(1.74, float(visible_ys.max()) * 1.02))
    axA.set_xlabel("Wall-clock training hours (log scale)", fontsize=11)
    axA.set_ylabel(f"Training loss (bits / byte, {SMOOTH_LABEL})", fontsize=11)
    axA.set_title("A. Full curve (log-x)", fontsize=11)
    axA.legend(fontsize=9, loc="upper right", framealpha=0.95)
    axA.grid(True, which="both", alpha=0.3)

    # Panel B — tail zoom (linear-x) where the strict order is the headline
    tail_lo = 40.0
    tail_ys = []
    for name in ORDER:
        xs, ys = data[name]
        mask = xs >= tail_lo
        tail_ys.extend(ys[mask].tolist())
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
                xytext=TAIL_LABEL_OFFSETS[name],
                textcoords="offset points",
                fontsize=8,
                color=COLORS[name],
                va="center",
                ha="right",
            )
    axB.set_xlim(tail_lo, max(520.0, max_x + 12.0))
    tail_min, tail_max = min(tail_ys), max(tail_ys)
    tail_pad = max(0.008, (tail_max - tail_min) * 0.08)
    axB.set_ylim(tail_min - tail_pad, tail_max + tail_pad)
    axB.set_xlabel("Wall-clock training hours", fontsize=11)
    axB.set_ylabel("Training loss (bits / byte)", fontsize=11)
    axB.set_title("B. Tail (h ≥ 40) — wall-clock order at the tail", fontsize=10)
    axB.grid(True, which="both", alpha=0.3)

    out = OUT / "figure_2_draft.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
