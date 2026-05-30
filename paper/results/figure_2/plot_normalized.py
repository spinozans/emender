#!/usr/bin/env python3
"""Render the normalized Figure 2 from the existing CSVs.

Reads E88_NDM.csv, FLA_GDN.csv, M2RNN_CMA.csv (produced by smooth.py
when run on the host with /tmp/pile_convergence_* logs present), and writes
figure_2_draft.png.

Normalized presentation conventions (shared with cma_flop_rate/plot.py):
  x-axis : wall-clock training hours (linear scale, full snapshot extent)
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

LABEL_X_PAD_H = 10.0
LABEL_RIGHT_PAD_H = 115.0
LABEL_MIN_GAP_BPB = 0.016
LABEL_Y_PAD_BPB = 0.014

WINDOW_STEPS = 100_000
Y_MIN, Y_MAX = 0.94, 1.32


def _full_window_start(steps: np.ndarray) -> int:
    """Return the first index whose trailing 100K-step window is fully populated."""
    if len(steps) <= 1:
        return 0
    log_every = int(np.median(np.diff(steps)))
    window_points = max(1, WINDOW_STEPS // log_every)
    return min(len(steps) - 1, window_points - 1)


def _first_visible_index(steps: np.ndarray, ys: np.ndarray) -> int:
    """Start curves only after the averaging window is full and y is in range."""
    start = _full_window_start(steps)
    in_range = np.nonzero((ys[start:] >= Y_MIN) & (ys[start:] <= Y_MAX))[0]
    if len(in_range) == 0:
        return start
    return start + int(in_range[0])


def load(path: Path):
    """Return post-warm-up wallclock_h and BPB arrays for SMOOTH_COLUMN.

    Underlying CSVs remain in native units (nats/token); the BPB conversion
    happens here at the display step so the training-log artefacts are not
    rewritten. The plotted prefix is removed until the trailing 100K-step
    window is fully populated; this avoids displaying the partial-window
    warm-up as if it were a stable 100K-step estimate.
    """
    steps, xs, ys = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            h = float(r["wallclock_h"])
            if h <= 0:
                continue
            steps.append(int(r["step"]))
            xs.append(h)
            ys.append(float(r[SMOOTH_COLUMN]) * NATS_TO_BPB)
    steps = np.array(steps)
    xs = np.array(xs)
    ys = np.array(ys)
    start = _first_visible_index(steps, ys)
    return {
        "steps": steps[start:],
        "wallclock_h": xs[start:],
        "bpb": ys[start:],
        "start_step": int(steps[start]),
        "start_h": float(xs[start]),
        "endpoint_h": float(xs[-1]),
        "endpoint_bpb": float(ys[-1]),
    }


def right_edge_label_positions(data):
    """Return readable right-edge label lanes ordered by endpoint BPB."""
    rows = sorted(
        data.items(),
        key=lambda item: item[1]["endpoint_bpb"],
        reverse=True,
    )
    endpoints = np.array([row[1]["endpoint_bpb"] for row in rows], dtype=float)
    if len(endpoints) <= 1:
        lanes = endpoints
    else:
        gaps = endpoints[:-1] - endpoints[1:]
        if np.all(gaps >= LABEL_MIN_GAP_BPB):
            lanes = endpoints
        else:
            center = float(np.mean(endpoints))
            offsets = (
                (len(endpoints) - 1) / 2.0 - np.arange(len(endpoints), dtype=float)
            ) * LABEL_MIN_GAP_BPB
            lanes = center + offsets

    high = Y_MAX - LABEL_Y_PAD_BPB
    low = Y_MIN + LABEL_Y_PAD_BPB
    if len(lanes) and lanes[0] > high:
        lanes = lanes - (lanes[0] - high)
    if len(lanes) and lanes[-1] < low:
        lanes = lanes + (low - lanes[-1])
    return {name: float(y) for (name, _), y in zip(rows, lanes)}


def main():
    fig, ax = plt.subplots(figsize=(8.6, 4.9))

    data = {name: load(FILES[name]) for name in ORDER}
    max_x = max(d["endpoint_h"] for d in data.values())
    label_x = max_x + LABEL_X_PAD_H
    label_lanes = right_edge_label_positions(data)

    for name in ORDER:
        d = data[name]
        xs, ys = d["wallclock_h"], d["bpb"]
        ax.plot(
            xs, ys,
            label=f"{name} ({PARAMS[name]})",
            color=COLORS[name],
            lw=2.4 if name == "M²RNN-CMA" else 1.9,
            zorder=5 if name == "M²RNN-CMA" else 3,
        )
        label_y = label_lanes[name]
        ax.plot(
            [xs[-1], label_x - 3.0],
            [ys[-1], label_y],
            color="#7f8790",
            lw=0.7,
            alpha=0.55,
            linestyle=(0, (3.0, 2.2)),
            dash_capstyle="round",
            zorder=2,
        )
        ax.text(
            label_x,
            label_y,
            name,
            fontsize=8,
            color=COLORS[name],
            va="center",
            ha="left",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 0.8,
            },
        )

    ax.set_xlim(0.0, max(520.0, max_x + LABEL_RIGHT_PAD_H))
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Wall-clock training hours", fontsize=11)
    ax.set_ylabel(f"Training loss (bits / byte, {SMOOTH_LABEL})", fontsize=11)
    ax.set_title("1.3 B racer after the 100K-step trailing window is populated", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)

    out = OUT / "figure_2_draft.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
