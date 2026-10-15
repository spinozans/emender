#!/usr/bin/env python3
"""Plot the receipted E97 training record; checkpoint payloads are never opened.

Run from any directory. Defaults read the three public-export tails and their
SOURCES.sha256 from /tmp/e97-150b-logs, plus checkpoint *names* in the three
listed workstation run directories. Frontier and GDN2 values are transcribed
from paper/main.typ (tab_frontier_anchors and tab_pair_identities). The output
receipt retains every plotted value and source identity for independent review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TOKENS_PER_STEP = 8 * 4 * 2048
TAIL_NAMES = ("tail_1282500.txt", "tail_1525000.txt", "tail_2300930.txt")
RUN_NAMES = (
    "emender_E97_1.3B_20260702_111457",
    "emender_E97_1.3B_20260706_083743",
    "emender_E97_1.3B_20260709_084606",
)
LOG_LINE = re.compile(r"^step (\d+) \| loss (\d+\.\d+) \| lr ")
CHECKPOINT_NAME = re.compile(r"checkpoint_step_(\d+)_loss_(\d+\.\d+)\.pt")
# Exact accepted-token counters, not Frontier worker-step extrapolations.
FRONTIER_ANCHORS = (
    (150_793_748_480, 2.4365),  # shared workstation seed / corpus boundary
    (199_615_447_040, 2.3178),
    (400_942_039_040, 2.2928),
    (513_013_841_920, 2.2798),
)
GDN2_STEP, GDN2_LOSS = 2_323_616, 2.4034
BLUE, ORANGE, RED = "#4477AA", "#EE7733", "#CC3311"


def read_window(path, expected_sha256):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Source checksum mismatch: {path}")
    records = {}
    for line in raw.decode().splitlines():
        match = LOG_LINE.match(line)
        if match:
            step, loss = int(match[1]), float(match[2])
            if step in records and records[step] != loss:
                raise ValueError(f"Conflicting loss at step {step}: {path}")
            records[step] = loss
    points = sorted(records.items())
    if len(points) < 2 or any(b[0] - a[0] != 25 for a, b in zip(points, points[1:])):
        raise ValueError(f"Expected a contiguous per-25-step window: {path}")
    return {"path": str(path), "sha256": actual, "step_loss": points}


def read_checkpoints(runs_dir):
    anchors = []
    for run in RUN_NAMES:
        # Directory enumeration only: never stat, hash, or load .pt contents.
        for path in sorted((runs_dir / run).glob("checkpoint_step_*_loss_*.pt")):
            match = CHECKPOINT_NAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Unexpected checkpoint name: {path.name}")
            anchors.append({"path": str(path), "step": int(match[1]), "loss": float(match[2])})
    anchors.sort(key=lambda a: a["step"])
    if len(anchors) != 28 or (anchors[0]["step"], anchors[-1]["step"]) != (1_065_000, 2_300_930):
        raise ValueError("Expected the 28 receipted workstation checkpoint names")
    if len({a["step"] for a in anchors}) != 28:
        raise ValueError("Duplicate checkpoint step")
    return anchors


def plot_record(receipt, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(7.5, 5.4), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.5))
    detail_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    ax = fig.add_subplot(grid[1, :])
    for detail, window in zip(detail_axes, receipt["windows"]):
        points = window["step_loss"]
        tokens = [step * TOKENS_PER_STEP / 1e9 for step, _ in points]
        losses = [loss for _, loss in points]
        for target in (detail, ax):
            target.plot(tokens, losses, color=BLUE, alpha=0.4, linewidth=0.4, zorder=1)
        detail.set_title(f"{tokens[0]:.2f}–{tokens[-1]:.2f} B tokens", pad=5)
        detail.set_xlim(tokens[0], tokens[-1])
        detail.set_ylim(1.95, 3.05)
        detail.tick_params(axis="x", labelrotation=0)
        detail.locator_params(axis="x", nbins=3)
        detail.grid(axis="y", alpha=0.15, linewidth=0.5)
    detail_axes[0].set_ylabel("Training loss (nats/token)")
    for detail in detail_axes[1:]:
        detail.tick_params(labelleft=False)

    anchors = receipt["workstation_checkpoints"]
    ax.scatter([a["step"] * TOKENS_PER_STEP / 1e9 for a in anchors],
               [a["loss"] for a in anchors], s=22, marker="o", facecolors="white",
               edgecolors=BLUE, linewidths=0.9, zorder=3)
    frontier = receipt["frontier_anchors"]
    ax.scatter([a["accepted_tokens"] / 1e9 for a in frontier[1:]],
               [a["loss"] for a in frontier[1:]], s=35, marker="D", color=RED, zorder=4)
    gd = receipt["gdn2_reference"]
    ax.scatter([gd["accepted_tokens"] / 1e9], [gd["loss"]], s=115, marker="*",
               color=ORANGE, edgecolors="white", linewidths=0.5, zorder=5)
    ax.annotate("GDN2 reference\n152.28 B, 2.4034", (gd["accepted_tokens"] / 1e9, gd["loss"]),
                xytext=(218, 2.58), fontsize=8, color="#99501F",
                arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.7})
    boundary = frontier[0]["accepted_tokens"] / 1e9
    ax.axvspan(boundary, 530, color=RED, alpha=0.025, zorder=0)
    ax.axvline(boundary, color="0.55", linestyle=":", linewidth=0.8, zorder=0)
    ax.text(boundary - 5, 3.015, "The Pile", ha="right", va="top", fontsize=8, color="0.4")
    ax.text(boundary + 5, 3.015, "CommaPile · Frontier", va="top", fontsize=8, color="0.4")
    ax.set(xlim=(60, 530), ylim=(1.95, 3.05), xlabel="Cumulative accepted tokens (billions)",
           ylabel="Training loss (nats/token)")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    handles = [
        Line2D([], [], color=BLUE, alpha=0.4, lw=0.6, label="E97 raw log windows (detail above)"),
        Line2D([], [], color=BLUE, marker="o", mfc="white", linestyle="none", markersize=4,
               label="E97 workstation checkpoints"),
        Line2D([], [], color=RED, marker="D", linestyle="none", markersize=4,
               label="E97 Frontier checkpoints"),
        Line2D([], [], color=ORANGE, marker="*", linestyle="none", markersize=8,
               label="GDN2 retained checkpoint (last-window loss)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("/tmp/e97-150b-logs"))
    parser.add_argument("--runs-dir", type=Path,
                        default=Path("/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("e97_recorded_loss.png"))
    args = parser.parse_args()
    source_manifest = args.logs_dir / "SOURCES.sha256"
    hashes = {name: digest for digest, name in
              (line.split() for line in source_manifest.read_text().splitlines() if line.strip())}
    receipt = {
        "tokens_per_workstation_step": TOKENS_PER_STEP,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "windows": [read_window(args.logs_dir / name, hashes[name]) for name in TAIL_NAMES],
        "workstation_checkpoints": read_checkpoints(args.runs_dir),
        "frontier_source": "paper/main.typ: tab_frontier_anchors (shared seed, then three Frontier anchors)",
        "frontier_anchors": [{"accepted_tokens": tokens, "loss": loss} for tokens, loss in FRONTIER_ANCHORS],
        "corpus_boundary_accepted_tokens": FRONTIER_ANCHORS[0][0],
        "gdn2_source": "paper/main.typ: tab_pair_identities (retained checkpoint last-window loss)",
        "gdn2_reference": {"step": GDN2_STEP, "accepted_tokens": GDN2_STEP * TOKENS_PER_STEP, "loss": GDN2_LOSS},
        "rendering": "Raw lines within each window; checkpoint markers only; no interpolation or lines across gaps.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_record(receipt, args.output)
    receipt_path = args.output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    for window in receipt["windows"]:
        points = window["step_loss"]
        print(f"{Path(window['path']).name}: {len(points)} records, steps {points[0][0]}–{points[-1][0]}")
    print(f"Saved {args.output} and {receipt_path}; 28 workstation anchors, 3 Frontier anchors, 1 GDN2 reference.")


if __name__ == "__main__":
    main()
