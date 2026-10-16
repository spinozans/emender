#!/usr/bin/env python3
"""plot_recorded_loss.py — E97 recorded training loss at every retained checkpoint.

Light version (operator ruling 2026-10-15: lighter touch, show the losses we
have, do not narrate the log gaps). One axis: recorded loss (nats/token, last
logged window at each retained checkpoint) versus cumulative accepted tokens.
Workstation checkpoints are filename-embedded losses; Frontier anchors are the
retained-checkpoint losses from tab_frontier_anchors; the GDN2 reference is
its single retained checkpoint. A light connector joins successive retained
losses; no interpolation is claimed beyond joining the known points.

Run from any directory. Receipt JSON retains every plotted value.
"""
import json
import re
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOK_PER_STEP_WS = 65_536          # 8 ranks x batch 4 x context 2048
TOK_PER_STEP_FRONTIER = 16_777_216 # 2048 ranks x batch 4 x context 2048

WS_RUNS = [
    "/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260702_111457",
    "/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260706_083743",
    "/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606",
]

# Frontier retained-checkpoint lineage (paper tab_frontier_anchors):
# (worker step delta base tokens B, accepted tokens, recorded loss)
FRONTIER = [
    (150_793_748_480, 2.4365),   # released workstation seed
    (199_615_447_040, 2.3178),   # first 256-node continuation
    (400_942_039_040, 2.2928),   # intermediate milestone
    (513_013_841_920, 2.2798),   # retained endpoint
]

GDN2 = (152_280_498_176, 2.4034)  # single retained GDN2 checkpoint


def workstation_points():
    pts = []
    for run in WS_RUNS:
        for f in glob.glob(run + "/checkpoint_step_*_loss_*.pt"):
            m = re.search(r"step_(\d+)_loss_([\d.]+)\.pt", f)
            if m:
                pts.append({"step": int(m.group(1)),
                            "loss": float(m.group(2)),
                            "source": pathlib.Path(f).name})
    pts.sort(key=lambda p: p["step"])
    assert len(pts) == 28, f"expected 28 workstation checkpoints, got {len(pts)}"
    for p in pts:
        p["tokens"] = p["step"] * TOK_PER_STEP_WS
    return pts


def main():
    ws = workstation_points()
    fr = [{"tokens": t, "loss": l, "source": "frontier-anchor"} for t, l in FRONTIER]

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=250)

    # GDN2 reference point
    ax.scatter([GDN2[0] / 1e9], [GDN2[1]], marker="*", s=160, color="#d97706",
               zorder=5, label="GDN2 retained checkpoint (152.28 B)")

    # E97 curve: light connector through the known losses, then markers
    all_pts = sorted(ws + fr, key=lambda p: p["tokens"])
    ax.plot([p["tokens"] / 1e9 for p in all_pts], [p["loss"] for p in all_pts],
            color="#3b82f6", lw=1.1, alpha=0.55, zorder=2)
    ax.scatter([p["tokens"] / 1e9 for p in ws], [p["loss"] for p in ws],
               s=22, color="#1d4ed8", zorder=4, label="E97 workstation checkpoints")
    ax.scatter([p["tokens"] / 1e9 for p in fr], [p["loss"] for p in fr],
               s=46, marker="D", color="#0f766e", zorder=4,
               label="E97 Frontier checkpoints")

    # corpus boundary at the Frontier handoff
    ax.axvline(150.794, color="#94a3b8", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.annotate("Pile / CommaPile", xy=(150.794, ax.get_ylim()[0]),
                xytext=(153, 2.245), fontsize=8, color="#64748b")

    ax.set_xlabel("Accepted training tokens (billions)")
    ax.set_ylabel("Recorded training loss (nats/token)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parent / "e97_recorded_loss.png"
    fig.savefig(out)
    print("wrote", out)

    receipt = {"workstation": ws, "frontier": fr, "gdn2": {"tokens": GDN2[0], "loss": GDN2[1]},
               "tok_per_step_ws": TOK_PER_STEP_WS,
               "note": "recorded loss at every retained checkpoint; connector joins known points only"}
    rpath = out.with_suffix(".receipt.json")
    rpath.write_text(json.dumps(receipt, indent=1))
    print("wrote", rpath)


if __name__ == "__main__":
    main()
