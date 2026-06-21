#!/usr/bin/env python3
"""Build the matched-token held-out BPB curve (emender vs gdn2) and the
who-leads-at-matched-tokens table from the offline eval CSVs.

task: offline-eval-references. Descriptive only (a curve) -- not an accept/reject verdict.
Both references were scored by scripts/eval_checkpoint.py on the SAME held-out
tensor (heldout_pile_tail_p50k_2048_1m.pt, md5 8e1198ab..., bytes_per_token
3.945132), y-mode swap applied, forward-only, fused kernel.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EMENDER_CSV = HERE / "emender_heldout_bpb.csv"
GDN2_CSV = HERE / "gdn2_heldout_bpb.csv"
COMBINED_CSV = HERE / "matched_token_bpb_curve.csv"
LEADER_CSV = HERE / "who_leads_at_matched_tokens.csv"
PLOT_PNG = HERE / "matched_token_bpb_curve.png"


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("split", "primary") != "primary":
                continue
            rows.append(
                {
                    "step": int(row["step"]),
                    "tokens": int(row["tokens"]),
                    "ce": float(row["ce"]),
                    "bpb": float(row["bpb"]),
                }
            )
    rows.sort(key=lambda r: r["tokens"])
    return rows


def main() -> int:
    em = load(EMENDER_CSV)
    gd = load(GDN2_CSV)
    print(f"emender points: {len(em)}  gdn2 points: {len(gd)}")

    em_tok = np.array([r["tokens"] for r in em], dtype=float)
    em_bpb = np.array([r["bpb"] for r in em], dtype=float)
    gd_tok = np.array([r["tokens"] for r in gd], dtype=float)
    gd_bpb = np.array([r["bpb"] for r in gd], dtype=float)

    # Combined long-form CSV (both models, one row per scored checkpoint).
    with COMBINED_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "step", "tokens", "ce", "bpb"])
        for r in em:
            w.writerow(["emender", r["step"], r["tokens"], f"{r['ce']:.8f}", f"{r['bpb']:.8f}"])
        for r in gd:
            w.writerow(["gdn2", r["step"], r["tokens"], f"{r['ce']:.8f}", f"{r['bpb']:.8f}"])

    # Matched-token comparison: linear interpolation of each curve onto a shared
    # token grid covering the OVERLAP of the two token ranges. Linear interp on a
    # BPB-vs-tokens curve is the standard matched-token read for monotone curves;
    # we only interpolate inside the measured range (no extrapolation).
    lo = max(em_tok.min(), gd_tok.min())
    hi = min(em_tok.max(), gd_tok.max())
    print(f"overlap token range: [{lo/1e6:.1f}M, {hi/1e6:.1f}M]")

    anchors = np.array([t for t in [200e6, 300e6, 400e6, 500e6, 600e6, 700e6, 800e6, 900e6]
                        if lo <= t <= hi] + [hi])
    anchors = np.unique(anchors)

    rows = []
    for t in anchors:
        e = float(np.interp(t, em_tok, em_bpb))
        g = float(np.interp(t, gd_tok, gd_bpb))
        delta = e - g  # >0 => gdn2 lower BPB (leads); <0 => emender leads
        leader = "gdn2" if delta > 0 else ("emender" if delta < 0 else "tie")
        rows.append((t, e, g, delta, leader))

    with LEADER_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tokens", "emender_bpb", "gdn2_bpb", "emender_minus_gdn2", "leader_lower_bpb"])
        for t, e, g, d, lead in rows:
            w.writerow([int(t), f"{e:.6f}", f"{g:.6f}", f"{d:+.6f}", lead])

    print("\n=== Who leads at matched tokens (lower held-out BPB wins) ===")
    print(f"{'tokens':>12} {'emender':>10} {'gdn2':>10} {'em-gdn2':>10} {'leader':>9}")
    for t, e, g, d, lead in rows:
        print(f"{t/1e6:>10.0f}M {e:>10.4f} {g:>10.4f} {d:>+10.4f} {lead:>9}")

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(em_tok / 1e6, em_bpb, "o-", color="#c0392b", label="emender (E97 + MLP)")
    ax.plot(gd_tok / 1e6, gd_bpb, "s-", color="#2471a3", label="gdn2-mlp")
    ax.axvspan(lo / 1e6, hi / 1e6, color="gray", alpha=0.08, label="matched-token overlap")
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("held-out BPB (pile-tail, y-mode, fused)")
    ax.set_title("Offline held-out BPB vs tokens — emender vs gdn2 (shared tensor)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for r in em:
        ax.annotate(f"{r['bpb']:.3f}", (r["tokens"] / 1e6, r["bpb"]),
                    textcoords="offset points", xytext=(0, 7), fontsize=7, color="#c0392b")
    for r in gd:
        ax.annotate(f"{r['bpb']:.3f}", (r["tokens"] / 1e6, r["bpb"]),
                    textcoords="offset points", xytext=(0, -12), fontsize=7, color="#2471a3")
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=130)
    print(f"\nwrote {COMBINED_CSV.name}, {LEADER_CSV.name}, {PLOT_PNG.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
