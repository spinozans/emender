#!/usr/bin/env python3
"""Combine the two REAL empirical modalities into the seed-maturity threshold
table (task: seed-maturity-threshold). CPU-only, no GPU, no mock data.

Modality A (NEW, this attempt): per-merge TRAIN-LOSS continuity harvested from the
  production DiLoCo run.logs (harvest_merge_continuity.py -> merge_continuity_
  summary.json). The detrended per-merge jump is Frankle's instability signal read
  in train-loss space at the beta=0 centroid.
Modality B (existing, gold standard): offline-scored HELD-OUT BPB degradation
  (consensus - single-GPU at matched total tokens) from the scaling-law
  degradation_summary.json and the seed-race seed_race_i4_degradation.csv.

Both are differential (robust to the absolute held-out tensor). They are
independent measurements (local train loss vs offline held-out bpb) and we report
both per cell so the in-basin/blow-up verdict is cross-checked, not single-source.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# --- Modality A: train-loss merge continuity ---------------------------------
cont = {r["run"]: r for r in json.loads(
    (HERE / "merge_continuity_summary.json").read_text())}

# --- Modality B: held-out BPB degradation ------------------------------------
sl = json.loads((REPO / "experiments/diloco_scaling_law/"
                 "degradation_summary.json").read_text())
heldout = defaultdict(list)
for r in sl:
    heldout[r["cell"]].append(r["degradation"])
seedrace = [float(r["degradation"]) for r in csv.DictReader(
    open(REPO / "experiments/diloco_seed_race_i4/"
         "seed_race_i4_degradation.csv"))
    if r.get("degradation") not in (None, "")]


def rng(vals):
    if not vals:
        return None
    return (min(vals), max(vals), sum(vals) / len(vals), len(vals))


# Map each run to its held-out cell (by maturity/island/beta).
HELDOUT_FOR = {
    "stab_k250":         rng(heldout["S0 scratch / I=4"]),
    "swell_i2_k250":     rng(heldout["S_well(528M) / I=2"]),
    "swell_i4_k250":     rng(heldout["S_well(528M) / I=4"]),
    "seed_race_i4":      rng(seedrace),
    "swell_i4_mom_k250": rng(heldout["S_well(528M) / I=4 +mom0.9"]),
    "outer_mom_i6":      None,  # live; held-out not yet offline-scored
}

ORDER = ["stab_k250", "swell_i2_k250", "swell_i4_k250", "seed_race_i4",
         "swell_i4_mom_k250", "outer_mom_i6"]

rows = []
print(f"{'cell':22s} {'maturity':10s} {'I':>2s} {'beta':>4s} {'merges':>6s} "
      f"{'dt-jump(train)':>15s} {'heldout-deg':>22s} {'verdict':>9s}")
print("-" * 100)
for run in ORDER:
    c = cont[run]
    h = HELDOUT_FOR[run]
    dt = c["mean_merge_jump_detrended"]
    sem = c["sem_merge_jump_detrended"]
    if h:
        hstr = f"[{h[0]:+.3f},{h[1]:+.3f}] m{h[2]:+.3f} n{h[3]}"
    else:
        hstr = "(live; not scored)"
    print(f"{run:22s} {c['seed_maturity']:10s} {str(c['islands']):>2s} "
          f"{str(c['outer_beta']):>4s} {str(c['n_merges_total']):>6s} "
          f"{dt:+.3f}+/-{sem:.3f}   {hstr:>22s} {c['verdict']:>9s}")
    rows.append({
        "cell": run,
        "seed_maturity": c["seed_maturity"],
        "seed_tokens": c["seed_tokens"],
        "provenance": c["provenance"],
        "islands": c["islands"],
        "outer_beta": c["outer_beta"],
        "outer_lr": c["outer_lr"],
        "K": c["K"],
        "n_merges": c["n_merges_total"],
        "loss_first": c["first_loss"],
        "loss_last": c["last_loss"],
        "loss_max": c["max_loss"],
        "train_merge_jump_detrended": dt,
        "train_merge_jump_sem": sem,
        "heldout_deg_min": h[0] if h else "",
        "heldout_deg_max": h[1] if h else "",
        "heldout_deg_mean": round(h[2], 4) if h else "",
        "heldout_deg_n": h[3] if h else "",
        "verdict": c["verdict"],
    })

out = HERE / "final_threshold_table.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out}")
