#!/usr/bin/env python3
"""Multi-seed aggregation of the grad-clip A/B (seeds 42,43,44).

Loads each seed's 4-cell results and computes, per seed and pooled, on BOTH the
non-averaged (primary) and schedule-free-averaged bases:
  - per-arm ΔBPB(clip-off − clip-on)
  - gap_clipon / gap_clipoff (emender − gdn2) and Δgap
Reports mean ± sample-std across seeds, sign consistency, and applies the task
decision rule (|Δgap| vs the lb-compare noise floor 0.01–0.09) to the pooled
mean. Also tallies skip / non-finite-loss / fused-guard across all 12 runs.
"""
import json, statistics as st
from pathlib import Path

THIS = Path(__file__).resolve().parent
NOISE_LO, NOISE_HI = 0.01, 0.09
SEEDS = [42, 43, 44]


def load_seed(seed):
    f = THIS / ("clip_ab_results.json" if seed == 42 else f"clip_ab_results_s{seed}.json")
    if not f.exists():
        return None
    rows = json.loads(f.read_text())
    # index by (arm, on/off) regardless of seed suffix in tag
    d = {}
    for r in rows:
        onoff = 'on' if float(r["grad_clip"]) > 0 else 'off'
        d[(r["arm"], onoff)] = r
    return d


def fval(cell, key):
    if cell is None:
        return None
    v = cell.get(key)
    return float(v) if v not in (None, "") else None


def seed_block(d, basis):
    e_on = fval(d.get(("emender-mlp", "on")), basis)
    e_off = fval(d.get(("emender-mlp", "off")), basis)
    g_on = fval(d.get(("gdn2-mlp", "on")), basis)
    g_off = fval(d.get(("gdn2-mlp", "off")), basis)
    if None in (e_on, e_off, g_on, g_off):
        return None
    gap_on = e_on - g_on
    gap_off = e_off - g_off
    return dict(
        emender_clipon=e_on, emender_clipoff=e_off, gdn2_clipon=g_on, gdn2_clipoff=g_off,
        dBPB_emender=round(e_off - e_on, 4), dBPB_gdn2=round(g_off - g_on, 4),
        gap_clipon=round(gap_on, 4), gap_clipoff=round(gap_off, 4),
        delta_gap=round(gap_off - gap_on, 4),
    )


def pooled(blocks, key):
    vals = [b[key] for b in blocks if b is not None]
    if not vals:
        return None
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return dict(mean=round(m, 4), std=round(s, 4), n=len(vals), vals=[round(v, 4) for v in vals])


def verdict(mean_dgap, vals):
    if mean_dgap is None:
        return "INCONCLUSIVE"
    mag = abs(mean_dgap)
    signs = set(1 if v < 0 else (0 if v == 0 else -1) for v in vals)  # -1==narrowing toward emender
    consistent_narrowing = all(v < 0 for v in vals)
    base = (f"mean Δgap={mean_dgap:+.4f} over seeds {vals}; "
            f"{'consistent narrowing toward emender' if consistent_narrowing else 'sign varies across seeds'}. ")
    if mag < NOISE_LO:
        return base + f"|mean Δgap| < {NOISE_LO} => VERDICT ROBUST (no movement)."
    if mag < NOISE_HI:
        return base + (f"|mean Δgap|={mag:.4f} within lb-compare noise band [{NOISE_LO},{NOISE_HI}] "
                       f"=> VERDICT ROBUST (shift within single-seed noise).")
    if mean_dgap < 0:
        return base + (f"|mean Δgap|={mag:.4f} EXCEEDS floor {NOISE_HI} AND narrows toward emender "
                       f"=> per decision rule, RE-OPEN emender verdict.")
    return base + f"|mean Δgap|={mag:.4f} EXCEEDS floor {NOISE_HI} but widens AGAINST emender => robust/strengthened."


seeds_data = {s: load_seed(s) for s in SEEDS}
report = {"seeds_loaded": [s for s in SEEDS if seeds_data[s] is not None]}

# instability + fused-guard tally across all loaded cells
allcells = [c for s in SEEDS if seeds_data[s] for c in seeds_data[s].values()]
report["n_runs"] = len(allcells)
report["total_skips"] = sum(int(c.get("skip_count", 0) or 0) for c in allcells)
report["any_nonfinite_loss_stop"] = any(bool(c.get("nonfinite_loss_stop")) for c in allcells)
report["all_fused_guard_ok"] = all(bool(c.get("fused_guard_ok")) for c in allcells)
report["all_completed_full_steps"] = all(str(c.get("steps_done")) == str(c.get("steps_budget")) for c in allcells)

for basis, label in [("heldout_bpb_nonavg", "nonavg"), ("heldout_bpb_avg", "avg")]:
    blocks = [seed_block(seeds_data[s], basis) for s in SEEDS if seeds_data[s]]
    blocks = [b for b in blocks if b is not None]
    seed_dgaps = [b["delta_gap"] for b in blocks]
    rep = dict(
        per_seed={s: seed_block(seeds_data[s], basis) for s in SEEDS if seeds_data[s]},
        dBPB_emender_off_minus_on=pooled(blocks, "dBPB_emender"),
        dBPB_gdn2_off_minus_on=pooled(blocks, "dBPB_gdn2"),
        gap_clipon=pooled(blocks, "gap_clipon"),
        gap_clipoff=pooled(blocks, "gap_clipoff"),
        delta_gap=pooled(blocks, "delta_gap"),
    )
    dg = rep["delta_gap"]
    rep["verdict"] = verdict(dg["mean"] if dg else None, seed_dgaps)
    report[label] = rep

print(json.dumps(report, indent=2))
(THIS / "clip_ab_multiseed_analysis.json").write_text(json.dumps(report, indent=2))
print("\nWROTE", THIS / "clip_ab_multiseed_analysis.json")
