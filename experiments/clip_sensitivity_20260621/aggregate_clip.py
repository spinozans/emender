#!/usr/bin/env python3
"""Aggregate the 4-cell grad-clip A/B into the task's decision rule.

Reads clip_ab_results.json (written by run_clip_ab.py) and computes, on BOTH the
non-averaged (primary, lb-compare-preferred) and schedule-free-averaged bases:
  - per-arm  ΔBPB(clip-off − clip-on)
  - gap_clipon  = BPB(emender) − BPB(gdn2)   under clip-on
  - gap_clipoff = BPB(emender) − BPB(gdn2)   under clip-off
  - Δgap = gap_clipoff − gap_clipon
and applies the task decision rule against the lb-compare BPB noise floor
(~0.01–0.09). Prints a verdict and emits a markdown table block.
"""
import json, sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
NOISE_LO, NOISE_HI = 0.01, 0.09          # lb-compare BPB noise band

rows = json.loads((THIS / "clip_ab_results.json").read_text())
by = {r["tag"]: r for r in rows}


def f(tag, key):
    v = by.get(tag, {}).get(key)
    return float(v) if v not in (None, "") else None


def block(basis_key, label):
    e_on, e_off = f("emender-mlp__clipon", basis_key), f("emender-mlp__clipoff", basis_key)
    g_on, g_off = f("gdn2-mlp__clipon", basis_key), f("gdn2-mlp__clipoff", basis_key)
    out = {"basis": label}
    out["emender_clipon"], out["emender_clipoff"] = e_on, e_off
    out["gdn2_clipon"], out["gdn2_clipoff"] = g_on, g_off
    out["dBPB_emender_off_minus_on"] = None if None in (e_on, e_off) else round(e_off - e_on, 4)
    out["dBPB_gdn2_off_minus_on"] = None if None in (g_on, g_off) else round(g_off - g_on, 4)
    gap_on = None if None in (e_on, g_on) else round(e_on - g_on, 4)
    gap_off = None if None in (e_off, g_off) else round(e_off - g_off, 4)
    out["gap_clipon_emender_minus_gdn2"] = gap_on
    out["gap_clipoff_emender_minus_gdn2"] = gap_off
    out["delta_gap"] = None if None in (gap_on, gap_off) else round(gap_off - gap_on, 4)
    return out


def verdict(dgap):
    if dgap is None:
        return "INCONCLUSIVE (a cell diverged / no BPB) — see per-cell divergence report"
    mag = abs(dgap)
    if mag < NOISE_LO:
        return (f"|Δgap|={mag:.4f} < noise floor low bound {NOISE_LO} => VERDICT ROBUST "
                f"(clip on/off does not move the emender−gdn2 gap beyond noise)")
    if mag < NOISE_HI:
        return (f"|Δgap|={mag:.4f} within noise band [{NOISE_LO},{NOISE_HI}] => VERDICT ROBUST "
                f"(gap shift is within lb-compare single-seed noise)")
    # beyond floor: direction matters
    if dgap < 0:
        return (f"|Δgap|={mag:.4f} > noise floor {NOISE_HI} AND negative (clip-off NARROWS gap "
                f"toward emender) => RE-OPEN emender verdict per decision rule")
    return (f"|Δgap|={mag:.4f} > noise floor {NOISE_HI} AND positive (clip-off WIDENS gap "
            f"against emender) => verdict robust / strengthened (clip-off hurts emender)")


report = {"cells": {t: {k: by[t].get(k) for k in (
    "grad_clip", "steps_done", "skip_count", "nonfinite_loss_stop",
    "fused_guard_ok", "grad_mean", "grad_median", "grad_max", "grad_frac_gt1",
    "heldout_bpb_nonavg", "heldout_bpb_avg", "final_loss_last100", "n_params", "wall_s")}
    for t in by}}
nonavg = block("heldout_bpb_nonavg", "non-averaged (primary)")
avg = block("heldout_bpb_avg", "schedule-free averaged")
report["nonavg"] = nonavg
report["avg"] = avg
report["verdict_nonavg"] = verdict(nonavg["delta_gap"])
report["verdict_avg"] = verdict(avg["delta_gap"])

print(json.dumps(report, indent=2))
(THIS / "clip_ab_analysis.json").write_text(json.dumps(report, indent=2))
print("\nWROTE", THIS / "clip_ab_analysis.json")
