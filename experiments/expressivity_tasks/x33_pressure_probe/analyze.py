#!/usr/bin/env python3
"""X33 analysis: coincidence/landmark tables from the probe's JSON outputs.

Prints the accuracy-vs-k and pressure-vs-k curves, the registered landmarks
(k_acc99 / k_acc75 / k_p50 / k_psat), the same-k B0-vs-B0x4 comparison, the
lead/lag distribution of first windowed saturation on collapsing examples,
and the per-shape breakdown. Pure read-only post-processing; all numbers in
REPORT.md trace to this script's stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ARMS = ("tape_B0", "tape_B0x4")
SWEEP = lambda arm: sorted(
    (int(k[7:]) for k in json.loads(
        (DATA / f"{arm}_per_k.json").read_text() if False else "{}")),
)  # placeholder; replaced below


def load(arm):
    per_k = json.loads((DATA / f"{arm}_per_k.json").read_text())
    ex = [json.loads(l) for l in
          (DATA / f"{arm}_per_example.jsonl").read_text().splitlines()]
    return per_k, ex


def sweep_ks(per_k, arm):
    if arm == "tape_B0":
        return [k for k in range(608, 865, 16)]
    own = [k for k in range(2816, 3073, 16)]
    return own


def pool_row(per_k, pool, k):
    return per_k.get(pool, {}).get(str(k))


def fmt(v, nd=4):
    if v is None:
        return "  --  "
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def sweep_table(arm):
    per_k, ex = load(arm)
    print(f"\n### {arm} — fine sweep (own window)")
    print("k      acc    meanP(acc)  fracP>=.5  fracP>=.9  fracSAT  med_first_sat  peak_live")
    for k in sweep_ks(per_k, arm):
        r = pool_row(per_k, f"sweep_k{k}", k)
        if not r:
            continue
        print(f"{k:<6d} {fmt(r['balanced_accuracy'])} "
              f"{fmt(r['mean_cum_pressure_accepts'])}      "
              f"{fmt(r['frac_accepts_pressure_ge_0.5'])}    "
              f"{fmt(r['frac_accepts_pressure_ge_0.9'])}    "
              f"{fmt(r['frac_accepts_sat_event'])}    "
              f"{fmt(r['median_first_sat_accepts'], 0):>8}      "
              f"{fmt(r['mean_peak_live'], 1)}")


def bin_table(arm):
    per_k, ex = load(arm)
    print(f"\n### {arm} — bins (per-k rows are pooled over ~4 words/k)")
    for pool in ("bin_id", "bin_2x", "bin_4x", "bin_9x"):
        rows = per_k.get(pool, {})
        ks = sorted(int(k) for k in rows)
        # decimate to <=16 rows for readability
        step = max(1, len(ks) // 16)
        print(f"\n-- {pool} (n_k={len(ks)}; every {step}th k shown)")
        print("k      acc    meanP(acc)  fracSAT  peak")
        for k in ks[::step]:
            r = rows[str(k)]
            print(f"{k:<6d} {fmt(r['balanced_accuracy'])} "
                  f"{fmt(r['mean_cum_pressure_accepts'])}    "
                  f"{fmt(r['frac_accepts_sat_event'])}   "
                  f"{fmt(r['mean_peak_live'], 1)}")


def landmarks(arm):
    per_k, ex = load(arm)
    out = {}
    for label, pool_fn in (
            ("own_sweep", lambda a: [(f"sweep_k{k}", k)
                                     for k in sweep_ks(per_k, a)]),):
        rows = []
        for pool, k in pool_fn(arm):
            r = pool_row(per_k, pool, k)
            if r:
                rows.append((k, r))
        acc99 = acc75 = p50 = psat = None
        for k, r in rows:
            if acc99 is None and r["balanced_accuracy"] < 0.99:
                acc99 = k
            if acc75 is None and r["balanced_accuracy"] < 0.75:
                acc75 = k
            m = r["mean_cum_pressure_accepts"]
            if m is not None:
                if p50 is None and m >= 0.5:
                    p50 = k
                if psat is None and m >= 0.9:
                    psat = k
        out[label] = {"k_acc99": acc99, "k_acc75": acc75,
                      "k_p50": p50, "k_psat": psat}
    print(f"\n### {arm} landmarks (own sweep pools)")
    print(json.dumps(out, indent=1))
    return out


def same_k_comparison():
    b0k, _ = load("tape_B0")
    b4k, _ = load("tape_B0x4")
    print("\n### Same-k comparison at B0's collapse band (B0 sweep points)")
    print("k      B0:acc  B0:P(acc)  B0:fracSAT | B0x4:acc  B0x4:P(acc)  B0x4:fracSAT")
    for k in range(608, 865, 16):
        r0 = pool_row(b0k, f"sweep_k{k}", k)
        r4 = pool_row(b4k, f"sweep_k{k}", k)
        if not (r0 and r4):
            continue
        print(f"{k:<6d} {fmt(r0['balanced_accuracy'])}  "
              f"{fmt(r0['mean_cum_pressure_accepts'])}     "
              f"{fmt(r0['frac_accepts_sat_event'])}   | "
              f"{fmt(r4['balanced_accuracy'])}  "
              f"{fmt(r4['mean_cum_pressure_accepts'])}      "
              f"{fmt(r4['frac_accepts_sat_event'])}")


def lead_lag(arm):
    _, ex = load(arm)
    rows = [e for e in ex if e["pool"].startswith("sweep_k")
            and e["label"] == 1 and e["first_sat"] >= 0]
    collapsing = [e for e in rows if not e["correct"]]
    holding = [e for e in rows if e["correct"]]
    print(f"\n### {arm} lead/lag on sweep accept words (first W64 saturation)")

    def summarize(name, group):
        if not group:
            print(f"  {name}: none")
            return
        lead_b = np.array([e["lead_b"] for e in group], dtype=float)
        lead_eos = np.array([e["lead_eos"] for e in group], dtype=float)
        print(f"  {name}: n={len(group)} "
              f"lead_b median={np.median(lead_b):.0f} "
              f"min={lead_b.min():.0f} max={lead_b.max():.0f} | "
              f"lead_eos median={np.median(lead_eos):.0f} "
              f"min={lead_eos.min():.0f} max={lead_eos.max():.0f}")

    summarize("collapsing (incorrect)", collapsing)
    summarize("holding (correct) with sat event", holding)
    # per-k medians for collapsing
    per_k = {}
    for e in collapsing:
        per_k.setdefault(e["k"], []).append(e["lead_b"])
    for k in sorted(per_k):
        print(f"    k={k}: median lead_b={np.median(per_k[k]):.0f} "
              f"(n={len(per_k[k])})")


def per_shape(arm, pools):
    _, ex = load(arm)
    print(f"\n### {arm} per-shape accuracy on {pools}")
    for pool in pools:
        sub = [e for e in ex if e["pool"] == pool]
        for shape in ("balanced", "leading_b", "count_mismatch", "trailing_a"):
            g = [e for e in sub if e["shape"] == shape]
            if not g:
                continue
            acc = sum(e["correct"] for e in g) / len(g)
            print(f"  {pool} {shape:<16s} n={len(g):<4d} acc={acc:.4f}")


def main():
    for arm in ARMS:
        sweep_table(arm)
    landmarks("tape_B0")
    landmarks("tape_B0x4")
    same_k_comparison()
    lead_lag("tape_B0")
    lead_lag("tape_B0x4")
    per_shape("tape_B0x4", ["sweep_k2816", "sweep_k3040", "bin_9x"])
    per_shape("tape_B0", ["bin_9x"])
    for arm in ARMS:
        bin_table(arm)


if __name__ == "__main__":
    main()
