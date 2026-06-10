#!/usr/bin/env python3
"""opt-synth cross-probe aggregator (OPT_SPEC §6).

Reads the four probes' shared-schema JCC_ROWS.jsonl, reconciles the frozen
specialist ceilings S_c across probes (§6.1), rebuilds ONE unified leaderboard
ranked by headline JCC = min_c r_c against the reconciled ceilings (§6.2),
runs the harness-consistency check on the shared GDN-2 control B (§6.2), and
emits the per-lever contribution table (§6.3).

No new training. Pure re-scoring of the committed per-(arm,seed) rows against a
single reconciled denominator so all four probes become directly comparable.

Reconciliation rule (§1.3 "best accuracy any single specialist reaches"):
  S_c = max over every probe's specialist/control ceiling for corner c.
  This is the HARDEST (most conservative) denominator; ΔJCC between any two
  arms scored under the same S_c is ceiling-invariant, so the per-lever
  verdicts are robust to this choice (verified in each probe's RESULTS).

Usage:
  python experiments/expressivity_tasks/aggregate_opt_synth.py
  python experiments/expressivity_tasks/aggregate_opt_synth.py --write   # write reconciled block into opt_ceilings.json
"""
import argparse
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CORNERS = ["recall", "counting", "step_growth", "track"]
TAU = 0.95
PROBES = ["headlr", "initspread", "norm", "minimal"]

# Per-probe declared specialist ceilings (from each RESULTS.md / opt_ceilings.json),
# the per-corner best-specialist accuracy each probe measured under the shared battery.
# Used only to RECONCILE the denominator; raw per_corner_acc is re-divided by the
# reconciled max.  headlr published per-length ceilings -> length-averaged here.
PROBE_CEILINGS = {
    "headlr":     {"recall": (0.998 + 0.977 + 0.854) / 3, "counting": (1.000 + 0.956 + 0.608) / 3,
                   "step_growth": (0.683 + 0.664 + 0.653) / 3, "track": (1.000 + 1.000 + 0.999) / 3},
    "initspread": {"recall": 0.9547390407986112, "counting": 0.8901273939344618,
                   "step_growth": 0.9987072414822048, "track": 0.9990285237630209},
    "norm":       {"recall": 0.95166015625, "counting": 0.8962461683485244,
                   "step_growth": 0.9714715745713974, "track": 0.9996846516927084},
    "minimal":    {"recall": 0.9607476128472222, "counting": 0.8497280544704862,
                   "step_growth": 0.978133307562934, "track": 0.9999177720811631},
}


def load_rows(probe):
    f = os.path.join(HERE, f"results_opt_{probe}", "JCC_ROWS.jsonl")
    rows = []
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reconcile_ceilings():
    """S_c = max over probes (the hardest/most-honest denominator)."""
    rec = {}
    prov = {}
    for c in CORNERS:
        best_p, best_v = None, -1.0
        for p in PROBES:
            v = PROBE_CEILINGS[p][c]
            if v > best_v:
                best_v, best_p = v, p
        rec[c] = best_v
        prov[c] = best_p
    return rec, prov


def rescore(rows, ceil):
    """seed-average JCC=min_c r_c per arm against reconciled ceilings."""
    by_arm = defaultdict(list)
    for r in rows:
        acc = r.get("per_corner_acc", {})
        if not all(c in acc for c in CORNERS):
            continue  # skip partial rows (e.g. LR-screen rows with only recall+counting)
        ratios = {c: min(acc[c] / ceil[c], 1.0) for c in CORNERS}
        jcc = min(ratios[c] for c in CORNERS)
        held = sum(1 for c in CORNERS if ratios[c] >= TAU)
        by_arm[r["arm"]].append((jcc, ratios, held, r.get("seed")))
    out = {}
    for arm, lst in by_arm.items():
        n = len(lst)
        jcc_mean = sum(x[0] for x in lst) / n
        jcc_sd = math.sqrt(sum((x[0] - jcc_mean) ** 2 for x in lst) / n) if n > 1 else 0.0
        se = jcc_sd / math.sqrt(n) if n > 1 else 0.0
        rc = {c: sum(x[1][c] for x in lst) / n for c in CORNERS}
        held = sum(x[2] for x in lst) / n
        out[arm] = {"jcc": jcc_mean, "se": se, "n": n, "rc": rc, "held": held}
    return out


# ---- R* re-run scoring (§6.4) -----------------------------------------------
EVAL_LENGTHS = ["128", "256", "512"]
CORNER_TASKS = {
    "recall": ["mqar_recall"],
    "counting": ["modular_counter", "dyck_depth_unbounded", "anbncn_viability"],
    "step_growth": ["modular_quadratic", "iterated_nonlinear_map"],
    "track": ["s5_permutation"],
}
RSTAR_ARM_DESC = {
    "b2_house": "house default (= B2 anchor)",
    "c5": "house + head_lr_compute_mult=5 (opt-headlr winner)",
    "klr20": "house + knob_lr_mult=20 (opt-initspread knob-LR)",
    "rstar": "house + compute_mult=5 + decay_init=slow (COMPOSED R*)",
}


def _per_length_avg_acc(run):
    le = run.get("length_extrap")
    if not le:
        fa = run.get("final_acc")
        return float(fa) if fa is not None else None
    accs = [float(le[T]["acc"]) for T in EVAL_LENGTHS if le.get(T) and "acc" in le[T]]
    if not accs:
        fa = run.get("final_acc")
        return float(fa) if fa is not None else None
    return sum(accs) / len(accs)


def _conv_cert_acc(run):
    """Accuracy-plateau gate (the faithful §1.5 cert on exact-algorithm tasks,
    matching every probe: rel-loss cert is noise-dominated near zero loss).
    Spread of eval_acc over the final 30% of steps; converged iff < 0.05."""
    steps = run.get("steps")
    if not steps or len(steps) < 3:
        return None, None
    last = steps[-1]["step"]
    tail = [s["eval_acc"] for s in steps if s["step"] >= 0.7 * last and "eval_acc" in s]
    if len(tail) < 2:
        return None, None
    spread = max(tail) - min(tail)
    return spread, (spread < 0.05)


def _canon_task(task):
    # strip trailing _K<digits> (modular_counter_K5 -> modular_counter)
    import re
    return re.sub(r"_K\d+$", "", task)


def score_rstar():
    import re
    rs_dir = os.path.join(HERE, "results_opt_synth")
    files = [f for f in os.listdir(rs_dir) if f.startswith("rs_") and f.endswith(".json")]
    if not files:
        print("No rs_*.json yet in results_opt_synth/ — run run_opt_synth_rstar.py first.")
        return None
    ceil, _ = reconcile_ceilings()
    # arm -> seed -> task -> acc ; arm -> seed -> task -> converged
    idx = defaultdict(lambda: defaultdict(dict))
    conv = defaultdict(lambda: defaultdict(dict))
    for f in files:
        d = json.load(open(os.path.join(rs_dir, f)))
        stem = f[len("rs_"):-len(".json")]
        task, arm, seedpart = stem.rsplit("__", 2)
        seed = int(seedpart.replace("seed", ""))
        acc = _per_length_avg_acc(d)
        if acc is None:
            continue
        ct = _canon_task(task)
        idx[arm][seed][ct] = acc
        _, ok = _conv_cert_acc(d)
        conv[arm][seed][ct] = ok

    def per_seed_jcc(arm, seed):
        tacc = idx[arm][seed]
        rc = {}
        for c, tasks in CORNER_TASKS.items():
            vals = [tacc[t] for t in tasks if t in tacc]
            if not vals:
                return None, None
            rc[c] = min((sum(vals) / len(vals)) / ceil[c], 1.0)
        return min(rc[c] for c in CORNER_TASKS), rc

    print("\n" + "=" * 78)
    print("R* RE-RUN (§6.4) — composed regime confirmation (reconciled S_c, identical harness)")
    print("=" * 78)
    summary = {}
    for arm in ["b2_house", "c5", "klr20", "rstar"]:
        if arm not in idx:
            continue
        seeds = sorted(idx[arm].keys())
        per = [(s, *per_seed_jcc(arm, s)) for s in seeds]
        per = [(s, j, rc) for (s, j, rc) in per if j is not None]
        if not per:
            continue
        jccs = [j for _, j, _ in per]
        n = len(jccs)
        mean = sum(jccs) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in jccs) / n) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        rc_mean = {c: sum(rc[c] for _, _, rc in per) / n for c in CORNER_TASKS}
        nconv = sum(1 for s in seeds for t in conv[arm][s] if conv[arm][s].get(t))
        ntot = sum(1 for s in seeds for t in conv[arm][s])
        summary[arm] = {"jcc": mean, "se": se, "n": n, "rc": rc_mean,
                        "conv": f"{nconv}/{ntot}"}
        print(f"\n  [{arm}] {RSTAR_ARM_DESC.get(arm,'')}")
        print(f"    JCC={mean:.3f} (SE {se:.3f}, n={n} seeds; per-seed {[f'{j:.3f}' for j in jccs]})")
        print(f"    r_c: recall={rc_mean['recall']:.3f} counting={rc_mean['counting']:.3f} "
              f"step={rc_mean['step_growth']:.3f} track={rc_mean['track']:.3f}  conv(acc-plateau)={nconv}/{ntot}")

    # verdict
    if "rstar" in summary:
        base = max((summary[a]["jcc"] for a in ("c5", "klr20") if a in summary), default=None)
        rstar = summary["rstar"]["jcc"]
        print("\n" + "-" * 78)
        if base is not None:
            tol = max(summary["rstar"]["se"], 0.01)
            if rstar >= base - tol:
                print(f"  COMPOSITION HOLDS: JCC(rstar)={rstar:.3f} >= max(single-lever)={base:.3f} "
                      f"(within tol {tol:.3f}). R* confirmed; carry to 1.3B.")
            else:
                print(f"  COMPOSITION REGRESSES: JCC(rstar)={rstar:.3f} < max(single-lever)={base:.3f}. "
                      f"Fall back to the best single lever (§6.4).")
        if "b2_house" in summary:
            print(f"  Lever lift vs B2 anchor: rstar {rstar - summary['b2_house']['jcc']:+.3f}, "
                  f"c5 {summary.get('c5',{}).get('jcc',float('nan')) - summary['b2_house']['jcc']:+.3f}, "
                  f"klr20 {summary.get('klr20',{}).get('jcc',float('nan')) - summary['b2_house']['jcc']:+.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write reconciled block into opt_ceilings.json")
    ap.add_argument("--rstar", action="store_true", help="score the R* re-run (results_opt_synth/rs_*.json)")
    args = ap.parse_args()

    if args.rstar:
        score_rstar()
        return

    ceil, prov = reconcile_ceilings()
    print("=" * 78)
    print("RECONCILED FROZEN CEILINGS S_c (max over probes; §6.1)")
    print("=" * 78)
    for c in CORNERS:
        print(f"  {c:12s} {ceil[c]:.4f}  (from opt-{prov[c]})")

    all_scored = {}
    b_consistency = {}
    print("\n" + "=" * 78)
    print("PER-PROBE re-score (against reconciled S_c)")
    print("=" * 78)
    for p in PROBES:
        rows = load_rows(p)
        scored = rescore(rows, ceil)
        for arm, v in scored.items():
            all_scored[f"{p}:{arm}"] = {**v, "probe": p, "arm": arm}
        # harness-consistency: the shared GDN-2 control B
        for cand in ("gdn2-default", "B_gdn2", "B_gdn_lr1e3", "B_lr0p001"):
            if cand in scored:
                b_consistency[p] = (cand, scored[cand])
                break

    # Unified leaderboard
    print("\n" + "=" * 78)
    print("UNIFIED LEADERBOARD (all arms, reconciled S_c, seed-avg JCC=min_c r_c)")
    print("=" * 78)
    print(f"{'probe:arm':40s} {'JCC':>6s} {'SE':>6s} {'n':>2s} {'recall':>7s} {'count':>7s} {'step':>6s} {'track':>6s} {'held':>4s}")
    for key, v in sorted(all_scored.items(), key=lambda kv: -kv[1]["jcc"]):
        rc = v["rc"]
        print(f"{key:40s} {v['jcc']:6.3f} {v['se']:6.3f} {v['n']:2d} "
              f"{rc['recall']:7.3f} {rc['counting']:7.3f} {rc['step_growth']:6.3f} {rc['track']:6.3f} {v['held']:4.1f}")

    # Harness-consistency check on B
    print("\n" + "=" * 78)
    print("HARNESS-CONSISTENCY CHECK — shared GDN-2 control B across probes (§6.2)")
    print("=" * 78)
    bvals = []
    for p in PROBES:
        if p in b_consistency:
            cand, v = b_consistency[p]
            bvals.append(v["jcc"])
            print(f"  opt-{p:11s} B={cand:16s} JCC={v['jcc']:.3f} (SE {v['se']:.3f}, n={v['n']})")
    if bvals:
        bm = sum(bvals) / len(bvals)
        spread = max(bvals) - min(bvals)
        print(f"  --> B JCC mean {bm:.3f}, spread {spread:.3f} across probes")
        print(f"      (spread reflects per-probe counting-witness set + B-LR differences; "
              f"the ranking is by ΔJCC within each probe, which is ceiling-invariant)")

    if args.write:
        path = os.path.join(HERE, "opt_ceilings.json")
        data = json.load(open(path))
        data["opt_synth_reconciled"] = {
            "note": "opt-synth reconciled ceilings (§6.1): per-corner MAX over the four probes' "
                    "specialist ceilings (hardest denominator). ΔJCC is ceiling-invariant so "
                    "per-lever verdicts are robust; this set is for the unified leaderboard.",
            "S_c": ceil,
            "provenance": prov,
            "tau": TAU,
            "probe_ceilings": PROBE_CEILINGS,
        }
        json.dump(data, open(path, "w"), indent=2)
        print(f"\nWrote reconciled block to {path}")


if __name__ == "__main__":
    main()
