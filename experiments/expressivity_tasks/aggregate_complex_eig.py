"""Aggregate the complex-eigenvalue capability battery (task complex-eig-capability).

Reads results_complex_eig/*.json and prints:
  (1) PERIODIC battery: per (task, arm) mean+-std of final_acc and per-length
      extrapolation acc; the complex-vs-real gap (and vs gdnneg anchor).
  (2) COEXISTENCE: modular_quadratic length-extrapolation curve per head config
      — does the hardtanh step-growth capability persist WITH complex transitions?

No mocks: reads the real run JSONs written by train_hybrid.py.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent

BATTERY_ARMS = ['complex', 'real', 'gdnneg']
COEXIST_CFGS = ['cplx_htanh', 'cplx_lin', 'real_htanh']
COEXIST_LENGTHS = ['128', '256', '512', '1024', '2048']
BATTERY_LENGTHS = ['128', '256', '512']


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float('nan'), float('nan'), 0
    m = sum(xs) / len(xs)
    if len(xs) > 1:
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        s = math.sqrt(v)
    else:
        s = 0.0
    return m, s, len(xs)


def load(out_dir: Path):
    runs = []
    for p in sorted(out_dir.glob('ce_*.json')):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        d['_file'] = p.name
        runs.append(d)
    return runs


def parse_label(fname: str):
    # ce_<task>[_K<k>]__<arm>__seed<seed>.json
    stem = fname[:-5] if fname.endswith('.json') else fname
    assert stem.startswith('ce_')
    body = stem[3:]
    left, arm, seedpart = body.split('__')
    seed = int(seedpart.replace('seed', ''))
    if '_K' in left:
        task, k = left.rsplit('_K', 1)
        K = int(k)
    else:
        task, K = left, 0
    return task, K, arm, seed


def battery_report(runs):
    # key: (task, K) -> arm -> list of run dicts
    by = defaultdict(lambda: defaultdict(list))
    coex_arms = set(COEXIST_CFGS)
    for d in runs:
        task, K, arm, seed = parse_label(d['_file'])
        if arm in coex_arms:
            continue
        by[(task, K)][arm].append(d)

    print("=" * 100)
    print("(1) PERIODIC BATTERY — final eval acc (train T=128), mean +/- std over seeds")
    print("    complex = rotation enabled | real = theta frozen to {0,pi} (matched) | gdnneg = prod real +/- anchor")
    print("=" * 100)
    hdr = f"{'task':<26}{'baseline':>9} | " + " | ".join(f"{a:>16}" for a in BATTERY_ARMS) + " || complex-real"
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for (task, K) in sorted(by.keys()):
        arms = by[(task, K)]
        base = arms[next(iter(arms))][0].get('random_baseline_acc', float('nan'))
        cells = {}
        for a in BATTERY_ARMS:
            accs = [r.get('final_acc') for r in arms.get(a, [])]
            cells[a] = mean_std(accs)
        tlabel = f"{task}" + (f" K={K}" if K else "")
        row = f"{tlabel:<26}{base:>9.3f} | "
        row += " | ".join(f"{cells[a][0]:>6.3f}+-{cells[a][1]:<5.3f}({cells[a][2]})" for a in BATTERY_ARMS)
        gap = cells['complex'][0] - cells['real'][0]
        row += f" || {gap:+.3f}"
        print(row)
        summary[tlabel] = {a: cells[a] for a in BATTERY_ARMS} | {'baseline': base, 'gap_complex_real': gap}

    # length extrapolation per task
    print("\n" + "=" * 100)
    print("    LENGTH-EXTRAPOLATION acc per arm (mean over seeds); train T=128")
    print("=" * 100)
    for (task, K) in sorted(by.keys()):
        arms = by[(task, K)]
        tlabel = f"{task}" + (f" K={K}" if K else "")
        print(f"\n  {tlabel}  (baseline {arms[next(iter(arms))][0].get('random_baseline_acc',float('nan')):.3f})")
        print(f"    {'arm':<10}" + "".join(f"T={L:<8}" for L in BATTERY_LENGTHS))
        for a in BATTERY_ARMS:
            rs = arms.get(a, [])
            cols = []
            for L in BATTERY_LENGTHS:
                accs = []
                for r in rs:
                    le = r.get('length_extrap', {}).get(L, {})
                    if 'acc' in le:
                        accs.append(le['acc'])
                m, s, n = mean_std(accs)
                cols.append(f"{m:.3f}" if n else "  -  ")
            print(f"    {a:<10}" + "".join(f"{c:<10}" for c in cols))
    return summary


def coexist_report(runs):
    # group by modulus K (p=7 easy control vs p=64 step-growth cliff)
    by = defaultdict(lambda: defaultdict(list))
    for d in runs:
        task, K, arm, seed = parse_label(d['_file'])
        if task != 'modular_quadratic' or arm not in COEXIST_CFGS:
            continue
        by[K][arm].append(d)
    if not by:
        return {}
    print("\n" + "=" * 100)
    print("(2) COEXISTENCE — modular_quadratic length-extrapolation (step-growth capability)")
    print("    cplx_htanh = complex + hardtanh subset (rotation + step-growth)")
    print("    cplx_lin   = complex, NO bounded subset (rotation only -> extrap should COLLAPSE at the cliff)")
    print("    real_htanh = real-only + hardtanh subset (step-growth, NO rotation)")
    print("    Axes COMPOSE iff cplx_htanh extrapolates ~ real_htanh; step-growth is REAL iff >> cplx_lin at the cliff.")
    print("=" * 100)
    summary = {}
    for K in sorted(by.keys()):
        arms = by[K]
        p = K if K > 0 else 7
        base = arms[next(iter(arms))][0].get('random_baseline_acc', float('nan'))
        regime = "CLIFF" if p >= 32 else "easy control"
        print(f"\n  modular_quadratic p={p}  ({regime}); baseline acc = {base:.3f}")
        print(f"  {'config':<14}" + "".join(f"T={L:<9}" for L in COEXIST_LENGTHS))
        ksum = {}
        for cfg in COEXIST_CFGS:
            rs = arms.get(cfg, [])
            cols = []
            rec = {}
            for L in COEXIST_LENGTHS:
                accs = []
                for r in rs:
                    le = r.get('length_extrap', {}).get(L, {})
                    if 'acc' in le:
                        accs.append(le['acc'])
                m, s, n = mean_std(accs)
                cols.append(f"{m:.3f}" if n else "  -  ")
                rec[L] = (m, s, n)
            print(f"  {cfg:<14}" + "".join(f"{c:<11}" for c in cols))
            ksum[cfg] = rec
        summary[f"p={p}"] = ksum
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=str(THIS / 'results_complex_eig'))
    ap.add_argument('--json_out', default=None)
    args = ap.parse_args()
    runs = load(Path(args.output_dir))
    print(f"loaded {len(runs)} runs from {args.output_dir}\n")
    bs = battery_report(runs)
    cs = coexist_report(runs)
    if args.json_out:
        json.dump({'battery': {k: {a: v[a][:2] if isinstance(v[a], tuple) else v[a]
                                    for a in v} for k, v in bs.items()},
                   'coexist': cs}, open(args.json_out, 'w'), indent=2, default=str)
        print(f"\nwrote {args.json_out}")


if __name__ == '__main__':
    main()
