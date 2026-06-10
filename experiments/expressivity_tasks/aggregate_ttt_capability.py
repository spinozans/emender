"""Aggregate the TTT-write (`refit`) capability battery (task `ttt-capability`).

Reads results_ttt_capability/ttt_*.json (written by train_hybrid.py) and prints:
  (1) Final eval ACC per (task, arm), mean+-std over seeds, with the two key gaps:
        refit-mom - refit-del   = the MOMENTUM / richer-inner-opt capability gap
                                  (matched A/B: same layer, momentum toggled)
        refit-mom - gdn2        = the TTT cell vs the production GDN-2 baseline
  (2) Final eval LOSS per (task, arm)  -> the convergent-loss-null check
      (do the arms converge to the same task loss at matched compute?).
  (3) Length-extrapolation acc curves per arm (train T=128, eval 128/256/512).

No mocks: reads the real run JSONs.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent

ARMS = ['refit-mom', 'refit-del', 'gdn2']
LENGTHS = ['128', '256', '512']


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float('nan'), float('nan'), 0
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s, len(xs)


def parse_label(fname: str):
    # ttt_<task>[_K<k>]__<arm>__seed<seed>.json
    stem = fname[:-5] if fname.endswith('.json') else fname
    assert stem.startswith('ttt_'), stem
    body = stem[4:]
    left, arm, seedpart = body.rsplit('__', 2)
    seed = int(seedpart.replace('seed', ''))
    if '_K' in left:
        task, k = left.rsplit('_K', 1)
        return task, int(k), arm, seed
    return left, 0, arm, seed


def load(out_dir: Path):
    runs = []
    for p in sorted(out_dir.glob('ttt_*.json')):
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  [warn] could not read {p.name}: {e}")
            continue
        d['_file'] = p.name
        runs.append(d)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=str(THIS / 'results_ttt_capability'))
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    runs = load(out_dir)
    print(f"Loaded {len(runs)} run JSONs from {out_dir}\n")

    by = defaultdict(lambda: defaultdict(list))  # (task,K) -> arm -> [run]
    for d in runs:
        task, K, arm, seed = parse_label(d['_file'])
        by[(task, K)][arm].append(d)

    bar = "=" * 104
    print(bar)
    print("(1) FINAL EVAL ACC (train T=128), mean +/- std over seeds")
    print("    refit-mom = momentum inner-opt | refit-del = momentum OFF (=gated-delta) | gdn2 = FLA prod")
    print(bar)
    hdr = (f"{'task':<24}{'base':>7} | " + " | ".join(f"{a:>17}" for a in ARMS)
           + " ||  mom-del   mom-gdn2")
    print(hdr)
    print("-" * len(hdr))
    acc_summary = {}
    for (task, K) in sorted(by.keys()):
        arms = by[(task, K)]
        base = arms[next(iter(arms))][0].get('random_baseline_acc', float('nan'))
        cells = {a: mean_std([r.get('final_acc') for r in arms.get(a, [])]) for a in ARMS}
        tlabel = f"{task}" + (f" K={K}" if K else "")
        row = f"{tlabel:<24}{base:>7.3f} | "
        row += " | ".join(f"{cells[a][0]:>6.3f}+-{cells[a][1]:<5.3f}({cells[a][2]})" for a in ARMS)
        g_md = cells['refit-mom'][0] - cells['refit-del'][0]
        g_mg = cells['refit-mom'][0] - cells['gdn2'][0]
        row += f" || {g_md:+.3f}  {g_mg:+.3f}"
        print(row)
        acc_summary[tlabel] = {a: cells[a] for a in ARMS} | {
            'baseline': base, 'gap_mom_minus_del': g_md, 'gap_mom_minus_gdn2': g_mg}

    print("\n" + bar)
    print("(2) FINAL EVAL LOSS (train T=128) — the CONVERGENT-LOSS-NULL check")
    print("    arms converging to the same task loss at matched compute => loss-null")
    print(bar)
    hdr2 = (f"{'task':<24} | " + " | ".join(f"{a:>17}" for a in ARMS)
            + " ||  del-mom   gdn2-mom")
    print(hdr2)
    print("-" * len(hdr2))
    for (task, K) in sorted(by.keys()):
        arms = by[(task, K)]
        cells = {a: mean_std([r.get('final_loss') for r in arms.get(a, [])]) for a in ARMS}
        tlabel = f"{task}" + (f" K={K}" if K else "")
        row = f"{tlabel:<24} | "
        row += " | ".join(f"{cells[a][0]:>7.4f}+-{cells[a][1]:<5.3f}" for a in ARMS)
        # loss gaps relative to refit-mom (positive => that arm has HIGHER/worse loss)
        d_del = cells['refit-del'][0] - cells['refit-mom'][0]
        d_gdn = cells['gdn2'][0] - cells['refit-mom'][0]
        row += f" || {d_del:+.4f}  {d_gdn:+.4f}"
        print(row)

    print("\n" + bar)
    print("(3) LENGTH-EXTRAPOLATION acc per arm (mean over seeds); train T=128")
    print(bar)
    for (task, K) in sorted(by.keys()):
        arms = by[(task, K)]
        tlabel = f"{task}" + (f" K={K}" if K else "")
        base = arms[next(iter(arms))][0].get('random_baseline_acc', float('nan'))
        print(f"\n  {tlabel}  (baseline {base:.3f})")
        print(f"    {'arm':<12}" + "".join(f"T={L:<8}" for L in LENGTHS))
        for a in ARMS:
            rs = arms.get(a, [])
            cols = []
            for L in LENGTHS:
                accs = [r.get('length_extrap', {}).get(L, {}).get('acc') for r in rs]
                m, s, n = mean_std(accs)
                cols.append(f"{m:.3f}" if n else "  -  ")
            print(f"    {a:<12}" + "".join(f"{c:<10}" for c in cols))

    # machine-readable summary
    out = out_dir / '_summary.json'
    json.dump({f"{t}": {a: list(v[a]) for a in ARMS} for t, v in
               {f"{k[0]}" + (f"_K{k[1]}" if k[1] else ""):
                {a: mean_std([r.get('final_acc') for r in by[k].get(a, [])]) for a in ARMS}
                for k in by}.items()},
              open(out, 'w'), indent=2)
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
