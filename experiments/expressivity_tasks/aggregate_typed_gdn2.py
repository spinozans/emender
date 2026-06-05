"""Aggregate typed-gdn-2-head CMA search + validation into report tables.

REAL JSONs only; no synthetic numbers. Emits markdown tables (stdout) and a
summary JSON for TYPED_GDN2_MIXTURE_CMA_RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
EVAL_TS = [128, 256, 512, 1024]
PROBE_LIST = ['mqar_recall', 's5_permutation', 'anbncn_viability',
              'flag_hold_recall', 'iterated_nonlinear_map', 'mixed_probe']
PROBE_SHORT = {
    'mqar_recall': 'recall(MQAR)', 's5_permutation': 's5/track',
    'anbncn_viability': 'count', 'flag_hold_recall': 'latch',
    'iterated_nonlinear_map': 'nonlin', 'mixed_probe': 'mixed',
}


def run_score(path: Path):
    """(mean-over-T acc, per-T dict) for one finished run; (None,{}) if absent."""
    if not path.exists():
        return None, {}
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None, {}
    le = d.get('length_extrap') or {}
    per_t, accs = {}, []
    for t in EVAL_TS:
        e = le.get(str(t))
        if isinstance(e, dict) and 'acc' in e:
            per_t[t] = float(e['acc'])
            accs.append(float(e['acc']))
    if not accs:
        return None, {}
    return float(np.mean(accs)), per_t


def arm_probe_score(out_dir, arm, probe, seeds, steps):
    """Mean over seeds of (mean-over-T acc) + averaged per-T."""
    ms, per_t_acc = [], {t: [] for t in EVAL_TS}
    for seed in seeds:
        lbl = f"tgdn2val_{arm}__{probe}__seed{seed}__s{steps}"
        m, per_t = run_score(out_dir / f'{lbl}.json')
        if m is not None:
            ms.append(m)
            for t, a in per_t.items():
                per_t_acc[t].append(a)
    if not ms:
        return None, {}
    return float(np.mean(ms)), {t: (float(np.mean(v)) if v else None)
                                for t, v in per_t_acc.items()}


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cma_dir', default=str(THIS / 'results' / 'typed_gdn2_cma'))
    ap.add_argument('--val_dir', default=str(THIS / 'results' / 'typed_gdn2_validate'))
    ap.add_argument('--arms', nargs='+',
                    default=['typed-gdn2-winner', 'e98-cma-winner', 'gdn-native-ref'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--out', default=str(THIS / 'results' / 'typed_gdn2_summary.json'))
    args = ap.parse_args()

    summary = {'cma': {}, 'validation': {}}
    cma_best = Path(args.cma_dir) / 'cma_best.json'
    print("## CMA winner\n")
    if cma_best.exists():
        b = json.load(open(cma_best))
        best = b.get('best', b)
        cfg = best.get('cfg', {})
        summary['cma'] = {
            'fitness': best.get('fitness'), 'mean': best.get('mean'),
            'min': best.get('min'), 'per_probe': best.get('per_probe'),
            'type_counts': cfg.get('type_counts'),
            'head_type_logits': cfg.get('head_type_logits'),
            'dim': cfg.get('dim'), 'actual_params': cfg.get('actual_params'),
            'lr': cfg.get('lr'),
        }
        print(f"- fitness={best.get('fitness')} (mean={best.get('mean')}, "
              f"min={best.get('min')})")
        print(f"- type_counts={cfg.get('type_counts')}")
        print(f"- head_type_logits={cfg.get('head_type_logits')}")
        print(f"- dim={cfg.get('dim')} params={cfg.get('actual_params')} "
              f"lr={cfg.get('lr')}")
        pp = best.get('per_probe') or {}
        if pp:
            print("\n| probe | search acc |")
            print("|---|---|")
            for p in PROBE_LIST:
                print(f"| {PROBE_SHORT[p]} | {fmt(pp.get(p))} |")
    else:
        print("(no CMA best yet)")

    print("\n## Validation (full-budget, multi-seed)\n")
    out_dir = Path(args.val_dir)
    header = "| probe | " + " | ".join(args.arms) + " |"
    print(header)
    print("|" + "---|" * (len(args.arms) + 1))
    arm_means = {a: [] for a in args.arms}
    val = {a: {} for a in args.arms}
    for p in PROBE_LIST:
        row = [PROBE_SHORT[p]]
        for a in args.arms:
            m, per_t = arm_probe_score(out_dir, a, p, args.seeds, args.steps)
            val[a][p] = {'mean': m, 'per_t': per_t}
            row.append(fmt(m))
            if m is not None:
                arm_means[a].append(m)
        print("| " + " | ".join(row) + " |")
    # worst-case + mean rows
    print("| **mean** | " + " | ".join(
        fmt(float(np.mean(arm_means[a])) if arm_means[a] else None) for a in args.arms) + " |")
    print("| **min (worst)** | " + " | ".join(
        fmt(float(np.min(arm_means[a])) if arm_means[a] else None) for a in args.arms) + " |")

    # per-T recall + s5 breakdown
    for probe in ('mqar_recall', 's5_permutation'):
        print(f"\n### {PROBE_SHORT[probe]} length-extrapolation (per-T acc)\n")
        print("| arm | " + " | ".join(f"T={t}" for t in EVAL_TS) + " |")
        print("|" + "---|" * (len(EVAL_TS) + 1))
        for a in args.arms:
            per_t = val[a].get(probe, {}).get('per_t', {})
            print(f"| {a} | " + " | ".join(fmt(per_t.get(t)) for t in EVAL_TS) + " |")

    summary['validation'] = val
    summary['arm_mean'] = {a: (float(np.mean(arm_means[a])) if arm_means[a] else None)
                           for a in args.arms}
    summary['arm_min'] = {a: (float(np.min(arm_means[a])) if arm_means[a] else None)
                          for a in args.arms}
    json.dump(summary, open(args.out, 'w'), indent=2)
    print(f"\n[summary -> {args.out}]")


if __name__ == '__main__':
    main()
