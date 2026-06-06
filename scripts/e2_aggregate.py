#!/usr/bin/env python3
"""E2 aggregator: seed-mean acc-by-precision tables + H1 monotonicity verdict.

Reads experiments/expressivity_tasks/results/e2_precision_sweep_20260604/
  e88linear_train{fp32,bf16}_seed{42,123,456}.json
and emits:
  * summary.json  (per-cell seed-mean/SD + per-seed raw + verdict)
  * a Markdown fragment on stdout (pasted into paper/review/E2_PRECISION_SWEEP.md)

Monotonicity test (per train_dtype, the eval-only / diagonal reading):
  H1 SUPPORTED  if acc(bf16) > acc(fp32) > acc(fp64)  (rounding helps)
  H1 DISFAVORED if acc(fp64) >= acc(fp32) >= acc(bf16) (precision helps/flat)
evaluated length-by-length, with a seed-mean delta and a sign tally.
"""
import os, sys, json, glob
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'experiments/expressivity_tasks/results/e2_precision_sweep_20260604')
SEEDS = [42, 123, 456]
TRAIN_DTYPES = ['fp32', 'bf16']
EVAL_DTYPES = ['fp64', 'fp32', 'bf16']
LENGTHS = [128, 256, 512, 1024]


def load(train_dtype, seed):
    p = os.path.join(OUT, f'e88linear_train{train_dtype}_seed{seed}.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def cell(train_dtype, eval_dtype, T):
    vals = []
    for s in SEEDS:
        d = load(train_dtype, s)
        if d is None:
            continue
        ps = d.get('precision_sweep', {}).get(eval_dtype, {}).get(str(T))
        if ps and 'acc' in ps:
            vals.append(ps['acc'])
    return vals


def main():
    summary = {'experiment': 'E2_precision_sweep', 'arm': 'e88-linear',
               'seeds': SEEDS, 'train_dtypes_present': [], 'cells': {}, 'verdict': {}}

    lines = []
    for td in TRAIN_DTYPES:
        present = [s for s in SEEDS if load(td, s) is not None]
        if not present:
            continue
        summary['train_dtypes_present'].append(td)
        summary['cells'][td] = {}
        lines.append(f"\n### train_dtype = {td}  (seeds present: {present})\n")
        lines.append("| eval dtype | T=128 | T=256 | T=512 | T=1024 |")
        lines.append("|---|---:|---:|---:|---:|")
        for ed in EVAL_DTYPES:
            summary['cells'][td][ed] = {}
            row = [f"| {ed} "]
            for T in LENGTHS:
                vals = cell(td, ed, T)
                if vals:
                    m, sd = float(np.mean(vals)), float(np.std(vals))
                    summary['cells'][td][ed][str(T)] = {
                        'mean': m, 'sd': sd, 'n': len(vals), 'raw': vals}
                    row.append(f"| {m:.4f}±{sd:.4f} ")
                else:
                    row.append("| — ")
            lines.append(''.join(row) + "|")

        # Monotonicity verdict per length (seed-mean)
        lines.append("")
        lines.append("Monotonicity vs eval precision (seed-mean acc; "
                     "bf16>fp32>fp64 ⇒ H1-support, fp64≥fp32≥bf16 ⇒ H1-disfavor):")
        lines.append("")
        lines.append("| T | fp64 | fp32 | bf16 | bf16−fp64 | reading |")
        lines.append("|---|---:|---:|---:|---:|---|")
        verdict_signs = []
        for T in LENGTHS:
            mv = {}
            for ed in EVAL_DTYPES:
                vals = cell(td, ed, T)
                mv[ed] = float(np.mean(vals)) if vals else float('nan')
            d64, d32, db16 = mv['fp64'], mv['fp32'], mv['bf16']
            delta = db16 - d64
            # classify
            if db16 > d32 > d64:
                reading = 'rounding helps (H1)'
                verdict_signs.append(1)
            elif d64 >= d32 >= db16:
                reading = 'precision helps/flat (¬H1)'
                verdict_signs.append(-1)
            else:
                reading = 'non-monotonic'
                verdict_signs.append(0)
            lines.append(f"| {T} | {d64:.4f} | {d32:.4f} | {db16:.4f} | "
                         f"{delta:+.4f} | {reading} |")
        summary['verdict'][td] = {
            'signs_by_length': dict(zip([str(t) for t in LENGTHS], verdict_signs)),
            'n_support_H1': sum(1 for s in verdict_signs if s == 1),
            'n_disfavor_H1': sum(1 for s in verdict_signs if s == -1),
            'n_nonmonotonic': sum(1 for s in verdict_signs if s == 0),
            'mean_bf16_minus_fp64': float(np.nanmean([
                np.mean(cell(td, 'bf16', T) or [np.nan]) -
                np.mean(cell(td, 'fp64', T) or [np.nan]) for T in LENGTHS])),
        }

    os.makedirs(OUT, exist_ok=True)
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=2)
    print('\n'.join(lines))
    print("\n=== VERDICT ===")
    for td, v in summary['verdict'].items():
        print(f"train={td}: H1-support@{v['n_support_H1']}/4 lengths, "
              f"H1-disfavor@{v['n_disfavor_H1']}/4, nonmono@{v['n_nonmonotonic']}/4, "
              f"mean(bf16−fp64)={v['mean_bf16_minus_fp64']:+.4f}")
    print(f"\nWrote {os.path.join(OUT, 'summary.json')}")


if __name__ == '__main__':
    main()
