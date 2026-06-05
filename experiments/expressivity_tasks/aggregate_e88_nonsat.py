"""Aggregate E88 NON-SATURATING results (task: e88-nonsat) into per-arm x length
accuracy tables (mean +/- std over seeds), plus a machine-readable summary.

For the COUNTING tasks (anbncn_viability, dyck_depth) the shared arms
(e88-tanh, e88-linear, lstm) are read from the committed probe1-counting JSONs
(`probe1_<task>__<arm>__seed*.json`), and the NEW non-saturating arms
(e88-relu, e88-softplus) from this task's `nonsat_<task>__<arm>__seed*.json`.
For S5/S3 every arm comes from `nonsat_<task>__*.json`.

Usage:
    python aggregate_e88_nonsat.py --task anbncn_viability
    python aggregate_e88_nonsat.py --task s5_permutation
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent

# Arms compared in this study, in display order.
ARM_ORDER = ['e88-linear', 'e88-tanh', 'e88-relu', 'e88-softplus', 'lstm']
ARM_KIND = {
    'e88-linear':   'E88 linear / affine state',
    'e88-tanh':     'E88 SATURATING nonlinear (tanh)',
    'e88-relu':     'E88 NON-SATURATING (relu)',
    'e88-softplus': 'E88 NON-SATURATING (softplus)',
    'lstm':         'LSTM additive counter [WGY+]',
}
# Counting arms that should be sourced from the committed probe1 JSONs when no
# nonsat_ JSON exists (avoids recomputing the shared baselines).
PROBE1_FALLBACK = {'e88-linear', 'e88-tanh', 'lstm'}


def _load_for_arm(rdir, task, arm):
    """Return list of (seed, length_extrap_dict, params, baseline) for an arm,
    preferring nonsat_ JSONs and falling back to probe1_ JSONs."""
    out = []
    pats = [f'nonsat_{task}__{arm}__seed*.json']
    if arm in PROBE1_FALLBACK:
        pats.append(f'probe1_{task}__{arm}__seed*.json')
    seen_seeds = set()
    for pat in pats:
        for f in sorted(rdir.glob(pat)):
            stem = f.stem
            seed = stem.rsplit('__seed', 1)[1]
            if seed in seen_seeds:
                continue  # nonsat_ takes precedence over probe1_ for same seed
            seen_seeds.add(seed)
            d = json.load(open(f))
            out.append((seed, d.get('length_extrap', {}), d.get('params'),
                        d.get('random_baseline_acc'), str(f.name)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='anbncn_viability')
    ap.add_argument('--results_dir', default=str(THIS / 'results'))
    ap.add_argument('--lengths', type=int, nargs='+', default=[128, 256, 512, 1024])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    acc = defaultdict(lambda: defaultdict(list))
    baseline = None
    params = {}
    seeds_seen = defaultdict(set)
    sources = defaultdict(list)
    for arm in ARM_ORDER:
        for seed, le, p, base, fname in _load_for_arm(rdir, args.task, arm):
            if base is not None:
                baseline = base
            if p is not None:
                params[arm] = p
            seeds_seen[arm].add(seed)
            sources[arm].append(fname)
            for L in args.lengths:
                e = le.get(str(L))
                if e and 'acc' in e:
                    acc[arm][L].append(e['acc'])

    def fmt(vals):
        if not vals:
            return '   --   '
        m = statistics.mean(vals)
        s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return f'{m:.3f}±{s:.3f}'

    lines = []
    lines.append(f'## {args.task} — accuracy at eval length (mean±std over seeds)')
    lines.append(f'random baseline = {baseline}')
    lines.append('')
    header = f'| {"arm":<13} | {"kind":<34} | {"params":>8} | ' + \
        ' | '.join(f'T={L:<8}' for L in args.lengths) + ' |'
    sep = '|' + '-' * 15 + '|' + '-' * 36 + '|' + '-' * 10 + '|' + \
        '|'.join('-' * 12 for _ in args.lengths) + '|'
    lines.append(header)
    lines.append(sep)
    summary = {'task': args.task, 'baseline': baseline, 'arms': {}}
    for arm in ARM_ORDER:
        if arm not in acc:
            continue
        p = params.get(arm)
        pstr = f'{p/1e6:.2f}M' if p else '?'
        row = f'| {arm:<13} | {ARM_KIND[arm]:<34} | {pstr:>8} | ' + \
            ' | '.join(f'{fmt(acc[arm][L]):<10}' for L in args.lengths) + ' |'
        lines.append(row)
        summary['arms'][arm] = {
            'kind': ARM_KIND[arm], 'params': p,
            'seeds': sorted(seeds_seen[arm]),
            'sources': sorted(sources[arm]),
            'acc': {str(L): {'mean': statistics.mean(acc[arm][L]) if acc[arm][L] else None,
                             'std': statistics.pstdev(acc[arm][L]) if len(acc[arm][L]) > 1 else 0.0,
                             'vals': acc[arm][L]}
                    for L in args.lengths},
        }
    text = '\n'.join(lines)
    print(text)
    out = args.out or str(rdir / f'nonsat_{args.task}_summary.json')
    json.dump(summary, open(out, 'w'), indent=2)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
