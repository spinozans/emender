"""Aggregate PROBE 1 counting-with-comparison results into a per-arm x length
accuracy table (mean +/- std over seeds), plus a machine-readable summary.

Usage:
    python aggregate_probe1.py --task dyck_depth
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent

ARM_ORDER = ['e88-linear', 'e88-tanh', 'gdn', 'm2rnn', 'relu_rnn', 'lstm']
ARM_KIND = {
    'e88-linear': 'LINEAR (eigenvalue-rich)',
    'e88-tanh':   'nonlinear-SATURATING (tanh)',
    'gdn':        'LINEAR (gated delta-net)',
    'm2rnn':      'matrix-memory RNN',
    'relu_rnn':   'ADDITIVE counter (ReLU-Elman) [WGY+]',
    'lstm':       'ADDITIVE counter (LSTM) [WGY+]',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='dyck_depth')
    ap.add_argument('--results_dir', default=str(THIS / 'results'))
    ap.add_argument('--lengths', type=int, nargs='+', default=[128, 256, 512, 1024])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    # arm -> length -> list[acc]
    acc = defaultdict(lambda: defaultdict(list))
    baseline = None
    params = {}
    seeds_seen = defaultdict(set)
    for f in sorted(rdir.glob(f'probe1_{args.task}__*.json')):
        d = json.load(open(f))
        # label: probe1_<task>__<arm>__seed<seed>
        stem = f.stem[len(f'probe1_{args.task}__'):]
        arm, seed = stem.rsplit('__seed', 1)
        baseline = d.get('random_baseline_acc', baseline)
        params[arm] = d.get('params')
        seeds_seen[arm].add(seed)
        le = d.get('length_extrap', {})
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
    header = f'| {"arm":<11} | {"kind":<34} | {"params":>8} | ' + \
        ' | '.join(f'T={L:<8}' for L in args.lengths) + ' |'
    sep = '|' + '-' * 13 + '|' + '-' * 36 + '|' + '-' * 10 + '|' + \
        '|'.join('-' * 12 for _ in args.lengths) + '|'
    lines.append(header)
    lines.append(sep)
    summary = {'task': args.task, 'baseline': baseline, 'arms': {}}
    for arm in ARM_ORDER:
        if arm not in acc:
            continue
        p = params.get(arm)
        pstr = f'{p/1e6:.2f}M' if p else '?'
        row = f'| {arm:<11} | {ARM_KIND[arm]:<34} | {pstr:>8} | ' + \
            ' | '.join(f'{fmt(acc[arm][L]):<10}' for L in args.lengths) + ' |'
        lines.append(row)
        summary['arms'][arm] = {
            'kind': ARM_KIND[arm], 'params': p,
            'seeds': sorted(seeds_seen[arm]),
            'acc': {str(L): {'mean': statistics.mean(acc[arm][L]) if acc[arm][L] else None,
                             'std': statistics.pstdev(acc[arm][L]) if len(acc[arm][L]) > 1 else 0.0,
                             'vals': acc[arm][L]}
                    for L in args.lengths},
        }
    text = '\n'.join(lines)
    print(text)
    out = args.out or str(rdir / f'probe1_{args.task}_summary.json')
    json.dump(summary, open(out, 'w'), indent=2)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
