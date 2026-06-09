"""Aggregate the phi-exploration sweep (task phi-explore).

Reads results_phi/phi_<task>__<phi>[__K<N>]__seed<seed>.json and reports, per
(task, scale K), the mean accuracy at each eval length for each phi, the
length-extrapolation cliff (acc@T128 - acc@Tmax), and the capability GAP vs the
phi='identity' linear baseline (best nonlinear phi - identity, at the longest
eval length). Prints a capability-vs-phi ranking on the primary modular_quadratic
cliff.

Usage:
    python experiments/expressivity_tasks/aggregate_phi.py            # human table
    python experiments/expressivity_tasks/aggregate_phi.py --json     # machine summary
"""
import argparse
import json
import glob
from collections import defaultdict

import numpy as np

LENS = [128, 256, 512, 1024, 2048]

# phi menu in report order, grouped by cost signature.
PHI_ORDER = [
    'identity', 'tanh', 'softsign', 'hardtanh', 'poly3',
    'relu', 'softplus', 'gelu', 'silu', 'signed_sqrt', 'learned',
]
PHI_GROUP = {
    'identity': 'LINEAR baseline',
    'tanh': 'bounded/saturating', 'softsign': 'bounded/saturating',
    'hardtanh': 'bounded/saturating', 'poly3': 'bounded/saturating',
    'relu': 'rectifying/unbounded', 'softplus': 'rectifying/unbounded',
    'gelu': 'smooth-gated', 'silu': 'smooth-gated',
    'signed_sqrt': 'compressive-unbounded',
    'learned': 'learned',
}

TASK_ORDER = ['modular_quadratic', 'dyck_depth_unbounded', 'iterated_nonlinear_map']


def parse_name(name):
    """phi_<task>__<phi>[__K<N>]__seed<seed> -> (task, phi, K)."""
    body = name[4:]  # strip 'phi_'
    segs = body.split('__')
    task, phi = segs[0], segs[1]
    K = 0
    for s in segs[2:]:
        if s.startswith('K'):
            K = int(s[1:])
    return task, phi, K


def load(out_dir='experiments/expressivity_tasks/results_phi'):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # (task,K)->phi->L->[acc]
    rb = {}
    nseed = defaultdict(lambda: defaultdict(set))
    for f in sorted(glob.glob(f'{out_dir}/phi_*.json')):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        name = f.split('/')[-1][:-5]
        task, phi, K = parse_name(name)
        key = (task, K)
        rb[key] = d.get('random_baseline_acc', float('nan'))
        nseed[key][phi].add(d.get('seed'))
        for L, vv in d.get('length_extrap', {}).items():
            if 'acc' in vv:
                data[key][phi][int(L)].append(vv['acc'])
    return data, rb, nseed


def verdict(g):
    return 'SEPARATION' if g > 0.05 else ('tie' if abs(g) <= 0.05 else 'WORSE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='experiments/expressivity_tasks/results_phi')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    data, rb, nseed = load(args.out_dir)

    summary = {}
    keys = sorted(data.keys(), key=lambda k: (TASK_ORDER.index(k[0]) if k[0] in TASK_ORDER else 99, k[1]))

    if not args.json:
        print("PHI-EXPLORATION SWEEP  (MLP in ALL arms, depth=4, per-step phi, length-extrap)\n")
    for task, K in keys:
        key = (task, K)
        ktag = f"  (mod={K})" if K else ""
        nsd = max((len(v) for v in nseed[key].values()), default=0)
        phi_means = {}
        for phi in PHI_ORDER:
            if phi not in data[key]:
                continue
            means = {}
            for L in LENS:
                a = data[key][phi].get(L, [])
                if a:
                    means[L] = float(np.mean(a))
            phi_means[phi] = means

        base = phi_means.get('identity', {})
        common = [L for L in LENS if L in base and all(L in phi_means[p] for p in phi_means)]
        Lmax = max(common) if common else None
        Lmin = min(common) if common else None

        rows = []
        for phi in PHI_ORDER:
            if phi not in phi_means:
                continue
            m = phi_means[phi]
            gap = (m[Lmax] - base[Lmax]) if (Lmax and 'identity' in phi_means) else float('nan')
            cliff = (m[Lmin] - m[Lmax]) if (Lmax and Lmin) else float('nan')
            rows.append((phi, m, gap, cliff))

        summary[f"{task}__K{K}"] = {
            'random_baseline': rb[key], 'seeds': nsd, 'Lmax': Lmax,
            'phi': {phi: {'means': m, 'gap_vs_identity': gap, 'cliff': cliff}
                    for phi, m, gap, cliff in rows},
        }

        if args.json:
            continue
        print(f"### {task}{ktag}   random={rb[key]:.3f}   seeds={nsd}   "
              f"(GAP/cliff @ T={Lmax})")
        hdr = f"  {'phi':14s}{'group':24s}" + "".join(f"T={L:<6d}" for L in LENS) + " GAP_vs_id  cliff"
        print(hdr)
        # rank within task by acc@Lmax (length-robustness), identity pinned context
        for phi, m, gap, cliff in sorted(rows, key=lambda r: -(r[1].get(Lmax, -1) if Lmax else -1)):
            row = f"  {phi:14s}{PHI_GROUP.get(phi,''):24s}"
            for L in LENS:
                row += f"{m[L]:.3f} " if L in m else "  -   "
            tag = verdict(gap) if phi != 'identity' else 'baseline'
            row += f" {gap:+.3f}[{tag}]" if phi != 'identity' else "  (baseline)"
            row += f"  {cliff:+.3f}" if cliff == cliff else ""
            print(row)
        print()

    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
