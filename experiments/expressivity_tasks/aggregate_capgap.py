"""Aggregate the capability-gap battery (task capability-gap-research).

Reads results_capgap/cg_<task>__<arm>__d<depth>[__K<N>]__seed<seed>.json and
prints, per (task, scale K), the mean accuracy at each eval length for each arm,
plus the LINEAR-vs-NONLINEAR gap (best nonlinear-in-time arm minus gdn-neg) and a
cliff metric (acc drop from train length T=128 to the longest eval length).

Handles BOTH rounds:
  Round 1 : 4-segment names   cg_<task>__<arm>__d<depth>__seed<seed>
  Round 2 : 5-segment names   cg_<task>__<arm>__d<depth>__K<N>__seed<seed>
and the Round-2 `count` arm (non-saturating relu-state counter, the
Weiss-Goldberg-Yahav-predicted counting winner).
"""
import json
import glob
from collections import defaultdict

import numpy as np

LENS = [128, 256, 512, 1024, 2048]
# nonlinear-in-time arms whose best is compared against the linear gdn-neg arm.
# `count` is non-saturating (relu-state) -> it is a *different* kind of nonlinearity
# than tanh-e97; we report it but track the tanh arms separately for the verdict.
ARMS = ['gdnneg', 'nlshell', 'e97delta', 'count']
ARM_LABEL = {
    'gdnneg':   'gdn-neg (LINEAR)      ',
    'nlshell':  'nlshell (NONLIN-clean)',
    'e97delta': 'e97delta (tanh-NONLIN)',
    'count':    'count (relu-NONLIN)   ',
}
TASK_ORDER = [
    'dyck_depth', 'dyck_depth_unbounded', 'modular_quadratic', 'monoid_track',  # GAP candidates
    's5_permutation', 'modular_quadratic_lin', 'iterated_nonlinear_map',         # controls
]
GAP = {'dyck_depth', 'dyck_depth_unbounded', 'modular_quadratic', 'monoid_track'}


def parse_name(name):
    """cg_<task>__<arm>__d<depth>[__K<N>]__seed<seed> -> (task, arm, K, seed)."""
    body = name[3:]  # strip 'cg_'
    segs = body.split('__')
    # segs: [task, arm, d<depth>, (K<N>)?, seed<seed>]
    task, arm = segs[0], segs[1]
    K = 0
    for s in segs[2:]:
        if s.startswith('K'):
            K = int(s[1:])
    return task, arm, K


def load(out_dir='experiments/expressivity_tasks/results_capgap'):
    # (task, K) -> arm -> L -> [acc]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rb = {}
    nseed = defaultdict(lambda: defaultdict(set))  # (task,K)->arm->{seeds}
    for f in sorted(glob.glob(f'{out_dir}/cg_*.json')):
        d = json.load(open(f))
        name = f.split('/')[-1][:-5]
        task, arm, K = parse_name(name)
        key = (task, K)
        rb[key] = d['random_baseline_acc']
        nseed[key][arm].add(d['seed'])
        for L, vv in d['length_extrap'].items():
            data[key][arm][int(L)].append(vv['acc'])
    return data, rb, nseed


def main():
    data, rb, nseed = load()
    print("CAPABILITY-GAP BATTERY  (MLP present in ALL arms, depth=4, 3 seeds)\n")
    # order: for each task, default (K=0) first then ascending K
    keys = sorted(data.keys(), key=lambda k: (TASK_ORDER.index(k[0]) if k[0] in TASK_ORDER else 99, k[1]))
    for task, K in keys:
        key = (task, K)
        tag = 'GAP-CANDIDATE' if task in GAP else 'control'
        ktag = f"  (scale={K})" if K else ""
        nsd = max((len(v) for v in nseed[key].values()), default=0)
        print(f"### {task}{ktag}   [{tag}]   random={rb[key]:.3f}   seeds={nsd}")
        hdr = f"  {'arm':24s}" + "".join(f"T={L:<7d}" for L in LENS)
        print(hdr)
        arm_means = {}
        for arm in ARMS:
            if arm not in data[key]:
                continue
            row = f"  {ARM_LABEL[arm]:24s}"
            means = {}
            for L in LENS:
                a = data[key][arm].get(L, [])
                if a:
                    means[L] = float(np.mean(a))
                    row += f"{np.mean(a):.3f}   "
                else:
                    row += "  -     "
            arm_means[arm] = means
            print(row)
        # gap: best nonlinear-in-time arm minus gdn-neg, at longest common length.
        # report BOTH the tanh-specific gap (the E97 cell) and the best-of-any-nonlinear gap.
        if 'gdnneg' in arm_means:
            gd = arm_means['gdnneg']
            tanh_arms = [a for a in ('nlshell', 'e97delta') if a in arm_means]
            any_nl = [a for a in ('nlshell', 'e97delta', 'count') if a in arm_means]
            common = [L for L in LENS if L in gd and all(L in arm_means[a] for a in any_nl)]
            if common and any_nl:
                Lmax = max(common)
                best_tanh = max((arm_means[a][Lmax] for a in tanh_arms), default=float('nan'))
                best_any = max(arm_means[a][Lmax] for a in any_nl)
                gap_tanh = best_tanh - gd[Lmax]
                gap_any = best_any - gd[Lmax]
                cliff = gd[min(common)] - gd[Lmax]

                def verdict(g):
                    return 'SEPARATION' if g > 0.05 else ('tie' if abs(g) <= 0.05 else 'gdn-neg wins')
                print(f"  -> @T={Lmax}: gdn-neg={gd[Lmax]:.3f}  "
                      f"best-tanh={best_tanh:.3f} GAP_tanh={gap_tanh:+.3f}[{verdict(gap_tanh)}]  "
                      f"best-any-NL={best_any:.3f} GAP_any={gap_any:+.3f}[{verdict(gap_any)}]  "
                      f"gdn-cliff(T{min(common)}->T{Lmax})={cliff:+.3f}")
        print()


if __name__ == '__main__':
    main()
