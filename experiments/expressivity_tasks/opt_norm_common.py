"""Shared metric helpers for the opt-norm probe (OPT_SPEC.md §1.3, §3).

Joint Capability Coverage (JCC): worst-corner held-ratio over FROZEN per-corner
specialist ceilings, at convergent loss. Used by both compute_opt_ceilings.py
(freezes S_c) and aggregate_opt_norm.py (scores every arm + emits JCC_ROWS.jsonl).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

# corner -> witness task keys (the filename task token, ktag included). The corner
# acc is the mean over its witnesses; counting averages three, step-growth two.
CORNERS = {
    'recall':      ['mqar_recall'],
    'counting':    ['modular_counter_K5', 'dyck_depth_unbounded', 'anbncn_viability'],
    'step_growth': ['modular_quadratic_K64', 'iterated_nonlinear_map'],
    'track':       ['s5_permutation'],
}
SCORED = ['recall', 'counting', 'step_growth', 'track']
SANITY = ['parity']
TAU = 0.95  # pre-registered held threshold (OPT_SPEC.md §1.3)

# frozen specialist that defines each corner ceiling S_c (OPT_SPEC.md §1.3):
# recall/track -> GDN-2 (the best-LR fla-gdn control); counting/step-growth -> the
# all-refit-del counting specialist.
CEILING_SPECIALIST = {
    'recall': 'B_gdn', 'track': 'B_gdn',
    'counting': 'spec_refit', 'step_growth': 'spec_refit',
}


def parse_label(fname: str):
    """optnorm_<task[_K<k>]>__<arm>__seed<seed>.json -> (task_key, arm, seed)."""
    stem = fname[:-5] if fname.endswith('.json') else fname
    assert stem.startswith('optnorm_'), stem
    body = stem[len('optnorm_'):]
    task_key, arm, seedpart = body.rsplit('__', 2)
    seed = int(seedpart.replace('seed', ''))
    return task_key, arm, seed


def load_runs(out_dir: Path):
    runs = []
    for p in sorted(out_dir.glob('optnorm_*.json')):
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  [warn] could not read {p.name}: {e}")
            continue
        d['_file'] = p.name
        runs.append(d)
    return runs


def conv_certificate(run):
    """Relative loss improvement over the FINAL 20% of training (OPT_SPEC.md §1.5):
    (L_80% - L_final)/L_80%. Reported for transparency. NOTE: on exact-algorithm
    tasks the eval loss decays toward 0, so this RELATIVE measure stays large even
    after the capability (accuracy) has saturated — it is a loss-to-zero artifact,
    not genuine non-convergence. The operational plateau gate is is_converged()
    (accuracy plateau), which honors §1.5's "compare plateaus, not progress" intent."""
    steps = run.get('steps', [])
    losses = [(s['step'], s.get('eval_loss')) for s in steps if s.get('eval_loss') is not None]
    if len(losses) < 3:
        return None
    total = losses[-1][0]
    cut = 0.8 * total
    l80 = next((l for (st, l) in losses if st >= cut), losses[-1][1])
    lf = losses[-1][1]
    if l80 is None or l80 == 0:
        return None
    return (l80 - lf) / abs(l80)


def acc_plateau_delta(run):
    """Absolute eval-accuracy change over the FINAL 20% of training. A run has
    PLATEAUED (stopped improving) iff this is small — the robust convergence signal
    for exact-algorithm tasks whose loss heads to 0 (where conv_certificate is
    artificially large). A run still climbing in the last 20% (e.g. acc 0.5->0.99)
    has a large delta and is correctly flagged non-converged / needs longer."""
    steps = run.get('steps', [])
    accs = [(s['step'], s.get('eval_acc')) for s in steps if s.get('eval_acc') is not None]
    if len(accs) < 3:
        return None
    total = accs[-1][0]
    cut = 0.8 * total
    a80 = next((a for (st, a) in accs if st >= cut), accs[-1][1])
    af = accs[-1][1]
    if a80 is None or af is None:
        return None
    return abs(af - a80)


def is_converged(run, acc_tol=0.02):
    """Converged iff the eval accuracy has plateaued over the final 20% of steps
    (|Δacc| < acc_tol). Robust to the loss-to-zero artifact in conv_certificate;
    a run still climbing is flagged for a longer-budget re-run (OPT_SPEC.md §1.5)."""
    d = acc_plateau_delta(run)
    if d is None:
        return None
    return d < acc_tol


def task_acc(run):
    """Mean length-extrapolation acc over the run's eval lengths (the algorithm-vs-
    memorization signal; OPT_SPEC.md §3.3)."""
    le = run.get('length_extrap', {})
    accs = [v.get('acc') for v in le.values() if isinstance(v, dict) and v.get('acc') is not None]
    if not accs:
        fa = run.get('final_acc')
        return fa
    return sum(accs) / len(accs)


def per_length_acc(run):
    le = run.get('length_extrap', {})
    return {L: v.get('acc') for L, v in le.items()
            if isinstance(v, dict) and v.get('acc') is not None}


def corner_acc_for_seed(runs_by_task, corner, seed):
    """Mean task_acc over a corner's witness tasks for one seed. None if no witness
    present for that seed."""
    vals = []
    for tk in CORNERS[corner]:
        for r in runs_by_task.get(tk, []):
            _, _, s = parse_label(r['_file'])
            if s == seed:
                a = task_acc(r)
                if a is not None:
                    vals.append(a)
    if not vals:
        return None
    return sum(vals) / len(vals)


def index_by_arm_task(runs):
    """arm -> task_key -> [run]."""
    idx = defaultdict(lambda: defaultdict(list))
    for r in runs:
        tk, arm, _ = parse_label(r['_file'])
        idx[arm][tk].append(r)
    return idx


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s, len(xs)
