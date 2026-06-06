"""Aggregate the unified-cell LEARNABILITY sweep.

Reads:
  - learn_*.json   (this study: spread arms klr{1,5,10,20} on 5 probes + mixed
                    baselines), produced by run_unified_learnability.py
  - unified_*.json (prior unified-cell sweep: standalone-probe presets, generic
                    learned-free, lstm), reused for the baseline columns.

Writes the experimental tables for paper/review/UNIFIED_LEARNABILITY_RESULTS.md:
  1. Per-probe accuracy @ T=128 / T=1024: spread arms vs generic learned-free vs
     best preset vs LSTM. (Does spread+knob-LR close the learnability gap?)
  2. Per-head (lambda,beta,gamma) DRIFT: init (spread) vs final. Do heads HOLD
     their corner or drift back to the generic center? Headline of the study.
  3. Which knob-LR multiplier worked.

Run: python experiments/expressivity_tasks/aggregate_unified_learnability.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / 'results'

SEEDS = [42, 123, 456]
PROBES = ['s5_permutation', 'anbncn_viability', 'iterated_nonlinear_map',
          'flag_hold_recall', 'mixed_probe']
PROBE_CORNER = {'s5_permutation': 'track', 'anbncn_viability': 'count',
                'iterated_nonlinear_map': 'nonlin', 'flag_hold_recall': 'latch',
                'mixed_probe': 'ALL'}
SPREAD_ARMS = ['unified-learned-spread-klr1', 'unified-learned-spread-klr5',
               'unified-learned-spread-klr10', 'unified-learned-spread-klr20']
PRESET_ARMS = ['unified-track', 'unified-count', 'unified-latch', 'unified-nonlin']

# Corner centroids in (lambda, beta, gamma) space (matches spread-init + generic).
CORNERS = {
    'track':  (0.9, 1.8, 0.05),
    'count':  (1.0, 0.0, 0.05),
    'latch':  (1.3, 0.0, 0.95),
    'nonlin': (0.9, 0.5, 0.95),
    'center': (0.95, 0.5, 0.5),   # the generic-init compromise regime
}
# Per-axis scale for nearest-centroid classification (normalize the 3 axes).
_SCALE = (0.4, 1.8, 1.0)


def load_learn(probe, arm, seed):
    p = RESULTS / f'learn_{probe}__{arm}__seed{seed}.json'
    return json.loads(p.read_text()) if p.exists() else None


def load_prior(probe, arm, seed):
    p = RESULTS / f'unified_{probe}__{arm}__seed{seed}.json'
    return json.loads(p.read_text()) if p.exists() else None


def acc_at(log, T):
    if log is None:
        return None
    le = log.get('length_extrap', {})
    if T in le and 'acc' in le[T]:
        return le[T]['acc']
    if T == '128':
        return log.get('final_acc')
    return None


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def g(x, nd=3):
    return f'{x:.{nd}f}' if x is not None else '--'


def fmt(m, s):
    return f'{m:.3f}±{s:.2f}' if m is not None else '  --  '


def nearest_corner(lam, beta, gam):
    best, bestd = None, 1e9
    for name, (cl, cb, cg) in CORNERS.items():
        d = ((lam - cl) / _SCALE[0]) ** 2 + ((beta - cb) / _SCALE[1]) ** 2 + ((gam - cg) / _SCALE[2]) ** 2
        if d < bestd:
            bestd, best = d, name
    return best


# ---------------------------------------------------------------------------
# Table 1: accuracy — does spread+knob-LR close the gap?
# ---------------------------------------------------------------------------
def best_preset_score(probe, T):
    """Best pinned-preset score on a probe at T (prior sweep for the 4 standalone
    probes; this sweep's mixed baselines for mixed_probe)."""
    loader = load_learn if probe == 'mixed_probe' else load_prior
    scores = {}
    for arm in PRESET_ARMS:
        m, _ = mean_std([acc_at(loader(probe, arm, s), T) for s in SEEDS])
        if m is not None:
            scores[arm] = m
    if not scores:
        return None, None
    best = max(scores, key=scores.get)
    return best.replace('unified-', ''), scores[best]


def generic_learned_score(probe, T):
    loader = load_learn if probe == 'mixed_probe' else load_prior
    m, _ = mean_std([acc_at(loader(probe, 'unified-learned-free', s), T) for s in SEEDS])
    return m


def lstm_score(probe, T):
    loader = load_learn if probe == 'mixed_probe' else load_prior
    m, _ = mean_std([acc_at(loader(probe, 'lstm', s), T) for s in SEEDS])
    return m


def accuracy_table(T):
    lines = [f'### Accuracy @ T={T}  (mean±std over seeds {SEEDS}; train T=128)', '']
    cols = ['probe (corner)', 'best preset', 'generic learned', 'LSTM',
            'spread klr1', 'spread klr5', 'spread klr10', 'spread klr20']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    for probe in PROBES:
        bp_name, bp = best_preset_score(probe, T)
        gen = generic_learned_score(probe, T)
        lstm = lstm_score(probe, T)
        cells = [f'{probe} ({PROBE_CORNER[probe]})',
                 f'{bp_name} {g(bp)}' if bp is not None else '--',
                 g(gen), g(lstm)]
        for arm in SPREAD_ARMS:
            m, s = mean_std([acc_at(load_learn(probe, arm, sd), T) for sd in SEEDS])
            cells.append(fmt(m, s))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def gap_closure_table(T):
    """Does the BEST spread arm match the best preset (within 0.05) where generic
    learned-free FAILED?"""
    lines = [f'### Gap closure @ T={T}: does spread+knob-LR match the best preset?', '']
    lines.append('| probe (corner) | best preset | generic learned (matches?) | '
                 'best spread arm (matches?) |')
    lines.append('|---|---|---|---|')
    for probe in PROBES:
        bp_name, bp = best_preset_score(probe, T)
        if bp is None:
            continue
        gen = generic_learned_score(probe, T) or 0.0
        gen_ok = 'YES' if gen >= bp - 0.05 else 'no'
        best_arm, best_val = None, -1
        for arm in SPREAD_ARMS:
            m, _ = mean_std([acc_at(load_learn(probe, arm, sd), T) for sd in SEEDS])
            if m is not None and m > best_val:
                best_val, best_arm = m, arm
        spread_ok = 'YES' if best_val >= bp - 0.05 else 'no'
        arm_tag = best_arm.replace('unified-learned-', '') if best_arm else '--'
        lines.append(f'| {probe} ({PROBE_CORNER[probe]}) | {bp_name} {g(bp)} | '
                     f'{g(gen)} ({gen_ok}) | {arm_tag} {g(best_val) if best_val>=0 else "--"} ({spread_ok}) |')
    lines.append('')
    return lines


# ---------------------------------------------------------------------------
# Table 2: knob DRIFT — do heads hold their corner or drift to center?
# ---------------------------------------------------------------------------
def pooled_knobs(probe, arm, which):
    """Pool per-head (lambda,beta,gamma) across layers+seeds. which: 'init'|'final'."""
    key = 'unified_knobs_init' if which == 'init' else 'unified_knobs'
    lam, beta, gam = [], [], []
    for seed in SEEDS:
        log = load_learn(probe, arm, seed)
        if not log or key not in log:
            continue
        for layer in log[key]:
            lam += layer['lambda']; beta += layer['beta']; gam += layer['gamma']
    return lam, beta, gam


def drift_table(arm):
    lines = [f'### Knob drift — arm `{arm}`  (per-head, pooled over layers+seeds)', '',
             'Each head starts on one of the four corners (round-robin). After '
             'training we classify each head to its NEAREST corner centroid '
             '(track/count/latch/nonlin/center) in normalized (lambda,beta,gamma) '
             'space. "held" = nearest corner is still a real corner (not center); '
             '"to center" = drifted to the generic compromise regime.', '']
    lines.append('| probe | n heads | mean lambda | mean beta | mean gamma | '
                 '%eig<0 | %held corner | %to center | mean |dlambda| | mean |dbeta| | mean |dgamma| |')
    lines.append('|' + '---|' * 11)
    for probe in PROBES:
        li, bi, gi = pooled_knobs(probe, arm, 'init')
        lf, bf, gf = pooled_knobs(probe, arm, 'final')
        if not lf:
            continue
        n = len(lf)
        eig = [l - b for l, b in zip(lf, bf)]
        cls = [nearest_corner(l, b, gm) for l, b, gm in zip(lf, bf, gf)]
        held = sum(1 for c in cls if c != 'center')
        to_center = sum(1 for c in cls if c == 'center')
        # drift magnitudes (init and final are aligned head-for-head when both present)
        if li and len(li) == n:
            dlam = sum(abs(a - b) for a, b in zip(lf, li)) / n
            dbeta = sum(abs(a - b) for a, b in zip(bf, bi)) / n
            dgam = sum(abs(a - b) for a, b in zip(gf, gi)) / n
        else:
            dlam = dbeta = dgam = None
        lines.append(
            f'| {probe} | {n} | {sum(lf)/n:.3f} | {sum(bf)/n:.3f} | {sum(gf)/n:.3f} | '
            f'{100*sum(1 for e in eig if e<0)/n:.0f}% | {100*held/n:.0f}% | {100*to_center/n:.0f}% | '
            f'{g(dlam)} | {g(dbeta)} | {g(dgam)} |')
    lines.append('')
    return lines


def corner_occupancy(arm, probe):
    """How heads distribute across corners at final, per probe (for the best arm)."""
    lf, bf, gf = pooled_knobs(probe, arm, 'final')
    if not lf:
        return None
    counts = defaultdict(int)
    for l, b, gm in zip(lf, bf, gf):
        counts[nearest_corner(l, b, gm)] += 1
    n = len(lf)
    return {k: (counts.get(k, 0), 100 * counts.get(k, 0) / n) for k in CORNERS}


def occupancy_table(arm):
    lines = [f'### Final corner occupancy — arm `{arm}` (nearest-centroid head counts)', '']
    lines.append('| probe | track | count | latch | nonlin | center |')
    lines.append('|---|---|---|---|---|---|')
    for probe in PROBES:
        occ = corner_occupancy(arm, probe)
        if occ is None:
            continue
        cells = [probe]
        for c in ['track', 'count', 'latch', 'nonlin', 'center']:
            n, pct = occ[c]
            cells.append(f'{n} ({pct:.0f}%)')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def count_done():
    n = 0
    for probe in PROBES:
        for arm in SPREAD_ARMS:
            for s in SEEDS:
                if load_learn(probe, arm, s):
                    n += 1
    return n


def main():
    out = ['<!-- AUTO-GENERATED (aggregate_unified_learnability.py). -->', '']
    out.append(f'_Spread-arm runs found: {count_done()} / {len(PROBES)*len(SPREAD_ARMS)*len(SEEDS)}._\n')
    out += ['## 1. Accuracy — does spread-init + knob-LR close the learnability gap?', '']
    out += accuracy_table('128')
    out += accuracy_table('1024')
    out += gap_closure_table('128')
    out += gap_closure_table('1024')
    out += ['## 2. Knob drift — do heads HOLD their corner or drift to center?', '']
    for arm in SPREAD_ARMS:
        out += drift_table(arm)
    out += ['## 3. Final corner occupancy (best-separating arms)', '']
    for arm in SPREAD_ARMS:
        out += occupancy_table(arm)
    text = '\n'.join(out)
    dst = THIS / 'UNIFIED_LEARNABILITY_EXPERIMENTAL.md'
    dst.write_text(text)
    print(text)
    print(f'\n[written] {dst}')


if __name__ == '__main__':
    main()
