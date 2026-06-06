"""Aggregate the E98 SIXTH-CORNER sweep: does adding the GATED-DELTA backbone give
the population real recall AND tracking, and is the gated-delta PRESET regime
(GDN-in-the-E98-cell) reachable — i.e. does it close the recall gap to GDN?

Reads results/e98sc_<probe>__<arm>__seed<seed>.json from run_e98_sixth_corner.py
and writes the [auto] tables for E98_SIXTH_CORNER_RESULTS.md:

  1. RECALL (MQAR) — gated-delta PRESET vs GDN bar vs leaky PRESET (5-corner corner)
     vs spread-6/5/4. THE KEY TEST: does the gated-delta preset approach GDN 0.95?
  2. S5 (track) — gated-delta PRESET (neg-eig) vs GDN+neg-eig bar vs spread arms.
     Does gated-delta ALSO track -> the dual recall+track backbone?
  3. 4-vs-5-vs-6 HEAD-TO-HEAD — spread-6 − spread-5 and spread-6 − spread-4 on every
     probe + mixed (does the gated-delta corner give the population real recall, and
     does adding it regress any exotic corner?).
  4. PER-HEAD (lambda,beta,gamma) corner occupancy for spread-6 (does the gated-delta
     group hold + carry recall?).

Run: python experiments/expressivity_tasks/aggregate_e98_sixth_corner.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / 'results'

SEEDS = [42, 123, 456]
EVAL_T = ['128', '256', '512', '1024']
PROBES = ['s5_permutation', 'anbncn_viability', 'iterated_nonlinear_map',
          'flag_hold_recall', 'mqar_recall', 'mixed_probe']
PROBE_CORNER = {'s5_permutation': 'track', 'anbncn_viability': 'count',
                'iterated_nonlinear_map': 'nonlin', 'flag_hold_recall': 'latch',
                'mqar_recall': 'recall', 'mixed_probe': 'ALL'}

# 6 placed corners + the collapse center, for the nearest-centroid head classifier.
CORNERS = {
    'track':       (0.9, 1.8, 0.05),
    'count':       (1.0, 0.0, 0.05),
    'latch':       (1.3, 0.0, 0.95),
    'nonlin':      (0.9, 0.5, 0.95),
    'leaky':       (0.9, 0.1, 0.05),
    'gated-delta': (0.99, 1.0, 0.05),
    'center':      (0.95, 0.5, 0.5),
}
CORNER_ORDER = ['track', 'count', 'latch', 'nonlin', 'leaky', 'gated-delta', 'center']
_SCALE = (0.4, 1.8, 1.0)


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def load(probe, arm, seed):
    return _load(f'e98sc_{probe}__{arm}__seed{seed}.json')


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


def arm_ms(probe, arm, T):
    return mean_std([acc_at(load(probe, arm, s), T) for s in SEEDS])


def fmt(m, s):
    return f'{m:.3f}±{s:.2f}' if m is not None else '  --  '


def g(x, nd=3):
    return f'{x:+.{nd}f}' if x is not None else '--'


def baseline_of(probe):
    for s in SEEDS:
        for arm in ('spread-6', 'spread-5', 'spread-4', 'gated-delta', 'gdn', 'gdn-neg'):
            log = load(probe, arm, s)
            if log and 'random_baseline_acc' in log:
                return log['random_baseline_acc']
    return None


def params_of(probe, arm):
    for s in SEEDS:
        log = load(probe, arm, s)
        if log and 'params' in log:
            return log['params']
    return None


def nearest_corner(lam, beta, gam):
    best, bestd = None, 1e9
    for name, (cl, cb, cg) in CORNERS.items():
        d = ((lam - cl) / _SCALE[0]) ** 2 + ((beta - cb) / _SCALE[1]) ** 2 + ((gam - cg) / _SCALE[2]) ** 2
        if d < bestd:
            bestd, best = d, name
    return best


# --- Table 1: recall -------------------------------------------------------
def recall_table():
    lines = ['### 1. RECALL (MQAR) — does the gated-delta PRESET close the gap to GDN where leaky could not?', '']
    base = baseline_of('mqar_recall')
    bstr = f'{base:.4f}' if base is not None else '--'
    lines.append(f'Random baseline ≈ {bstr} (1/vocab). MQAR pairs scale with length. '
                 f'GDN is the workhorse bar; leaky was the 5-corner recall corner (the WORST).')
    lines.append('')
    arms = ['gdn', 'gated-delta', 'leaky', 'spread-6', 'spread-5', 'spread-4']
    cols = ['arm'] + [f'@{T}' for T in EVAL_T] + ['params']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    for arm in arms:
        cells = [arm]
        for T in EVAL_T:
            cells.append(fmt(*arm_ms('mqar_recall', arm, T)))
        p = params_of('mqar_recall', arm)
        cells.append(f'{p/1e6:.2f}M' if p else '--')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    # explicit gap line
    gd, _ = arm_ms('mqar_recall', 'gated-delta', '128')
    gdn, _ = arm_ms('mqar_recall', 'gdn', '128')
    lk, _ = arm_ms('mqar_recall', 'leaky', '128')
    if gd is not None and gdn is not None:
        lines.append(f'**@128: gated-delta preset {gd:.3f} vs GDN {gdn:.3f} '
                     f'(gap {gd-gdn:+.3f}); leaky preset {lk:.3f}. '
                     f'gated-delta − leaky = {gd-lk:+.3f}.**')
        lines.append('')
    return lines


# --- Table 2: S5 track (dual capability) -----------------------------------
def s5_table():
    lines = ['### 2. S5 (track) — does gated-delta ALSO track (neg-eig)? -> the dual recall+track backbone', '']
    base = baseline_of('s5_permutation')
    bstr = f'{base:.4f}' if base is not None else '--'
    lines.append(f'Random baseline ≈ {bstr}. GDN+neg-eig is the length-robust track bar.')
    lines.append('')
    arms = ['gdn-neg', 'gated-delta', 'spread-6', 'spread-5', 'spread-4']
    cols = ['arm'] + [f'@{T}' for T in EVAL_T]
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    for arm in arms:
        cells = [arm]
        for T in EVAL_T:
            cells.append(fmt(*arm_ms('s5_permutation', arm, T)))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


# --- Table 3: 4-vs-5-vs-6 head-to-head -------------------------------------
def headtohead_table(T):
    lines = [f'### 3. 4-vs-5-vs-6 HEAD-TO-HEAD @ T={T} — spread-6 vs spread-5 vs spread-4 '
             f'(+Δ on recall = real gain; −Δ on an exotic probe = regression)', '']
    cols = ['probe (corner)', 'baseline', 'spread-4', 'spread-5', 'spread-6',
            'Δ(6−5)', 'Δ(6−4)', 'verdict']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    win = lose = tie = 0
    for probe in PROBES:
        s4m, s4s = arm_ms(probe, 'spread-4', T)
        s5m, s5s = arm_ms(probe, 'spread-5', T)
        s6m, s6s = arm_ms(probe, 'spread-6', T)
        d65 = (s6m - s5m) if (s5m is not None and s6m is not None) else None
        d64 = (s6m - s4m) if (s4m is not None and s6m is not None) else None
        v = '--'
        if d65 is not None:
            if d65 > 0.01:
                v = '6 stronger'; win += 1
            elif d65 < -0.01:
                v = 'REGRESSION vs5'; lose += 1
            else:
                v = 'tie'; tie += 1
        base = baseline_of(probe)
        bstr = f'{base:.3f}' if base is not None else '--'
        lines.append(f'| {probe} ({PROBE_CORNER[probe]}) | {bstr} | {fmt(s4m, s4s)} | '
                     f'{fmt(s5m, s5s)} | {fmt(s6m, s6s)} | {g(d65)} | {g(d64)} | {v} |')
    lines.append('')
    lines.append(f'**@T={T}: spread-6 beats spread-5 on {win} / regresses {lose} / ties {tie} '
                 f'of {len(PROBES)} probes.**')
    lines.append('')
    return lines


# --- Table 4: per-head occupancy for spread-6 ------------------------------
def occupancy_table(arm, ncorners):
    lines = [f'### 4. PER-HEAD corner occupancy — {arm} '
             f'(nearest-centroid over {ncorners} corners, pooled layers+seeds)', '']
    placed = CORNER_ORDER[:ncorners]
    cols = ['probe', 'n heads'] + placed + ['center', 'covered']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    for probe in PROBES:
        lf, bf, gf = [], [], []
        for seed in SEEDS:
            log = load(probe, arm, seed)
            if not log or 'unified_knobs' not in log:
                continue
            for layer in log['unified_knobs']:
                lf += layer['lambda']; bf += layer['beta']; gf += layer['gamma']
        if not lf:
            continue
        counts = defaultdict(int)
        for l, b, gm in zip(lf, bf, gf):
            counts[nearest_corner(l, b, gm)] += 1
        n = len(lf)
        covered = sum(1 for c in placed if counts.get(c, 0) >= 0.05 * n)
        cells = [probe, str(n)]
        for c in placed + ['center']:
            cells.append(f"{counts.get(c,0)} ({100*counts.get(c,0)/n:.0f}%)")
        cells.append(f'{covered}/{ncorners}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def gd_group_table():
    """For spread-6: mean (lambda,beta,gamma,eig) of the heads that land in the
    gated-delta corner on the recall/track/mixed probes — does the placed
    gated-delta group hold its clean-overwrite point (eig~0)?"""
    lines = ['### 4b. spread-6 gated-delta-group knobs (heads classified to the gated-delta corner)', '']
    lines.append('| probe | n | mean λ | mean β | mean γ | mean eig(λ−β) |')
    lines.append('|---|---|---|---|---|---|')
    for probe in ('mqar_recall', 's5_permutation', 'mixed_probe'):
        lam, beta, gam = [], [], []
        for seed in SEEDS:
            log = load(probe, 'spread-6', seed)
            if not log or 'unified_knobs' not in log:
                continue
            for layer in log['unified_knobs']:
                for l, b, gm in zip(layer['lambda'], layer['beta'], layer['gamma']):
                    if nearest_corner(l, b, gm) == 'gated-delta':
                        lam.append(l); beta.append(b); gam.append(gm)
        if not lam:
            lines.append(f'| {probe} | 0 | -- | -- | -- | -- |')
            continue
        n = len(lam)
        ml = sum(lam)/n; mb = sum(beta)/n; mg = sum(gam)/n
        lines.append(f'| {probe} | {n} | {ml:.2f} | {mb:.2f} | {mg:.2f} | {ml-mb:+.2f} |')
    lines.append('')
    return lines


def main():
    out = []
    out.append('## [auto] E98 sixth-corner aggregated tables')
    out.append('')
    out.append(f'Params (≈): spread-4 {params_of("s5_permutation","spread-4")}, '
               f'spread-5 {params_of("s5_permutation","spread-5")}, '
               f'spread-6 {params_of("s5_permutation","spread-6")}, '
               f'gated-delta {params_of("mqar_recall","gated-delta")}, '
               f'gdn {params_of("mqar_recall","gdn")}, '
               f'gdn-neg {params_of("s5_permutation","gdn-neg")}.')
    out.append('')
    out += recall_table()
    out += s5_table()
    for T in ('128', '1024'):
        out += headtohead_table(T)
    out += occupancy_table('spread-6', 6)
    out += occupancy_table('spread-5', 5)
    out += gd_group_table()
    text = '\n'.join(out)
    print(text)
    (THIS / 'E98_SIXTH_CORNER_AUTO.md').write_text(text)
    print(f'\n[wrote] {THIS / "E98_SIXTH_CORNER_AUTO.md"}')


if __name__ == '__main__':
    main()
