"""Aggregate the E98 FIVE-CORNER sweep: is a 5-type head population stronger than
4, and is the leaky-linear associative-memory corner the recall-relevant one?

Reads results/e98fc_<probe>__<arm>__seed<seed>.json produced by
run_e98_five_corner.py and writes the [auto] tables for E98_FIVE_CORNER_RESULTS.md:

  1. RECALL — leaky preset / spread-5 / GDN vs spread-4 and the exotic presets on
     MQAR (does the workhorse solve recall where the exotic corners fail?).
  2. 4-vs-5 HEAD-TO-HEAD — spread-5 − spread-4 on every probe + mixed-5 (is 5
     stronger, and does adding the workhorse HURT any exotic corner? = regression).
  3. PER-HEAD (lambda,beta,gamma) corner occupancy for spread-5 (does the
     leaky-linear group hold its corner; what mixture).

Run: python experiments/expressivity_tasks/aggregate_e98_five_corner.py
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
                'mqar_recall': 'leaky (recall)', 'mixed_probe': 'ALL-5'}
EXOTIC_PRESETS = ['track', 'count', 'latch', 'nonlin']

# 5 corners + the collapse center, for the nearest-centroid head classifier.
CORNERS = {
    'track':  (0.9, 1.8, 0.05),
    'count':  (1.0, 0.0, 0.05),
    'latch':  (1.3, 0.0, 0.95),
    'nonlin': (0.9, 0.5, 0.95),
    'leaky':  (0.9, 0.1, 0.05),
    'center': (0.95, 0.5, 0.5),
}
_SCALE = (0.4, 1.8, 1.0)


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def load(probe, arm, seed):
    return _load(f'e98fc_{probe}__{arm}__seed{seed}.json')


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
        for arm in ('spread-5', 'spread-4', 'leaky', 'gdn'):
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


def pooled_knobs(probe, arm, which):
    key = 'unified_knobs_init' if which == 'init' else 'unified_knobs'
    lam, beta, gam = [], [], []
    for seed in SEEDS:
        log = load(probe, arm, seed)
        if not log or key not in log:
            continue
        for layer in log[key]:
            lam += layer['lambda']; beta += layer['beta']; gam += layer['gamma']
    return lam, beta, gam


# --- Table 1: recall -------------------------------------------------------
def recall_table():
    lines = ['### 1. RECALL (MQAR) — does the leaky-linear workhorse solve it where the exotic corners fail?', '']
    base = baseline_of('mqar_recall')
    bstr = f'{base:.4f}' if base is not None else '--'
    lines.append(f'Random baseline ≈ {bstr} (1/vocab). MQAR pairs scale with length '
                 f'(T//16): 8 pairs @128 → 64 @1024.')
    lines.append('')
    arms = ['leaky', 'spread-5', 'gdn', 'spread-4'] + EXOTIC_PRESETS
    cols = ['arm'] + [f'@{T}' for T in EVAL_T]
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    for arm in arms:
        cells = [arm]
        for T in EVAL_T:
            cells.append(fmt(*arm_ms('mqar_recall', arm, T)))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


# --- Table 2: 4-vs-5 head-to-head -----------------------------------------
def headtohead_table(T):
    lines = [f'### 2. 4-vs-5 HEAD-TO-HEAD @ T={T} — spread-5 − spread-4 (+ = 5-type wins; − on an exotic probe = regression)', '']
    cols = ['probe (corner)', 'baseline', 'spread-4', 'spread-5', 'Δ(5−4)', 'verdict']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    win = lose = tie = 0
    for probe in PROBES:
        s4m, s4s = arm_ms(probe, 'spread-4', T)
        s5m, s5s = arm_ms(probe, 'spread-5', T)
        d = (s5m - s4m) if (s4m is not None and s5m is not None) else None
        v = '--'
        if d is not None:
            if d > 0.01:
                v = '5 stronger'; win += 1
            elif d < -0.01:
                v = 'REGRESSION'; lose += 1
            else:
                v = 'tie'; tie += 1
        base = baseline_of(probe)
        bstr = f'{base:.3f}' if base is not None else '--'
        lines.append(f'| {probe} ({PROBE_CORNER[probe]}) | {bstr} | {fmt(s4m, s4s)} | '
                     f'{fmt(s5m, s5s)} | {g(d)} | {v} |')
    lines.append('')
    lines.append(f'**@T={T}: spread-5 wins {win} / regresses {lose} / ties {tie} of {len(PROBES)} probes.**')
    lines.append('')
    return lines


# --- Table 3: per-head occupancy for spread-5 -----------------------------
def occupancy_table(arm):
    lines = [f'### 3. PER-HEAD corner occupancy — {arm} (nearest-centroid over 5 corners, pooled layers+seeds)', '']
    cols = ['probe', 'n heads', 'track', 'count', 'latch', 'nonlin', 'leaky', 'center', 'covered']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    ncorners = 5 if arm == 'spread-5' else 4
    corner_names = ['track', 'count', 'latch', 'nonlin'] + (['leaky'] if ncorners == 5 else [])
    for probe in PROBES:
        lf, bf, gf = pooled_knobs(probe, arm, 'final')
        if not lf:
            continue
        counts = defaultdict(int)
        for l, b, gm in zip(lf, bf, gf):
            counts[nearest_corner(l, b, gm)] += 1
        n = len(lf)
        covered = sum(1 for c in corner_names if counts.get(c, 0) >= 0.05 * n)
        cells = [probe, str(n)]
        for c in ['track', 'count', 'latch', 'nonlin', 'leaky', 'center']:
            cells.append(f"{counts.get(c,0)} ({100*counts.get(c,0)/n:.0f}%)")
        cells.append(f'{covered}/{ncorners}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def leaky_group_table():
    """For spread-5: mean (lambda,beta,gamma,eig) of the heads that land in the
    leaky corner on the recall probe — does the placed leaky group hold?"""
    lines = ['### 3b. spread-5 leaky-group knobs on MQAR (heads classified to the leaky corner)', '']
    lines.append('| stat | n | mean λ | mean β | mean γ | mean eig(λ−β) |')
    lines.append('|---|---|---|---|---|---|')
    for arm in ('spread-5',):
        for probe in ('mqar_recall', 'mixed_probe'):
            lam, beta, gam = [], [], []
            for seed in SEEDS:
                log = load(probe, arm, seed)
                if not log or 'unified_knobs' not in log:
                    continue
                for layer in log['unified_knobs']:
                    for l, b, gm in zip(layer['lambda'], layer['beta'], layer['gamma']):
                        if nearest_corner(l, b, gm) == 'leaky':
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
    out.append('## [auto] E98 five-corner aggregated tables')
    out.append('')
    out.append(f'Params (≈): spread-4 {params_of("s5_permutation","spread-4")}, '
               f'spread-5 {params_of("s5_permutation","spread-5")}, '
               f'leaky {params_of("mqar_recall","leaky")}, '
               f'gdn {params_of("mqar_recall","gdn")}.')
    out.append('')
    out += recall_table()
    for T in ('128', '1024'):
        out += headtohead_table(T)
    out += occupancy_table('spread-5')
    out += occupancy_table('spread-4')
    out += leaky_group_table()
    text = '\n'.join(out)
    print(text)
    (THIS / 'E98_FIVE_CORNER_AUTO.md').write_text(text)
    print(f'\n[wrote] {THIS / "E98_FIVE_CORNER_AUTO.md"}')


if __name__ == '__main__':
    main()
