"""Aggregate the E98 = E97-split-gate-on-unified sweep and compare to the
E88-based unified cell (the prior, coupled erase=write recurrence).

Reads:
  - e98_*.json    (this study: split-gated arms produced by run_e98_on_e97.py)
  - learn_*.json  (E88-based winning form: unified-learned-spread-klr20; mixed
                   generic baseline unified-learned-free)
  - unified_*.json(E88-based standalone presets + standalone generic learned-free)
  - spec_*.json   (E88-based fixed-type-population: unified-fixedpop)

Writes paper/review tables for E98_ON_E97_RESULTS.md:
  1. Corner re-confirmation on the SPLIT-GATED cell: each e98 preset wins its
     corner (acc@128/@1024), next to the E88-based preset (did the split gate
     help/keep the corner?).
  2. Learnability re-confirmation: e98-learned-spread-klr20 on all five probes +
     knob drift + corner occupancy (does the split-gated cell still hold 4/4?).
  3. E97-vs-E88 head-to-head: spread/generic/fixedpop, split vs no-split, on the
     four corners + the mixed task. Does building on E97 IMPROVE the span?

Run: python experiments/expressivity_tasks/aggregate_e98_on_e97.py
"""
from __future__ import annotations

import json
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
CORNER_PRESET = {'s5_permutation': 'track', 'anbncn_viability': 'count',
                 'flag_hold_recall': 'latch', 'iterated_nonlinear_map': 'nonlin'}

CORNERS = {
    'track':  (0.9, 1.8, 0.05),
    'count':  (1.0, 0.0, 0.05),
    'latch':  (1.3, 0.0, 0.95),
    'nonlin': (0.9, 0.5, 0.95),
    'center': (0.95, 0.5, 0.5),
}
_SCALE = (0.4, 1.8, 1.0)


# --- loaders -------------------------------------------------------------
def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def load_e98(probe, arm, seed):
    return _load(f'e98_{probe}__{arm}__seed{seed}.json')


def load_e88_spread(probe, seed):
    return _load(f'learn_{probe}__unified-learned-spread-klr20__seed{seed}.json')


def load_e88_preset(probe, seed):
    """E88-based standalone preset on its own corner."""
    corner = CORNER_PRESET.get(probe)
    if corner is None:
        return None
    return _load(f'unified_{probe}__unified-{corner}__seed{seed}.json')


def load_e88_generic(probe, seed):
    if probe == 'mixed_probe':
        return _load(f'learn_mixed_probe__unified-learned-free__seed{seed}.json')
    return _load(f'unified_{probe}__unified-learned-free__seed{seed}.json')


def load_e88_fixedpop(probe, seed):
    return _load(f'spec_{probe}__unified-fixedpop__seed{seed}.json')


# --- helpers -------------------------------------------------------------
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


def params_of(loadfn, probe):
    for s in SEEDS:
        log = loadfn(probe, s)
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


def pooled_knobs(loadfn, probe, which):
    key = 'unified_knobs_init' if which == 'init' else 'unified_knobs'
    lam, beta, gam = [], [], []
    for seed in SEEDS:
        log = loadfn(probe, seed)
        if not log or key not in log:
            continue
        for layer in log[key]:
            lam += layer['lambda']; beta += layer['beta']; gam += layer['gamma']
    return lam, beta, gam


# --- Table 1: corner re-confirmation on the split-gated cell --------------
def corner_table(T):
    lines = [f'### Corner re-confirmation @ T={T} — does each split-gated preset win its corner?', '']
    cols = ['probe (corner)', 'E98 preset (split)', 'E88 preset', 'E98 spread', 'E88 spread']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    for probe in PROBES:
        corner = CORNER_PRESET.get(probe)
        if corner is None:
            continue
        e98p = mean_std([acc_at(load_e98(probe, f'e98-{corner}', s), T) for s in SEEDS])
        e88p = mean_std([acc_at(load_e88_preset(probe, s), T) for s in SEEDS])
        e98s = mean_std([acc_at(load_e98(probe, 'e98-learned-spread-klr20', s), T) for s in SEEDS])
        e88s = mean_std([acc_at(load_e88_spread(probe, s), T) for s in SEEDS])
        lines.append(f'| {probe} ({corner}) | {fmt(*e98p)} | {fmt(*e88p)} | '
                     f'{fmt(*e98s)} | {fmt(*e88s)} |')
    lines.append('')
    return lines


# --- Table 2: learnability re-confirmation (split-gated spread arm) -------
def learnability_table(T):
    lines = [f'### Learnability @ T={T} — E98 spread (split) on every probe', '']
    cols = ['probe (corner)', 'E98 spread (split)', 'E88 spread', 'E98 generic', 'E98 fixedpop']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    for probe in PROBES:
        e98s = mean_std([acc_at(load_e98(probe, 'e98-learned-spread-klr20', s), T) for s in SEEDS])
        e88s = mean_std([acc_at(load_e88_spread(probe, s), T) for s in SEEDS])
        gen = fixp = (None, None)
        if probe == 'mixed_probe':
            gen = mean_std([acc_at(load_e98(probe, 'e98-learned-free', s), T) for s in SEEDS])
            fixp = mean_std([acc_at(load_e98(probe, 'e98-fixedpop', s), T) for s in SEEDS])
        lines.append(f'| {probe} ({PROBE_CORNER[probe]}) | {fmt(*e98s)} | {fmt(*e88s)} | '
                     f'{fmt(*gen)} | {fmt(*fixp)} |')
    lines.append('')
    return lines


def occupancy_table(arm_label, loadfn):
    lines = [f'### Final corner occupancy — {arm_label} (nearest-centroid head counts, pooled layers+seeds)', '']
    lines.append('| probe | n heads | track | count | latch | nonlin | center | covered |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for probe in PROBES:
        lf, bf, gf = pooled_knobs(loadfn, probe, 'final')
        if not lf:
            continue
        counts = defaultdict(int)
        for l, b, gm in zip(lf, bf, gf):
            counts[nearest_corner(l, b, gm)] += 1
        n = len(lf)
        covered = sum(1 for c in ['track', 'count', 'latch', 'nonlin']
                      if counts.get(c, 0) >= 0.05 * n)
        cells = [probe, str(n)]
        for c in ['track', 'count', 'latch', 'nonlin', 'center']:
            cells.append(f"{counts.get(c,0)} ({100*counts.get(c,0)/n:.0f}%)")
        cells.append(f'{covered}/4')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


# --- Table 3: E97-vs-E88 head-to-head (delta) -----------------------------
def headtohead_table(T):
    lines = [f'### E97-split vs E88-coupled head-to-head @ T={T} (E98 − E88, +=split helps)', '']
    cols = ['probe (corner)', 'spread E98', 'spread E88', 'Δ spread',
            'preset E98', 'preset E88', 'Δ preset']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * len(cols))
    win = lose = tie = 0
    for probe in PROBES:
        e98s, _ = mean_std([acc_at(load_e98(probe, 'e98-learned-spread-klr20', s), T) for s in SEEDS])
        e88s, _ = mean_std([acc_at(load_e88_spread(probe, s), T) for s in SEEDS])
        ds = (e98s - e88s) if (e98s is not None and e88s is not None) else None
        if ds is not None:
            if ds > 0.01: win += 1
            elif ds < -0.01: lose += 1
            else: tie += 1
        corner = CORNER_PRESET.get(probe)
        if corner:
            e98p, _ = mean_std([acc_at(load_e98(probe, f'e98-{corner}', s), T) for s in SEEDS])
            e88p, _ = mean_std([acc_at(load_e88_preset(probe, s), T) for s in SEEDS])
            dp = (e98p - e88p) if (e98p is not None and e88p is not None) else None
        else:
            e98p = e88p = dp = None
        lines.append(f'| {probe} ({PROBE_CORNER[probe]}) | {g(e98s)} | {g(e88s)} | '
                     f'{("%+.3f"%ds) if ds is not None else "--"} | '
                     f'{g(e98p)} | {g(e88p)} | {("%+.3f"%dp) if dp is not None else "--"} |')
    lines.append('')
    lines.append(f'_Spread head-to-head @ T={T}: split-gate WINS {win}, LOSES {lose}, '
                 f'TIES {tie} (|Δ|>0.01)._')
    lines.append('')
    return lines


def params_table():
    lines = ['### Parameter cost (depth 4, dim 384, 32 heads, N=V=32)', '',
             '| arm | params | note |', '|---|---|---|']
    rows = [
        ('E98 spread (split)', params_of(lambda p, s: load_e98(p, 'e98-learned-spread-klr20', s), 'mixed_probe'),
         'unified + 2 split-gate projections (erase b*k, value w*v)'),
        ('E88 spread (no split)', params_of(lambda p, s: load_e88_spread(p, s), 'mixed_probe'),
         'unified cell, coupled erase=write=k'),
        ('E98 fixedpop (split)', params_of(lambda p, s: load_e98(p, 'e98-fixedpop', s), 'mixed_probe'), ''),
        ('E88 fixedpop (no split)', params_of(lambda p, s: load_e88_fixedpop(p, s), 'mixed_probe'), ''),
    ]
    for name, pr, note in rows:
        lines.append(f'| {name} | {pr:,} | {note} |' if pr else f'| {name} | -- | {note} |')
    lines.append('')
    return lines


def count_done():
    arms = ['e98-learned-spread-klr20', 'e98-track', 'e98-count', 'e98-latch',
            'e98-nonlin', 'e98-learned-free', 'e98-fixedpop']
    n = 0
    for probe in PROBES:
        for arm in arms:
            for s in SEEDS:
                if load_e98(probe, arm, s):
                    n += 1
    return n


def main():
    out = ['<!-- AUTO-GENERATED (aggregate_e98_on_e97.py). -->', '']
    out.append(f'_E98 runs found: {count_done()}._\n')
    out += ['## 1. Corner re-confirmation on the split-gated cell', '']
    out += corner_table('128')
    out += corner_table('1024')
    out += ['## 2. Learnability (winning form: spread-init + knob-LR 20x) on the split-gated cell', '']
    out += learnability_table('128')
    out += learnability_table('1024')
    out += occupancy_table('E98 spread (split)', lambda p, s: load_e98(p, 'e98-learned-spread-klr20', s))
    out += occupancy_table('E88 spread (no split)', lambda p, s: load_e88_spread(p, s))
    out += ['## 3. E97-split vs E88-coupled — does the split gate help?', '']
    out += headtohead_table('128')
    out += headtohead_table('1024')
    out += params_table()
    text = '\n'.join(out)
    dst = THIS / 'E98_ON_E97_EXPERIMENTAL.md'
    dst.write_text(text)
    print(text)
    print(f'\n[written] {dst}')


if __name__ == '__main__':
    main()
