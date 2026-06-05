"""Aggregate the unified-cell sweep into the expressivity table, the un-cribbing
demo, and the emergent-specialization knob distributions.

Reads experiments/expressivity_tasks/results/unified_*.json and writes a
markdown report to paper/review/UNIFIED_CELL_RESULTS.md (the experimental
sections; the kernel-correctness + verdict prose is maintained by hand).
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / 'results'
ROOT = THIS.parents[1]

ARMS = ['unified-track', 'unified-count', 'unified-latch', 'unified-nonlin',
        'unified-learned-free', 'unified-learned-clamp', 'unified-e88base', 'lstm']
PROBES = ['s5_permutation', 'anbncn_viability', 'iterated_nonlinear_map', 'flag_hold_recall']
PROBE_CORNER = {'s5_permutation': 'track', 'anbncn_viability': 'count',
                'iterated_nonlinear_map': 'nonlinear', 'flag_hold_recall': 'latch'}
SEEDS = [42, 123, 456]
EVAL_TS = ['128', '256', '512', '1024']


def load(probe, arm, seed):
    p = RESULTS / f'unified_{probe}__{arm}__seed{seed}.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def acc_at(log, T):
    if log is None:
        return None
    if T == '128':
        # final_acc is the train-length (128) eval; prefer length_extrap['128'] if present
        le = log.get('length_extrap', {})
        if '128' in le and 'acc' in le['128']:
            return le['128']['acc']
        return log.get('final_acc')
    le = log.get('length_extrap', {})
    if T in le and 'acc' in le[T]:
        return le[T]['acc']
    return None


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def fmt(m, s):
    if m is None:
        return '  --  '
    return f'{m:.3f}±{s:.2f}'


def baseline(probe):
    log = None
    for arm in ARMS:
        for seed in SEEDS:
            log = load(probe, arm, seed)
            if log:
                return log.get('random_baseline_acc')
    return None


def expressivity_table(T):
    lines = [f'### Expressivity @ T={T}  (mean±std over seeds {SEEDS}; train T=128)', '']
    header = '| arm | ' + ' | '.join(f'{p}<br>({PROBE_CORNER[p]})' for p in PROBES) + ' |'
    lines.append(header)
    lines.append('|' + '---|' * (len(PROBES) + 1))
    # baseline row
    base = '| _random baseline_ | ' + ' | '.join(
        (f'{baseline(p):.3f}' if baseline(p) is not None else '--') for p in PROBES) + ' |'
    lines.append(base)
    for arm in ARMS:
        cells = []
        for probe in PROBES:
            m, s = mean_std([acc_at(load(probe, arm, seed), T) for seed in SEEDS])
            cells.append(fmt(m, s))
        lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def best_preset_per_probe(T):
    """For each probe, which pinned preset (corner arm) wins, and does LEARNED match it?"""
    preset_arms = ['unified-track', 'unified-count', 'unified-latch', 'unified-nonlin']
    lines = [f'### Does LEARNED match the best preset on each probe? @ T={T}', '']
    lines.append('| probe (corner) | best preset (acc) | LEARNED-free (acc) | E88-base (acc) | LSTM (acc) | LEARNED matches best? |')
    lines.append('|---|---|---|---|---|---|')
    for probe in PROBES:
        scores = {}
        for arm in preset_arms:
            m, _ = mean_std([acc_at(load(probe, arm, seed), T) for seed in SEEDS])
            scores[arm] = m if m is not None else -1
        best_arm = max(scores, key=scores.get)
        best_score = scores[best_arm]
        learned, _ = mean_std([acc_at(load(probe, 'unified-learned-free', seed), T) for seed in SEEDS])
        e88, _ = mean_std([acc_at(load(probe, 'unified-e88base', seed), T) for seed in SEEDS])
        lstm, _ = mean_std([acc_at(load(probe, 'lstm', seed), T) for seed in SEEDS])
        learned = learned if learned is not None else 0.0
        matches = 'YES' if learned >= best_score - 0.05 else 'no'

        def g(x):
            return f'{x:.3f}' if x is not None else '--'
        lines.append(f'| {probe} ({PROBE_CORNER[probe]}) | {best_arm.replace("unified-","")} '
                     f'({g(best_score)}) | {g(learned)} | {g(e88)} | '
                     f'{g(lstm)} | {matches} |')
    lines.append('')
    return lines


def uncribbing_demo():
    lines = ['## Un-cribbing demo (controlled headline): lambda FREE vs CLAMPED to (0,1)', '',
             'Same LEARNED cell, identical recipe; the ONLY difference is whether the gain '
             'lambda may reach/exceed 1 (`lam_max=1.5`, FREE) or is clamped to (0,1) '
             '(`lam_max=1.0`, the cribbed E88 regime). Counting and latching require '
             'eigenvalue magnitude >=1; clamping should kill them.', '']
    lines.append('| probe (corner) | T | LEARNED-free | LEARNED-clamp | delta (free-clamp) |')
    lines.append('|---|---|---|---|---|')
    for probe in ['anbncn_viability', 'flag_hold_recall', 's5_permutation', 'iterated_nonlinear_map']:
        for T in ['128', '512', '1024']:
            mf, sf = mean_std([acc_at(load(probe, 'unified-learned-free', seed), T) for seed in SEEDS])
            mc, sc = mean_std([acc_at(load(probe, 'unified-learned-clamp', seed), T) for seed in SEEDS])
            if mf is None or mc is None:
                continue
            lines.append(f'| {probe} ({PROBE_CORNER[probe]}) | {T} | {mf:.3f}±{sf:.2f} | '
                         f'{mc:.3f}±{sc:.2f} | {mf-mc:+.3f} |')
    lines.append('')
    return lines


def emergent_specialization():
    """Aggregate per-head learned (lambda,beta,gamma,eig_along) from learned-free arm."""
    lines = ['## Emergent specialization (Run C): per-head learned knobs', '',
             'Per-head (lambda, beta, gamma, eig_along=lambda-beta) after training the '
             'LEARNED-free cell, pooled over all heads/layers/seeds, reported per probe. '
             'Corner signatures: track=eig_along<0 (reflection); count=lambda~1 & beta~0; '
             'latch=lambda>1 & gamma high (tanh); nonlinear=gamma high & lambda<1.', '']
    for probe in PROBES:
        lams, betas, gammas, eigs = [], [], [], []
        for seed in SEEDS:
            log = load(probe, 'unified-learned-free', seed)
            if not log or 'unified_knobs' not in log:
                continue
            for layer in log['unified_knobs']:
                lams += layer['lambda']; betas += layer['beta']
                gammas += layer['gamma']; eigs += layer['eig_along']
        if not lams:
            continue
        def desc(v):
            return f'mean={sum(v)/len(v):.3f} min={min(v):.3f} max={max(v):.3f}'
        n_neg_eig = sum(1 for e in eigs if e < 0)
        n_count = sum(1 for l, b in zip(lams, betas) if l > 0.9 and b < 0.3)
        n_latch = sum(1 for l, gv in zip(lams, gammas) if l > 1.0 and gv > 0.6)
        lines.append(f'### {probe} ({PROBE_CORNER[probe]}) — {len(lams)} heads pooled')
        lines.append(f'- lambda : {desc(lams)}')
        lines.append(f'- beta   : {desc(betas)}')
        lines.append(f'- gamma  : {desc(gammas)}')
        lines.append(f'- eig_along (lambda-beta): {desc(eigs)}  '
                     f'({100*n_neg_eig/len(eigs):.0f}% heads reflecting, eig<0)')
        lines.append(f'- heads in count-corner (lambda>0.9 & beta<0.3): {n_count} '
                     f'({100*n_count/len(lams):.0f}%); latch-corner (lambda>1 & gamma>0.6): '
                     f'{n_latch} ({100*n_latch/len(lams):.0f}%)')
        lines.append('')
    return lines


def main():
    out = ['<!-- AUTO-GENERATED experimental sections (aggregate_unified.py). -->',
           '<!-- Re-run: python experiments/expressivity_tasks/aggregate_unified.py -->', '']
    n_done = sum(1 for p in PROBES for a in ARMS for s in SEEDS if load(p, a, s))
    out.append(f'_Runs found: {n_done} / {len(PROBES)*len(ARMS)*len(SEEDS)}._\n')
    out += ['## Expressivity table (Run A)', '']
    out += expressivity_table('128')
    out += expressivity_table('512')
    out += expressivity_table('1024')
    out += best_preset_per_probe('128')
    out += best_preset_per_probe('512')
    out += uncribbing_demo()
    out += emergent_specialization()
    text = '\n'.join(out)
    dst = THIS / 'UNIFIED_CELL_EXPERIMENTAL.md'
    dst.write_text(text)
    print(text)
    print(f'\n[written] {dst}')


if __name__ == '__main__':
    main()
