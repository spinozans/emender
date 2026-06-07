"""Aggregate the E97-RAW x GDN HYBRID expressivity ratio-sweep (task e97-gdn-hybrid).

Reads results/e97gdn_<probe>__<arm>__seed<seed>.json (run_e97_gdn_hybrid.py) and
emits the ratio-sweep tables for paper/review/E97_RAW_GDN_HYBRID_RESULTS.md:

  1. PER-PROBE tables: arm (all-e97raw -> all-gdn) x eval-length, mean+/-std (3 seeds).
  2. DECISION tables:
       RECALL  — mqar acc vs ratio (does recall recover toward GDN's bar?).
       LATCH   — flag_hold acc T=128 vs T=1024 vs ratio (does GDN fix the
                 e97-raw latch length-extrapolation collapse?).
       COUNT   — anbncn acc vs ratio (does count SURVIVE as GDN layers replace e97?).
  3. Param counts per arm (shape-matched; GDN layers are ~5x cheaper).

Run: python experiments/expressivity_tasks/aggregate_e97_gdn_hybrid.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / 'results'

SEEDS = [42, 123, 456]
EVAL_T = ['128', '256', '512', '1024']
PROBES = ['s5_permutation', 'anbncn_viability', 'iterated_nonlinear_map',
          'flag_hold_recall', 'mqar_recall']
PROBE_CAP = {'s5_permutation': 'TRACK (S5 state-tracking)',
             'anbncn_viability': 'COUNT (a^n b^n c^n)',
             'iterated_nonlinear_map': 'NONLIN (iterated map)',
             'flag_hold_recall': 'LATCH (flag-hold)',
             'mqar_recall': 'RECALL (MQAR)'}
# ordered from pure e97-raw backbone -> pure GDN
ARMS = ['all-e97raw', 'h3to1', 'h1to1', 'h1to3', 'all-gdn']
ARM_DESC = {'all-e97raw': 'E97 E97 E97 E97 (0/4 gdn)',
            'h3to1': 'E97 E97 E97 gdn (1/4 gdn)',
            'h1to1': 'E97 gdn E97 gdn (2/4 gdn)',
            'h1to3': 'E97 gdn gdn gdn (3/4 gdn)',
            'all-gdn': 'gdn gdn gdn gdn (4/4 gdn)'}


def _load(probe, arm, seed):
    p = RESULTS / f'e97gdn_{probe}__{arm}__seed{seed}.json'
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


def arm_ms(probe, arm, T):
    return mean_std([acc_at(_load(probe, arm, s), T) for s in SEEDS])


def fmt(m, s):
    return '   —   ' if m is None else f'{m:.3f}±{s:.3f}'


def baseline(probe):
    for s in SEEDS:
        for arm in ARMS:
            log = _load(probe, arm, s)
            if log and log.get('random_baseline_acc') is not None:
                return log['random_baseline_acc']
    return None


def params(arm):
    for s in SEEDS:
        for probe in PROBES:
            log = _load(probe, arm, s)
            if log and log.get('params') is not None:
                return log['params']
    return None


def n_done(probe):
    return sum(1 for arm in ARMS for s in SEEDS if _load(probe, arm, s) is not None)


def at(probe, arm, T):
    m, _ = arm_ms(probe, arm, T)
    return m


def v(x):
    return f'{x:.3f}' if x is not None else '—'


def f(x):
    return f'{x:+.3f}' if x is not None else '—'


def main():
    lines = []
    total = sum(n_done(p) for p in PROBES)
    lines.append(f'## [auto] E97-raw x GDN hybrid ratio-sweep '
                 f'({total}/{len(PROBES)*len(ARMS)*len(SEEDS)} runs done)\n')

    # Param table
    lines.append('### Param counts per arm (shape-matched dim256/h32/N32, depth4)\n')
    lines.append('| arm | pattern | params |')
    lines.append('|---|---|---|')
    for arm in ARMS:
        p = params(arm)
        lines.append(f'| {arm} | {ARM_DESC[arm]} | {p:,} |' if p else
                     f'| {arm} | {ARM_DESC[arm]} | — |')
    lines.append('')

    # Per-probe tables
    lines.append('### Per-probe accuracy (mean±std over seeds 42/123/456)\n')
    for probe in PROBES:
        bl = baseline(probe)
        done = n_done(probe)
        bl_str = f'{bl:.3f}' if bl is not None else 'n/a'
        lines.append(f'#### {PROBE_CAP[probe]} — `{probe}`  '
                     f'(random baseline={bl_str})  [{done}/{len(ARMS)*len(SEEDS)}]\n')
        lines.append('| arm | ' + ' | '.join(f'T={t}' for t in EVAL_T) + ' |')
        lines.append('|' + '---|' * (len(EVAL_T) + 1))
        for arm in ARMS:
            row = [arm] + [fmt(*arm_ms(probe, arm, t)) for t in EVAL_T]
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    # DECISION tables
    lines.append('## [auto] Decision tables\n')

    lines.append('### RECALL recovery — does adding GDN recover MQAR? (acc vs ratio)\n')
    lines.append('| arm | T=128 | T=256 | T=512 | T=1024 |')
    lines.append('|---|---|---|---|---|')
    for arm in ARMS:
        lines.append(f'| {arm} | ' + ' | '.join(v(at('mqar_recall', arm, t)) for t in EVAL_T) + ' |')
    lines.append('')

    lines.append('### LATCH length-extrapolation — does GDN fix the e97-raw collapse?\n')
    lines.append('| arm | T=128 | T=1024 | drop (T1024−T128) |')
    lines.append('|---|---|---|---|')
    for arm in ARMS:
        a128, a1024 = at('flag_hold_recall', arm, '128'), at('flag_hold_recall', arm, '1024')
        drop = (a1024 - a128) if (a128 is not None and a1024 is not None) else None
        lines.append(f'| {arm} | {v(a128)} | {v(a1024)} | {f(drop)} |')
    lines.append('')

    lines.append('### COUNT retention — does count survive as GDN replaces e97-raw?\n')
    lines.append('| arm | T=128 | T=256 | T=512 | T=1024 |')
    lines.append('|---|---|---|---|---|')
    for arm in ARMS:
        lines.append(f'| {arm} | ' + ' | '.join(v(at('anbncn_viability', arm, t)) for t in EVAL_T) + ' |')
    lines.append('')

    lines.append('### TRACK (S5) — secondary; GDN & e97-raw both weak\n')
    lines.append('| arm | T=128 | T=1024 |')
    lines.append('|---|---|---|')
    for arm in ARMS:
        lines.append(f'| {arm} | {v(at("s5_permutation", arm, "128"))} | {v(at("s5_permutation", arm, "1024"))} |')
    lines.append('')

    out = '\n'.join(lines)
    print(out)
    (THIS / '_e97gdn_auto_tables.md').write_text(out)
    print(f'\n[written] {THIS / "_e97gdn_auto_tables.md"}')


if __name__ == '__main__':
    main()
