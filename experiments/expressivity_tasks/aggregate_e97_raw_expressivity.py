"""Aggregate the E97-RAW EXPRESSIVITY battery.

Reads results/e97raw_<probe>__<arm>__seed<seed>.json (run_e97_raw_expressivity.py)
and emits, for paper/review/E97_RAW_EXPRESSIVITY_RESULTS.md:

  1. PER-PROBE tables: arm x eval-length (mean+/-std acc over 3 seeds), with the
     task random baseline.
  2. ISOLATION deltas for the three hypotheses:
       H1 (recall):  e97-raw vs GDN, and e97 vs e97-raw, on mqar_recall.
       H2 (latch+count): e97-raw/e97 vs e97-linear on flag_hold_recall + anbncn.
       H3 (track):  e97-raw on s5_permutation (does it reach the track regime?).

Run: python experiments/expressivity_tasks/aggregate_e97_raw_expressivity.py
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
PROBE_CAP = {'s5_permutation': 'TRACK (state-tracking)',
             'anbncn_viability': 'COUNT (a^n b^n c^n)',
             'iterated_nonlinear_map': 'NONLIN (iterated map)',
             'flag_hold_recall': 'LATCH (flag-hold)',
             'mqar_recall': 'RECALL (MQAR)'}
ARMS = ['e97-raw', 'e97', 'e97-linear', 'gdn']


def _load(probe, arm, seed):
    p = RESULTS / f'e97raw_{probe}__{arm}__seed{seed}.json'
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
    if m is None:
        return '   —   '
    return f'{m:.3f}±{s:.3f}'


def baseline(probe):
    for s in SEEDS:
        for arm in ARMS:
            log = _load(probe, arm, s)
            if log and log.get('random_baseline_acc') is not None:
                return log['random_baseline_acc']
    return None


def n_done(probe):
    return sum(1 for arm in ARMS for s in SEEDS if _load(probe, arm, s) is not None)


def main():
    lines = []
    lines.append('## [auto] Per-probe results (acc, mean±std over seeds 42/123/456)\n')
    for probe in PROBES:
        bl = baseline(probe)
        done = n_done(probe)
        bl_str = f'{bl:.3f}' if bl is not None else 'n/a'
        lines.append(f'### {PROBE_CAP[probe]} — `{probe}`  '
                     f'(random baseline acc={bl_str})  [{done}/12 runs]\n')
        header = '| arm | ' + ' | '.join(f'T={t}' for t in EVAL_T) + ' |'
        sep = '|' + '---|' * (len(EVAL_T) + 1)
        lines.append(header)
        lines.append(sep)
        for arm in ARMS:
            row = [arm]
            for t in EVAL_T:
                m, s = arm_ms(probe, arm, t)
                row.append(fmt(m, s))
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    # ---- Hypothesis deltas (train-length T=128 + extrapolation T=512) ----
    lines.append('## [auto] Hypothesis isolation deltas\n')

    def at(probe, arm, T):
        m, s = arm_ms(probe, arm, T)
        return m

    lines.append('### H1 — recall: raw-write drops the overwrite recall needs\n')
    lines.append('| length | e97-raw | e97 (delta) | gdn (ref) | e97 − e97-raw | gdn − e97-raw |')
    lines.append('|---|---|---|---|---|---|')
    for t in EVAL_T:
        r, e, g = at('mqar_recall', 'e97-raw', t), at('mqar_recall', 'e97', t), at('mqar_recall', 'gdn', t)
        d_de = (e - r) if (e is not None and r is not None) else None
        d_gd = (g - r) if (g is not None and r is not None) else None
        def f(x): return f'{x:+.3f}' if x is not None else '—'
        def v(x): return f'{x:.3f}' if x is not None else '—'
        lines.append(f'| T={t} | {v(r)} | {v(e)} | {v(g)} | {f(d_de)} | {f(d_gd)} |')
    lines.append('')

    lines.append('### H2 — tanh squash helps latch + count\n')
    lines.append('| probe @T | e97-raw | e97 | e97-linear | (tanh) − (linear) |')
    lines.append('|---|---|---|---|---|')
    for probe in ['flag_hold_recall', 'anbncn_viability']:
        for t in ['128', '512']:
            raw, e97, lin = at(probe, 'e97-raw', t), at(probe, 'e97', t), at(probe, 'e97-linear', t)
            # tanh arms = e97-raw & e97; linear arm = e97-linear. Use e97 (delta+tanh)
            # vs e97-linear (delta+identity) for the clean tanh isolation.
            d = (e97 - lin) if (e97 is not None and lin is not None) else None
            def v(x): return f'{x:.3f}' if x is not None else '—'
            def f(x): return f'{x:+.3f}' if x is not None else '—'
            lines.append(f'| {probe} @T={t} | {v(raw)} | {v(e97)} | {v(lin)} | {f(d)} |')
    lines.append('')

    lines.append('### H3 — does e97-raw reach the S5 track regime?\n')
    lines.append('| length | e97-raw | e97 | e97-linear | gdn | baseline |')
    lines.append('|---|---|---|---|---|---|')
    bl = baseline('s5_permutation')
    for t in EVAL_T:
        vals = [at('s5_permutation', a, t) for a in ARMS]
        def v(x): return f'{x:.3f}' if x is not None else '—'
        lines.append(f'| T={t} | ' + ' | '.join(v(x) for x in vals) +
                     f' | {v(bl)} |')
    lines.append('')

    out = '\n'.join(lines)
    print(out)
    (THIS / '_e97raw_auto_tables.md').write_text(out)
    print(f'\n[written] {THIS / "_e97raw_auto_tables.md"}')


if __name__ == '__main__':
    main()
