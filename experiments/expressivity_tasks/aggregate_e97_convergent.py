"""Aggregate the E97-CONVERGENT 2x3 matrix — BOTH axes (task e97-convergent).

EXPRESSIVITY: reads results_convergent/e97conv_<probe>__<arm>__seed<seed>.json
(run_e97_convergent.py) for the 6 arms (backbone{raw,delta} x recall{none,gdn,gdn-neg}).
LM: reads results_convergent (e97_gdn_hybrid_lm)/<arm>_mlp1.0_s0.json
(run_e97_convergent_lm.py).

Emits the tables for paper/review/E97_CONVERGENT_CELL_RESULTS.md:
  1. Param counts per arm.
  2. 5-capability matrix (T=128 + T=1024 length-extrap), 3 seeds.
  3. DECISION A — does gdn-neg give RECALL (mqar) AND TRACK (s5) in ONE head?
  4. DECISION B — is e97-delta the better BACKBONE? (delta − raw per capability)
  5. DECISION C — LM axis: does convergence cost LM loss? (train/held/bpb per arm)

Run: python experiments/expressivity_tasks/aggregate_e97_convergent.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / 'results_convergent'
LM_RESULTS = THIS.parents[0] / 'e97_gdn_hybrid_lm' / 'results_convergent'

SEEDS = [42, 123, 456]
EVAL_T = ['128', '256', '512', '1024']
PROBES = ['s5_permutation', 'anbncn_viability', 'iterated_nonlinear_map',
          'flag_hold_recall', 'mqar_recall']
PROBE_CAP = {'s5_permutation': 'TRACK (S5)',
             'anbncn_viability': 'COUNT (a^n b^n c^n)',
             'iterated_nonlinear_map': 'NONLIN (iter-map)',
             'flag_hold_recall': 'LATCH (flag-hold)',
             'mqar_recall': 'RECALL (MQAR)'}
ARMS = ['raw-none', 'raw-gdn', 'raw-gdnneg',
        'delta-none', 'delta-gdn', 'delta-gdnneg']
ARM_DESC = {
    'raw-none':     'E97 E97 E97 E97 / raw_write=1 (backbone only)',
    'raw-gdn':      'E97 gdn E97 gdn / raw, allow_neg=0',
    'raw-gdnneg':   'E97 gdn E97 gdn / raw, allow_neg=1',
    'delta-none':   'E97 E97 E97 E97 / raw_write=0 (backbone only)',
    'delta-gdn':    'E97 gdn E97 gdn / delta, allow_neg=0',
    'delta-gdnneg': 'E97 gdn E97 gdn / delta, allow_neg=1',
}
LM_ARMS = ['raw-none', 'raw-gdn', 'raw-gdnneg',
           'delta-none', 'delta-gdn', 'delta-gdnneg', 'gdn2-ref']


def _load(probe, arm, seed):
    p = RESULTS / f'e97conv_{probe}__{arm}__seed{seed}.json'
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


def at(probe, arm, T):
    m, _ = arm_ms(probe, arm, T)
    return m


def fmt(m, s):
    return '   —   ' if m is None else f'{m:.3f}±{s:.3f}'


def v(x):
    return f'{x:.3f}' if x is not None else '—'


def f(x):
    return f'{x:+.3f}' if x is not None else '—'


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


def lm_load(arm):
    p = LM_RESULTS / f'{arm}_mlp1.0_s0.json'
    return json.loads(p.read_text()) if p.exists() else None


def main():
    lines = []
    total = sum(n_done(p) for p in PROBES)
    full = len(PROBES) * len(ARMS) * len(SEEDS)
    lines.append(f'## [auto] E97-CONVERGENT 2x3 matrix — expressivity ({total}/{full} runs done)\n')

    lines.append('### Param counts per arm (shape-matched dim256/h32/N32, depth4)\n')
    lines.append('| arm | pattern / config | params |')
    lines.append('|---|---|---|')
    for arm in ARMS:
        p = params(arm)
        lines.append(f'| {arm} | {ARM_DESC[arm]} | ' + (f'{p:,}' if p else '—') + ' |')
    lines.append('')

    # 5-capability matrix at T=128 and T=1024
    for T in ['128', '1024']:
        lines.append(f'### Capability matrix @ T={T} (mean±std over seeds 42/123/456)\n')
        hdr = '| arm | ' + ' | '.join(PROBE_CAP[p] for p in PROBES) + ' |'
        lines.append(hdr)
        lines.append('|' + '---|' * (len(PROBES) + 1))
        bl = {p: baseline(p) for p in PROBES}
        for arm in ARMS:
            row = [arm] + [fmt(*arm_ms(p, arm, T)) for p in PROBES]
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('| _random baseline_ | ' +
                     ' | '.join(v(bl[p]) for p in PROBES) + ' |')
        lines.append('')

    # DECISION A — gdn-neg: recall AND track in one head?
    lines.append('## [auto] DECISION A — does gdn-neg deliver RECALL *and* TRACK in one head?\n')
    lines.append('MQAR (recall) and S5 (track), best over T (max length-extrap acc). '
                 'gdn-neg wins this question iff a single recall arm is high on BOTH.\n')
    lines.append('| arm | RECALL (mqar, max-T) | TRACK (s5, max-T) | both? |')
    lines.append('|---|---|---|---|')

    def maxT(probe, arm):
        vals = [at(probe, arm, t) for t in EVAL_T]
        vals = [x for x in vals if x is not None]
        return max(vals) if vals else None
    for arm in ARMS:
        r = maxT('mqar_recall', arm)
        tr = maxT('s5_permutation', arm)
        both = '✓' if (r is not None and tr is not None and r > 0.5 and tr > 0.5) else ''
        lines.append(f'| {arm} | {v(r)} | {v(tr)} | {both} |')
    lines.append('')

    # DECISION B — delta vs raw backbone (per-capability delta at T=1024 and T=128)
    lines.append('## [auto] DECISION B — is e97-delta the better BACKBONE? (delta − raw)\n')
    for T in ['128', '1024']:
        lines.append(f'### delta − raw per capability @ T={T} (matched recall-head)\n')
        lines.append('| recall-head | ' + ' | '.join(PROBE_CAP[p] for p in PROBES) + ' |')
        lines.append('|' + '---|' * (len(PROBES) + 1))
        for head in ['none', 'gdn', 'gdnneg']:
            r_arm, d_arm = f'raw-{head}', f'delta-{head}'
            row = [head]
            for p in PROBES:
                rv, dv = at(p, r_arm, T), at(p, d_arm, T)
                row.append(f((dv - rv) if (rv is not None and dv is not None) else None))
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    # LATCH length-extrap collapse (does delta backbone / gdn fix raw's collapse?)
    lines.append('### LATCH length-extrap (flag-hold) — T=128 vs T=1024 (raw collapse fix?)\n')
    lines.append('| arm | T=128 | T=1024 | drop |')
    lines.append('|---|---|---|---|')
    for arm in ARMS:
        a128, a1024 = at('flag_hold_recall', arm, '128'), at('flag_hold_recall', arm, '1024')
        drop = (a1024 - a128) if (a128 is not None and a1024 is not None) else None
        lines.append(f'| {arm} | {v(a128)} | {v(a1024)} | {f(drop)} |')
    lines.append('')

    # DECISION C — LM axis
    lines.append('## [auto] DECISION C — LM axis (REAL Pile, +MLP r1.0, token-matched, bf16)\n')
    lm_done = sum(1 for a in LM_ARMS if lm_load(a) is not None)
    lines.append(f'_{lm_done}/{len(LM_ARMS)} LM arms done._\n')
    lines.append('| arm | params | core | train (last10%) | held-out nats | held BPB |')
    lines.append('|---|---|---|---|---|---|')
    for arm in LM_ARMS:
        lg = lm_load(arm)
        if lg is None:
            lines.append(f'| {arm} | — | — | — | — | — |')
            continue
        fin = lg.get('final', {})
        lines.append(
            f"| {arm} | {lg.get('params', 0):,} | {lg.get('params_noembed', 0):,} | "
            f"{fin.get('train_loss_last10pct', float('nan')):.4f} | "
            f"{fin.get('heldout_nats', float('nan')):.4f} | "
            f"{fin.get('heldout_bpb', float('nan')):.4f} |")
    lines.append('')

    out = '\n'.join(lines)
    print(out)
    (THIS / 'e97_convergent_auto_tables.md').write_text(out)


if __name__ == '__main__':
    main()
