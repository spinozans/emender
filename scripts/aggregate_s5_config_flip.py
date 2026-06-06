#!/usr/bin/env python3
"""Assemble the CONFIG-FLIP 2x2 (task s5-config-flip).

Pure reader. Combines the four cells of the linear_state x config cross:

           tanh (linear_state=0)        linear (linear_state=1)
  config T   A = e88-tanh   [eval]        D = e88-cfgT-linear [flip]
  config L   C = e88-cfgL-tanh [flip]     B = e88-linear [eval]

A and B come VERBATIM from the s5sym-eval summary.json (cited, not rerun); their
raw per-seed JSONs are e88-tanh_*.json / e88-linear_*.json. C and D come from
this task's raw per-seed JSONs (e88-cfgL-tanh_*.json / e88-cfgT-linear_*.json),
re-aggregated here with the same seed-mean/SD recipe as aggregate_s5_symmetric.

Emits eval/config_flip_summary.json and prints the 2x2 + extrapolation + S3.
"""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / 'experiments/expressivity_tasks/results/s5_symmetric_20260603'
EVAL = RESULTS / 'eval'
WINNERS = RESULTS / 'winners'
SEEDS = [42, 123, 456]
TAGS = ['S5', 'S3']
LENGTHS = ['128', '256', '512', '1024']

# cell -> (arm label used in filenames, config tag, knob)
CELLS = {
    'A': {'arm': 'e88-tanh',        'config': 'T', 'knob': 'tanh',   'source': 'eval (cited)'},
    'B': {'arm': 'e88-linear',      'config': 'L', 'knob': 'linear', 'source': 'eval (cited)'},
    'C': {'arm': 'e88-cfgL-tanh',   'config': 'L', 'knob': 'tanh',   'source': 'flip (this task)'},
    'D': {'arm': 'e88-cfgT-linear', 'config': 'T', 'knob': 'linear', 'source': 'flip (this task)'},
}


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s, len(xs)


def load(arm, tag, seed):
    p = EVAL / f'{arm}_{tag}_seed{seed}.json'
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def param_count(arm):
    """Real param count from any per-seed JSON for this arm."""
    for tag in TAGS:
        for seed in SEEDS:
            d = load(arm, tag, seed)
            if d and 'params' in d:
                return d['params']
    return None


def aggregate_arm(arm):
    out = {}
    for tag in TAGS:
        per_seed = {s: load(arm, tag, s) for s in SEEDS}
        per_T = {}
        for T in LENGTHS:
            accs = [((per_seed[s] or {}).get('length_extrap', {}) or {}).get(T, {}).get('acc')
                    if per_seed[s] else None for s in SEEDS]
            m, sd, n = mean_std(accs)
            per_T[T] = {'mean': m, 'std': sd, 'n': n,
                        'seeds': {str(s): a for s, a in zip(SEEDS, accs)}}
        out[tag] = per_T
    return out


def fmt(m, sd):
    if m is None:
        return '  NA  '
    return f'{m:.4f}±{sd:.4f}' if sd is not None else f'{m:.4f}'


def main():
    data = {}
    for cell, meta in CELLS.items():
        arm = meta['arm']
        data[cell] = {**meta, 'params': param_count(arm), 'agg': aggregate_arm(arm)}

    out = EVAL / 'config_flip_summary.json'
    json.dump(data, open(out, 'w'), indent=2)

    def g(cell, tag, T):
        return data[cell]['agg'][tag][T]['mean'], data[cell]['agg'][tag][T]['std']

    print('\n=== PARAM COUNTS ===')
    for cell in 'ABCD':
        m = data[cell]
        print(f'  {cell}  {m["arm"]:18} config {m["config"]} + {m["knob"]:6} '
              f'params={m["params"]}  [{m["source"]}]')

    print('\n=== 2x2 on S5@T128 (seed-mean ± SD) ===')
    print(f'{"":10} {"tanh (ls=0)":>22} {"linear (ls=1)":>22}')
    for cfg, (tcell, lcell) in [('config T', ('A', 'D')), ('config L', ('C', 'B'))]:
        tm, ts = g(tcell, 'S5', '128'); lm, ls = g(lcell, 'S5', '128')
        print(f'{cfg:10} {tcell}:{fmt(tm,ts):>20} {lcell}:{fmt(lm,ls):>20}')

    for tag in TAGS:
        print(f'\n=== {tag} length-extrapolation grid (seed-mean ± SD) ===')
        print(f'{"cell":5}{"arm":18}{"cfg":4}{"knob":7}' + ''.join(f'{"T"+T:>14}' for T in LENGTHS))
        for cell in 'ABCD':
            m = data[cell]
            row = f'{cell:5}{m["arm"]:18}{m["config"]:4}{m["knob"]:7}'
            for T in LENGTHS:
                mm, sd = g(cell, tag, T)
                row += f'{fmt(mm,sd):>14}'
            print(row)

    print('\n=== DISENTANGLEMENT (S5@T128) ===')
    A = g('A', 'S5', '128')[0]; B = g('B', 'S5', '128')[0]
    C = g('C', 'S5', '128')[0]; D = g('D', 'S5', '128')[0]
    if None not in (A, B, C, D):
        print(f'  KNOB at config L: linear(B)={B:.4f} vs tanh(C)={C:.4f}  -> Δ(lin-tanh)={B-C:+.4f}')
        print(f'  KNOB at config T: linear(D)={D:.4f} vs tanh(A)={A:.4f}  -> Δ(lin-tanh)={D-A:+.4f}')
        print(f'  CONFIG at tanh:   cfgL(C)={C:.4f} vs cfgT(A)={A:.4f}    -> Δ(L-T)={C-A:+.4f}')
        print(f'  CONFIG at linear: cfgL(B)={B:.4f} vs cfgT(D)={D:.4f}    -> Δ(L-T)={B-D:+.4f}')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
