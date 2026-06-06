#!/usr/bin/env python3
"""Aggregate the s5sym-eval winner-eval JSONs into a seed-mean summary.

Reads the 24 raw per-seed train_hybrid JSONs under
results/s5_symmetric_20260603/eval/ and emits, for each arm x task, the
mean +/- std (over seeds {42,123,456}) of the length-extrapolation accuracy at
T in {128,256,512,1024}. Pure reader: no training, no mock numbers.

Writes results/s5_symmetric_20260603/eval/summary.json and prints a human table.
"""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / 'experiments/expressivity_tasks/results/s5_symmetric_20260603/eval'
ARMS = ['e88-tanh', 'e88-linear', 'm2rnn', 'gdn']
SEEDS = [42, 123, 456]
TAGS = ['S5', 'S3']
LENGTHS = ['128', '256', '512', '1024']


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) > 1:
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        s = math.sqrt(v)
    else:
        s = 0.0
    return m, s, len(xs)


def load(arm, tag, seed):
    p = EVAL / f'{arm}_{tag}_seed{seed}.json'
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for tag in TAGS:
            per_T = {}
            for T in LENGTHS:
                accs = []
                for seed in SEEDS:
                    d = load(arm, tag, seed)
                    if d is None:
                        accs.append(None)
                        continue
                    le = (d.get('length_extrap') or {}).get(T) or {}
                    accs.append(le.get('acc'))
                m, s, n = mean_std(accs)
                per_T[T] = {'mean': m, 'std': s, 'n': n,
                            'seeds': {str(seed): a for seed, a in zip(SEEDS, accs)}}
            summary[arm][tag] = per_T

    out = EVAL / 'summary.json'
    json.dump(summary, open(out, 'w'), indent=2)

    def g(arm, tag, T):
        return summary[arm][tag][T]['mean']

    # Emender slot = better of the two E88 arms on S5@T128.
    et = g('e88-tanh', 'S5', '128')
    el = g('e88-linear', 'S5', '128')
    if et is None or el is None:
        emender_arm, emender_val = None, None
    elif et >= el:
        emender_arm, emender_val = 'e88-tanh', et
    else:
        emender_arm, emender_val = 'e88-linear', el

    print('\n=== S5 length-extrapolation accuracy (seed-mean) ===')
    hdr = f'{"arm":12} ' + ' '.join(f'T{T:>5}' for T in LENGTHS)
    print(hdr)
    for arm in ARMS:
        row = f'{arm:12} '
        for T in LENGTHS:
            m = g(arm, 'S5', T)
            row += f'{(f"{m:.4f}" if m is not None else "  NA "):>6} '
        print(row)

    print('\n=== S3 control length-extrapolation accuracy (seed-mean) ===')
    print(hdr)
    for arm in ARMS:
        row = f'{arm:12} '
        for T in LENGTHS:
            m = g(arm, 'S3', T)
            row += f'{(f"{m:.4f}" if m is not None else "  NA "):>6} '
        print(row)

    print('\n=== Emender slot (better E88 on S5@T128) ===')
    print(f'  e88-tanh   S5@T128 = {et}')
    print(f'  e88-linear S5@T128 = {el}')
    print(f'  -> Emender = {emender_arm} ({emender_val})')

    m2 = g('m2rnn', 'S5', '128')
    gd = g('gdn', 'S5', '128')
    print('\n=== Ordering on S5@T128 ===')
    print(f'  Emender({emender_arm})={emender_val}  M2RNN-CMA={m2}  GDN={gd}')
    if None not in (emender_val, m2, gd):
        ordered = sorted([('Emender', emender_val), ('M2RNN-CMA', m2), ('GDN', gd)],
                         key=lambda kv: kv[1], reverse=True)
        print('  rank: ' + ' > '.join(f'{k}({v:.4f})' for k, v in ordered))
        emender_top = emender_val > m2 and emender_val > gd
        print(f'  Emender on top? {emender_top}')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
