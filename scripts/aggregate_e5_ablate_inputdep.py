#!/usr/bin/env python3
"""Aggregate the E5 input-dependence ablation vs the e88-linear baseline.

Reads:
  * Baseline (REUSED verbatim from s5sym-eval): e88-linear is the symmetric-CMA
    winner with linear_state=1, use_gate=1, decay_mode=mamba.
      results/s5_symmetric_20260603/eval/e88-linear_{S5,S3}_seed{42,123,456}.json
  * Ablations (this task):
      results/e5_ablate_inputdep_20260604/eval/{use_gate0,decay_const}_{S5,S3}_seed*.json

Each accuracy is length_extrap[T]['acc'] (the held-out length-extrapolation acc
train_hybrid writes after training). Produces seed-mean ± SD per (arm, task, T)
and a markdown ablation table. NO mocks.
"""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_EVAL = REPO / 'experiments/expressivity_tasks/results/s5_symmetric_20260603/eval'
ABL_EVAL = REPO / 'experiments/expressivity_tasks/results/e5_ablate_inputdep_20260604/eval'
OUT = ABL_EVAL / 'summary.json'

SEEDS = [42, 123, 456]
LENGTHS = ['128', '256', '512', '1024']
TASKS = ['S5', 'S3']

# (display-name, eval-dir, file-prefix)
ARMS = [
    ('e88-linear baseline (use_gate=1, mamba)', BASE_EVAL, 'e88-linear'),
    ('A: use_gate=0 (no output gate)', ABL_EVAL, 'use_gate0'),
    ('B: decay_mode=constant (input-indep transition)', ABL_EVAL, 'decay_const'),
]


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0, len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var), len(xs)


def load_acc(eval_dir, prefix, task, seed, T):
    f = eval_dir / f'{prefix}_{task}_seed{seed}.json'
    if not f.exists():
        return None
    try:
        d = json.load(open(f))
    except Exception:
        return None
    le = d.get('length_extrap') or {}
    return (le.get(T) or {}).get('acc')


def main():
    summary = {}
    for name, eval_dir, prefix in ARMS:
        summary[prefix] = {'name': name, 'dir': str(eval_dir)}
        for task in TASKS:
            summary[prefix][task] = {}
            for T in LENGTHS:
                accs = [load_acc(eval_dir, prefix, task, s, T) for s in SEEDS]
                m, sd, n = mean_sd(accs)
                summary[prefix][task][T] = {
                    'mean': m, 'std': sd, 'n': n,
                    'seeds': {str(s): a for s, a in zip(SEEDS, accs)},
                }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT, 'w'), indent=2)

    def cell(prefix, task, T):
        e = summary[prefix][task][T]
        if e['mean'] is None:
            return '  —  '
        return f"{e['mean']:.4f}±{e['std']:.4f}(n{e['n']})"

    for task in TASKS:
        rand = 1 / 120 if task == 'S5' else 1 / 6
        print(f'\n### {task} seed-mean acc ± SD (random = {rand:.4f})\n')
        print('| Arm | T=128 | T=256 | T=512 | T=1024 |')
        print('|-----|------:|------:|------:|------:|')
        for name, _, prefix in ARMS:
            cells = ' | '.join(cell(prefix, task, T) for T in LENGTHS)
            print(f'| {name} | {cells} |')
    print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
