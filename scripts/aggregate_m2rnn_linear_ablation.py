#!/usr/bin/env python3
"""Aggregate the M2RNN linear/nonlinear state-nonlinearity ablation (task m2rnn-linear-ablation).

Reads every {arm}_{tag}_seed{seed}.json under the ablation eval dir, computes
mean/std over seeds of the length-extrapolation accuracy at {128,256,512,1024}
for each arm x task, and writes:
  * summary.json          — nested arm -> task -> length -> {mean,std,n,seeds}
  * TABLE.md              — markdown table (acc% mean±std), with the E88 knob
                            reference row pulled from the s5sym-eval summary.
NO mocks: every number is read from the real train_hybrid length_extrap output.
"""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ABL = REPO / 'experiments/expressivity_tasks/results/m2rnn_linear_ablation_20260604'
EVAL_DIR = ABL / 'eval'
S5SYM = REPO / 'experiments/expressivity_tasks/results/s5_symmetric_20260603/eval/summary.json'

ARMS = ['m2rnn-nonlinear', 'm2rnn-linear']
SEEDS = [42, 123, 456]
TAGS = ['S5', 'S3']
LENS = ['128', '256', '512', '1024']


def load_acc(arm, tag, seed):
    f = EVAL_DIR / f'{arm}_{tag}_seed{seed}.json'
    d = json.load(open(f))
    le = d['length_extrap']
    return {L: le[L]['acc'] for L in LENS}


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n > 1:
        var = sum((x - m) ** 2 for x in xs) / (n - 1)
        s = math.sqrt(var)
    else:
        s = 0.0
    return m, s


def main():
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for tag in TAGS:
            summary[arm][tag] = {}
            for L in LENS:
                accs, seedmap = [], {}
                for seed in SEEDS:
                    a = load_acc(arm, tag, seed)[L]
                    accs.append(a)
                    seedmap[str(seed)] = a
                m, s = mean_std(accs)
                summary[arm][tag][L] = {'mean': m, 'std': s, 'n': len(accs), 'seeds': seedmap}

    (EVAL_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))

    # E88 knob reference (tanh vs linear) from the sibling s5sym-eval.
    e88 = json.load(open(S5SYM))

    def fmt(d, arm, tag):
        cells = []
        for L in LENS:
            x = d[arm][tag][L]
            cells.append(f"{x['mean']*100:.1f}±{x['std']*100:.1f}")
        return cells

    lines = []
    lines.append('| Family | Arm | Task | T=128 | T=256 | T=512 | T=1024 |')
    lines.append('|---|---|---|---|---|---|---|')
    for tag in TAGS:
        for arm in ARMS:
            c = fmt(summary, arm, tag)
            lines.append(f"| raw-write | {arm} | {tag} | " + " | ".join(c) + " |")
    for tag in TAGS:
        for arm in ('e88-tanh', 'e88-linear'):
            c = fmt(e88, arm, tag)
            lines.append(f"| delta-corr (ref) | {arm} | {tag} | " + " | ".join(c) + " |")
    (ABL / 'TABLE.md').write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {EVAL_DIR/'summary.json'} and {ABL/'TABLE.md'}")


if __name__ == '__main__':
    main()
