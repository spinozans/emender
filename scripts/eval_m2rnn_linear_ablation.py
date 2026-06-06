#!/usr/bin/env python3
"""M2RNN raw-write state-nonlinearity ablation orchestrator (task m2rnn-linear-ablation).

Symmetric counterpart to scripts/eval_s5_symmetric_winners.py, restricted to the
M2RNN-CMA winner and its linear-state ablation:

  arms:  m2rnn-nonlinear  (Z = tanh(h W + k v^T), as built)
         m2rnn-linear     (Z =      h W + k v^T,  --linear_state 1)
  seeds: {42, 123, 456}
  tasks: S5  (--task s5_permutation, train T=128, 20000 steps)
         S3  (--task s3_permutation, train T=128, 10000 steps, solvable control)
  grid:  eval lengths {128, 256, 512, 1024} (length extrapolation, end of training)
  optim: schedule-free AdamW, batch 32, seq_len 128 (identical to the search arms)

Shape/lr come verbatim from winners/m2rnn.args.json. The nonlinear-arm JSONs are
typically pre-seeded by copying the s5sym-eval m2rnn winner runs (behaviorally
identical to --linear_state 0); `already_done` then skips them and only the
linear arm trains. Pass --force-nonlinear to re-train the nonlinear arm too.

GPUs: 6,7 ONLY (CUDA_VISIBLE_DEVICES per process). Idempotent: a job whose output
JSON already has a populated `length_extrap` block with all four lengths is skipped.
NO mocks: every accuracy is the real eval_acc / length_extrap field train_hybrid writes.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ABL = REPO / 'experiments/expressivity_tasks/results/m2rnn_linear_ablation_20260604'
EVAL_DIR = ABL / 'eval'
LOG_DIR = EVAL_DIR / 'logs'
TRAIN = REPO / 'experiments/expressivity_tasks/train_hybrid.py'

SEEDS = [42, 123, 456]
GPUS = [6, 7]
EVAL_LENGTHS = ['128', '256', '512', '1024']

# arm -> linear_state flag value
ARMS = {'m2rnn-nonlinear': 0, 'm2rnn-linear': 1}
WINNER = {'dim': 512, 'depth': 5, 'n_heads': 19, 'n_state': 32,
          'lr': 0.0016057620284487264}

# (task_flag, train steps, short tag)
TASKS = [
    ('s5_permutation', 20000, 'S5'),
    ('s3_permutation', 10000, 'S3'),
]

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def build_cmd(arm, linear_state, task, steps, seed, label):
    p = WINNER
    return [
        sys.executable, str(TRAIN),
        '--task', task,
        '--layer_pattern', 'm2rnn',
        '--dim', str(p['dim']),
        '--depth', str(p['depth']),
        '--n_heads', str(p['n_heads']),
        '--n_state', str(p['n_state']),
        '--expansion', '1.0',
        '--lr', str(p['lr']),
        '--optimizer', 'schedulefree',
        '--steps', str(steps),
        '--seq_len', '128',
        '--batch_size', '32',
        '--K', '5',
        '--seed', str(seed),
        '--label', label,
        '--output_dir', str(EVAL_DIR),
        '--eval_lengths', *EVAL_LENGTHS,
        '--eval_lengths_n_batches', '8',
        '--linear_state', str(linear_state),
    ]


def already_done(label):
    out = EVAL_DIR / f'{label}.json'
    if not out.exists():
        return False
    try:
        d = json.load(open(out))
    except Exception:
        return False
    le = d.get('length_extrap') or {}
    return all(str(t) in le and 'acc' in le[str(t)] for t in EVAL_LENGTHS)


def run_job(job, gpu_pool):
    arm, linear_state, task, steps, tag, seed = job
    label = f'{arm}_{tag}_seed{seed}'
    if already_done(label):
        log(f'SKIP  {label} (already complete)')
        return (label, 'skipped', None)
    gpu = gpu_pool.get()
    try:
        cmd = build_cmd(arm, linear_state, task, steps, seed, label)
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        logf = LOG_DIR / f'{label}.log'
        t0 = time.time()
        log(f'START {label}  gpu={gpu}  steps={steps}')
        with open(logf, 'w') as lf:
            lf.write('CMD: ' + ' '.join(cmd) + '\n\n')
            lf.flush()
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=str(REPO), env=env)
        dt = time.time() - t0
        ok = already_done(label) and rc == 0
        status = 'ok' if ok else f'FAIL(rc={rc})'
        log(f'DONE  {label}  gpu={gpu}  {status}  {dt/60:.1f}min')
        return (label, status, dt)
    finally:
        gpu_pool.put(gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force-nonlinear', action='store_true',
                    help='re-train the nonlinear arm instead of reusing seeded JSONs')
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    arms = dict(ARMS)
    if args.force_nonlinear:
        # remove pre-seeded nonlinear JSONs so they re-train
        for tag in ('S5', 'S3'):
            for s in SEEDS:
                f = EVAL_DIR / f'm2rnn-nonlinear_{tag}_seed{s}.json'
                if f.exists():
                    f.unlink()

    jobs = []
    for task, steps, tag in TASKS:           # longest (S5) first
        for arm, ls in arms.items():
            for seed in SEEDS:
                jobs.append((arm, ls, task, steps, tag, seed))

    log(f'{len(jobs)} jobs ({len(arms)} arms x {len(SEEDS)} seeds x {len(TASKS)} tasks); '
        f'GPUs {GPUS}')

    gpu_pool = queue.Queue()
    for g in GPUS:
        gpu_pool.put(g)

    results = []
    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        futs = [ex.submit(run_job, j, gpu_pool) for j in jobs]
        for f in as_completed(futs):
            results.append(f.result())

    ok = [r for r in results if r[1] == 'ok']
    skip = [r for r in results if r[1] == 'skipped']
    bad = [r for r in results if r[1] not in ('ok', 'skipped')]
    log(f'ALL DONE: {len(ok)} ok, {len(skip)} skipped, {len(bad)} failed')
    if bad:
        for label, status, _ in bad:
            log(f'  FAILED: {label} {status}')
        sys.exit(1)


if __name__ == '__main__':
    main()
