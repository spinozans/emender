#!/usr/bin/env python3
"""Full winner-eval orchestrator for task s5sym-eval.

Runs the FOUR symmetric-CMA winners (e88-tanh, e88-linear, m2rnn, gdn) at full
fidelity as the current paper-§6 state-tracking probe:

  * 3 seeds {42, 123, 456}
  * S5  : --task s5_permutation, train T=128, 20000 steps
  * S3  : --task s3_permutation, train T=128, 10000 steps  (solvable control)
  * eval grid {128, 256, 512, 1024} (length-extrapolation, end of training)
  * schedule-free AdamW, batch 32, seq_len 128 (identical to the search arms)

24 REAL train_hybrid.py runs (4 arms x 3 seeds x {S5, S3}), round-robin across
GPUs 0-7. NO mocks: every accuracy is the real eval_acc / length_extrap field
train_hybrid writes. The per-arm shape/lr come verbatim from the CMA winners
under results/s5_symmetric_20260603/winners/*.args.json. The E88 BL-1 structural
knobs (linear_state, use_gate) are forwarded exactly as fixed in the search.

Idempotent: a job whose output JSON already exists with a populated
`length_extrap` block is skipped, so the orchestrator can be re-launched to
resume after an interruption.
"""
import json
import os
import subprocess
import sys
import threading
import queue
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / 'experiments/expressivity_tasks/results/s5_symmetric_20260603'
WINNERS = RESULTS / 'winners'
EVAL_DIR = RESULTS / 'eval'
LOG_DIR = EVAL_DIR / 'logs'
TRAIN = REPO / 'experiments/expressivity_tasks/train_hybrid.py'

SEEDS = [42, 123, 456]
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
LAYER = {'e88': 'E88', 'm2rnn': 'm2rnn', 'fla-gdn': 'fla-gdn'}
ARMS = ['e88-tanh', 'e88-linear', 'm2rnn', 'gdn']
EVAL_LENGTHS = ['128', '256', '512', '1024']

# (task_flag, train-length steps, short tag)
TASKS = [
    ('s5_permutation', 20000, 'S5'),
    ('s3_permutation', 10000, 'S3'),
]

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        ts = time.strftime('%H:%M:%S')
        print(f'[{ts}] {msg}', flush=True)


def load_winner(arm):
    return json.load(open(WINNERS / f'{arm}.args.json'))


def build_cmd(arm, w, task, steps, seed, label):
    p = w['params']
    model = w['model']  # e88 / m2rnn / fla-gdn
    level = LAYER[model]
    cmd = [
        sys.executable, str(TRAIN),
        '--task', task,
        '--layer_pattern', level,
        '--dim', str(int(p['dim'])),
        '--depth', str(int(p['depth'])),
        '--n_heads', str(int(p['n_heads'])),
        '--n_state', str(int(p['n_state'])),
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
    ]
    if model == 'e88':
        if 'linear_state' in p:
            cmd += ['--linear_state', str(int(p['linear_state']))]
        if 'use_gate' in p:
            cmd += ['--use_gate', str(int(p['use_gate']))]
    return cmd


def already_done(label):
    out = EVAL_DIR / f'{label}.json'
    if not out.exists():
        return False
    try:
        d = json.load(open(out))
    except Exception:
        return False
    le = d.get('length_extrap') or {}
    # require all four lengths present with a numeric acc
    return all(str(t) in le and 'acc' in le[str(t)] for t in EVAL_LENGTHS)


def run_job(job, gpu_pool):
    arm, w, task, steps, tag, seed = job
    label = f'{arm}_{tag}_seed{seed}'
    if already_done(label):
        log(f'SKIP  {label} (already complete)')
        return (label, 'skipped', None)
    gpu = gpu_pool.get()
    try:
        cmd = build_cmd(arm, w, task, steps, seed, label)
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
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    winners = {arm: load_winner(arm) for arm in ARMS}

    # Build job list. Longest jobs (S5, 20k) first so the tail is short jobs.
    jobs = []
    for task, steps, tag in TASKS:
        for arm in ARMS:
            for seed in SEEDS:
                jobs.append((arm, winners[arm], task, steps, tag, seed))

    log(f'{len(jobs)} jobs total ({len(ARMS)} arms x {len(SEEDS)} seeds x '
        f'{len(TASKS)} tasks); {len(GPUS)} GPUs')

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
