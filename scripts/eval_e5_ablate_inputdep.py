#!/usr/bin/env python3
"""E5 — Ablate the input-dependence confound on e88-linear (task e5-ablate-inputdep).

Tests the LEADING competing explanation for the S5 state-tracking win
(Cirone 2024; Grazzi 2025; Khavari 2025; DeltaProduct/Siems 2025): that a
linear-state model's state-tracking comes from INPUT-DEPENDENCE + eigenvalue
range, NOT state nonlinearity or rounding.

Baseline = the e88-linear symmetric-CMA winner (linear_state=1, use_gate=1,
decay_mode=mamba). Baseline numbers are REUSED verbatim from the committed
s5sym-eval run under results/s5_symmetric_20260603/eval/ (e88-linear_*).

This driver trains the two cleanly-implementable ablations of input-dependence:

  A) use_gate=0       -- remove the input-dependent OUTPUT gate (g_proj).
                         Per the task spec part (a).
  B) decay_mode=constant
                      -- remove the input-dependent TRANSITION (selectivity):
                         the per-step decay becomes a learned per-head CONSTANT
                         sigmoid(decay_logit) in (0,1) instead of the
                         input-dependent exp(-exp(A_log)*softplus(a_proj(x)+dt)).
                         This is the DIRECT test of input-dependence in the
                         recurrence and is param-matched to baseline.

Eigenvalue-range note (task part b): the E88 recurrence transition is
new_state = decay * S + outer, with decay = exp(g), g = -exp(A_log)*softplus(.)
< 0 (mamba) OR sigmoid(decay_logit) (constant) -- BOTH strictly in (0,1). The
code has NO path to negative or complex eigenvalues, so the eigenvalues are
ALREADY constrained to [0,1] by construction. Constraining further is a no-op;
allowing negative eigenvalues (the Grazzi/Khavari state-tracking regime) would
require a model code change and is documented as such in the writeup.

For EACH ablation arm:
  * 3 seeds {42, 123, 456}
  * S5  : --task s5_permutation, train T=128, 20000 steps
  * S3  : --task s3_permutation, train T=128, 10000 steps  (solvable control)
  * eval grid {128, 256, 512, 1024} (length-extrapolation, end of training)
  * schedule-free AdamW, batch 32, seq_len 128 (identical recipe to baseline)

12 REAL train_hybrid.py runs (2 arms x 3 seeds x {S5, S3}). NO mocks. ONLY
GPUs 4,5 (CUDA_VISIBLE_DEVICES set per job). Co-location on 4,5 is expected and
fine. Idempotent: a job whose output JSON already has a populated length_extrap
block for all four lengths is skipped, so the orchestrator can be re-launched.
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
RESULTS = REPO / 'experiments/expressivity_tasks/results/e5_ablate_inputdep_20260604'
EVAL_DIR = RESULTS / 'eval'
LOG_DIR = EVAL_DIR / 'logs'
TRAIN = REPO / 'experiments/expressivity_tasks/train_hybrid.py'

# e88-linear symmetric-CMA winner config (verbatim from
# results/s5_symmetric_20260603/winners/e88-linear.args.json).
WINNER = {
    'dim': 256,
    'depth': 5,
    'n_heads': 38,
    'n_state': 32,
    'lr': 0.0026571129141058,
    'linear_state': 1,
}

SEEDS = [42, 123, 456]
# ONLY GPUs 4,5. Three slots per GPU => 6 concurrent jobs (models are ~8M/~8GB
# on 48GB cards; co-residence with other agents' jobs is expected and fine).
GPU_SLOTS = [4, 4, 4, 5, 5, 5]
EVAL_LENGTHS = ['128', '256', '512', '1024']

# (task_flag, train-length steps, short tag)
TASKS = [
    ('s5_permutation', 20000, 'S5'),
    ('s3_permutation', 10000, 'S3'),
]

# Ablation arms: (arm-name, extra-flags forwarded to train_hybrid).
# All arms keep the winner shape/lr and linear_state=1; only the named knob moves.
ARMS = [
    # A: remove input-dependent output gate (task part a).
    ('use_gate0', ['--use_gate', '0']),
    # B: remove input-dependent transition (selectivity) -- learned constant decay.
    ('decay_const', ['--use_gate', '1', '--decay_mode', 'constant']),
]

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        ts = time.strftime('%H:%M:%S')
        print(f'[{ts}] {msg}', flush=True)


def build_cmd(arm, extra, task, steps, seed, label):
    cmd = [
        sys.executable, str(TRAIN),
        '--task', task,
        '--layer_pattern', 'E88',
        '--dim', str(WINNER['dim']),
        '--depth', str(WINNER['depth']),
        '--n_heads', str(WINNER['n_heads']),
        '--n_state', str(WINNER['n_state']),
        '--expansion', '1.0',
        '--lr', str(WINNER['lr']),
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
        '--linear_state', str(WINNER['linear_state']),
    ]
    cmd += extra
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
    return all(str(t) in le and 'acc' in le[str(t)] for t in EVAL_LENGTHS)


def run_job(job, gpu_pool):
    arm, extra, task, steps, tag, seed = job
    label = f'{arm}_{tag}_seed{seed}'
    if already_done(label):
        log(f'SKIP  {label} (already complete)')
        return (label, 'skipped', None)
    gpu = gpu_pool.get()
    try:
        cmd = build_cmd(arm, extra, task, steps, seed, label)
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

    # Longest jobs (S5, 20k) first so the tail is short S3 jobs.
    jobs = []
    for task, steps, tag in TASKS:
        for arm, extra in ARMS:
            for seed in SEEDS:
                jobs.append((arm, extra, task, steps, tag, seed))

    log(f'{len(jobs)} jobs total ({len(ARMS)} arms x {len(SEEDS)} seeds x '
        f'{len(TASKS)} tasks); GPU slots {GPU_SLOTS}')

    gpu_pool = queue.Queue()
    for g in GPU_SLOTS:
        gpu_pool.put(g)

    results = []
    with ThreadPoolExecutor(max_workers=len(GPU_SLOTS)) as ex:
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
