#!/usr/bin/env python3
"""Eigenvalue-causal-test orchestrator (task: eigenvalue-causal-test).

Trains+evaluates the two CAUSAL arms that FLIP the transition-operator
along-key eigenvalue sign, to test whether eigenvalue-sign is the S5 lever:

  ARM A  gdnNEG    : GDN winner config + allow_neg_eigval=True
                     (beta in (0,2) -> along-key eig g(1-beta) can go NEGATIVE).
                     PREDICT: S5 markedly IMPROVES vs GDN baseline 0.5446.
  ARM B  e88clamp  : e88-linear winner config + pos_eigval_clamp=True
                     (decay multiplies the WHOLE operator decay*(I-kk^T) so the
                     along-key eig becomes decay*(1-1)=0 >= 0, delta read-modify
                     -write preserved). PREDICT: S5 markedly DROPS vs 0.9997.
  ARM B2 e88raw    : (secondary, labeled) e88-linear + raw_write=True
                     (drop delta-correction; A_t=decay*I, all eigenvalues
                     decay>0). Positive-eigenvalue route. PREDICT: S5 DROPS.

Baselines (gdn 0.5446, e88-linear 0.9997 @ S5 T128) are CITED, NOT rerun.

Each arm: 3 seeds {42,123,456}; S5 (s5_permutation, 20000 steps) + S3 control
(s3_permutation, 10000 steps); eval grid {128,256,512,1024}. Real
train_hybrid.py runs, bf16, schedule-free AdamW, the EXACT winner shape/lr.

GPU policy: dynamically dispatch onto IDLE GPUs only (free mem < IDLE_MB).
Never preempt a busy GPU. Re-checks availability before every launch, so it
naturally scales as other experiments free GPUs. Idempotent: a finished job
(JSON with full length_extrap grid) is skipped, so it can resume.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'experiments/expressivity_tasks'
WINNERS = EXP / 'results/s5_symmetric_20260603/winners'
OUT = EXP / 'results/eigenvalue_causal_20260604'
EVAL_DIR = OUT / 'eval'
LOG_DIR = OUT / 'logs'
TRAIN = EXP / 'train_hybrid.py'

SEEDS = [42, 123, 456]
EVAL_LENGTHS = ['128', '256', '512', '1024']
IDLE_MB = 2000          # a GPU with < this many MiB used counts as idle
POLL_SECS = 30
CANDIDATE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

# (task_flag, steps, tag)
TASKS = [('s5_permutation', 20000, 'S5'), ('s3_permutation', 10000, 'S3')]

# arm -> (winner_file, extra train_hybrid flags)
ARMS = {
    'gdnNEG':   ('gdn',        ['--gdn_allow_neg_eigval', '1']),
    'e88clamp': ('e88-linear', ['--linear_state', '1', '--use_gate', '1',
                                '--e88_pos_eigval_clamp', '1']),
    'e88raw':   ('e88-linear', ['--linear_state', '1', '--use_gate', '1',
                                '--e88_raw_write', '1']),
}
LAYER = {'e88': 'E88', 'm2rnn': 'm2rnn', 'fla-gdn': 'fla-gdn'}

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_winner(arm_file):
    return json.load(open(WINNERS / f'{arm_file}.args.json'))


def build_cmd(arm, extra, task, steps, seed, label):
    w = load_winner(ARMS[arm][0])
    p = w['params']
    level = LAYER[w['model']]
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
    ] + extra
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


def gpu_free_mem():
    """Return {gpu_index: used_MiB} from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,noheader,nounits'], text=True)
    except Exception as e:
        log(f'nvidia-smi failed: {e}')
        return {}
    used = {}
    for line in out.strip().splitlines():
        idx, mem = [s.strip() for s in line.split(',')]
        used[int(idx)] = int(mem)
    return used


def run_job(job, gpu, results):
    arm, extra, task, steps, tag, seed, label = job
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
        dt = (time.time() - t0) / 60
        ok = already_done(label) and rc == 0
        status = 'ok' if ok else f'FAIL(rc={rc})'
        log(f'DONE  {label}  gpu={gpu}  {status}  {dt:.1f}min')
        with _lock:
            results[label] = status
    finally:
        with _lock:
            _claimed.discard(gpu)


_claimed = set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['gdnNEG', 'e88clamp'],
                    choices=list(ARMS.keys()),
                    help='Which arms to run. Default: core (gdnNEG e88clamp). '
                         'Add e88raw for the secondary positive-eigenvalue route.')
    ap.add_argument('--idle_mb', type=int, default=IDLE_MB)
    ap.add_argument('--gpus', type=int, nargs='+', default=CANDIDATE_GPUS)
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build job list. S5 (long) first so the tail is the short S3 jobs.
    jobs = []
    for task, steps, tag in TASKS:
        for arm in args.arms:
            extra = ARMS[arm][1]
            for seed in SEEDS:
                label = f'{arm}_{tag}_seed{seed}'
                jobs.append((arm, extra, task, steps, tag, seed, label))

    pending = [j for j in jobs if not already_done(j[-1])]
    skipped = [j for j in jobs if already_done(j[-1])]
    for j in skipped:
        log(f'SKIP  {j[-1]} (already complete)')
    log(f'{len(jobs)} jobs ({len(args.arms)} arms x {len(SEEDS)} seeds x '
        f'{len(TASKS)} tasks); {len(pending)} to run, {len(skipped)} done. '
        f'idle<{args.idle_mb}MiB GPUs from {args.gpus}')

    results = {}
    threads = []
    while pending or any(t.is_alive() for t in threads):
        used = gpu_free_mem()
        with _lock:
            idle = [g for g in args.gpus
                    if used.get(g, 10**9) < args.idle_mb and g not in _claimed]
        while pending and idle:
            gpu = idle.pop(0)
            job = pending.pop(0)
            with _lock:
                _claimed.add(gpu)
            t = threading.Thread(target=run_job, args=(job, gpu, results))
            t.start()
            threads.append(t)
            time.sleep(8)  # stagger launches; let mem register before re-poll
        time.sleep(POLL_SECS)

    for t in threads:
        t.join()

    ok = sum(1 for s in results.values() if s == 'ok')
    bad = {k: v for k, v in results.items() if v != 'ok'}
    log(f'ALL DONE: {ok} ok, {len(skipped)} skipped, {len(bad)} failed')
    for k, v in bad.items():
        log(f'  FAILED: {k} {v}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
