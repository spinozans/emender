#!/usr/bin/env python3
"""CONFIG-FLIP 2x2 orchestrator for task s5-config-flip.

Disentangles the E88 state-nonlinearity KNOB (linear_state) from the CMA-found
CONFIG (geometry/lr) by running the two missing cells of a 2x2 cross-flip.

The symmetric eval (task s5sym-eval) already produced two of the four cells:
  * A = config T + tanh   (linear_state=0)  == arm e88-tanh    [HAVE, do not rerun]
  * B = config L + linear (linear_state=1)  == arm e88-linear  [HAVE, do not rerun]
where
  * config T = e88-tanh   winner: dim256/depth5/H39/N32, lr=0.002949971704778276
  * config L = e88-linear winner: dim256/depth5/H38/N32, lr=0.0026571129141058

This driver runs the two FLIPS (the other diagonal):
  * C = config L + tanh   (linear_state=0)  -> arm e88-cfgL-tanh
  * D = config T + linear (linear_state=1)  -> arm e88-cfgT-linear

EVERYTHING else is identical to scripts/eval_s5_symmetric_winners.py so A/B/C/D
are directly comparable:
  * 3 seeds {42, 123, 456}
  * S5 : --task s5_permutation, train T=128, 20000 steps
  * S3 : --task s3_permutation, train T=128, 10000 steps (solvable control)
  * eval grid {128, 256, 512, 1024}, 8 eval batches per length
  * schedule-free AdamW, batch 32, seq_len 128, expansion 1.0, K 5
ONLY linear_state differs from the source config in each flip; nothing else is
re-tuned. The geometry (dim/depth/n_heads/n_state) and lr come VERBATIM from the
source winner args.json.

2 flips x {S5, S3} x 3 seeds = 12 REAL train_hybrid.py runs. NO mocks: every
accuracy is the real length_extrap field train_hybrid writes.

GPU SAFETY GATE: only GPUs reading <2GB used are used (round-robin); the idle
subset is detected at launch and logged to eval/logs/flip_gpus_used.txt.

Idempotent: a job whose output JSON already exists with a populated
`length_extrap` block (all four lengths) is skipped, so the orchestrator can be
re-launched to resume after an interruption.
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
EVAL_LENGTHS = ['128', '256', '512', '1024']

# (task_flag, train-length steps, short tag)  -- identical to the eval recipe
TASKS = [
    ('s5_permutation', 20000, 'S5'),
    ('s3_permutation', 10000, 'S3'),
]

# The two flips. Each names the SOURCE winner whose geometry/lr is reused
# verbatim, and the linear_state value to OVERRIDE it with.
#   arm name          source winner args      linear_state override
FLIPS = [
    # C = config L (e88-linear geometry/lr) + tanh state
    ('e88-cfgL-tanh',   'e88-linear', 0),
    # D = config T (e88-tanh geometry/lr) + linear state
    ('e88-cfgT-linear', 'e88-tanh',   1),
]

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        ts = time.strftime('%H:%M:%S')
        print(f'[{ts}] {msg}', flush=True)


def detect_idle_gpus(threshold_mib=2048):
    """Return the list of GPU indices reading < threshold MiB used."""
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used',
         '--format=csv,noheader,nounits']).decode()
    idle, busy = [], []
    for line in out.strip().splitlines():
        idx_s, mem_s = [x.strip() for x in line.split(',')]
        idx, mem = int(idx_s), int(mem_s)
        if mem < threshold_mib:
            idle.append(idx)
        else:
            busy.append((idx, mem))
    return idle, busy


def load_winner(arm):
    return json.load(open(WINNERS / f'{arm}.args.json'))


def build_cmd(arm, w, linear_state, task, steps, seed, label):
    """Identical to eval_s5_symmetric_winners.build_cmd, except linear_state is
    forced to the flip value rather than read from the source config."""
    p = w['params']
    model = w['model']  # always 'e88' for the flips
    assert model == 'e88', f'flip only defined for e88, got {model}'
    cmd = [
        sys.executable, str(TRAIN),
        '--task', task,
        '--layer_pattern', 'E88',
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
        # FLIP: only linear_state changes from the source config.
        '--linear_state', str(int(linear_state)),
        # use_gate is held identical to both source winners (both use_gate=1).
        '--use_gate', str(int(p['use_gate'])),
    ]
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
    arm, w, linear_state, task, steps, tag, seed = job
    label = f'{arm}_{tag}_seed{seed}'
    if already_done(label):
        log(f'SKIP  {label} (already complete)')
        return (label, 'skipped', None)
    gpu = gpu_pool.get()
    try:
        cmd = build_cmd(arm, w, linear_state, task, steps, seed, label)
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        logf = LOG_DIR / f'{label}.log'
        t0 = time.time()
        log(f'START {label}  gpu={gpu}  steps={steps}  linear_state={linear_state}')
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

    # --- GPU SAFETY GATE -------------------------------------------------
    idle, busy = detect_idle_gpus()
    for idx, mem in busy:
        log(f'GPU_BUSY: GPU {idx} has {mem}MiB used (>=2048MiB) -- excluded.')
    if not idle:
        log('FLIP_ABORTED: no idle GPU (<2GB) detected. No training launched.')
        sys.exit(2)
    gate_msg = (f'GPU_GATE_PASS: idle GPUs = {idle}. '
                f'Round-robin across {len(idle)} GPU(s) for the 12 flip runs.')
    log(gate_msg)
    with open(LOG_DIR / 'flip_gpus_used.txt', 'w') as f:
        f.write(gate_msg + '\n')

    # Source winners (geometry/lr come verbatim from these).
    src = {arm: load_winner(arm) for _, arm, _ in FLIPS}

    # Build job list. Longest jobs (S5, 20k) first so the tail is short jobs.
    jobs = []
    for task, steps, tag in TASKS:
        for arm, src_arm, linear_state in FLIPS:
            for seed in SEEDS:
                jobs.append((arm, src[src_arm], linear_state, task, steps, tag, seed))

    log(f'{len(jobs)} jobs total ({len(FLIPS)} flips x {len(SEEDS)} seeds x '
        f'{len(TASKS)} tasks); {len(idle)} idle GPUs')

    gpu_pool = queue.Queue()
    for g in idle:
        gpu_pool.put(g)

    results = []
    with ThreadPoolExecutor(max_workers=len(idle)) as ex:
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
