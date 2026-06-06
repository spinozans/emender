#!/usr/bin/env python3
"""E2 orchestrator: run the precision-sweep jobs across GPUs 2,3 ONLY.

Jobs = {train_dtype in (fp32, bf16)} x {seed in 42,123,456} = 6 runs.
Each job = e2_precision_sweep.py: train one e88-linear model at train_dtype,
then eval the SAME serial model at fp64/fp32/bf16 over T{128,256,512,1024}.

  * fp32-train rows = the EVAL-ONLY regime (the task's required fallback):
    fixed fp32 weights, vary only the eval rounding dtype.
  * bf16-train rows = the TRAIN-PER-PRECISION diagonal (bf16 matches the
    published §6 recipe). fp64 TRAINING is infeasible on RTX 6000 Ada
    (fp64 ~ 1/64 fp32 throughput) so it is reported eval-only; the
    fp32-train + fp64-eval cell is the idealized-linear proxy.

Idempotent: a completed output JSON (with a populated precision_sweep block)
is skipped, so the orchestrator can resume after interruption.
ONLY GPUs 2 and 3 are ever touched.
"""
import os, sys, json, time, subprocess, queue, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'experiments/expressivity_tasks/results/e2_precision_sweep_20260604')
LOGDIR = os.path.join(OUT, 'logs')
SWEEP = os.path.join(REPO, 'scripts/e2_precision_sweep.py')

GPUS = [2, 3]              # NON-NEGOTIABLE: only GPUs 2,3
SEEDS = [42, 123, 456]
TRAIN_DTYPES = ['fp32', 'bf16']
STEPS = 12000

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def done(path):
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path))
    except Exception:
        return False
    ps = d.get('precision_sweep') or {}
    return all(ed in ps and all(str(T) in ps[ed] and 'acc' in ps[ed][str(T)]
                                for T in (128, 256, 512, 1024))
               for ed in ('fp64', 'fp32', 'bf16'))


def run_job(job, pool):
    td, seed = job
    label = f'e88linear_train{td}_seed{seed}'
    out = os.path.join(OUT, f'{label}.json')
    if done(out):
        log(f'SKIP {label} (complete)')
        return (label, 'skipped')
    gpu = pool.get()
    try:
        cmd = [sys.executable, SWEEP, '--seed', str(seed), '--train_dtype', td,
               '--steps', str(STEPS), '--out', out,
               '--save_ckpt', os.path.join(OUT, f'{label}.pt')]
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        logf = os.path.join(LOGDIR, f'{label}.log')
        t0 = time.time()
        log(f'START {label} gpu={gpu} steps={STEPS}')
        with open(logf, 'w') as lf:
            lf.write('CMD: ' + ' '.join(cmd) + f'\nCUDA_VISIBLE_DEVICES={gpu}\n\n')
            lf.flush()
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=REPO, env=env)
        ok = (rc == 0) and done(out)
        log(f'DONE  {label} gpu={gpu} rc={rc} ok={ok} {(time.time()-t0)/60:.1f}min')
        return (label, 'ok' if ok else f'FAIL(rc={rc})')
    finally:
        pool.put(gpu)


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    jobs = [(td, s) for td in TRAIN_DTYPES for s in SEEDS]
    log(f'{len(jobs)} jobs on GPUs {GPUS}')
    pool = queue.Queue()
    for g in GPUS:
        pool.put(g)
    results = []
    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        futs = [ex.submit(run_job, j, pool) for j in jobs]
        for f in as_completed(futs):
            results.append(f.result())
    ok = [r for r in results if r[1] == 'ok']
    skip = [r for r in results if r[1] == 'skipped']
    bad = [r for r in results if r[1] not in ('ok', 'skipped')]
    log(f'ALL DONE: {len(ok)} ok, {len(skip)} skipped, {len(bad)} failed')
    for label, st in bad:
        log(f'  FAILED {label}: {st}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
