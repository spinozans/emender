#!/usr/bin/env python3
"""Orchestrate the E99 1.3B matched controls across idle GPUs (run-matched-1-3b).

Runs 3 arms x N seeds, ONE job per GPU at a time (a 1.3B fp32 job peaks ~44 GB,
so never co-locate two on a 48 GB card). Enforces the aggregate GPU-hour ceiling
from experiments/e99_1p3b_sanity/budget_caps.json BEFORE launching; HARD-STOPS
and logs if the projected aggregate would exceed it. Each child enforces its own
per-job walltime hard-stop (e99_lm_controls.py --walltime_safety).
"""
import os, sys, json, time, subprocess, argparse, datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(_THIS, 'e99_lm_controls.py')
RESULTS = os.path.join(_THIS, 'results')

# Aggregate ceiling from budget_caps.json: matched_controls aggregate 5.25 GPU-h.
AGG_CEILING_GPU_MIN = 315.0


def log(msg):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f'[{ts}] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='fla-gdn,e98-cma-lm,typed-gdn2-lm')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--train_minutes', type=float, default=15.0)
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--outer_timeout_min', type=float, default=22.0)
    args = ap.parse_args()

    arms = args.arms.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    gpus = args.gpus.split(',')
    os.makedirs(RESULTS, exist_ok=True)

    jobs = [(a, s) for a in arms for s in seeds]
    n_jobs = len(jobs)
    proj_gpu_min = n_jobs * args.train_minutes
    log(f'{n_jobs} jobs ({len(arms)} arms x {len(seeds)} seeds) x '
        f'{args.train_minutes} min = {proj_gpu_min:.0f} projected GPU-min '
        f'(ceiling {AGG_CEILING_GPU_MIN:.0f}).')
    if proj_gpu_min > AGG_CEILING_GPU_MIN:
        log(f'HARD STOP (pre-launch): projected {proj_gpu_min:.0f} GPU-min > '
            f'ceiling {AGG_CEILING_GPU_MIN:.0f}. Reduce seeds/arms. Aborting.')
        sys.exit(2)

    free_gpus = list(gpus)
    running = {}   # gpu -> (proc, arm, seed, t0)
    queue = list(jobs)
    done = []
    spent_gpu_min = 0.0
    outer_to_s = args.outer_timeout_min * 60.0

    while queue or running:
        # launch while GPUs free and budget remains
        while queue and free_gpus:
            if spent_gpu_min + args.train_minutes > AGG_CEILING_GPU_MIN + 1e-6:
                log(f'HARD STOP: launching next job would exceed ceiling '
                    f'({spent_gpu_min:.0f}+{args.train_minutes:.0f} > '
                    f'{AGG_CEILING_GPU_MIN:.0f} GPU-min). {len(queue)} jobs '
                    f'left UNRUN, logged.')
                queue = []
                break
            arm, seed = queue.pop(0)
            gpu = free_gpus.pop(0)
            logf = os.path.join(RESULTS, f'{arm}_s{seed}.log')
            fh = open(logf, 'w')
            proc = subprocess.Popen(
                [sys.executable, DRIVER, '--config', arm, '--seed', str(seed),
                 '--gpu', gpu, '--train_minutes', str(args.train_minutes),
                 '--outdir', RESULTS],
                stdout=fh, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu})
            running[gpu] = (proc, arm, seed, time.time(), fh)
            log(f'LAUNCH {arm} seed {seed} on GPU {gpu} (pid {proc.pid}).')

        time.sleep(10)

        # reap finished / kill overrun
        for gpu in list(running.keys()):
            proc, arm, seed, t0, fh = running[gpu]
            elapsed = time.time() - t0
            rc = proc.poll()
            if rc is None and elapsed > outer_to_s:
                proc.kill()
                log(f'HARD STOP {arm} seed {seed} GPU {gpu}: outer timeout '
                    f'{elapsed:.0f}s > {outer_to_s:.0f}s -> killed.')
                rc = -9
            if rc is not None:
                fh.close()
                used_min = (time.time() - t0) / 60.0
                spent_gpu_min += min(used_min, args.train_minutes * 1.2)
                done.append((arm, seed, rc, round(used_min, 1)))
                log(f'DONE {arm} seed {seed} GPU {gpu} rc={rc} '
                    f'({used_min:.1f} min). spent~{spent_gpu_min:.0f} GPU-min.')
                del running[gpu]
                free_gpus.append(gpu)

    log(f'ALL DONE. {len(done)} jobs. ~{spent_gpu_min:.0f} GPU-min '
        f'(~{spent_gpu_min/60.0:.2f} GPU-h) of {AGG_CEILING_GPU_MIN/60.0:.2f} '
        f'GPU-h ceiling.')
    summary = dict(jobs=[dict(arm=a, seed=s, rc=rc, minutes=m)
                         for a, s, rc, m in done],
                   spent_gpu_min=round(spent_gpu_min, 1),
                   ceiling_gpu_min=AGG_CEILING_GPU_MIN,
                   train_minutes=args.train_minutes,
                   timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with open(os.path.join(RESULTS, 'orchestrator_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
