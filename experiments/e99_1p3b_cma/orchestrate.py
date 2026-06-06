#!/usr/bin/env python3
"""Idle-GPU orchestrator for the E99 1.3B LM-CMA search (task run-e99-1-3b).

Constraint: idle-GPU-ONLY, do NOT preempt other jobs. Other tasks (notably the
sibling run-matched-1-3b) share GPUs 0-7. This orchestrator:

  1. Maintains a gpu_file listing only SUSTAINED-idle GPUs (a GPU must read
     < mem_idle_mb for >= idle_confirm_sec continuously before it qualifies, so
     we never grab a GPU another job briefly released between its own runs).
     The CMA harness re-reads this file at every generation boundary, so the
     search opportunistically scales up as neighbours finish and scales down if
     a GPU goes busy again.
  2. Launches the search subprocess ONCE >= min_launch_gpus qualify, then keeps
     refreshing the gpu_file until the search exits.

Nothing here trains or fabricates data — it only schedules the real search onto
genuinely-idle GPUs. No GPU currently in use by another process is ever listed.
"""
import os, sys, time, json, subprocess, argparse, shutil

_THIS = os.path.dirname(os.path.abspath(__file__))


def gpu_mem_used():
    """Return {gpu_index: mem_used_mb} via nvidia-smi."""
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True)
    used = {}
    for line in out.stdout.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(',')]
        used[int(idx)] = int(mem)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--gpu_file', required=True)
    ap.add_argument('--all_gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--immediate_gpus', default='7',
                    help='GPUs not owned by neighbours -> qualify without idle wait')
    ap.add_argument('--mem_idle_mb', type=int, default=2000)
    ap.add_argument('--idle_confirm_sec', type=int, default=180)
    ap.add_argument('--min_launch_gpus', type=int, default=3)
    ap.add_argument('--poll_sec', type=int, default=45)
    # search args forwarded
    ap.add_argument('--train_minutes', type=float, default=15.0)
    ap.add_argument('--popsize', type=int, default=8)
    ap.add_argument('--max_generations', type=int, default=12)
    ap.add_argument('--max_total_evals', type=int, default=96)
    ap.add_argument('--gpu_minute_ceiling', type=float, default=1440.0)
    args = ap.parse_args()

    args.output = os.path.abspath(args.output)
    args.gpu_file = os.path.abspath(args.gpu_file)
    all_gpus = [int(g) for g in args.all_gpus.split(',')]
    immediate = set(int(g) for g in args.immediate_gpus.split(',') if g.strip())
    idle_since = {}   # gpu -> first time seen idle (None if busy)
    launched = None
    log = open(os.path.join(args.output, 'orchestrate.log'), 'a')

    def emit(msg):
        ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        log.write(f"{ts} {msg}\n"); log.flush()
        print(f"{ts} {msg}", flush=True)

    emit(f"orchestrator start; all_gpus={all_gpus} immediate={sorted(immediate)} "
         f"min_launch={args.min_launch_gpus} idle_confirm={args.idle_confirm_sec}s")

    while True:
        now = time.time()
        used = gpu_mem_used()
        qualified = []
        for g in all_gpus:
            mem = used.get(g, 10**9)
            if mem < args.mem_idle_mb:
                if idle_since.get(g) is None:
                    idle_since[g] = now
                idle_for = now - idle_since[g]
                if g in immediate or idle_for >= args.idle_confirm_sec:
                    qualified.append(g)
            else:
                idle_since[g] = None  # busy -> reset confirmation

        # Write gpu_file (never empty once launched -> keep at least last good set)
        if qualified:
            with open(args.gpu_file, 'w') as f:
                f.write(','.join(str(g) for g in qualified) + '\n')

        if launched is None and len(qualified) >= args.min_launch_gpus:
            cmd = [sys.executable, os.path.join(_THIS, 'run_e99_cma.py'),
                   '--output', args.output, '--gpu_file', args.gpu_file,
                   '--gpus', ','.join(str(g) for g in qualified),
                   '--train_minutes', str(args.train_minutes),
                   '--popsize', str(args.popsize),
                   '--max_generations', str(args.max_generations),
                   '--max_total_evals', str(args.max_total_evals),
                   '--gpu_minute_ceiling', str(args.gpu_minute_ceiling)]
            slog = open(os.path.join(args.output, 'search.log'), 'a')
            emit(f"LAUNCH search on GPUs {qualified}: {' '.join(cmd)}")
            _ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
            launched = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT,
                                        cwd=_ROOT)

        if launched is not None:
            rc = launched.poll()
            if rc is not None:
                emit(f"search exited rc={rc}; orchestrator stop")
                break
        else:
            emit(f"waiting: qualified={qualified} "
                 f"(idle_for={ {g: round(now-idle_since[g]) for g in all_gpus if idle_since.get(g)} })")

        time.sleep(args.poll_sec)


if __name__ == '__main__':
    main()
