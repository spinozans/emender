"""PROBE 1 — counting-with-comparison suite runner.

Runs the linear-vs-nonlinear counting probe across all arms, 3 seeds, training
at T=128 and evaluating length extrapolation at T in {128,256,512,1024}.

Arms (param-matched ~8M, same recipe across all):
  e88-linear  : E88, linear_state=1  (eigenvalue-rich LINEAR state)
  e88-tanh    : E88, linear_state=0  (saturating NONLINEAR state; WGY false-neg)
  gdn         : fla-gdn              (gated delta-net; linear state)
  m2rnn       : m2rnn                (matrix-memory RNN)
  relu_rnn    : ReLU-Elman RNN       (additive counter; WGY POSITIVE control)
  lstm        : LSTM                 (gated additive counter; WGY POSITIVE control)

Dynamic GPU scheduling: starts on the GPUs that are free now (2,3 per the task)
and opportunistically grabs any OTHER GPU whose used-memory is < FREE_MEM_MIB as
running experiments finish. Never preempts a busy GPU; re-checks before each
launch. Skips jobs whose output JSON already exists (safe to resume).
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]

FREE_MEM_MIB = 2000  # a GPU is "idle" iff used memory < this

# Arm name -> model-shape CLI args (param-matched ~8M).
ARMS = {
    'e88-linear': ['--layer_pattern', 'E88', '--dim', '384', '--n_heads', '32',
                   '--n_state', '32', '--linear_state', '1'],
    'e88-tanh':   ['--layer_pattern', 'E88', '--dim', '384', '--n_heads', '32',
                   '--n_state', '32', '--linear_state', '0'],
    'gdn':        ['--layer_pattern', 'fla-gdn', '--dim', '640', '--n_heads', '32',
                   '--n_state', '32'],
    'm2rnn':      ['--layer_pattern', 'm2rnn', '--dim', '384', '--n_heads', '32',
                   '--n_state', '32'],
    'relu_rnn':   ['--layer_pattern', 'relu_rnn', '--dim', '704', '--expansion', '1.0'],
    'lstm':       ['--layer_pattern', 'lstm', '--dim', '448', '--expansion', '1.0'],
}

# Task -> per-task CLI knobs (K is forwarded only if the task reads it; these
# counting tasks ignore K, so we leave it default).
TASKS = {
    'dyck_depth': {},
    'anbncn_viability': {},
}

SEEDS = [42, 123, 456]


@dataclass
class Job:
    task: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"probe1_{self.task}__{self.arm}__seed{self.seed}"


def gpu_used_mib() -> dict[int, int]:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def build_cmd(job: Job, args, out_dir: Path) -> list[str]:
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.task,
        *ARMS[job.arm],
        '--depth', str(args.depth),
        '--steps', str(args.steps),
        '--seq_len', '128',
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--optimizer', 'schedulefree',
        '--disable_autocast',           # fp32: exact counting + fair across arms
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', '128', '256', '512', '1024',
        '--eval_lengths_n_batches', str(args.eval_n_batches),
    ]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['dyck_depth'],
                    choices=list(TASKS.keys()))
    ap.add_argument('--arms', nargs='+', default=list(ARMS.keys()))
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--start_gpus', type=int, nargs='+', default=[2, 3],
                    help='GPUs to use immediately (must be idle now).')
    ap.add_argument('--max_gpus', type=int, default=8,
                    help='Upper bound on GPU index to ever consider grabbing.')
    ap.add_argument('--output_dir', default=str(THIS / 'results'))
    ap.add_argument('--poll', type=float, default=20.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = []
    for task, arm, seed in itertools.product(args.tasks, args.arms, args.seeds):
        j = Job(task, arm, seed)
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs across arms={args.arms} tasks={args.tasks} "
          f"seeds={args.seeds} steps={args.steps}", flush=True)

    running: dict[int, tuple[Job, subprocess.Popen, object]] = {}
    pending = list(jobs)
    allowed = set(args.start_gpus)  # grows as other GPUs become idle

    while pending or running:
        # Reap finished jobs.
        for gpu in list(running):
            job, proc, logf = running[gpu]
            if proc.poll() is not None:
                logf.close()
                status = 'ok' if proc.returncode == 0 else f'FAIL({proc.returncode})'
                print(f"[done] gpu{gpu} {job.label} -> {status}", flush=True)
                del running[gpu]

        if pending:
            used = gpu_used_mib()
            # Any GPU that is currently idle is allowed (starts with start_gpus).
            for g, m in used.items():
                if g < args.max_gpus and m < FREE_MEM_MIB:
                    allowed.add(g)
            busy_by_us = set(running)
            for gpu in sorted(allowed):
                if not pending:
                    break
                if gpu in busy_by_us:
                    continue
                if used.get(gpu, 10**9) >= FREE_MEM_MIB:
                    continue  # someone else is using it now — never preempt
                job = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(out_dir / f'{job.label}.log', 'w')
                cmd = build_cmd(job, args, out_dir)
                proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                        stdout=logf, stderr=subprocess.STDOUT)
                running[gpu] = (job, proc, logf)
                print(f"[run ] gpu{gpu} {job.label}", flush=True)
                time.sleep(3)  # let it claim memory before re-polling

        time.sleep(args.poll)

    print("[complete] all jobs finished", flush=True)


if __name__ == '__main__':
    main()
