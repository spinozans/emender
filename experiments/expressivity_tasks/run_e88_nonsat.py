"""E88 NON-SATURATING STATE runner (task: e88-nonsat).

Tests the hypothesis that replacing E88's saturating ``tanh`` state with a
NON-SATURATING nonlinearity (relu / softplus) gives E88 a genuine unbounded
counter, while keeping its width-efficient design.

Arms (param-matched; the four E88 arms differ ONLY in --state_activation, so the
state nonlinearity is the single isolated variable):
  e88-tanh      E88 + tanh      saturating nonlinear  (DEFAULT; bounded |S|)
  e88-linear    E88 + identity  affine / linear state
  e88-relu      E88 + relu      NON-SATURATING (clamp <0, unbounded >0)  [NEW]
  e88-softplus  E88 + softplus  NON-SATURATING smooth (unbounded >0)     [NEW]
  lstm          LSTM            additive non-saturating counter reference (WGY+)

Task families:
  COUNTING : anbncn_viability (1a, unbounded a^n b^n c^n) and dyck_depth (1b)
             -- reuses the probe1-counting tasks. The probe1 JSONs for the
             SHARED arms (e88-tanh/e88-linear/lstm) are reused as-is; here we
             only run the NEW non-saturating arms on counting.
  FINITE-STATE : s5_permutation and s3_permutation -- run for ALL arms.

3 seeds {42,123,456}. Train T=128. Eval grid T in {128,256,512,1024}.

Dynamic idle-GPU scheduler (copied from run_probe1_counting): starts on the GPUs
that are free now and opportunistically grabs any OTHER GPU whose used memory is
< FREE_MEM_MIB as running jobs finish. Never preempts a busy GPU; re-checks
before each launch. Skips jobs whose output JSON already exists (safe to resume).
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]

FREE_MEM_MIB = 2000  # a GPU is "idle" iff used memory < this

# E88 shape is param-matched to probe1 (~7.9M); the four E88 arms share it and
# differ only in the state nonlinearity. lstm reference matches probe1 (~8.05M).
E88_SHAPE = ['--layer_pattern', 'E88', '--dim', '384', '--n_heads', '32',
             '--n_state', '32']
ARMS = {
    'e88-tanh':     [*E88_SHAPE, '--state_activation', 'tanh'],
    'e88-linear':   [*E88_SHAPE, '--state_activation', 'identity'],
    'e88-relu':     [*E88_SHAPE, '--state_activation', 'relu'],
    'e88-softplus': [*E88_SHAPE, '--state_activation', 'softplus'],
    'lstm':         ['--layer_pattern', 'lstm', '--dim', '448', '--expansion', '1.0'],
}

# The two NEW non-saturating E88 arms; the others already have probe1 JSONs for
# the counting tasks and are only (re)run for S5/S3.
NONSAT_ARMS = ['e88-relu', 'e88-softplus']

# Per-task training budget + eval config. Counting matches probe1 exactly (so the
# new arms are directly comparable to the committed probe1 JSONs). S5/S3 are
# harder finite-state word problems and get a larger step budget.
TASKS = {
    'anbncn_viability': dict(steps=3000, eval_n_batches=8, arms='nonsat'),
    'dyck_depth':       dict(steps=3000, eval_n_batches=8, arms='nonsat'),
    's5_permutation':   dict(steps=10000, eval_n_batches=4, arms='all'),
    's3_permutation':   dict(steps=6000, eval_n_batches=4, arms='all'),
}

SEEDS = [42, 123, 456]
EVAL_LENGTHS = ['128', '256', '512', '1024']


@dataclass
class Job:
    task: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"nonsat_{self.task}__{self.arm}__seed{self.seed}"


def gpu_used_mib() -> dict:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def build_cmd(job: Job, args, out_dir: Path) -> list:
    cfg = TASKS[job.task]
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.task,
        *ARMS[job.arm],
        '--depth', str(args.depth),
        '--steps', str(cfg['steps']),
        '--seq_len', '128',
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--optimizer', 'schedulefree',
        '--disable_autocast',           # fp32: exact counting + fair across arms
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', *EVAL_LENGTHS,
        '--eval_lengths_n_batches', str(cfg['eval_n_batches']),
    ]
    return cmd


def plan_jobs(args, out_dir: Path) -> list:
    jobs = []
    for task in args.tasks:
        cfg = TASKS[task]
        arms = NONSAT_ARMS if cfg['arms'] == 'nonsat' else list(ARMS.keys())
        arms = [a for a in arms if a in args.arms] if args.arms else arms
        for arm, seed in itertools.product(arms, args.seeds):
            j = Job(task, arm, seed)
            if (out_dir / f'{j.label}.json').exists():
                print(f"[skip] {j.label} (exists)", flush=True)
                continue
            jobs.append(j)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS.keys()),
                    choices=list(TASKS.keys()))
    ap.add_argument('--arms', nargs='+', default=None,
                    help='Restrict to these arms (default: per-task arm set).')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--start_gpus', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                    help='GPUs to use immediately (must be idle now).')
    ap.add_argument('--max_gpus', type=int, default=8,
                    help='Upper bound on GPU index to ever consider grabbing.')
    ap.add_argument('--output_dir', default=str(THIS / 'results'))
    ap.add_argument('--poll', type=float, default=20.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = plan_jobs(args, out_dir)
    print(f"[plan] {len(jobs)} jobs across tasks={args.tasks} seeds={args.seeds}",
          flush=True)
    for j in jobs:
        print(f"   - {j.label} (steps={TASKS[j.task]['steps']})", flush=True)

    running: dict = {}
    pending = list(jobs)
    allowed = set(args.start_gpus)

    while pending or running:
        for gpu in list(running):
            job, proc, logf = running[gpu]
            if proc.poll() is not None:
                logf.close()
                status = 'ok' if proc.returncode == 0 else f'FAIL({proc.returncode})'
                print(f"[done] gpu{gpu} {job.label} -> {status}", flush=True)
                del running[gpu]

        if pending:
            used = gpu_used_mib()
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
                time.sleep(3)

        time.sleep(args.poll)

    print("[complete] all jobs finished", flush=True)


if __name__ == '__main__':
    main()
