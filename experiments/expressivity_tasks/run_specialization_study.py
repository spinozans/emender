"""SPECIALIZATION-STUDY sweep runner.

Studies ways to FORCE heterogeneous per-head specialization (HORIZONTAL head-type
hybridization) in ONE unified cell, on the existing validated Triton kernel (no
kernel rewrite). Three families of approach, all on the same kernel:

  1. SPECIALIZATION-PRESSURE REGULARIZER (primary) on a GENERIC-init learned cell
     (`unified-learned-free`, the cell that did NOT self-specialize). Variants:
       pull           (a) pull-to-nearest-corner
       anticenter     (b) repel-center / reward-corner
       coverage       (c) population diversity (anti-collapse)
       pull_cov       (a)+(c)
       anticenter_cov (b)+(c)
     Weight swept in {0.1, 0.3, 1.0} (annealed in over the first half of training).
  2. TYPE-DICTIONARY: K shared learnable prototype knobs + per-head soft weights
     (`unified-dict4`, `unified-dict8`).
  3. FIXED-TYPE POPULATION (floor): heads hard-assigned to corners, projections
     only learn (`unified-fixedpop`).

Reference arms (generic-init learned, spread-init+knob-LR, LSTM, presets) are
REUSED from the prior unified-cell + unified-learnability sweeps (results/ dir:
`unified_*.json` and `learn_*.json`) -- not re-run here.

Probes: s5_permutation (track), anbncn_viability (count),
iterated_nonlinear_map (nonlin), flag_hold_recall (latch), mixed_probe (ALL).

Idle-GPU scheduling (used-mem < FREE_MEM_MIB; never preempt), resumable (skips
existing JSON). 3 seeds {42,123,456}, train T=128, eval {128,256,512,1024},
~8M params, fp32, schedule-free AdamW. Matches the prior sweeps' hyperparameters
so the reused baseline columns are comparable.
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

FREE_MEM_MIB = 2000  # a GPU is "idle" iff used memory < this (task rule: <2GB)

SHARED = ['--dim', '384', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

REG_VARIANTS = ['pull', 'anticenter', 'coverage', 'pull_cov', 'anticenter_cov']
REG_WEIGHTS = [0.1, 0.3, 1.0]

# Non-regularizer arms: (arm_name, layer_args, extra_train_args)
STRUCT_ARMS: dict[str, tuple[list[str], list[str]]] = {
    'unified-dict4':    (['--layer_pattern', 'unified-dict4', *SHARED], []),
    'unified-dict8':    (['--layer_pattern', 'unified-dict8', *SHARED], []),
    'unified-fixedpop': (['--layer_pattern', 'unified-fixedpop', *SHARED], []),
}

PROBES = {
    's5_permutation': [],
    'anbncn_viability': [],
    'iterated_nonlinear_map': [],
    'flag_hold_recall': ['--K', '4'],
    'mixed_probe': ['--K', '4'],
}

SEEDS = [42, 123, 456]


def reg_arm_name(variant: str, weight: float) -> str:
    w = ('%g' % weight).replace('.', 'p')
    return f'reg-{variant}-w{w}'


@dataclass
class Job:
    probe: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"spec_{self.probe}__{self.arm}__seed{self.seed}"


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
    if job.arm.startswith('reg-'):
        # regularizer arm: generic-init learned-free cell + train-time penalty
        _, variant, wtag = job.arm.split('-', 2)
        weight = float(wtag[1:].replace('p', '.'))
        layer_args = ['--layer_pattern', 'unified-learned-free', *SHARED]
        extra = ['--spec_reg', variant, '--spec_reg_weight', str(weight),
                 '--spec_reg_anneal', str(args.spec_reg_anneal)]
    else:
        layer_args, extra = STRUCT_ARMS[job.arm]
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.probe,
        *layer_args,
        *PROBES[job.probe],
        *extra,
        '--depth', str(args.depth),
        '--steps', str(args.steps),
        '--seq_len', '128',
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--optimizer', 'schedulefree',
        '--disable_autocast',
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', '128', '256', '512', '1024',
        '--eval_lengths_n_batches', str(args.eval_n_batches),
    ]
    return cmd


def all_arms() -> list[str]:
    arms = [reg_arm_name(v, w) for v, w in itertools.product(REG_VARIANTS, REG_WEIGHTS)]
    arms += list(STRUCT_ARMS.keys())
    return arms


def plan_jobs(args) -> list[Job]:
    jobs: list[Job] = []
    arms = args.arms or all_arms()
    for probe in args.probes:
        for arm, seed in itertools.product(arms, args.seeds):
            jobs.append(Job(probe, arm, seed))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', nargs='+', default=list(PROBES.keys()), choices=list(PROBES.keys()))
    ap.add_argument('--arms', nargs='+', default=None,
                    help='Subset of arm names (default: all reg + struct arms).')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--spec_reg_anneal', type=float, default=0.5)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--max_gpus', type=int, default=8)
    ap.add_argument('--output_dir', default=str(THIS / 'results'))
    ap.add_argument('--poll', type=float, default=15.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = plan_jobs(args)
    jobs = []
    for j in all_jobs:
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs (of {len(all_jobs)}); probes={args.probes} "
          f"seeds={args.seeds} steps={args.steps} arms={len(args.arms or all_arms())}", flush=True)

    running: dict[int, tuple[Job, subprocess.Popen, object]] = {}
    pending = list(jobs)

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
            for gpu in range(args.max_gpus):
                if not pending:
                    break
                if gpu in running:
                    continue
                if used.get(gpu, 10**9) >= FREE_MEM_MIB:
                    continue  # busy — never preempt
                job = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(out_dir / f'{job.label}.log', 'w')
                proc = subprocess.Popen(build_cmd(job, args, out_dir), cwd=str(ROOT),
                                        env=env, stdout=logf, stderr=subprocess.STDOUT)
                running[gpu] = (job, proc, logf)
                print(f"[run ] gpu{gpu} {job.label}", flush=True)
                time.sleep(3)

        time.sleep(args.poll)

    print("[complete] all jobs finished", flush=True)


if __name__ == '__main__':
    main()
