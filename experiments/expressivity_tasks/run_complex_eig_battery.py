"""Complex-eigenvalue capability battery (task complex-eig-capability).

THE question: does the COMPLEX eigenvalue (rotation lambda = r*e^{i*theta}) unlock
a periodic / mod-k / positional capability that the REAL-eigenvalue cell
(positive+negative real lambda) cannot reach? REAL tasks, REAL training.

Matched A/B (identical params/kernel/compute — built from the SAME
ComplexEigHeadLayer, fp32):
  complex  : cplx_real_only=0  -> phase free (rotation enabled)
  real     : cplx_real_only=1  -> theta snapped to {0,pi} & frozen (rotation removed)
Plus a production anchor:
  gdnneg   : fla-gdn allow_neg_eigval=1 -> the real positive+negative head in prod

Two batteries:
  (1) PERIODIC battery  (frac=0, pure eigenvalue-type axis):
      positional_clock K={6,12}   native mod-K clock (filler input, track phase)
      periodic_pattern K=6        repeating-motif next-symbol prediction
      modular_counter  K=5        mod-k running counter
      parity                      mod-2 CONTROL (real reflection should suffice)
  (2) COEXISTENCE (criterion 2): modular_quadratic length-extrapolation with the
      hardtanh subset ON *and* complex transitions ON — do the two axes compose?
        cplx_htanh : complex + hardtanh subset frac=0.25  (step-growth + rotation)
        cplx_lin   : complex,  no bounded subset           (rotation only)
        real_htanh : real-only + hardtanh subset frac=0.25 (step-growth, no rotation)

All arms: dim 256, 32 heads, n_state 32, depth 4, mlp_ratio 2.0 (fixed O(depth)
nonlinear readout present in BOTH arms), schedule-free AdamW, fp32 (disable_autocast
— exact algorithmic tasks; complex ops run in complex64 regardless). Train T=128,
eval length-extrapolation. 3 seeds. Idle-GPU scheduling; resumable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]

GPUS = [0, 1, 2, 3, 4, 6]    # idle GPUs (GPU 7 left free per constraint; 5 partly used)
SLOTS_PER_GPU = 2            # eager scan holds a T-step autograd graph -> heavier
MEM_CAP_MIB = 40000

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0',
          '--mlp_ratio', '2.0', '--depth', '4', '--disable_autocast',
          '--gdn_allow_neg_eigval', '1', '--optimizer', 'schedulefree']

# arm -> (layer level, extra flags). complex/real are the matched A/B (same layer);
# gdnneg is the production real positive+negative anchor.
# complex/real route through the STABLE eager scan (--cplx_force_sequential 1):
# the chunked kernel overflows fp32 (1/cp decay-folding) when |lambda| is driven
# small within a chunk -> NaN, which corrupts BOTH arms mid-training. chunked<->eager
# parity is validated to ~4e-7 (complex-eig-validate), so the eager path measures the
# same capability without the overflow artifact. gdnneg uses FLA's stable kernel.
ARMS = {
    'complex': ('complex-eig', ['--cplx_real_only', '0', '--cplx_force_sequential', '1']),
    'real':    ('complex-eig', ['--cplx_real_only', '1', '--cplx_force_sequential', '1']),
    'gdnneg':  ('fla-gdn', []),
}

# periodic battery: task -> K (0 = task default / no K)
BATTERY = [
    ('positional_clock', 6),
    ('positional_clock', 12),
    ('periodic_pattern', 6),
    ('modular_counter', 5),
    ('parity', 0),
]
BATTERY_EVAL = ['128', '256', '512']   # eager scan: long-T eval is costly; 3 lengths suffice for extrap

# coexistence: modular_quadratic, three head configs (all complex-eig layer)
COEXIST = {
    'cplx_htanh': ['--cplx_real_only', '0', '--cplx_force_sequential', '1',
                   '--cplx_nonlin_subset_frac', '0.25', '--cplx_nonlin_subset_phi', 'hardtanh'],
    'cplx_lin':   ['--cplx_real_only', '0', '--cplx_force_sequential', '1',
                   '--cplx_nonlin_subset_frac', '0.0'],
    'real_htanh': ['--cplx_real_only', '1', '--cplx_force_sequential', '1',
                   '--cplx_nonlin_subset_frac', '0.25', '--cplx_nonlin_subset_phi', 'hardtanh'],
}
COEXIST_EVAL = ['128', '256', '512', '1024', '2048']

SEEDS = [42, 123, 456]


@dataclass
class Job:
    task: str
    arm: str
    seed: int
    K: int = 0
    extra: list = field(default_factory=list)
    level: str = 'complex-eig'
    eval_lengths: list = field(default_factory=lambda: list(BATTERY_EVAL))
    tag: str = ''

    @property
    def label(self) -> str:
        ktag = f"_K{self.K}" if self.K > 0 else ""
        return f"ce_{self.task}{ktag}__{self.arm}__seed{self.seed}"


def build_battery_jobs():
    jobs = []
    for task, K in BATTERY:
        for arm, (level, extra) in ARMS.items():
            for seed in SEEDS:
                jobs.append(Job(task=task, arm=arm, seed=seed, K=K,
                                extra=list(extra), level=level,
                                eval_lengths=list(BATTERY_EVAL)))
    return jobs


def build_coexist_jobs(K=0):
    jobs = []
    for cfg, extra in COEXIST.items():
        for seed in SEEDS:
            jobs.append(Job(task='modular_quadratic', arm=cfg, seed=seed, K=K,
                            extra=list(extra), level='complex-eig',
                            eval_lengths=list(COEXIST_EVAL)))
    return jobs


def gpu_used_mib():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def build_cmd(job: Job, args, out_dir: Path):
    pattern = [job.level] * args.depth
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.task,
        '--layer_pattern', *pattern,
        *SHARED,
        *job.extra,
        '--steps', str(args.steps),
        '--seq_len', '128',
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', *job.eval_lengths,
        '--eval_lengths_n_batches', str(args.eval_n_batches),
    ]
    if job.K > 0:
        cmd += ['--K', str(job.K)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['battery', 'coexist', 'both'], default='both')
    ap.add_argument('--coexist_K', type=int, default=0,
                    help='modulus p for the coexistence modular_quadratic task. 0=default '
                         'p=7 (easy, MLP solves it -> axes-dont-conflict check). Use 64 to '
                         'hit the step-growth length-extrapolation CLIFF (capgap regime).')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--output_dir', default=str(THIS / 'results_complex_eig'))
    ap.add_argument('--poll', type=float, default=10.0)
    args = ap.parse_args()

    global SEEDS
    if args.seeds:
        SEEDS = args.seeds

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = []
    if args.which in ('battery', 'both'):
        all_jobs += build_battery_jobs()
    if args.which in ('coexist', 'both'):
        all_jobs += build_coexist_jobs(K=args.coexist_K)

    jobs = []
    for j in all_jobs:
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs (of {len(all_jobs)}); which={args.which} "
          f"steps={args.steps} seeds={SEEDS} gpus={GPUS}", flush=True)

    running = []
    pending = list(jobs)
    env_base = dict(os.environ, PYTHONPATH=str(ROOT))

    while pending or running:
        still = []
        for gpu, job, proc, logf in running:
            if proc.poll() is not None:
                logf.close()
                status = 'ok' if proc.returncode == 0 else f'FAIL({proc.returncode})'
                print(f"[done] gpu{gpu} {job.label} -> {status}", flush=True)
            else:
                still.append((gpu, job, proc, logf))
        running = still

        if pending:
            used = gpu_used_mib()
            slots = {g: 0 for g in GPUS}
            for gpu, _, _, _ in running:
                slots[gpu] = slots.get(gpu, 0) + 1
            for gpu in GPUS:
                if not pending:
                    break
                if slots.get(gpu, 0) >= SLOTS_PER_GPU:
                    continue
                if used.get(gpu, 10**9) >= MEM_CAP_MIB:
                    continue
                job = pending.pop(0)
                env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(out_dir / f'{job.label}.log', 'w')
                proc = subprocess.Popen(build_cmd(job, args, out_dir), cwd=str(ROOT),
                                        env=env, stdout=logf, stderr=subprocess.STDOUT)
                running.append((gpu, job, proc, logf))
                slots[gpu] = slots.get(gpu, 0) + 1
                print(f"[run ] gpu{gpu} ({slots[gpu]}/{SLOTS_PER_GPU}) {job.label}", flush=True)
                time.sleep(3)

        time.sleep(args.poll)

    print("[complete] all jobs finished", flush=True)


if __name__ == '__main__':
    main()
