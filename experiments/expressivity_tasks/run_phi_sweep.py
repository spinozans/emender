"""Phi-exploration sweep (task phi-explore).

Sweep the PER-STEP state nonlinearity phi in

    S_t = phi( diag(g_t) S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T )

on the depth-growing capability battery, under LENGTH EXTRAPOLATION (train T=128,
test to 16x = T=2048), 3 seeds, with a SwiGLU MLP present in every arm. Every arm
is one PhiShellLayer stack differing ONLY in phi; phi='identity' is the linear
gdn-neg baseline realized in the exact same code path.

Primary cliff (where per-step tanh was proven to separate +0.18..0.21):
    modular_quadratic  K in {32,48,64}
Secondary (cost-signature controls):
    iterated_nonlinear_map   CONTRACTING logistic map  (bounded depth -> no gap expected)
    dyck_depth_unbounded     UNBOUNDED counting        (rectifying phi should help, saturating hurt)

Idle-GPU scheduling; resumable (skips existing JSON).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
SLOTS_PER_GPU = 4   # ~2.6 GB/job with gradient checkpointing -> 4 fit beside other agents
MEM_CAP_MIB = 38000

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

# The phi menu, grouped by cost signature (see paper/review/PHI_EXPLORATION_RESULTS.md).
PHIS = [
    'identity',     # LINEAR gdn-neg baseline (same code path)
    'tanh',         # bounded saturating (the proven per-step winner)
    'softsign',     # bounded saturating, ALGEBRAIC/rational (x/(1+|x|))
    'hardtanh',     # bounded saturating, piecewise-linear
    'poly3',        # bounded saturating, low-degree POLYNOMIAL
    'relu',         # rectifying, unbounded one-sided
    'softplus',     # smooth rectifier, unbounded one-sided
    'gelu',         # smooth gated, non-monotone near 0
    'silu',         # smooth gated (swish), non-monotone near 0
    'signed_sqrt',  # odd, compressive, UNBOUNDED magnitude (never saturates)
    'learned',      # small LEARNED elementwise scalar MLP (init = identity)
]

# task -> list of K values (0 = task default modulus / params)
TASKS = {
    'modular_quadratic': [32, 48, 64],
    'iterated_nonlinear_map': [0],
    'dyck_depth_unbounded': [0],
}

SEEDS = [42, 123, 456]


@dataclass
class Job:
    task: str
    phi: str
    seed: int
    K: int = 0

    @property
    def label(self) -> str:
        ktag = f"__K{self.K}" if self.K > 2 else ""
        return f"phi_{self.task}__{self.phi}{ktag}__seed{self.seed}"


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
    pattern = ['phi-shell'] * args.depth
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.task,
        '--layer_pattern', *pattern,
        *SHARED,
        '--phi', job.phi,
        '--split_edit', str(args.split_edit),
        '--gdn_allow_neg_eigval', '1',
        '--mlp_ratio', str(args.mlp_ratio),
        '--depth', str(args.depth),
        '--steps', str(args.steps),
        '--seq_len', '128',
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--optimizer', 'schedulefree',
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', '128', '256', '512', '1024', '2048',
        '--eval_lengths_n_batches', str(args.eval_n_batches),
        '--eval_interval', str(args.eval_interval),
    ]
    if job.K > 2:
        cmd += ['--K', str(job.K)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS.keys()))
    ap.add_argument('--mq_ks', type=int, nargs='+', default=None,
                    help='Override modular_quadratic modulus list (default 32 48 64).')
    ap.add_argument('--phis', nargs='+', default=PHIS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch_size', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--mlp_ratio', type=float, default=2.0)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--eval_interval', type=int, default=500)
    ap.add_argument('--split_edit', type=int, default=0, choices=[0, 1],
                    help='Use the E97 SPLIT-EDIT recurrence substrate (default 0 = '
                         'plain gated-delta). Split-edit is the substrate proven to '
                         'separate on the cliff.')
    ap.add_argument('--output_dir', default=str(THIS / 'results_phi'))
    ap.add_argument('--gpus', type=int, nargs='+', default=GPUS)
    ap.add_argument('--slots_per_gpu', type=int, default=SLOTS_PER_GPU)
    ap.add_argument('--poll', type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = []
    for t in args.tasks:
        ks = TASKS.get(t, [0])
        if t == 'modular_quadratic' and args.mq_ks is not None:
            ks = args.mq_ks
        for k in ks:
            for phi in args.phis:
                for s in args.seeds:
                    all_jobs.append(Job(t, phi, s, k))
    jobs = []
    for j in all_jobs:
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs (of {len(all_jobs)}); tasks={args.tasks} "
          f"phis={args.phis} seeds={args.seeds} steps={args.steps} gpus={args.gpus} "
          f"slots={args.slots_per_gpu}", flush=True)

    running: list[tuple[int, Job, subprocess.Popen, object]] = []
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
            slots = {g: 0 for g in args.gpus}
            for gpu, _, _, _ in running:
                slots[gpu] = slots.get(gpu, 0) + 1
            for gpu in args.gpus:
                if not pending:
                    break
                if slots.get(gpu, 0) >= args.slots_per_gpu:
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
                print(f"[run ] gpu{gpu} ({slots[gpu]}/{args.slots_per_gpu}) {job.label}", flush=True)
                time.sleep(3)

        time.sleep(args.poll)

    print("[complete] all jobs finished", flush=True)


if __name__ == '__main__':
    main()
