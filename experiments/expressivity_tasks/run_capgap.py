"""Capability-gap battery (task capability-gap-research).

THE question (Erik 2026-06-08): WITH an MLP present in BOTH arms, is
nonlinearity-IN-TIME (in the recurrence) useless, or is there a capability
where the best nonlinear-in-time+MLP separates from the best
linear-recurrence(gdn-neg)+MLP? The separating task must need NONLINEAR
compositional depth that GROWS WITH SEQUENCE LENGTH (a fixed-depth MLP cannot
supply it) and be BEYOND gdn-neg's free group/counting capability.

Every run here interleaves a SwiGLU MLP after each mixer (--mlp_ratio) so the
fixed O(depth) nonlinear readout is present in ALL arms — the whole point.

Arms (pure within-layer head type, 32 heads, all the SAME plumbing):
  gdnneg   : gdn2_recall (idx0), allow_neg_eigval=1  -> LINEAR-in-time recurrence
  nlshell  : gdn2_nonlin_shell (idx5)                -> NONLINEAR-in-time, SAME GDN
                                                        plumbing (the clean A/B; the
                                                        only delta is state tanh)
  e97delta : e97_delta (idx7), tanh state            -> the tanh-e97 nonlinear-in-time
                                                        cell from the 1.3B LM line

Tasks:
  GAP CANDIDATES (need O(T) nonlinear depth):
    dyck_depth          counting w/ zero-floor comparison (Weiss-Goldberg-Yahav)
    modular_quadratic   x_t=(x_{t-1}^2+c_t) mod p  (nonlinear, non-invertible, non-contracting)
    monoid_track        non-group monoid (random non-invertible maps)
  CONTROLS (no gap expected):
    s5_permutation        GROUP (invertible) -> gdn-neg solves for free
    modular_quadratic_lin LINEAR modular counter (invertible) -> gdn-neg solves
    iterated_nonlinear_map CONTRACTING logistic map -> bounded effective depth

3 seeds; train T=128; eval length-extrap T in {128,256,512,1024,2048}.
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

GPUS = [2, 3, 4, 5, 6, 7]   # leave 0,1 for whatever is already running
SLOTS_PER_GPU = 3
MEM_CAP_MIB = 40000

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

L = -30.0
def _logits(active):
    v = [L] * 8
    for i in active:
        v[i] = 0.0
    return ','.join(str(x) for x in v)

# arm -> (head_type_logits over 8 TYPE_NAMES, gdn_allow_neg_eigval)
# TYPE_NAMES = [gdn2_recall, e97_track, count, latch, nonlin, gdn2_nonlin_shell, e97_raw, e97_delta]
ARMS: dict[str, tuple[str, int]] = {
    'gdnneg':   (_logits([0]), 1),   # LINEAR-in-time recurrence
    'nlshell':  (_logits([5]), 1),   # NONLINEAR-in-time, same GDN plumbing
    'e97delta': (_logits([7]), 1),   # tanh-e97 nonlinear-in-time (SATURATING)
    'count':    (_logits([2]), 1),   # UnifiedCell count corner (NON-saturating, relu-state)
}

# task -> extra CLI flags (the new tasks use their own defaults)
TASKS = {
    'dyck_depth': [],
    'dyck_depth_unbounded': [],   # Round 2: UNBOUNDED counting (magnitude > train band)
    'modular_quadratic': [],
    'monoid_track': [],
    's5_permutation': [],
    'modular_quadratic_lin': [],
    'iterated_nonlinear_map': [],
}

SEEDS = [42, 123, 456]


@dataclass
class Job:
    task: str
    arm: str
    seed: int
    depth: int
    K: int = 0   # 0 = use task default; >2 = scale stress (N for monoid, p for modular)

    @property
    def label(self) -> str:
        ktag = f"__K{self.K}" if self.K > 2 else ""
        return f"cg_{self.task}__{self.arm}__d{self.depth}{ktag}__seed{self.seed}"


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
    logits, neg = ARMS[job.arm]
    pattern = ['typed-gdn2'] * job.depth
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.task,
        '--layer_pattern', *pattern,
        *SHARED,
        f'--head_type_logits={logits}',
        '--gdn_allow_neg_eigval', str(neg),
        '--mlp_ratio', str(args.mlp_ratio),
        *TASKS[job.task],
        '--depth', str(job.depth),
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
    ]
    if job.K > 2:
        cmd += ['--K', str(job.K)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS.keys()))
    ap.add_argument('--arms', nargs='+', default=list(ARMS.keys()), choices=list(ARMS.keys()))
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--depths', type=int, nargs='+', default=[4])
    ap.add_argument('--Ks', type=int, nargs='+', default=[0],
                    help='Scale stress sweep: N for monoid_track, p for modular_quadratic. '
                         '0 = task default.')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--mlp_ratio', type=float, default=2.0)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--output_dir', default=str(THIS / 'results_capgap'))
    ap.add_argument('--poll', type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = [Job(t, a, s, d, k) for t in args.tasks for a in args.arms
                for s in args.seeds for d in args.depths for k in args.Ks]
    jobs = []
    for j in all_jobs:
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs (of {len(all_jobs)}); tasks={args.tasks} "
          f"arms={args.arms} seeds={args.seeds} depths={args.depths} steps={args.steps} "
          f"mlp_ratio={args.mlp_ratio} gpus={GPUS}", flush=True)

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
