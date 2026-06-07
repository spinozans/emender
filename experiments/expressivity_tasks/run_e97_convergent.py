"""E97-CONVERGENT CELL — expressivity battery for the 2x3 matrix (task e97-convergent).

Builds the CONVERGENT-CELL candidate and tests the two findings that reshaped the
target architecture (supersedes the fragile messages to e97-gdn-hybrid):

  1. e97-DELTA (delta-correction, raw_write=0) vs e97-RAW (raw_write=1) as BACKBONE.
     Study A found e97-delta strictly dominates raw on expressivity (count + length-
     robust latch + nonlin + 3x track) at ~0.02-0.06 nats LM cost. Does the delta
     backbone fix raw's latch-extrapolation collapse and lift nonlin in the hybrid?
  2. gdn-neg (allow_neg_eigval=1) as recall head. Track needs NEGATIVE eigenvalues;
     gdn-neg solved S5 at 0.998 while vanilla gdn does recall 0.95. Does gdn-neg
     deliver recall AND track from ONE head, or trade one for the other?

MATRIX (depth-4 interleave, shape-matched dim=256 n_heads=32 n_state=32 exp=1.0):
  backbone in {e97-raw (raw_write=1), e97-delta (raw_write=0)} ; both state tanh,
  recall-head in {none, gdn, gdn-neg}:
    none    : E97 E97 E97 E97        (4/0 backbone, study-A endpoint)
    gdn     : E97 fla-gdn E97 fla-gdn (2/4 recall, allow_neg_eigval=0)  <- C's best 2/4 mix
    gdn-neg : E97 fla-gdn E97 fla-gdn (2/4 recall, allow_neg_eigval=1)
  -> 6 arms.

RECALL HEAD = fla-gdn (FLAGatedDeltaNetLayer == fused FLA chunk_gated_delta_rule).
allow_neg_eigval is a scalar (beta*=2 -> along-key eigenvalue can go negative); gdn
and gdn-neg are THE SAME fused kernel differing only in that scalar (preflight item 1:
no unfused path; gdn2_nonlin_shell is NOT used in any arm). Recall heads carry FEWER
params than backbone layers, so any capability gained is a CONSERVATIVE result.

PRECISION: bf16 autocast for ALL arms (E97 uses a BOUNDED tanh state -> faithful in
bf16, verified by parity_e97_bf16.py; the fla-gdn chunk kernel rejects fp32). E97
runs the PyTorch reference recurrence (NOT --use_triton_e88), same as Study A / C.

Probes (existing battery): s5_permutation (track), anbncn_viability (count),
iterated_nonlinear_map (nonlin), flag_hold_recall K=4 (latch), mqar_recall (recall).
3 seeds {42,123,456}; train T=128; eval length-extrap T in {128,256,512,1024}.
GPUs default 4,5,6,7 (sibling C owns 0-3). Idle-GPU scheduling; resumable.
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

GPUS = [4, 5, 6, 7]   # sibling C owns 0-3; expand once they free
SLOTS_PER_GPU = 3     # split-edit recurrence is latency-bound; 3/GPU saturates util
MEM_CAP_MIB = 30000   # never add to a GPU already above this (no preempt)

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

BACKBONE = ['E97', 'E97', 'E97', 'E97']
INTERLEAVE = ['E97', 'fla-gdn', 'E97', 'fla-gdn']


@dataclass
class Arm:
    name: str
    pattern: list[str]
    raw_write: int          # 1 = e97-raw, 0 = e97-delta
    allow_neg: int | None   # fla-gdn allow_neg_eigval; None when no recall head


ARMS: dict[str, Arm] = {
    'raw-none':     Arm('raw-none',     BACKBONE,   1, None),
    'raw-gdn':      Arm('raw-gdn',      INTERLEAVE, 1, 0),
    'raw-gdnneg':   Arm('raw-gdnneg',   INTERLEAVE, 1, 1),
    'delta-none':   Arm('delta-none',   BACKBONE,   0, None),
    'delta-gdn':    Arm('delta-gdn',    INTERLEAVE, 0, 0),
    'delta-gdnneg': Arm('delta-gdnneg', INTERLEAVE, 0, 1),
}
ARM_NAMES = list(ARMS.keys())

PROBES = {
    's5_permutation': [],
    'anbncn_viability': [],
    'iterated_nonlinear_map': [],
    'flag_hold_recall': ['--K', '4'],
    'mqar_recall': [],
}

SEEDS = [42, 123, 456]


@dataclass
class Job:
    probe: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"e97conv_{self.probe}__{self.arm}__seed{self.seed}"


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
    arm = ARMS[job.arm]
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.probe,
        '--layer_pattern', *arm.pattern,
        *SHARED,
        *PROBES[job.probe],
        '--state_activation', 'tanh',
        '--e88_raw_write', str(arm.raw_write),
        '--depth', str(args.depth),
        '--steps', str(args.steps),
        '--seq_len', '128',
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--optimizer', 'schedulefree',
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', '128', '256', '512', '1024',
        '--eval_lengths_n_batches', str(args.eval_n_batches),
    ]
    if arm.allow_neg is not None:
        cmd += ['--gdn_allow_neg_eigval', str(arm.allow_neg)]
    return cmd


def plan_jobs(args) -> list[Job]:
    jobs: list[Job] = []
    for probe in args.probes:
        for arm in args.arms:
            for seed in args.seeds:
                jobs.append(Job(probe, arm, seed))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', nargs='+', default=list(PROBES.keys()), choices=list(PROBES.keys()))
    ap.add_argument('--arms', nargs='+', default=ARM_NAMES, choices=ARM_NAMES)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--gpus', type=int, nargs='+', default=GPUS)
    ap.add_argument('--output_dir', default=str(THIS / 'results'))
    ap.add_argument('--poll', type=float, default=15.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus

    all_jobs = plan_jobs(args)
    jobs = []
    for j in all_jobs:
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs (of {len(all_jobs)}); probes={args.probes} "
          f"arms={args.arms} seeds={args.seeds} steps={args.steps} gpus={gpus}", flush=True)

    running: list[tuple[int, Job, subprocess.Popen, object]] = []
    pending = list(jobs)

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
            slots = {g: 0 for g in gpus}
            for gpu, _, _, _ in running:
                slots[gpu] = slots.get(gpu, 0) + 1
            for gpu in gpus:
                if not pending:
                    break
                if slots.get(gpu, 0) >= SLOTS_PER_GPU:
                    continue
                if used.get(gpu, 10**9) >= MEM_CAP_MIB:
                    continue
                job = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
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
