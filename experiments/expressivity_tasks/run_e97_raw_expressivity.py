"""E97-RAW EXPRESSIVITY BATTERY — characterize the 1.3B LM CMA-ES winner.

e97-raw is the #1 cell on the 1.3B LM CMA-ES leaderboard (avg-loss 5.9511) but
has NEVER been run on the expressivity battery. This sweep runs the full battery
on e97-raw and the two ablations that isolate its two design choices, plus GDN as
the recall reference bar.

Arms (split-gated E88FLAHybrid, ~7.4M, dim=256 / 32 heads / N=V=32):
  * e97-raw    = E97 (use_split_edit) + tanh state-squash + RAW WRITE (no delta)
  * e97        = E97 + tanh state-squash + DELTA correction   <- isolates raw-write
  * e97-linear = E97 + IDENTITY state (no tanh) + DELTA        <- isolates the tanh squash
  * gdn        = GatedDeltaNet reference on recall (the parameter-efficient
                 recall workhorse; ~1.6M at matched shape — a conservative bar)

Design-choice isolation:
  e97-raw vs e97        -> RAW-WRITE  vs delta-correction (state nonlinearity held = tanh)
  e97       vs e97-linear -> TANH squash vs identity      (delta-correction held on)

Probes (existing battery, experiments/expressivity_tasks/tasks/):
  s5_permutation (track), anbncn_viability (count), iterated_nonlinear_map (nonlin),
  flag_hold_recall (latch), mqar_recall (recall).

REAL training, fp32 for the E97 arms (--disable_autocast); GDN runs bf16 (its
chunked delta kernel rejects float32). schedule-free AdamW, depth 4. 3 seeds
{42,123,456}; train T=128; eval (length extrapolation) at T in {128,256,512,1024}.
ONLY GPUs 0,1,2,3 (a sibling study e97-raw-plus-mlp owns GPUs 4-7). Idle-GPU
scheduling (used-mem < FREE_MEM_MIB); never preempt a busy GPU. Resumable (skips
existing JSON).
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

GPUS = [0, 1, 2, 3]  # TASK CONSTRAINT: only GPUs 0-3 (sibling study owns 4-7)
SLOTS_PER_GPU = 3    # the split-edit recurrence is latency-bound (~30% util solo);
                     # 3 concurrent jobs/GPU saturate to 100% util (~2.5x throughput,
                     # ~12GB) — measured on these GPUs at B32/T128/dim256.
MEM_CAP_MIB = 30000  # do not add a job to a GPU whose used mem already exceeds this
                     # (leaves headroom; never preempt the sibling study on 4-7).

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

# arm -> (layer CLI args, extra train args, disable_autocast?).
# DEFAULT = FUSED: the three E97 arms (tanh / identity state, kernel-compatible) run
# bf16 autocast through the fused split-edit Triton fwd/bwd kernel (--use_triton_e88),
# ~43-58x faster than the eager fp32 T-scan with VERIFIED numerical parity (fused bf16
# vs eager bf16, fwd+bwd, rel-L2 < 1e-2 at T up to 1024; see verify_e97_fused_parity.py
# and paper/review/E97_FUSED_LM_KERNEL_NOTE.md). GDN runs bf16 via its own FLA chunked
# delta kernel. Pass --eager-fp32 to reproduce the original fp32 eager E97 arms.
# NOTE: only tanh/identity state is kernel-compatible; relu/softplus (non-saturating)
# MUST stay eager fp32 — the kernel raises rather than silently run tanh.
ARMS: dict[str, tuple[list[str], list[str], bool]] = {
    'e97-raw':    (['--layer_pattern', 'E97', *SHARED], ['--state_activation', 'tanh',     '--e88_raw_write', '1', '--use_triton_e88'], False),
    'e97':        (['--layer_pattern', 'E97', *SHARED], ['--state_activation', 'tanh',     '--e88_raw_write', '0', '--use_triton_e88'], False),
    'e97-linear': (['--layer_pattern', 'E97', *SHARED], ['--state_activation', 'identity', '--e88_raw_write', '0', '--use_triton_e88'], False),
    'gdn':        (['--layer_pattern', 'gdn', *SHARED], [], False),
}

# Eager fp32 fallback arms (original behaviour): no Triton, autocast disabled.
ARMS_EAGER_FP32: dict[str, tuple[list[str], list[str], bool]] = {
    'e97-raw':    (['--layer_pattern', 'E97', *SHARED], ['--state_activation', 'tanh',     '--e88_raw_write', '1'], True),
    'e97':        (['--layer_pattern', 'E97', *SHARED], ['--state_activation', 'tanh',     '--e88_raw_write', '0'], True),
    'e97-linear': (['--layer_pattern', 'E97', *SHARED], ['--state_activation', 'identity', '--e88_raw_write', '0'], True),
    'gdn':        (['--layer_pattern', 'gdn', *SHARED], [], False),
}

PROBES = {
    's5_permutation': [],
    'anbncn_viability': [],
    'iterated_nonlinear_map': [],
    'flag_hold_recall': ['--K', '4'],
    'mqar_recall': [],
}

# All four arms run on every probe: the three E97 arms give the H1/H2/H3 isolation
# on every capability; GDN gives a full reference profile (it is the recall bar on
# mqar, and a useful contrast on the state-tracking / counting probes it is known
# to be weak on).
ARM_NAMES = ['e97-raw', 'e97', 'e97-linear', 'gdn']

SEEDS = [42, 123, 456]


@dataclass
class Job:
    probe: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"e97raw_{self.probe}__{self.arm}__seed{self.seed}"


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
    arm_table = ARMS_EAGER_FP32 if getattr(args, 'eager_fp32', False) else ARMS
    layer_args, extra, disable_autocast = arm_table[job.arm]
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
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', '128', '256', '512', '1024',
        '--eval_lengths_n_batches', str(args.eval_n_batches),
    ]
    if disable_autocast:
        cmd.append('--disable_autocast')
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
    ap.add_argument('--output_dir', default=str(THIS / 'results'))
    ap.add_argument('--poll', type=float, default=15.0)
    ap.add_argument('--eager-fp32', dest='eager_fp32', action='store_true',
                    help='Reproduce the ORIGINAL eager fp32 E97 arms (no Triton, '
                         'autocast disabled). Default is the FUSED bf16 split-edit '
                         'Triton path (~43-58x faster, parity-verified).')
    args = ap.parse_args()
    print(f"[mode] E97 arms run {'EAGER fp32 (legacy)' if args.eager_fp32 else 'FUSED bf16 Triton (default, --use_triton_e88)'}", flush=True)

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
          f"arms={args.arms} seeds={args.seeds} steps={args.steps} gpus={GPUS}", flush=True)

    # running: list of (gpu, Job, proc, logf). Multiple jobs may share a GPU.
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
            slots = {g: 0 for g in GPUS}
            for gpu, _, _, _ in running:
                slots[gpu] = slots.get(gpu, 0) + 1
            for gpu in GPUS:
                if not pending:
                    break
                if slots.get(gpu, 0) >= SLOTS_PER_GPU:
                    continue  # GPU full
                if used.get(gpu, 10**9) >= MEM_CAP_MIB:
                    continue  # leave headroom / never preempt
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
