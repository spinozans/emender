"""E97-RAW x GDN HYBRID — expressivity ratio-sweep (task e97-gdn-hybrid).

Study A (E97_RAW_EXPRESSIVITY_RESULTS) found e97-raw is the #1 LM cell but a
COUNT/LATCH specialist that FAILS recall (0.14 vs GDN 0.95) and S5 track. A's #1
recommendation: add a GDN gated-delta recall head. This sweep builds the HYBRID
and asks: does adding GDN recurrence to the e97-raw backbone RECOVER recall (and
fix latch length-extrapolation) while KEEPING count — and at what mix ratio?

Implementation (the cleanest faithful one): INTERLEAVED LAYERS. The e97-raw cell
(E88FLAHybrid use_split_edit=True, raw_write=1, state tanh — the validated LM
winner) is the backbone; GDN gated-delta (`gdn` token == GatedDeltaNet, the same
recall reference bar study A used) layers are interleaved. The within-layer typed
head mixture (typed_head_mixture) was REJECTED for this study because its e97 head
is `e97_track` (the UnifiedCell 'track' corner), NOT the raw-write `e97-raw` LM
winner — interleaving uses the actual e97-raw cell.

Ratio = fraction of GDN layers in the depth-4 stack:
  all-e97raw : E97 E97 E97 E97   (0/4 gdn)   <- study-A e97-raw endpoint, bf16
  h3to1      : E97 E97 E97 gdn   (1/4 gdn)
  h1to1      : E97 gdn  E97 gdn  (2/4 gdn)
  h1to3      : E97 gdn  gdn gdn  (3/4 gdn)
  all-gdn    : gdn gdn  gdn gdn  (4/4 gdn)   <- study-A GDN recall bar, bf16

Shape-matched (dim=256, n_heads=32, n_state=32, expansion=1.0) per study A's
protocol — GDN layers are ~5x cheaper than e97-raw layers, so the hybrid arms
carry FEWER params than all-e97raw (reported per arm). This makes any recall
recovery a CONSERVATIVE result (capability gained while shedding params).

PRECISION: bf16 for ALL arms (the GDN chunked-delta kernel rejects float32, so a
fp32 hybrid is impossible; running every arm in bf16 keeps the sweep internally
consistent and the endpoints directly comparable to the hybrid arms). e97-raw
uses a BOUNDED tanh state, so bf16 is faithful for it.

Probes (existing battery): s5_permutation (track), anbncn_viability (count),
iterated_nonlinear_map (nonlin), flag_hold_recall K=4 (latch), mqar_recall (recall).
3 seeds {42,123,456}; train T=128; eval (length-extrap) T in {128,256,512,1024}.
ONLY GPUs 0,1,2,3. Idle-GPU scheduling; resumable (skips existing JSON).
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

GPUS = [0, 1, 2, 3]   # TASK CONSTRAINT: only GPUs 0-3 (sibling owns 4-7)
SLOTS_PER_GPU = 3     # split-edit recurrence is latency-bound; 3/GPU saturates util
MEM_CAP_MIB = 30000   # do not add a job to a GPU already above this (never preempt)

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

# e97-raw structural flags forwarded to EVERY E97 layer in the pattern.
E97_RAW = ['--state_activation', 'tanh', '--e88_raw_write', '1']

# arm -> layer_pattern (depth 4). All bf16 (NO --disable_autocast).
ARMS: dict[str, list[str]] = {
    'all-e97raw': ['E97', 'E97', 'E97', 'E97'],
    'h3to1':      ['E97', 'E97', 'E97', 'gdn'],
    'h1to1':      ['E97', 'gdn', 'E97', 'gdn'],
    'h1to3':      ['E97', 'gdn', 'gdn', 'gdn'],
    'all-gdn':    ['gdn', 'gdn', 'gdn', 'gdn'],
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
        return f"e97gdn_{self.probe}__{self.arm}__seed{self.seed}"


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
    pattern = ARMS[job.arm]
    # E97 structural flags are forwarded only to E97-family layers by train_hybrid;
    # harmless when an arm is pure-gdn (no E97 layer consumes them).
    extra = E97_RAW if any(l == 'E97' for l in pattern) else []
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.probe,
        '--layer_pattern', *pattern,
        *SHARED,
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
          f"arms={args.arms} seeds={args.seeds} steps={args.steps} gpus={GPUS}", flush=True)

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
