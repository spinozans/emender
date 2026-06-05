"""UNIFIED-CELL expressivity + un-cribbing sweep runner.

Trains the unified parameterized matrix-recurrence cell (and baselines) across
the four capability probes, 3 seeds, T=128 train, length-extrapolation eval at
T in {128,256,512,1024}. Param-matched ~8M, fp32, schedule-free AdamW.

Arms (Run A expressivity + Run B un-cribbing + Run C emergent):
  4 pinned presets : unified-track / -count / -latch / -nonlin
  LEARNED (free)   : unified-learned-free   (lambda in (0,1.5], gamma-mix phi)
  LEARNED (clamp)  : unified-learned-clamp  (lambda CLAMPED to (0,1)) -- un-cribbing control
  E88-baseline     : unified-e88base        (cribbed lambda<1, beta=1, tanh)
  LSTM             : gated additive counter baseline

Probes (capability corners):
  s5_permutation         (track)     anbncn_viability       (count)
  iterated_nonlinear_map (nonlinear) flag_hold_recall        (latch)

Dynamic GPU scheduling: uses ONLY idle GPUs (used-mem < FREE_MEM_MIB), re-checks
before each launch, never preempts a busy GPU. Skips jobs whose output JSON
already exists (resumable). Per-head learned (lambda,beta,gamma) are dumped into
each LEARNED-arm JSON under log['unified_knobs'] for the emergent-specialization
analysis (Run C).
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
ARMS = {
    'unified-track':        ['--layer_pattern', 'unified-track', *SHARED],
    'unified-count':        ['--layer_pattern', 'unified-count', *SHARED],
    'unified-latch':        ['--layer_pattern', 'unified-latch', *SHARED],
    'unified-nonlin':       ['--layer_pattern', 'unified-nonlin', *SHARED],
    'unified-learned-free': ['--layer_pattern', 'unified-learned-free', *SHARED],
    'unified-learned-clamp':['--layer_pattern', 'unified-learned-clamp', *SHARED],
    'unified-e88base':      ['--layer_pattern', 'unified-e88base', *SHARED],
    'lstm':                 ['--layer_pattern', 'lstm', '--dim', '448', '--expansion', '1.0'],
}

# probe -> extra CLI knobs (K forwarded only where the task reads it)
PROBES = {
    's5_permutation': [],
    'anbncn_viability': [],
    'iterated_nonlinear_map': [],
    'flag_hold_recall': ['--K', '4'],
}

SEEDS = [42, 123, 456]


@dataclass
class Job:
    probe: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"unified_{self.probe}__{self.arm}__seed{self.seed}"


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
    return [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.probe,
        *ARMS[job.arm],
        *PROBES[job.probe],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', nargs='+', default=list(PROBES.keys()), choices=list(PROBES.keys()))
    ap.add_argument('--arms', nargs='+', default=list(ARMS.keys()))
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--max_gpus', type=int, default=8)
    ap.add_argument('--output_dir', default=str(THIS / 'results'))
    ap.add_argument('--poll', type=float, default=15.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = []
    for probe, arm, seed in itertools.product(args.probes, args.arms, args.seeds):
        j = Job(probe, arm, seed)
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs; arms={len(args.arms)} probes={len(args.probes)} "
          f"seeds={args.seeds} steps={args.steps}", flush=True)

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
