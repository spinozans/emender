"""E98 = E97 split-gate (decoupled erase b*k / value-write w*v) ON TOP of the
unified capability-span + horizontal specialization. Sweep runner.

Re-confirms, on the SPLIT-GATED cell:
  (1) the four capability corners  -- each pinned-preset arm wins its corner
      (e98-track @ s5, e98-count @ anbncn, e98-latch @ flag_hold, e98-nonlin @
      iterated_nonlinear_map);
  (2) learnability with the WINNING specialization form from SPECIALIZATION_STUDY
      (spread-init + knob-specific LR), now with the split gate:
      e98-learned-spread @ knob_lr_mult=20 on all five probes (4 corners + mixed).
  (3) reference arms for the E97-vs-E88 comparison: e98-learned-free (generic-init
      + split) and e98-fixedpop (fixed placement + split) on the mixed probe.

The E88-BASED unified arms (no split gate) are NOT re-run here -- they are reused
from the UNIFIED_LEARNABILITY / SPECIALIZATION_STUDY results in results/ (the
`learn_*` / `unified_*` JSONs). aggregate_e98_on_e97.py reads both for the
head-to-head: does the split gate IMPROVE the capability span / mixed accuracy?

REAL training, fp32, schedule-free AdamW, ~depth 4, 32 heads, N=V=32, idle-GPU
scheduling (used-mem < FREE_MEM_MIB). 3 seeds {42,123,456}; train T=128; eval
(length extrapolation) at T in {128,256,512,1024}. Resumable (skips existing JSON).
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

# arm -> (layer CLI, extra train args). The split gate is selected by the e98-* pattern.
ARMS: dict[str, tuple[list[str], list[str]]] = {
    # pinned-corner presets (split-gate on) -- each wins its own corner
    'e98-track':  (['--layer_pattern', 'e98-track', *SHARED], []),
    'e98-count':  (['--layer_pattern', 'e98-count', *SHARED], []),
    'e98-latch':  (['--layer_pattern', 'e98-latch', *SHARED], []),
    'e98-nonlin': (['--layer_pattern', 'e98-nonlin', *SHARED], []),
    # WINNING specialization form: spread-init + knob-LR 20x (split-gate on)
    'e98-learned-spread-klr20': (['--layer_pattern', 'e98-learned-spread', *SHARED], ['--knob_lr_mult', '20']),
    # reference arms for the E97-vs-E88 comparison (split-gate on)
    'e98-learned-free': (['--layer_pattern', 'e98-learned-free', *SHARED], []),
    'e98-fixedpop':     (['--layer_pattern', 'e98-fixedpop', *SHARED], []),
}

PROBES = {
    's5_permutation': [],
    'anbncn_viability': [],
    'iterated_nonlinear_map': [],
    'flag_hold_recall': ['--K', '4'],
    'mixed_probe': ['--K', '4'],
}

# Which arms run on which probes.
#   * the winning learnability arm runs on EVERY probe (the capability span test)
#   * each pinned preset runs ONLY on its own corner (preset-wins-corner)
#   * the reference arms (generic + fixedpop) run on the mixed probe
CORNER_OF = {
    's5_permutation': 'e98-track',
    'anbncn_viability': 'e98-count',
    'flag_hold_recall': 'e98-latch',
    'iterated_nonlinear_map': 'e98-nonlin',
}
SPREAD_ARM = 'e98-learned-spread-klr20'
MIXED_REF_ARMS = ['e98-learned-free', 'e98-fixedpop']

SEEDS = [42, 123, 456]


@dataclass
class Job:
    probe: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"e98_{self.probe}__{self.arm}__seed{self.seed}"


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
    layer_args, extra = ARMS[job.arm]
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


def plan_jobs(args) -> list[Job]:
    jobs: list[Job] = []
    for probe in args.probes:
        # winning learnability arm on every probe
        for seed in args.seeds:
            jobs.append(Job(probe, SPREAD_ARM, seed))
        # pinned preset only on its own corner
        if probe in CORNER_OF:
            for seed in args.seeds:
                jobs.append(Job(probe, CORNER_OF[probe], seed))
        # reference arms on the mixed probe
        if probe == 'mixed_probe':
            for arm, seed in itertools.product(MIXED_REF_ARMS, args.seeds):
                jobs.append(Job(probe, arm, seed))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', nargs='+', default=list(PROBES.keys()), choices=list(PROBES.keys()))
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

    all_jobs = plan_jobs(args)
    jobs = []
    for j in all_jobs:
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs (of {len(all_jobs)}); probes={args.probes} "
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
                    continue  # busy -- never preempt
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
