"""E98 FIVE-CORNER — does a 5-type head population beat 4? Sweep runner.

Adds the leaky-linear ASSOCIATIVE-MEMORY workhorse (lambda<1, small beta ->
positive along-key eigenvalue in (0,1), phi=identity; the GDN/Mamba fading
key-value-recall regime) as a 5th placed corner, and an MQAR multi-query
associative-recall probe + a 5-capability mixed task that demands recall.

Arms (split-gated E98 unified cell, ~7.4M, dim=256 / 32 heads / N=V=32):
  * spread-4  = e98-learned-spread  (the current 4 exotic corners)   } HEAD-TO-HEAD,
  * spread-5  = e98-learned-spread5 (adds the leaky-linear corner)    } every probe
  * e98-leaky = leaky-linear PRESET alone (confirm it WINS recall)
  * gdn       = GatedDeltaNet reference on recall (the workhorse is the right tool)
  * the four exotic PRESETS on recall (confirm they FAIL it -> recall is covered
    ONLY by the leaky-linear corner)

Both spread arms use the WINNING learnability form: spread-init + knob_lr_mult=20.

Probes: s5_permutation, anbncn_viability, iterated_nonlinear_map, flag_hold_recall,
mqar_recall (NEW), mixed_probe (now 5-capability).

REAL training, fp32 (the unified arms; GDN runs bf16 — its Triton chunk kernels
reject fp32, noted in the report). schedule-free AdamW, depth 4. 3 seeds
{42,123,456}; train T=128; eval (length extrapolation) at T in {128,256,512,1024}.
Idle-GPU scheduling (used-mem < FREE_MEM_MIB); never preempt a busy GPU (runs in
parallel with cma-capability). Resumable (skips existing JSON).
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

FREE_MEM_MIB = 2000  # a GPU is "idle" iff used memory < this (task rule: <2GB)

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

# arm -> (layer CLI args, extra train args, disable_autocast?). fp32 for unified
# arms; GDN must run bf16 (its chunked delta kernel rejects float32).
ARMS: dict[str, tuple[list[str], list[str], bool]] = {
    'spread-4':  (['--layer_pattern', 'e98-learned-spread',  *SHARED], ['--knob_lr_mult', '20'], True),
    'spread-5':  (['--layer_pattern', 'e98-learned-spread5', *SHARED], ['--knob_lr_mult', '20'], True),
    'leaky':     (['--layer_pattern', 'e98-leaky',           *SHARED], [], True),
    'track':     (['--layer_pattern', 'e98-track',           *SHARED], [], True),
    'count':     (['--layer_pattern', 'e98-count',           *SHARED], [], True),
    'latch':     (['--layer_pattern', 'e98-latch',           *SHARED], [], True),
    'nonlin':    (['--layer_pattern', 'e98-nonlin',          *SHARED], [], True),
    'gdn':       (['--layer_pattern', 'gdn',                 *SHARED], [], False),
}

PROBES = {
    's5_permutation': [],
    'anbncn_viability': [],
    'iterated_nonlinear_map': [],
    'flag_hold_recall': ['--K', '4'],
    'mqar_recall': [],
    'mixed_probe': ['--K', '4'],
}

# spread-4 and spread-5 run on EVERY probe (the 4-vs-5 head-to-head).
HEADTOHEAD = ['spread-4', 'spread-5']
# Recall-only arms: the leaky PRESET (wins), GDN reference (right tool), and the
# four EXOTIC presets (fail -> recall covered only by the leaky-linear corner).
RECALL_ONLY = ['leaky', 'gdn', 'track', 'count', 'latch', 'nonlin']

SEEDS = [42, 123, 456]


@dataclass
class Job:
    probe: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"e98fc_{self.probe}__{self.arm}__seed{self.seed}"


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
    layer_args, extra, disable_autocast = ARMS[job.arm]
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
        for arm in HEADTOHEAD:
            for seed in args.seeds:
                jobs.append(Job(probe, arm, seed))
        if probe == 'mqar_recall':
            for arm in RECALL_ONLY:
                for seed in args.seeds:
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
