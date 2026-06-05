"""UNIFIED-CELL LEARNABILITY sweep runner.

Tests whether a SINGLE learned unified cell can be trained to SELF-ORGANIZE onto
the four capability corners, via two interventions on top of the validated
unified-cell Triton kernel (no kernel rewrite):

  (1) INIT SPREAD   : per-head knobs spread ACROSS corners at init
                      (arm `unified-learned-spread`, lam_max=1.5, gamma_mix phi).
  (2) KNOB-SPECIFIC LR: lambda/beta/igain/gamma in a separate optimizer group at
                      a higher LR (train_hybrid --knob_lr_mult), swept {1,5,10,20}.
                      klr=1 is the init-spread-ALONE ablation.

Arms:
  unified-learned-spread @ knob_lr_mult in {1,5,10,20}   (the new interventions)
  baselines on the MIXED probe (new task): 4 presets, generic learned-free, lstm.
  The 4 standalone-probe baselines (presets / generic learned-free / lstm) are
  REUSED from the prior unified-cell sweep (run_unified_cell.py) -- same results/
  dir + label convention, so aggregate_unified_learnability.py reads both.

Probes: s5_permutation, anbncn_viability, iterated_nonlinear_map,
        flag_hold_recall (latch), mixed_probe (all four at once).

Optional length curriculum (--curriculum 128,256,512,1024) is the SECONDARY
lever, run only if init-spread+knob-LR proves insufficient.

Idle-GPU scheduling (used-mem < FREE_MEM_MIB), resumable (skips existing JSON).
3 seeds {42,123,456}, ~8M params, fp32, schedule-free AdamW.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]

FREE_MEM_MIB = 2000  # a GPU is "idle" iff used memory < this (task rule: <2GB)

SHARED = ['--dim', '384', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

# arm name -> (layer_pattern + shared CLI, extra train args e.g. knob_lr_mult)
ARMS: dict[str, tuple[list[str], list[str]]] = {
    'unified-learned-spread-klr1':  (['--layer_pattern', 'unified-learned-spread', *SHARED], ['--knob_lr_mult', '1']),
    'unified-learned-spread-klr5':  (['--layer_pattern', 'unified-learned-spread', *SHARED], ['--knob_lr_mult', '5']),
    'unified-learned-spread-klr10': (['--layer_pattern', 'unified-learned-spread', *SHARED], ['--knob_lr_mult', '10']),
    'unified-learned-spread-klr20': (['--layer_pattern', 'unified-learned-spread', *SHARED], ['--knob_lr_mult', '20']),
    # mixed-probe baselines (the standalone-probe versions are reused from prior sweep)
    'unified-track':        (['--layer_pattern', 'unified-track', *SHARED], []),
    'unified-count':        (['--layer_pattern', 'unified-count', *SHARED], []),
    'unified-latch':        (['--layer_pattern', 'unified-latch', *SHARED], []),
    'unified-nonlin':       (['--layer_pattern', 'unified-nonlin', *SHARED], []),
    'unified-learned-free': (['--layer_pattern', 'unified-learned-free', *SHARED], []),
    'lstm':                 (['--layer_pattern', 'lstm', '--dim', '448', '--expansion', '1.0'], []),
}

SPREAD_ARMS = [a for a in ARMS if a.startswith('unified-learned-spread')]
MIXED_BASELINES = ['unified-track', 'unified-count', 'unified-latch',
                   'unified-nonlin', 'unified-learned-free', 'lstm']

PROBES = {
    's5_permutation': [],
    'anbncn_viability': [],
    'iterated_nonlinear_map': [],
    'flag_hold_recall': ['--K', '4'],
    'mixed_probe': ['--K', '4'],
}

SEEDS = [42, 123, 456]


@dataclass
class Job:
    probe: str
    arm: str
    seed: int
    curriculum: str | None = None

    @property
    def label(self) -> str:
        suffix = '' if not self.curriculum else '__curr'
        return f"learn_{self.probe}__{self.arm}__seed{self.seed}{suffix}"


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
    if job.curriculum:
        cmd += ['--curriculum', job.curriculum]
    return cmd


def plan_jobs(args) -> list[Job]:
    jobs: list[Job] = []
    curr = args.curriculum  # None unless explicitly enabled
    for probe in args.probes:
        # spread arms (the interventions) on every probe
        for arm, seed in itertools.product(SPREAD_ARMS, args.seeds):
            jobs.append(Job(probe, arm, seed, curriculum=curr))
        # baselines only on the MIXED probe (standalone-probe baselines reused
        # from the prior unified-cell sweep)
        if probe == 'mixed_probe':
            for arm, seed in itertools.product(MIXED_BASELINES, args.seeds):
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
    ap.add_argument('--curriculum', type=str, default=None,
                    help='Enable the secondary length-curriculum lever for spread '
                         'arms, e.g. 128,256,512,1024. Default None.')
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
          f"seeds={args.seeds} steps={args.steps} curriculum={args.curriculum}", flush=True)

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
