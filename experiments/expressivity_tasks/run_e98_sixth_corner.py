"""E98 SIXTH CORNER — does adding the GATED-DELTA backbone beat the 5-corner pop?

The 5-corner study found leaky-linear is the WORST recall corner; the real recall
workhorse is GATED-DELTA (GDN solved MQAR 0.951 vs unified-best ~0.29, 4.6x fewer
params), and GDN+neg-eig was the S5 length-robust winner. So gated-delta does BOTH
recall AND tracking — the genuine LM backbone — and no placed corner occupied it.
This sweep adds it.

The gated-delta corner = the GDN operating point expressible in the VALIDATED E98
split-gated unified cell (NO kernel rewrite): beta~=1 (full delta / clean
overwrite), INPUT-DEPENDENT gated decay lambda_t in (0,1) (Mamba/GDN style, fed to
the kernel's existing time-varying lam_t input via a decay gate), identity phi
(linear state). along-key eig = lambda_t - 1 in (-1,0]: ~0 clean erase-then-write
(recall) when lambda_t~1, negative (reflection / S5 track) when lambda_t<1.

Arms (split-gated E98 unified cell, ~7.4M, dim=256 / 32 heads / N=V=32):
  * spread-4      = e98-learned-spread  (the 4 exotic corners)          } HEAD-TO-HEAD
  * spread-5      = e98-learned-spread5 (adds leaky-linear)             } on EVERY probe
  * spread-6      = e98-learned-spread6 (adds gated-delta backbone)     }
  * gated-delta   = e98-gated-delta PRESET (input-dep decay) — does it now SOLVE
                    recall (approach GDN 0.95)?  AND help S5 (neg-eig)?  {mqar, s5}
  * leaky         = e98-leaky PRESET on recall (the 5-corner recall corner: WORST)  {mqar}
  * gdn           = GatedDeltaNet reference on recall (the 0.95 bar)               {mqar}
  * gdn-neg       = fla GatedDeltaNet + allow_neg_eigval on S5 (the track bar)     {s5}

All three spread arms use the WINNING learnability form: spread-init + knob_lr_mult=20.

Probes: s5_permutation, anbncn_viability, iterated_nonlinear_map, flag_hold_recall,
mqar_recall, mixed_probe (the 6-corner population on the 5-capability mixed stream;
the 6th corner is a HEAD TYPE serving recall+track, both already in the mix).

REAL training, fp32 (unified arms; GDN runs bf16 — its chunk kernels reject fp32).
schedule-free AdamW, depth 4. 3 seeds {42,123,456}; train T=128; eval (length
extrapolation) at T in {128,256,512,1024}. Idle/co-locatable GPUs only (used mem <
COLOC_MEM_MIB); never preempt a busy GPU. Resumable (skips existing JSON).
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

# A GPU is co-locatable iff used memory < this. The 48GB cards here idle at ~2.9GB
# (small cma-capability models); an 8M co-tenant needs ~1-2GB, so co-location is
# safe well below the card limit. We only SKIP heavily-loaded cards (never preempt).
COLOC_MEM_MIB = 38000

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

# arm -> (layer CLI args, extra train args, disable_autocast?). fp32 for unified
# arms; GDN refs must run bf16 (their chunked delta kernels reject float32).
ARMS: dict[str, tuple[list[str], list[str], bool]] = {
    'spread-4':     (['--layer_pattern', 'e98-learned-spread',  *SHARED], ['--knob_lr_mult', '20'], True),
    'spread-5':     (['--layer_pattern', 'e98-learned-spread5', *SHARED], ['--knob_lr_mult', '20'], True),
    'spread-6':     (['--layer_pattern', 'e98-learned-spread6', *SHARED], ['--knob_lr_mult', '20'], True),
    'gated-delta':  (['--layer_pattern', 'e98-gated-delta',     *SHARED], [], True),
    'leaky':        (['--layer_pattern', 'e98-leaky',           *SHARED], [], True),
    'gdn':          (['--layer_pattern', 'gdn',                 *SHARED], [], False),
    'gdn-neg':      (['--layer_pattern', 'fla-gdn',             *SHARED], ['--gdn_allow_neg_eigval', '1'], False),
}

PROBES = {
    's5_permutation': [],
    'anbncn_viability': [],
    'iterated_nonlinear_map': [],
    'flag_hold_recall': ['--K', '4'],
    'mqar_recall': [],
    'mixed_probe': ['--K', '4'],
}

# All three spread arms run on EVERY probe (the 4-vs-5-vs-6 head-to-head).
HEADTOHEAD = ['spread-4', 'spread-5', 'spread-6']
# Probe-specific reference/preset arms.
#   mqar: the gated-delta PRESET (does it now SOLVE recall?), the leaky PRESET
#         (5-corner recall corner, the WORST), and the GDN bar (0.95).
#   s5  : the gated-delta PRESET (does it ALSO track, with neg-eig?) and the
#         GDN+neg-eig track bar.
EXTRA = {
    'mqar_recall': ['gated-delta', 'leaky', 'gdn'],
    's5_permutation': ['gated-delta', 'gdn-neg'],
}

SEEDS = [42, 123, 456]


@dataclass
class Job:
    probe: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"e98sc_{self.probe}__{self.arm}__seed{self.seed}"


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
        for arm in EXTRA.get(probe, []):
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
    ap.add_argument('--coloc_mem_mib', type=int, default=COLOC_MEM_MIB)
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
                if used.get(gpu, 10**9) >= args.coloc_mem_mib:
                    continue  # heavily loaded -- never preempt
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
