"""E97 WITHIN-LAYER expressivity battery (task e97-within-layer).

THE AUTHORITATIVE within-layer study. Each layer is a single TypedHeadMixtureLayer
holding many heads of MIXED types IN PARALLEL (head-type fractions), summed into the
residual — NOT interleaved whole-layers (the retired e97-gdn-hybrid / e97-convergent
approach). The mixer is `--layer_pattern typed-gdn2 ...` (every layer typed), so the
head-type composition is purely within-layer.

Matrix (backbone x recall), expressed as per-layer head-type fractions over the
TypedHeadMixtureLayer TYPE_NAMES
  [gdn2_recall, e97_track, count, latch, nonlin, gdn2_nonlin_shell, e97_raw, e97_delta]:

  backbone in {e97_raw (idx6), e97_delta (idx7)}
  recall   in {none, gdn (idx0, allow_neg_eigval=0), gdn-neg (idx0, allow_neg_eigval=1)}

  -> 6 matrix configs. "none" == the PURE backbone arm (all heads the backbone type).
  + 1 reference: pure gdn-neg (the recall/associative-memory workhorse bar, GDN ~0.95).

Half/half mixes use equal logits (0) on the two active type indices and -30 (~ -inf)
elsewhere -> softmax 0.5/0.5 -> 16/16 of n_heads=32. Pure arms put all mass on one
index. gdn vs gdn-neg is the --gdn_allow_neg_eigval flag (plumbed into the typed layer).

ALL E97 heads run FUSED (TypedHeadMixtureLayer use_triton_e97=True default + bf16
autocast in train_hybrid); the loud-guard / no-eager-fallback is proven separately by
verify_e97_within_layer_heads.py. REAL training, REAL data — no mocks.

Probes (existing battery, experiments/expressivity_tasks/tasks/):
  s5_permutation (TRACK), anbncn_viability (COUNT), iterated_nonlinear_map (NONLIN),
  flag_hold_recall K=4 (LATCH), mqar_recall (RECALL).

3 seeds {42,123,456}; train T=128; eval length-extrap T in {128,256,512,1024}.
Idle-GPU scheduling across ALL 8 GPUs (used-mem cap; never preempt a busy GPU).
Resumable (skips existing JSON).
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

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]   # task constraint: all 8 GPUs
SLOTS_PER_GPU = 3                  # tiny models (~6M); the split-edit recurrence is
                                   # latency-bound, 3/GPU saturates util.
MEM_CAP_MIB = 40000               # skip a GPU already above this (leave headroom;
                                   # never preempt another user's job).

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']
TYPED = ['--layer_pattern', 'typed-gdn2', 'typed-gdn2', 'typed-gdn2', 'typed-gdn2']

# config -> (head_type_logits over 8 TYPE_NAMES, gdn_allow_neg_eigval).
# idx: 0 gdn2_recall .. 6 e97_raw, 7 e97_delta.
L = -30.0  # ~ -inf in softmax -> 0 heads
def _logits(active):
    v = [L] * 8
    for i in active:
        v[i] = 0.0
    return ','.join(str(x) for x in v)

CONFIGS: dict[str, tuple[str, int]] = {
    # backbone e97_raw (idx6)
    'raw_none':     (_logits([6]),    1),  # 32 e97_raw            (pure backbone)
    'raw_gdn':      (_logits([0, 6]), 0),  # 16 gdn  + 16 e97_raw
    'raw_gdnneg':   (_logits([0, 6]), 1),  # 16 gdnN + 16 e97_raw
    # backbone e97_delta (idx7)
    'delta_none':   (_logits([7]),    1),  # 32 e97_delta          (pure backbone)
    'delta_gdn':    (_logits([0, 7]), 0),  # 16 gdn  + 16 e97_delta
    'delta_gdnneg': (_logits([0, 7]), 1),  # 16 gdnN + 16 e97_delta
    # reference recall bar
    'gdnneg_ref':   (_logits([0]),    1),  # 32 gdn-neg (recall workhorse)
}

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
    config: str
    seed: int

    @property
    def label(self) -> str:
        return f"wl_{self.probe}__{self.config}__seed{self.seed}"


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
    logits, neg = CONFIGS[job.config]
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.probe,
        *TYPED, *SHARED,
        # '=' form: the logit string can start with '-30', which argparse would
        # otherwise mis-parse as a new flag ("expected one argument").
        f'--head_type_logits={logits}',
        '--gdn_allow_neg_eigval', str(neg),
        *PROBES[job.probe],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', nargs='+', default=list(PROBES.keys()), choices=list(PROBES.keys()))
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
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

    all_jobs = [Job(p, c, s) for p in args.probes for c in args.configs for s in args.seeds]
    jobs = []
    for j in all_jobs:
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs (of {len(all_jobs)}); configs={args.configs} "
          f"probes={args.probes} seeds={args.seeds} steps={args.steps} gpus={GPUS}", flush=True)

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
