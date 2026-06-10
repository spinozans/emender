"""mlp-mem capability battery (task nlmem-capability, NONLIN_MEMORY_SPEC.md §8.1).

THE question: does the NONLINEAR MLP-memory cell (`mlp-mem`: state = params of a
1-hidden-layer MLP, read `W2 sigma(W1 q)` nonlinear in the query) store mappings a
LINEAR matrix memory (GDN-2 / gated-delta, read `S q` linear in the query) provably
cannot — canonically a non-bilinear key->value association (XOR) or a mod-k nonlinear
map? REAL tasks, REAL training, matched compute.

Matched A/B — identical FLA-GatedDeltaNet shell (projections / short-conv / L2-norm /
output-gate / o_norm / o_proj), the ONLY change is the recurrent cell:
  mlpmem : level 'mlp-mem'  -> MLP fast-weight memory (nonlinear read)   [the head]
  gdn2   : level 'fla-gdn'  -> matrix gated-delta memory (linear read)   [the baseline]
                              with allow_neg_eigval=1 (the GDN-2 reflect/decay anchor).
Both: same dim/heads/n_state/depth/mlp_ratio/steps/batch, fp32 (disable_autocast —
exact algorithmic tasks; mlp-mem runs its fused fp32 sequential Triton kernel). Params
are matched to <1% by construction (mlp-mem adds only two per-head scalar gates).

Battery (task -> K):
  boolean_assoc        (0) NON-bilinear XOR association  <- THE separator probe (§8.1.1)
  boolean_assoc_lin    (0) linear single-bit recall      <- matched linear control
  mqar_recall          (0) multi-query recall capacity   <- crosstalk probe (§8.1.2)
  assoc_recall         (0) single-query recall           <- linear recall control
  modular_counter      (5) mod-k running counter (LINEAR group action)  <- mod-k control
  modular_quadratic    (0) x_t=(x^2+c) mod p nonlinear    <- nonlinear mod-k sep (§8.1.4)
  iterated_nonlinear_map (0) logistic map state-nonlin    <- nonlinear-state sep (§8.1.4)
  parity               (0) mod-2 running parity           <- mod-2 control

Mechanism-localization ablation (mlp_ratio=0, the post-head SwiGLU readout REMOVED) on
the discriminating subset {boolean_assoc, boolean_assoc_lin, modular_quadratic}: if a
nonlinear-read win appears ONLY when the model-level MLP is removed, the memory
nonlinearity is real but REDUNDANT with the readout — the convergent-loss-null mechanism
this lab keeps finding (e97-within-layer, complex-eig-capability).

GPUs are claimed via the gpu-broker (this process must already hold a lease; pass the
granted GLOBAL ids with --gpus). 3 seeds. Idle-GPU scheduling; resumable (skips
existing <label>.json).
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

SLOTS_PER_GPU = 1            # mlp-mem holds a T-step sequential-scan autograd graph -> heavy
MEM_CAP_MIB = 60000

SHARED = ['--dim', '256', '--n_heads', '4', '--n_state', '32', '--expansion', '1.0',
          '--depth', '4', '--disable_autocast', '--optimizer', 'schedulefree']

# arm -> (layer level, extra flags). EXACT matched A/B: both arms are the SAME
# MlpMemHeadLayer shell (identical projections/conv/gate/o_proj, 0% param diff); the
# ONLY difference is the recurrent cell — mlpmem = nonlinear MLP fast-weight memory,
# gdn2 = spec §2.3 degenerate LINEAR corner (FLA chunked gated-delta, the GDN-2 anchor
# with allow_neg_eigval reflect/decay eigenvalues on by default).
ARMS = {
    'mlpmem': ('mlp-mem', []),
    'gdn2':   ('gdn-matched', []),
}

# battery: (task, K)   K=0 -> task default / no K
BATTERY = [
    ('boolean_assoc', 0),
    ('boolean_assoc_lin', 0),
    ('mqar_recall', 0),
    ('assoc_recall', 0),
    ('modular_counter', 5),
    ('modular_quadratic', 0),
    ('iterated_nonlinear_map', 0),
    ('parity', 0),
]
BATTERY_EVAL = ['128']            # train T=128 (mlp-mem sequential scan: single-length eval for speed)

# mechanism-localization: mlp_ratio=0 on the discriminating subset
ABLATE_TASKS = [('boolean_assoc', 0), ('boolean_assoc_lin', 0), ('modular_quadratic', 0)]

SEEDS = [42, 123, 456]


@dataclass
class Job:
    task: str
    arm: str
    seed: int
    K: int = 0
    extra: list = field(default_factory=list)
    level: str = 'mlp-mem'
    mlp_ratio: float = 2.0
    eval_lengths: list = field(default_factory=lambda: list(BATTERY_EVAL))

    @property
    def label(self) -> str:
        ktag = f"_K{self.K}" if self.K > 0 else ""
        mtag = "" if self.mlp_ratio == 2.0 else f"_mlp{self.mlp_ratio:g}"
        return f"mm_{self.task}{ktag}{mtag}__{self.arm}__seed{self.seed}"


def build_jobs(mlp_ratio: float, tasks):
    jobs = []
    for task, K in tasks:
        for arm, (level, extra) in ARMS.items():
            for seed in SEEDS:
                jobs.append(Job(task=task, arm=arm, seed=seed, K=K, extra=list(extra),
                                level=level, mlp_ratio=mlp_ratio,
                                eval_lengths=list(BATTERY_EVAL)))
    return jobs


def gpu_used_mib(gpus):
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def build_cmd(job: Job, args, out_dir: Path):
    pattern = [job.level] * args.depth
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.task,
        '--layer_pattern', *pattern,
        *SHARED,
        '--mlp_ratio', str(job.mlp_ratio),
        *job.extra,
        '--steps', str(args.steps),
        '--seq_len', '128',
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', *job.eval_lengths,
        '--eval_lengths_n_batches', str(args.eval_n_batches),
    ]
    if job.K > 0:
        cmd += ['--K', str(job.K)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', required=True,
                    help='csv of GLOBAL nvidia GPU ids this process holds a broker lease on')
    ap.add_argument('--which', choices=['battery', 'ablate', 'both'], default='both')
    ap.add_argument('--tasks', nargs='+', default=None,
                    help='restrict to these task names (smoke / partial reruns)')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--output_dir', default=str(THIS / 'results_mlp_mem'))
    ap.add_argument('--poll', type=float, default=10.0)
    args = ap.parse_args()

    gpus = [int(g) for g in str(args.gpus).split(',') if g.strip() != '']
    assert gpus, "no GPUs given"

    global SEEDS
    if args.seeds:
        SEEDS = args.seeds

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = []
    if args.which in ('battery', 'both'):
        all_jobs += build_jobs(2.0, BATTERY)
    if args.which in ('ablate', 'both'):
        all_jobs += build_jobs(0.0, ABLATE_TASKS)

    if args.tasks:
        keep = set(args.tasks)
        all_jobs = [j for j in all_jobs if j.task in keep]

    jobs = []
    for j in all_jobs:
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
            continue
        jobs.append(j)
    print(f"[plan] {len(jobs)} jobs (of {len(all_jobs)}); which={args.which} "
          f"steps={args.steps} seeds={SEEDS} gpus={gpus}", flush=True)

    running = []
    pending = list(jobs)
    # Parent already holds the broker lease; per-child CUDA_VISIBLE_DEVICES uses the
    # GLOBAL leased ids. Drop any inherited (broker-remapped) CUDA_VISIBLE_DEVICES so
    # the absolute global id we set per child is honoured.
    env_base = dict(os.environ, PYTHONPATH=str(ROOT))
    env_base.pop('CUDA_VISIBLE_DEVICES', None)

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
            used = gpu_used_mib(gpus)
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
