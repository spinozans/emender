"""TTT-write (`refit`) capability battery — task `ttt-capability`.

THE question: does the richer EMENDATION of the TTT inner-optimization write head
(`refit` — one heavy-ball / momentum-delta inner-optimizer step per token) unlock
any REAL capability the GDN-2 (gated-delta) baseline cannot reach — especially the
gaps nothing has solved yet: mod-k counting (`modular_counter`) and the nonlinear /
step-growth and unbounded-counting cliffs (`modular_quadratic`, `dyck_depth_unbounded`)?

Matched A/B (identical layer class / params / kernel / compute — all three arms are
the SAME typed-gdn2 layer with ALL heads forced onto the `refit` slot, fp32):

  refit-mom : head_type_logits mass on refit (idx 8), --refit_has_mom 1
              -> full momentum-delta inner optimizer (the `refit` cell).
  refit-del : same layer, --refit_has_mom 0
              -> momentum OFF = the gated-delta / e97 special case (delta = ONE
                 inner step). The CLEAN matched ablation: momentum is the ONLY
                 removed degree of freedom, params identical.

Plus a production anchor:

  gdn2      : fla-gdn allow_neg_eigval=1 -> the chunked FLA GatedDeltaNet that the
              `refit` delta special case is the same algorithmic class as.

refit-mom vs refit-del isolates *exactly* "does the richer (momentum) inner
optimization buy capability"; gdn2 is the external production baseline.

Battery (REAL deterministic generators, REAL training, schedule-free AdamW, fp32):
  HARD GAPS (the open ones; long budget, length-extrap eval):
    modular_counter K=5          mod-k running counter (TC0-outside for unbounded T)
    modular_quadratic p=64       nonlinear non-invertible step-growth cliff
    dyck_depth_unbounded         UNBOUNDED counting (a^n b^n-style depth)
  CONTROLS (cheap; sanity that the arms learn and that GDN keeps its home turf):
    parity                       mod-2 saturation control (all arms should solve)
    mqar_recall                  associative recall — GDN's strength; refit must not lose it

All arms: dim 256, 32 heads, n_state 32, depth 4, mlp_ratio 2.0 (the FIXED O(depth)
nonlinear readout present in every arm), schedule-free AdamW, fp32 (disable_autocast
— exact algorithmic tasks). Train T=128, eval length-extrapolation. 3 seeds.

GPU scheduling is BROKER-AWARE: it reads the leased physical GPU ids from
CUDA_VISIBLE_DEVICES (set by scripts/gpu_lease.sh) and round-robins jobs across
ONLY those leased GPUs. Run under a lease:

    eval "$(scripts/gpu_lease.sh 4)"
    python experiments/expressivity_tasks/run_ttt_capability_battery.py

Resumable: a run whose output JSON already exists is skipped.
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

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0',
          '--mlp_ratio', '2.0', '--depth', '4', '--disable_autocast',
          '--gdn_allow_neg_eigval', '1', '--optimizer', 'schedulefree']

# All-refit allocation: softmax over 9 typed slots with a big value on idx 8 (refit)
# -> largest-remainder gives every head to refit.
REFIT_LOGITS = '0,0,0,0,0,0,0,0,30'

# arm -> (layer level, extra flags). refit-mom / refit-del are the matched A/B
# (same typed-gdn2 layer, momentum toggled); gdn2 is the production anchor.
ARMS = {
    'refit-mom': ('typed-gdn2', ['--head_type_logits', REFIT_LOGITS, '--refit_has_mom', '1']),
    'refit-del': ('typed-gdn2', ['--head_type_logits', REFIT_LOGITS, '--refit_has_mom', '0']),
    'gdn2':      ('fla-gdn', []),
}

# task -> (K arg or 0, steps, eval_lengths). K maps per train_hybrid: modular_counter->K,
# modular_quadratic->p modulus. Hard gaps get the long budget; controls are cheap.
BATTERY = [
    ('modular_counter',      5,  8000, ['128', '256', '512']),
    ('modular_quadratic',    64, 8000, ['128', '256', '512']),
    ('dyck_depth_unbounded', 0,  8000, ['128', '256', '512']),
    ('parity',               0,  5000, ['128', '256']),
    ('mqar_recall',          0,  5000, ['128', '256']),
]

SEEDS = [42, 123, 456]
LR = 5e-4


@dataclass
class Job:
    task: str
    arm: str
    seed: int
    K: int
    steps: int
    eval_lengths: list
    level: str
    extra: list = field(default_factory=list)

    @property
    def label(self) -> str:
        ktag = f"_K{self.K}" if self.K > 0 else ""
        return f"ttt_{self.task}{ktag}__{self.arm}__seed{self.seed}"


def build_jobs():
    jobs = []
    for task, K, steps, evals in BATTERY:
        for arm, (level, extra) in ARMS.items():
            for seed in SEEDS:
                jobs.append(Job(task=task, arm=arm, seed=seed, K=K, steps=steps,
                                eval_lengths=list(evals), level=level, extra=list(extra)))
    return jobs


def build_cmd(job: Job, args, out_dir: Path):
    pattern = [job.level] * args.depth
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.task,
        '--layer_pattern', *pattern,
        *SHARED,
        *job.extra,
        '--steps', str(job.steps),
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


def leased_gpus():
    """Physical GPU ids the broker leased to this shell (CUDA_VISIBLE_DEVICES).

    Each child is launched with CUDA_VISIBLE_DEVICES set to ONE of these physical
    ids. CUDA_VISIBLE_DEVICES is read absolutely (not composed) by each process,
    so passing a leased physical id to the child targets exactly that GPU.
    """
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is empty — lease GPUs first:\n"
            "    eval \"$(scripts/gpu_lease.sh 4)\"\n"
            "then re-run. (Never hand-pick GPUs; route through the broker.)")
    return [g for g in cvd.split(',') if g.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps_scale', type=float, default=1.0,
                    help='Multiply every task step budget (smoke=0.05).')
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--slots_per_gpu', type=int, default=2)
    ap.add_argument('--output_dir', default=str(THIS / 'results_ttt_capability'))
    ap.add_argument('--poll', type=float, default=5.0)
    args = ap.parse_args()

    global SEEDS
    if args.seeds:
        SEEDS = args.seeds

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus = leased_gpus()
    jobs = build_jobs()
    if args.steps_scale != 1.0:
        for j in jobs:
            j.steps = max(50, int(j.steps * args.steps_scale))

    # Skip already-finished runs (resumable).
    pending = []
    for j in jobs:
        out = out_dir / f'{j.label}.json'
        if out.exists():
            print(f"[skip] {j.label} (exists)", flush=True)
        else:
            pending.append(j)

    print(f"Leased GPUs: {gpus}  slots/gpu={args.slots_per_gpu}  "
          f"jobs: {len(pending)}/{len(jobs)} pending", flush=True)

    # slot -> (gpu_id). Build the concurrency pool.
    slots = []
    for g in gpus:
        for _ in range(args.slots_per_gpu):
            slots.append(g)

    running = {}  # slot_idx -> (proc, job, t0)
    queue = list(pending)
    t_start = time.time()
    done = 0

    def launch(slot_idx, job):
        gpu = slots[slot_idx]
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        logf = open(out_dir / f'{job.label}.log', 'w')
        cmd = build_cmd(job, args, out_dir)
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                cwd=str(ROOT))
        running[slot_idx] = (proc, job, time.time(), logf)
        print(f"[launch gpu{gpu} slot{slot_idx}] {job.label} ({job.steps} steps)", flush=True)

    while queue or running:
        # Fill free slots.
        for slot_idx in range(len(slots)):
            if slot_idx not in running and queue:
                launch(slot_idx, queue.pop(0))
        # Poll for completions.
        time.sleep(args.poll)
        for slot_idx in list(running.keys()):
            proc, job, t0, logf = running[slot_idx]
            if proc.poll() is not None:
                logf.close()
                dt = time.time() - t0
                # A job counts as ok if it wrote a valid JSON, even if it caught a
                # spurious SIGTERM during interpreter teardown (rc=-15 after the
                # "Saved to ..." line). Missing JSON => real failure, picked up on a
                # resumable re-run.
                has_json = (out_dir / f'{job.label}.json').exists()
                ok = (proc.returncode == 0 or proc.returncode == -15) and has_json
                done += 1
                status = 'ok' if ok else f'FAIL(rc={proc.returncode},json={has_json})'
                print(f"[{status} {dt:.0f}s] {job.label}  "
                      f"({done}/{len(pending)} done, {time.time()-t_start:.0f}s elapsed)",
                      flush=True)
                del running[slot_idx]

    print(f"\nBattery complete: {done} runs in {time.time()-t_start:.0f}s -> {out_dir}",
          flush=True)


if __name__ == '__main__':
    main()
