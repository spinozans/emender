"""opt-headlr — per-head-type learning-rate sweep (OPT_SPEC §5.1).

THE question: on the FROZEN within-layer GDN+nonlin mixture, do RECALL-class heads
(gdn2_recall) and COMPUTE-class heads (counting/step-growth/nonlin) want DIFFERENT
learning rates? A single LR forces a compromise that drops a corner; distinct
per-head-type LRs may let both sides converge onto their corners SIMULTANEOUSLY.
The metric is Joint Capability Coverage (JCC = worst-corner ratio vs frozen
specialist ceilings) at convergent loss (OPT_SPEC §1.3/§1.4).

Substrate `M` (frozen; OPT_SPEC §2.2 "house mixture"): 50% gdn2_recall (neg-eigval
on → recall+track) + 25% refit-del (momentum-off = the gated-delta / e97-delta
EXACT special case → counting) + 25% nonlin (UnifiedCellLayer nonlin corner →
step-growth), at n_heads=32, dim 256, depth 4, mlp_ratio 2.0, fp32.

  NOTE on the counting slice: §2.2 lists "e97_delta/refit-del for counting". The
  e97_delta sub-block's fused split-edit kernel is bf16-only and cannot run under
  the §2.1-mandated fp32 (`--disable_autocast`); refit with momentum OFF is the
  documented EXACT equivalent (refit μ=0 ≡ e97 delta; ttt-triton) and is fp32 +
  fused (proven in the TTT battery). We use refit-del so the whole substrate stays
  fp32 as the spec requires.

Lever (the ONLY thing swept): the LR-multiplier RATIO between recall-class and
compute-class head params (recall-class held at base unless the falsifier):

  arm               recall-mult  compute-mult
  headlr_uniform(B2)   1x           1x       (= substrate default; B2 control)
  headlr_c2            1x           2x
  headlr_c5            1x           5x
  headlr_c10           1x          10x
  headlr_c20           1x          20x
  headlr_rslow         0.5x        10x       (recall gentler, compute driven; CMA pattern)
  headlr_inverted     10x           1x       (PRE-REGISTERED FALSIFIER: drive recall)

Controls (OPT_SPEC §4.1):
  B  (gdn2-default)   fla-gdn allow_neg_eigval=1, default regime. "Reasonably tuned,
                      NOT hobbled": a base-LR sanity sweep {3e-4,5e-4,1e-3}; the best
                      LR is the §1.4 baseline. Also serves as the RECALL + TRACK
                      specialist ceiling (gdn-neg owns both; OPT_SPEC §1.3).
  spec_refit          all-refit-del specialist → COUNTING ceiling.
  spec_nonlin         all-nonlin specialist     → STEP-GROWTH ceiling.

Battery (REAL deterministic generators, REAL training, schedule-free AdamW, fp32):
  recall      mqar_recall                                       (8000)
  counting    modular_counter K=5 / dyck_depth_unbounded        (8000)
              anbncn_viability                                  (5000)
  step-growth modular_quadratic p=64 / iterated_nonlinear_map   (8000)
  track       s5_permutation                                    (8000)
  sanity      parity                                            (5000)

Heavy budget is 8000 steps (the proven-plateau TTT-battery budget at this exact
shape, so the controls cross-check). The §1.5 CONVERGENCE CERTIFICATE — relative
eval-loss improvement over the final 20% of steps, computed by aggregate_opt_headlr.py
— is the REAL convergence gate; any arm whose certificate fails (>2%) is flagged for
a longer-budget re-run (`--heavy_steps 12000`).

GPU scheduling is BROKER-AWARE (reads CUDA_VISIBLE_DEVICES from scripts/gpu_lease.sh
and round-robins jobs over the leased ids). Run under a lease:

    eval "$(scripts/gpu_lease.sh 6)"
    python experiments/expressivity_tasks/run_opt_headlr_battery.py

Resumable: a run whose output JSON already exists is skipped.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]

# Shared shape/harness (OPT_SPEC §2.1/§2.3) — identical to the TTT battery so the
# control numbers cross-check.
SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0',
          '--mlp_ratio', '2.0', '--depth', '4', '--disable_autocast',
          '--gdn_allow_neg_eigval', '1', '--optimizer', 'schedulefree']

# House-mixture head_type_logits: softmax → largest-remainder over 9 typed slots.
# idx0=gdn2_recall, idx4=nonlin, idx8=refit. log-counts give EXACT 16/8/8 at 32 heads.
HOUSE_LOGITS = f"{math.log(16):.8f},-30,-30,-30,{math.log(8):.8f},-30,-30,-30,{math.log(8):.8f}"
REFIT_LOGITS = '0,0,0,0,0,0,0,0,30'   # all heads → refit (idx8)
# NOTE: leading token MUST be non-negative or argparse reads it as a flag; a big
# positive peak on idx4 (nonlin) with 0 elsewhere softmaxes to all-nonlin just the same.
NONLIN_LOGITS = '0,0,0,0,30,0,0,0,0'  # all heads → nonlin (idx4)

# Lever arms on the house mixture (level typed-gdn2; refit slice momentum OFF).
HOUSE_BASE = ['--head_type_logits', HOUSE_LOGITS, '--refit_has_mom', '0']
LEVER_ARMS = {
    'headlr_uniform':  (1.0, 1.0),   # = B2 (substrate default)
    'headlr_c2':       (1.0, 2.0),
    'headlr_c5':       (1.0, 5.0),
    'headlr_c10':      (1.0, 10.0),
    'headlr_c20':      (1.0, 20.0),
    'headlr_rslow':    (0.5, 10.0),
    'headlr_inverted': (10.0, 1.0),  # pre-registered falsifier
}

# corner -> witness tasks (OPT_SPEC §3.1). 'sanity' is not scored in the headline.
SCORED_BATTERY = [
    # (task, K, steps_tier, eval_lengths, corner)
    ('mqar_recall',            0,  'heavy', ['128', '256', '512'], 'recall'),
    ('modular_counter',        5,  'heavy', ['128', '256', '512'], 'counting'),
    ('dyck_depth_unbounded',   0,  'heavy', ['128', '256', '512'], 'counting'),
    ('anbncn_viability',       0,  'light', ['128', '256'],        'counting'),
    ('modular_quadratic',      64, 'heavy', ['128', '256', '512'], 'step_growth'),
    ('iterated_nonlinear_map', 0,  'heavy', ['128', '256', '512'], 'step_growth'),
    ('s5_permutation',         0,  'heavy', ['128', '256', '512'], 'track'),
    ('parity',                 0,  'light', ['128', '256'],        'sanity'),
]
TASK_BY_NAME = {t[0]: t for t in SCORED_BATTERY}

# Which tasks each specialist needs to establish its corner ceiling.
SPEC_REFIT_TASKS = ['modular_counter', 'dyck_depth_unbounded', 'anbncn_viability']
SPEC_NONLIN_TASKS = ['modular_quadratic', 'iterated_nonlinear_map']
# B base-LR sanity probe set (confirms 5e-4 does not hobble the baseline).
B_SANITY_TASKS = ['mqar_recall', 'modular_counter']
B_SANITY_LRS = [3e-4, 1e-3]   # 5e-4 is the full-battery primary baseline

SEEDS = [42, 123, 456]
BASE_LR = 5e-4


@dataclass
class Job:
    task: str
    arm: str
    seed: int
    K: int
    steps: int
    eval_lengths: list
    level: str
    lr: float
    extra: list = field(default_factory=list)

    @property
    def label(self) -> str:
        ktag = f"_K{self.K}" if self.K > 0 else ""
        lrtag = "" if abs(self.lr - BASE_LR) < 1e-12 else f"_lr{self.lr:g}"
        return f"oh_{self.task}{ktag}__{self.arm}{lrtag}__seed{self.seed}"


def _steps(tier, heavy_steps, light_steps):
    return heavy_steps if tier == 'heavy' else light_steps


def build_jobs(heavy_steps, light_steps):
    jobs = []

    def add(task, arm, level, extra, lr, tasks_ok=None):
        if tasks_ok is not None and task not in tasks_ok:
            return
        name, K, tier, evals, corner = TASK_BY_NAME[task]
        for seed in SEEDS:
            jobs.append(Job(task=name, arm=arm, seed=seed, K=K,
                            steps=_steps(tier, heavy_steps, light_steps),
                            eval_lengths=list(evals), level=level, lr=lr,
                            extra=list(extra)))

    # 1) Lever arms — full scored battery.
    for arm, (rmult, cmult) in LEVER_ARMS.items():
        extra = HOUSE_BASE + ['--head_lr_recall_mult', str(rmult),
                              '--head_lr_compute_mult', str(cmult)]
        for task in TASK_BY_NAME:
            add(task, arm, 'typed-gdn2', extra, BASE_LR)

    # 2) B (GDN-2 default) — full battery at base LR (primary baseline + recall/track ceiling).
    for task in TASK_BY_NAME:
        add(task, 'gdn2-default', 'fla-gdn', [], BASE_LR)
    # B base-LR sanity sweep (a few tasks at the alt LRs).
    for lr in B_SANITY_LRS:
        for task in B_SANITY_TASKS:
            add(task, 'gdn2-default', 'fla-gdn', [], lr, tasks_ok=B_SANITY_TASKS)

    # 3) Specialists for the frozen ceilings (own-corner tasks only).
    for task in SPEC_REFIT_TASKS:
        add(task, 'spec_refit', 'typed-gdn2',
            ['--head_type_logits', REFIT_LOGITS, '--refit_has_mom', '0'], BASE_LR)
    for task in SPEC_NONLIN_TASKS:
        add(task, 'spec_nonlin', 'typed-gdn2',
            ['--head_type_logits', NONLIN_LOGITS], BASE_LR)

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
        '--lr', str(job.lr),
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
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is empty — lease GPUs first:\n"
            "    eval \"$(scripts/gpu_lease.sh 6)\"\n"
            "then re-run. (Never hand-pick GPUs; route through the broker.)")
    return [g for g in cvd.split(',') if g.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heavy_steps', type=int, default=8000,
                    help='Step budget for hard-gap tasks (default 8000 = proven-plateau '
                         'TTT budget; the convergence certificate is the real gate).')
    ap.add_argument('--light_steps', type=int, default=5000)
    ap.add_argument('--steps_scale', type=float, default=1.0,
                    help='Multiply every step budget (smoke=0.02).')
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--slots_per_gpu', type=int, default=3)
    ap.add_argument('--only_arms', type=str, default=None,
                    help='Comma-sep arm names to restrict to (debug).')
    ap.add_argument('--only_tasks', type=str, default=None,
                    help='Comma-sep task names to restrict to (e.g. re-run slow tasks '
                         'at a longer --heavy_steps).')
    ap.add_argument('--force', action='store_true',
                    help='Re-run even if the output JSON exists (overwrites).')
    ap.add_argument('--output_dir', default=str(THIS / 'results_opt_headlr'))
    ap.add_argument('--poll', type=float, default=5.0)
    args = ap.parse_args()

    global SEEDS
    if args.seeds:
        SEEDS = args.seeds

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus = leased_gpus()
    jobs = build_jobs(args.heavy_steps, args.light_steps)
    if args.only_arms:
        keep = set(args.only_arms.split(','))
        jobs = [j for j in jobs if j.arm in keep]
    if args.only_tasks:
        keept = set(args.only_tasks.split(','))
        jobs = [j for j in jobs if j.task in keept]
    if args.steps_scale != 1.0:
        for j in jobs:
            j.steps = max(50, int(j.steps * args.steps_scale))

    pending = []
    for j in jobs:
        if (out_dir / f'{j.label}.json').exists() and not args.force:
            print(f"[skip] {j.label} (exists)", flush=True)
        else:
            pending.append(j)

    print(f"Leased GPUs: {gpus}  slots/gpu={args.slots_per_gpu}  "
          f"jobs: {len(pending)}/{len(jobs)} pending", flush=True)

    slots = []
    for g in gpus:
        for _ in range(args.slots_per_gpu):
            slots.append(g)

    running = {}
    queue = list(pending)
    t_start = time.time()
    done = 0

    def launch(slot_idx, job):
        gpu = slots[slot_idx]
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        logf = open(out_dir / f'{job.label}.log', 'w')
        proc = subprocess.Popen(build_cmd(job, args, out_dir), stdout=logf,
                                stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        running[slot_idx] = (proc, job, time.time(), logf)
        print(f"[launch gpu{gpu} slot{slot_idx}] {job.label} ({job.steps} steps)", flush=True)

    while queue or running:
        for slot_idx in range(len(slots)):
            if slot_idx not in running and queue:
                launch(slot_idx, queue.pop(0))
        time.sleep(args.poll)
        for slot_idx in list(running.keys()):
            proc, job, t0, logf = running[slot_idx]
            if proc.poll() is not None:
                logf.close()
                dt = time.time() - t0
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
