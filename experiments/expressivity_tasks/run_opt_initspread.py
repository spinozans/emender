"""opt-initspread — Lever 2 (init-spread / placement) probe runner.

OPT_SPEC.md §5.2. THE question for this probe: on the FROZEN GDN+nonlin
within-layer substrate (the `typed-gdn2` layer), does TUNING THE PLACEMENT — how
heads are allocated across the typed corners, with the spread-init's knob-LR
companion — let ONE trained mixture hold all capability corners simultaneously
(JCC) better than the GDN-2 control `B`?

The substrate `M` is frozen (OPT_SPEC §2.2): typed-gdn2, n_heads=32,
gdn_allow_neg_eigval=1, dim256/n_state32/depth4/mlp_ratio2.0, fp32 (disable
autocast), schedule-free AdamW. The probe varies ONLY the training regime around
it — here the head PLACEMENT (head_type_logits allocation across the typed slots)
and its knob-LR companion (knob_lr_mult). No kernel changes; existing fused cells.

Placement is the typed-substrate analog of the SPECIALIZATION_STUDY spread-init:
on the UnifiedCell the placement lived in a `corner_mixture`/spread-init; on
typed-gdn2 the UnifiedCell sub-block's corner_mixture is AUTO-DERIVED from the
head allocation (typed_head_mixture.py:373), so the placement lever IS the
head_type_logits allocation. The finding under test (memory
`unified-cell-pressure-vs-placement`): only PLACEMENT (not pressure) covers the
corners — spread/fixed-pop init + knob-specific LR.

Counting head: refit-del = refit (TYPE idx 8) with --refit_has_mom 0, i.e. the
momentum-OFF gated-delta special case (== e97_delta exactly, per
ttt-capability). It is fp32-clean; the e97_delta slot (idx 7) crashes under
--disable_autocast (the fused split-edit kernel needs bf16 input but its
projections are fp32). OPT_SPEC §2.2 explicitly sanctions "e97_delta/refit-del
for counting", so refit-del is the faithful, fp32-safe choice.

TYPE_NAMES (typed_head_mixture.py):
  [gdn2_recall, e97_track, count, latch, nonlin, gdn2_nonlin_shell,
       0           1         2      3      4          5
   e97_raw, e97_delta, refit]
      6         7         8

Arms (all typed-gdn2 unless noted):
  B_gdn2          : fla-gdn neg-eigval (CONTROL `B` + recall/track ceiling).
                    LR-screened {3e-4,5e-4,1e-3} seed42, best LR run 3 seeds.
  alldelta        : all refit-del heads (counting/step-growth SPECIALIST+ceiling).
  spread_center   : uniform allocation over the 6 fp32-safe corners, klr1
                    (naive no-skew placement reference).
  house_klr1 (=B2): 50% gdn2_recall + 25% nonlin + 25% refit-del, klr1
                    (the §2.2 house mixture, default regime = substrate-default B2).
  house_klr20     : SAME allocation, knob_lr_mult=20 (the spread-init+knob-LR LEVER).
  uniform4_klr20  : 25% each of recall/track/nonlin/refit-del, klr20.
  skew_klr20      : CMA difficulty-skew (starve recall, feed hard corners), klr20.
  recallheavy_klr20: 62.5% recall + 18.75% nonlin + 18.75% refit-del, klr20
                    (over-allocate recall — placement falsifier).

Battery (OPT_SPEC §3): scored corners recall(mqar_recall),
counting(modular_counter K5 + dyck_depth_unbounded), step-growth(modular_quadratic
K64), track(s5_permutation); parity = saturation control. Train T=128, eval
length-extrap {128,256,512} (controls {128,256}). 3 seeds {42,123,456}.

GPU scheduling is BROKER-AWARE (reads CUDA_VISIBLE_DEVICES from the lease,
round-robins over leased ids). Run under a lease:

    eval "$(scripts/gpu_lease.sh 8)"
    python experiments/expressivity_tasks/run_opt_initspread.py

Resumable: a run whose output JSON exists is skipped.
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

# Frozen substrate shape (OPT_SPEC §2.3) — identical to the TTT battery so the
# control numbers cross-check.
SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0',
          '--mlp_ratio', '2.0', '--depth', '4', '--disable_autocast',
          '--gdn_allow_neg_eigval', '1', '--optimizer', 'schedulefree']

N_HEADS = 32
# TYPE_NAMES slot indices used here.
RECALL, E97_TRACK, COUNT, LATCH, NONLIN, SHELL, E97_RAW, E97_DELTA, REFIT = range(9)
NEG = -30.0  # logit for an unused slot (softmax ~ 0)


def logits_from_fracs(fracs: dict[int, float]) -> str:
    """`--head_type_logits=<9 comma-sep logits>` placing softmax mass = `fracs[slot]`
    on each active slot, NEG elsewhere. logit = ln(frac); softmax over {ln f_i,
    NEG...} recovers f_i. The `=` join is REQUIRED: the NEG logits make the value
    start with '-', which bare-arg argparse would mis-read as a flag."""
    out = [NEG] * 9
    for slot, f in fracs.items():
        out[slot] = math.log(f)
    return '--head_type_logits=' + ','.join(f'{v:.6f}' for v in out)


# Placement allocations (fractions over the 32 heads). refit-del is the counting
# head (REFIT slot + --refit_has_mom 0). nonlin is the step-growth corner.
HOUSE = {RECALL: 0.50, NONLIN: 0.25, REFIT: 0.25}
UNIFORM4 = {RECALL: 0.25, E97_TRACK: 0.25, NONLIN: 0.25, REFIT: 0.25}
# CMA difficulty-skew: starve the corner GDN already aces (recall), feed the hard
# corners (counting/refit, step-growth/nonlin, track). 6/8/9/9 heads -> 32.
SKEW = {RECALL: 0.20, E97_TRACK: 0.26, NONLIN: 0.27, REFIT: 0.27}
RECALLHEAVY = {RECALL: 0.625, NONLIN: 0.1875, REFIT: 0.1875}
# spread_center: uniform over the 6 fp32-safe corners (avoid the e97_raw/e97_delta/
# shell fused-bf16 slots that crash under --disable_autocast).
SAFE6 = {RECALL: 1/6, E97_TRACK: 1/6, COUNT: 1/6, LATCH: 1/6, NONLIN: 1/6, REFIT: 1/6}
ALLDELTA = {REFIT: 1.0}

# arm -> (level, extra flags, knob_lr_mult). refit-del arms carry --refit_has_mom 0.
REFIT_DEL = ['--refit_has_mom', '0']
ARMS: dict[str, tuple[str, list[str], float]] = {
    # control + ceiling source (recall, track)
    'B_gdn2':            ('fla-gdn', [], 1.0),
    # counting / step-growth specialist + ceiling source
    'alldelta':          ('typed-gdn2', [logits_from_fracs(ALLDELTA), *REFIT_DEL], 1.0),
    # placement family
    'spread_center':     ('typed-gdn2', [logits_from_fracs(SAFE6), *REFIT_DEL], 1.0),
    'house_klr1':        ('typed-gdn2', [logits_from_fracs(HOUSE), *REFIT_DEL], 1.0),
    'house_klr20':       ('typed-gdn2', [logits_from_fracs(HOUSE), *REFIT_DEL], 20.0),
    'uniform4_klr20':    ('typed-gdn2', [logits_from_fracs(UNIFORM4), *REFIT_DEL], 20.0),
    'skew_klr20':        ('typed-gdn2', [logits_from_fracs(SKEW), *REFIT_DEL], 20.0),
    'recallheavy_klr20': ('typed-gdn2', [logits_from_fracs(RECALLHEAVY), *REFIT_DEL], 20.0),
}

# Specialist/ceiling arms (subset of ARMS run regardless of LR sweep).
CEILING_ARMS = ['B_gdn2', 'alldelta']

# task -> (K arg or 0, steps, eval_lengths). Hard gaps get the long budget
# (OPT_SPEC §3.2); parity is the cheap saturation control.
BATTERY = [
    ('mqar_recall',          0,  12000, ['128', '256', '512']),  # recall
    ('modular_counter',      5,  12000, ['128', '256', '512']),  # counting
    ('dyck_depth_unbounded', 0,  12000, ['128', '256', '512']),  # counting
    ('modular_quadratic',    64, 12000, ['128', '256', '512']),  # step-growth
    ('s5_permutation',       0,  12000, ['128', '256', '512']),  # track
    ('parity',               0,  5000,  ['128', '256']),         # control (sanity)
]

SEEDS = [42, 123, 456]
LR = 5e-4
B_LR_SCREEN = [3e-4, 5e-4, 1e-3]  # GDN-2 control LR sanity sweep (§4.1)
FORCE_LR = None  # set by --force_lr to run every arm at one explicit LR


@dataclass
class Job:
    task: str
    arm: str
    seed: int
    K: int
    steps: int
    eval_lengths: list
    level: str
    extra: list
    knob_lr: float
    lr: float

    @property
    def label(self) -> str:
        ktag = f"_K{self.K}" if self.K > 0 else ""
        lrtag = "" if abs(self.lr - LR) < 1e-12 else f"_lr{self.lr:g}"
        return f"opt_{self.task}{ktag}__{self.arm}{lrtag}__seed{self.seed}"


def build_jobs(arms_subset, tasks_subset, seeds, b_lr_screen=True,
               b_full_lr_screen=False, steps_override=None):
    jobs = []
    for task, K, steps, evals in BATTERY:
        if tasks_subset and task not in tasks_subset:
            continue
        if steps_override and task in steps_override:
            steps = steps_override[task]
        for arm in arms_subset:
            level, extra, klr = ARMS[arm]
            # GDN-2 control: LR screen for "reasonably tuned, not hobbled" (§4.1).
            #   b_full_lr_screen: every screened LR at every seed (fair 3-seed B at
            #     its best LR, with SE_seed).
            #   b_lr_screen: base LR at all seeds + the other LRs at seed[0] only.
            if arm == 'B_gdn2' and b_full_lr_screen:
                lrs_by_seed = {s: [LR] + [l for l in B_LR_SCREEN if l != LR] for s in seeds}
            elif arm == 'B_gdn2' and b_lr_screen:
                lrs_by_seed = {}
                for s in seeds:
                    lrs_by_seed[s] = [LR] + ([l for l in B_LR_SCREEN if l != LR] if s == seeds[0] else [])
            else:
                lrs_by_seed = {s: [LR] for s in seeds}
            if FORCE_LR is not None:
                lrs_by_seed = {s: [FORCE_LR] for s in seeds}
            for seed in seeds:
                for lr in lrs_by_seed[seed]:
                    jobs.append(Job(task=task, arm=arm, seed=seed, K=K, steps=steps,
                                    eval_lengths=list(evals), level=level,
                                    extra=list(extra), knob_lr=klr, lr=lr))
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
    if job.knob_lr != 1.0:
        cmd += ['--knob_lr_mult', str(job.knob_lr)]
    if job.K > 0:
        cmd += ['--K', str(job.K)]
    return cmd


def leased_gpus():
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is empty — lease GPUs first:\n"
            "    eval \"$(scripts/gpu_lease.sh 8)\"\n"
            "then re-run. (Never hand-pick GPUs; route through the broker.)")
    return [g for g in cvd.split(',') if g.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps_scale', type=float, default=1.0,
                    help='Multiply every task step budget (smoke=0.12).')
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--arms', nargs='+', default=None,
                    help='Subset of arm names (default: all).')
    ap.add_argument('--tasks', nargs='+', default=None,
                    help='Subset of task names (default: full battery).')
    ap.add_argument('--no_b_lr_screen', action='store_true',
                    help='Skip the GDN-2 LR sweep (base LR only).')
    ap.add_argument('--b_full_lr_screen', action='store_true',
                    help='Run every GDN-2 LR at every seed (fair 3-seed best-LR B).')
    ap.add_argument('--force_lr', type=float, default=None,
                    help='Run every arm at this single explicit LR (label encodes it).')
    ap.add_argument('--steps_override', nargs='*', default=None,
                    help='Per-task step overrides as task:steps (e.g. modular_counter:32000).')
    ap.add_argument('--slots_per_gpu', type=int, default=2)
    ap.add_argument('--output_dir', default=str(THIS / 'results_opt_initspread'))
    ap.add_argument('--poll', type=float, default=5.0)
    args = ap.parse_args()

    global FORCE_LR
    FORCE_LR = args.force_lr
    seeds = args.seeds or SEEDS
    arms = args.arms or list(ARMS.keys())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_override = None
    if args.steps_override:
        steps_override = {}
        for kv in args.steps_override:
            t, s = kv.split(':')
            steps_override[t] = int(s)

    gpus = leased_gpus()
    jobs = build_jobs(arms, args.tasks, seeds, b_lr_screen=not args.no_b_lr_screen,
                      b_full_lr_screen=args.b_full_lr_screen, steps_override=steps_override)
    if args.steps_scale != 1.0:
        for j in jobs:
            j.steps = max(50, int(j.steps * args.steps_scale))

    pending = []
    for j in jobs:
        out = out_dir / f'{j.label}.json'
        if out.exists():
            print(f"[skip] {j.label} (exists)", flush=True)
        else:
            pending.append(j)

    print(f"Leased GPUs: {gpus}  slots/gpu={args.slots_per_gpu}  "
          f"jobs: {len(pending)}/{len(jobs)} pending  arms={arms}", flush=True)

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
        cmd = build_cmd(job, args, out_dir)
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                cwd=str(ROOT))
        running[slot_idx] = (proc, job, time.time(), logf)
        print(f"[launch gpu{gpu} slot{slot_idx}] {job.label} ({job.steps} steps, lr={job.lr:g})",
              flush=True)

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
