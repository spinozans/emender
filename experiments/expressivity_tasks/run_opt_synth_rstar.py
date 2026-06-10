"""opt-synth — R* composition re-run at small scale (OPT_SPEC §6.4).

Composes the per-lever winners onto the minimal-core house mixture and re-runs
the full scored battery at the §2.3 small shape (3 seeds), at IDENTICAL harness,
to confirm the composed JCC before paying for 1.3B. Because levers can interact
(additivity is NOT assumed), this is a real re-run, not an assumption.

Composition decision (synth, from the four probe RESULTS):
  - PLACEMENT (opt-initspread GO lever): the house mixture — 50% gdn2_recall
    (neg-eig on -> recall+track) + 25% refit-del (mom-off = e97_delta exact ->
    counting) + 25% nonlin (-> step-growth), n_heads=32. This IS the minimal
    load-bearing core (opt-minimal: every recurrent piece load-bearing; MLP
    removable for capability but KEPT here because it matters for LM bpb at 1.3B).
  - PER-HEAD-TYPE LR (opt-headlr GO lever): head_lr_compute_mult=5 (the c5 winner;
    10-20x backfires). Chosen OVER initspread's knob_lr_mult=20 because (a) c5 is
    the larger, more reliable pure contribution (+0.021 vs the knob-LR +0.015,
    which is at the edge of seed noise), (b) the two LR levers are MUTUALLY
    EXCLUSIVE in the trainer's param-grouping (train_hybrid.py:489-532), and
    (c) both drive the compute side, so stacking risks the documented compute-side
    over-drive collapse. initspread's *actual* GO lever was PLACEMENT (already in
    the house mixture), not the knob-LR companion.
  - DECAY INIT (opt-norm, NULL lever but free + directionally-correct):
    decay_init=slow (retention->1) — the one positive parameterization axis
    (sub-Delta* but never harmful; the only knob opt-norm forwards).

Arms (identical harness; isolates the interaction the synth must confirm):
  b2_house  house default, no lever           (anchor = B2)
  c5        house + head_lr_compute_mult=5     (opt-headlr winner, re-measured)
  klr20     house + knob_lr_mult=20            (opt-initspread knob-LR, re-measured)
  rstar     house + head_lr_compute_mult=5 + decay_init=slow   (the composed R*)

The aggregate step (aggregate_opt_synth.py --rstar) scores these against the
reconciled ceilings and reports whether JCC(rstar) >= max(JCC(c5), JCC(klr20))
(=> composition holds) or falls back to the best single lever.

Broker-leased GPUs only:  eval "$(scripts/gpu_lease.sh 8)" && python <this>
Resumable: a job whose output JSON exists is skipped.
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

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0',
          '--mlp_ratio', '2.0', '--depth', '4', '--disable_autocast',
          '--gdn_allow_neg_eigval', '1', '--optimizer', 'schedulefree']

# House placement: 16 gdn2_recall / 8 nonlin / 8 refit on 32 heads (exact via log-counts).
HOUSE_LOGITS = f"{math.log(16):.8f},-30,-30,-30,{math.log(8):.8f},-30,-30,-30,{math.log(8):.8f}"
HOUSE_BASE = ['--head_type_logits', HOUSE_LOGITS, '--refit_has_mom', '0']

# Each arm = the extra flags layered on the house mixture.
ARMS = {
    'b2_house': [],
    'c5':       ['--head_lr_recall_mult', '1.0', '--head_lr_compute_mult', '5.0'],
    'klr20':    ['--knob_lr_mult', '20.0'],
    'rstar':    ['--head_lr_recall_mult', '1.0', '--head_lr_compute_mult', '5.0',
                 '--decay_init', 'slow'],
}

# (task, K, steps, eval_lengths, corner). modular_counter/modular_quadratic at 16000
# (still-climbing at 8000 per all four probes); other heavy 8000; light 5000.
BATTERY = [
    ('mqar_recall',            0,  8000,  ['128', '256', '512'], 'recall'),
    ('modular_counter',        5,  16000, ['128', '256', '512'], 'counting'),
    ('dyck_depth_unbounded',   0,  8000,  ['128', '256', '512'], 'counting'),
    ('anbncn_viability',       0,  5000,  ['128', '256'],        'counting'),
    ('modular_quadratic',      64, 16000, ['128', '256', '512'], 'step_growth'),
    ('iterated_nonlinear_map', 0,  8000,  ['128', '256', '512'], 'step_growth'),
    ('s5_permutation',         0,  8000,  ['128', '256', '512'], 'track'),
    ('parity',                 0,  5000,  ['128', '256'],        'sanity'),
]
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
    extra: list = field(default_factory=list)

    @property
    def label(self):
        ktag = f"_K{self.K}" if self.K > 0 else ""
        return f"rs_{self.task}{ktag}__{self.arm}__seed{self.seed}"


def build_jobs(steps_scale):
    jobs = []
    for arm, lever in ARMS.items():
        extra = HOUSE_BASE + lever
        for task, K, steps, evals, corner in BATTERY:
            for seed in SEEDS:
                jobs.append(Job(task, arm, seed, K, max(50, int(steps * steps_scale)),
                                list(evals), list(extra)))
    return jobs


def build_cmd(job, args, out_dir):
    cmd = ['python', str(THIS / 'train_hybrid.py'),
           '--task', job.task,
           '--layer_pattern', *(['typed-gdn2'] * args.depth),
           *SHARED, *job.extra,
           '--steps', str(job.steps), '--seq_len', '128',
           '--batch_size', str(args.batch_size), '--lr', str(BASE_LR),
           '--seed', str(job.seed), '--label', job.label,
           '--output_dir', str(out_dir),
           '--eval_lengths', *job.eval_lengths,
           '--eval_lengths_n_batches', str(args.eval_n_batches)]
    if job.K > 0:
        cmd += ['--K', str(job.K)]
    return cmd


def leased_gpus():
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit("CUDA_VISIBLE_DEVICES empty — lease first: "
                         'eval "$(scripts/gpu_lease.sh 8)"')
    return [g for g in cvd.split(',') if g.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--slots_per_gpu', type=int, default=3)
    ap.add_argument('--steps_scale', type=float, default=1.0)
    ap.add_argument('--only_arms', type=str, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--output_dir', default=str(THIS / 'results_opt_synth'))
    ap.add_argument('--poll', type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = leased_gpus()
    jobs = build_jobs(args.steps_scale)
    if args.only_arms:
        keep = set(args.only_arms.split(','))
        jobs = [j for j in jobs if j.arm in keep]

    pending = [j for j in jobs
               if args.force or not (out_dir / f'{j.label}.json').exists()]
    print(f"Leased GPUs: {gpus} slots/gpu={args.slots_per_gpu}  "
          f"{len(pending)}/{len(jobs)} pending", flush=True)

    slots = [g for g in gpus for _ in range(args.slots_per_gpu)]
    running, queue, t0all, done = {}, list(pending), time.time(), 0

    def launch(si, job):
        env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(slots[si])
        logf = open(out_dir / f'{job.label}.log', 'w')
        proc = subprocess.Popen(build_cmd(job, args, out_dir), stdout=logf,
                                stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        running[si] = (proc, job, time.time(), logf)
        print(f"[launch gpu{slots[si]} slot{si}] {job.label} ({job.steps} steps)", flush=True)

    while queue or running:
        for si in range(len(slots)):
            if si not in running and queue:
                launch(si, queue.pop(0))
        time.sleep(args.poll)
        for si in list(running.keys()):
            proc, job, t0, logf = running[si]
            if proc.poll() is not None:
                logf.close()
                has = (out_dir / f'{job.label}.json').exists()
                ok = (proc.returncode in (0, -15)) and has
                done += 1
                print(f"[{'ok' if ok else f'FAIL(rc={proc.returncode},json={has})'} "
                      f"{time.time()-t0:.0f}s] {job.label} ({done}/{len(pending)}, "
                      f"{time.time()-t0all:.0f}s elapsed)", flush=True)
                del running[si]
    print(f"\nR* battery complete: {done} runs in {time.time()-t0all:.0f}s -> {out_dir}",
          flush=True)


if __name__ == '__main__':
    main()
