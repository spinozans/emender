"""opt-norm — Lever 3 probe: normalization placement, gate-bias init, decay /
A_log / dt_bias init on the GDN+nonlin mixture (OPT_SPEC.md §5.3).

THE question: with the function class FROZEN, does a better JOINT init of the
gate / decay / norm parameterization let ONE trained mixture hold all capability
corners (recall + counting + step-growth + track) at convergent loss — i.e. beat
the GDN-2 incumbent on the worst-corner JCC metric (§1.3) by the pre-registered
bar (§1.4)?  Or is the convergent-loss null (architecture exhausted) ALSO robust
to this optimization lever?

Substrate `M` (FROZEN; OPT_SPEC.md §2.2 house mixture, fp32-clean variant):
  50% gdn2_recall (allow_neg_eigval -> recall+track) + 25% nonlin + 25% refit-del.
  (refit with momentum OFF = the gated-delta special case = the §2.2-named
   counting substrate; the e97_delta fused split-edit kernel rejects fp32, which
   §2.1 mandates, so refit-del is the fp32-clean realization of the same write
   rule the TTT battery validated.)

Lever arms (one knob off the §2.2/§5.1 defaults at a time; combined re-tested):
  decay/A_log/dt_bias init : decay_slow (retention->1) / decay_fast (retention->0)
  gate-bias init           : gate_open (+2) / gate_closed (-2)   [vs default no-bias]
  decay eigenvalue cap     : lam_1.0 / lam_1.3                   [vs 1.585 default]
  reflection depth (track) : beta_1.5 / beta_2.0                 [vs 2.747 default]
  norm placement           : norm_off (unified head-norm OFF)    [vs ON default]
  combined_best            : best single value of each axis, RE-TESTED (added after
                             the single-axis sweep; levers may interact).

Controls / specialists (OPT_SPEC.md §4.1, §1.3 frozen ceilings):
  B  = GDN-2 fair-default  : fla-gdn allow_neg_eigval, LR sanity {3e-4,5e-4,1e-3}
                             (reasonably tuned, NOT hobbled; best LR reported as B).
                             Also the recall+track specialist => ceilings S_recall,
                             S_track.
  B2 = substrate-default   : the house mixture trained with the DEFAULT regime
                             (no lever). JCC(R)-JCC(B2) is the lever's pure effect.
  spec_refit = all-refit-del : counting + step-growth specialist => ceilings
                             S_counting, S_step_growth.

Battery (REAL deterministic generators, REAL training, schedule-free AdamW, fp32):
  recall      : mqar_recall                                   (12000 steps)
  counting    : modular_counter K5 / dyck_depth_unbounded     (12000) +
                anbncn_viability                              (5000)
  step-growth : modular_quadratic K64 / iterated_nonlinear_map (12000)
  track       : s5_permutation                                (12000)
  sanity      : parity                                        (5000; all arms solve)

Train T=128; eval length-extrapolation {128,256,512} (hard) / {128,256} (control).
3 seeds {42,123,456} for every JCC-scored arm; the two non-default B LRs use seed
42 only (LR selection). Resumable: a run whose JSON exists is skipped.

GPU scheduling is BROKER-AWARE (reads CUDA_VISIBLE_DEVICES from the lease, round-
robins jobs over leased GPUs). Run under a lease:

    eval "$(scripts/gpu_lease.sh 8)"
    python experiments/expressivity_tasks/run_opt_norm.py --slots_per_gpu 3
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

# Shared shape / harness (OPT_SPEC.md §2.1, §2.3) — identical across all arms.
SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0',
          '--mlp_ratio', '2.0', '--depth', '4', '--disable_autocast',
          '--optimizer', 'schedulefree']

# House mixture (OPT_SPEC.md §2.2): 50% gdn2_recall(idx0) + 25% nonlin(idx4) +
# 25% refit(idx8). softmax(30, .., 29.30685, .., 29.30685)->0.5/0.25/0.25 on 32 heads.
HOUSE = '30,0,0,0,29.30685,0,0,0,29.30685'
# All-refit specialist (counting/step-growth ceiling): all mass on refit (idx8).
ALL_REFIT = '0,0,0,0,0,0,0,0,30'

# Frozen house defaults so every typed arm is matched except its swept knob.
HOUSE_BASE = ['--head_type_logits', HOUSE, '--gdn_allow_neg_eigval', '1',
              '--refit_has_mom', '0', '--lam_max', '1.585', '--beta_max', '2.747']


def typed(*overrides):
    """typed-gdn2 house-mixture flag list with knob overrides appended (last wins)."""
    return ['typed-gdn2', list(HOUSE_BASE) + list(overrides)]


# arm name -> (level, extra flags, lr, seeds). lr default 5e-4 (battery base, §3.2).
DEF_SEEDS = [42, 123, 456]
LR = 5e-4
ARMS = {
    # --- controls / specialists ---
    'B_gdn_lr5e4':  ('fla-gdn', ['--gdn_allow_neg_eigval', '1'], 5e-4, DEF_SEEDS),
    'B_gdn_lr3e4':  ('fla-gdn', ['--gdn_allow_neg_eigval', '1'], 3e-4, [42]),
    'B_gdn_lr1e3':  ('fla-gdn', ['--gdn_allow_neg_eigval', '1'], 1e-3, [42]),
    'spec_refit':   ('typed-gdn2', ['--head_type_logits', ALL_REFIT,
                                    '--gdn_allow_neg_eigval', '1', '--refit_has_mom', '0'],
                     5e-4, DEF_SEEDS),
    'B2_default':   (*typed(), 5e-4, DEF_SEEDS),
    # --- lever arms (one knob each off the house default) ---
    'decay_slow':   (*typed('--decay_init', 'slow'), 5e-4, DEF_SEEDS),
    'decay_fast':   (*typed('--decay_init', 'fast'), 5e-4, DEF_SEEDS),
    'gate_open':    (*typed('--gate_bias_init', '2.0'), 5e-4, DEF_SEEDS),
    'gate_closed':  (*typed('--gate_bias_init', '-2.0'), 5e-4, DEF_SEEDS),
    'lam_1.0':      (*typed('--lam_max', '1.0'), 5e-4, DEF_SEEDS),
    'lam_1.3':      (*typed('--lam_max', '1.3'), 5e-4, DEF_SEEDS),
    'beta_1.5':     (*typed('--beta_max', '1.5'), 5e-4, DEF_SEEDS),
    'beta_2.0':     (*typed('--beta_max', '2.0'), 5e-4, DEF_SEEDS),
    'norm_off':     (*typed('--unified_head_norm', '0'), 5e-4, DEF_SEEDS),
    # --- combined arm: filled by compose_combined() after the single-axis sweep ---
    # 'combined_best': added programmatically below if OPT_NORM_COMBINED is set.
}

# task -> (K arg or 0, steps, eval_lengths). Hard gaps get 8000 steps — the
# TTT-battery budget that demonstrably plateaus (the convergence certificate §1.5
# is the per-run guard; any row failing <2% is flagged for a longer re-run).
BATTERY = [
    ('mqar_recall',            0,  8000, ['128', '256', '512']),
    ('modular_counter',        5,  8000, ['128', '256', '512']),
    ('dyck_depth_unbounded',   0,  8000, ['128', '256', '512']),
    ('modular_quadratic',      64, 8000, ['128', '256', '512']),
    ('iterated_nonlinear_map', 0,  8000, ['128', '256', '512']),
    ('s5_permutation',         0,  8000, ['128', '256', '512']),
    ('anbncn_viability',       0,  5000, ['128', '256']),
    ('parity',                 0,  5000, ['128', '256']),
]


def compose_combined(spec_str):
    """Add a 'combined_best' arm from a comma list of override flags, e.g.
    OPT_NORM_COMBINED='--decay_init slow --gate_bias_init 2.0 --lam_max 1.3'."""
    overrides = spec_str.split()
    ARMS['combined_best'] = (*typed(*overrides), 5e-4, DEF_SEEDS)


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
    lr: float

    @property
    def label(self) -> str:
        ktag = f"_K{self.K}" if self.K > 0 else ""
        return f"optnorm_{self.task}{ktag}__{self.arm}__seed{self.seed}"


def build_jobs(arm_filter=None):
    jobs = []
    for task, K, steps, evals in BATTERY:
        for arm, (level, extra, lr, seeds) in ARMS.items():
            if arm_filter and arm not in arm_filter:
                continue
            for seed in seeds:
                jobs.append(Job(task=task, arm=arm, seed=seed, K=K, steps=steps,
                                eval_lengths=list(evals), level=level,
                                extra=list(extra), lr=lr))
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
            "    eval \"$(scripts/gpu_lease.sh 8)\"\n"
            "then re-run. (Never hand-pick GPUs; route through the broker.)")
    return [g for g in cvd.split(',') if g.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps_scale', type=float, default=1.0,
                    help='Multiply every task step budget (smoke=0.02).')
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--slots_per_gpu', type=int, default=3)
    ap.add_argument('--arms', type=str, default=None,
                    help='Comma list to restrict arms (default: all).')
    ap.add_argument('--output_dir', default=str(THIS / 'results_opt_norm'))
    ap.add_argument('--poll', type=float, default=5.0)
    args = ap.parse_args()

    combined = os.environ.get('OPT_NORM_COMBINED', '').strip()
    if combined:
        compose_combined(combined)
        print(f"[combined] added combined_best arm: {combined}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_filter = set(args.arms.split(',')) if args.arms else None
    gpus = leased_gpus()
    jobs = build_jobs(arm_filter)
    if args.steps_scale != 1.0:
        for j in jobs:
            j.steps = max(40, int(j.steps * args.steps_scale))

    pending = []
    for j in jobs:
        if (out_dir / f'{j.label}.json').exists():
            print(f"[skip] {j.label} (exists)", flush=True)
        else:
            pending.append(j)

    print(f"Leased GPUs: {gpus}  slots/gpu={args.slots_per_gpu}  "
          f"jobs: {len(pending)}/{len(jobs)} pending", flush=True)

    slots = [g for g in gpus for _ in range(args.slots_per_gpu)]
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
                ok = (proc.returncode in (0, -15)) and has_json
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
