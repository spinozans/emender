"""emender-real-cap — expressivity battery: REAL Emender vs pure GDN-2.

Does a SMALL sprinkle (4/64, 8/64) of nonlinear EMENDMENT heads (e97_delta split-edit,
per-step tanh = the depth-capability head) add expressivity that a SEA of pure GDN-2
(gdn2_recall, neg-eig) cannot reach? Head-to-head accuracy on a length-EXTRAPOLATION
battery (the honest separation metric). bf16 (matched precision), fused (loud guard).

Arms (dim=512, n_heads=64 -> exact 4/64 & 8/64 fractions; head_dim=8 = the battery shape):
  gdn2        fla-gdn, neg-eig                 = pure GDN-2 CONTROL (the incumbent)
  gdn2typed   64 gdn2_recall (typed plumbing)  = isolates typed-vs-fla (attribution)
  emender4    60 gdn2_recall + 4 e97_delta-tanh = REAL Emender (6.25% nonlin)
  emender8    56 gdn2_recall + 8 e97_delta-tanh = REAL Emender (12.5% nonlin)
  shell4      60 gdn2_recall + 4 nonlin-shell   = throughput-friendly head CONTROL
                                                   (expect capability-INERT: phi-explore)

Battery (REAL deterministic generators, REAL training, schedule-free AdamW, bf16):
  modular_quadratic  K=64  step-growth / THE CLIFF  (primary separation: e97_delta +0.18..0.21)
  iterated_nonlinear_map   step-growth (secondary)
  s5_permutation           state-tracking (S5)
  modular_counter    K=5   counting
  mqar_recall              recall (TIE expected; sprinkle must NOT break recall)

Eval at T in {128,256,512} (train 128) = length extrapolation. 3 seeds. Resumable.

  eval "$(scripts/gpu_lease.sh 8)" && python experiments/emender_real_cap/run_cap_battery.py
"""
from __future__ import annotations
import argparse, math, os, subprocess, time
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
TRAIN = ROOT / 'experiments' / 'expressivity_tasks' / 'train_hybrid.py'

def L(counts):
    names = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin',
             'gdn2_nonlin_shell', 'e97_raw', 'e97_delta', 'refit']
    return ','.join(f'{math.log(counts[t]):.6f}' if counts.get(t, 0) > 0 else '-30'
                    for t in names)


# Two scales, SAME sparse fractions (6.25% / 12.5%):
#   large : dim512 nh64 -> exact 4/64 & 8/64 (the literal task fractions; the 1.3B nh).
#   small : dim256 nh32 -> exact 2/32 & 4/32 (the DOCUMENTED separation regime where
#           e97_delta split-edit beats gdn-neg on modular_quadratic; phi-explore).
# head_dim=dim/nh=8 and n_state=32 are IDENTICAL across both (only the SEA size differs).
SCALES = {
    'large': dict(dim=512, n_heads=64, n_small=4, n_big=8),
    'small': dict(dim=256, n_heads=32, n_small=2, n_big=4),
}


def make_arms(scale):
    sc = SCALES[scale]
    nh, ns, nb = sc['n_heads'], sc['n_small'], sc['n_big']
    return {
        'gdn2':      ('fla-gdn',    []),
        'gdn2typed': ('typed-gdn2', ['--head_type_logits', L({'gdn2_recall': nh})]),
        'emender4':  ('typed-gdn2', ['--head_type_logits', L({'gdn2_recall': nh - ns, 'e97_delta': ns}),
                                     '--e97_state_nonlin', 'tanh', '--use_chunked_e97_delta', '0']),
        'emender8':  ('typed-gdn2', ['--head_type_logits', L({'gdn2_recall': nh - nb, 'e97_delta': nb}),
                                     '--e97_state_nonlin', 'tanh', '--use_chunked_e97_delta', '0']),
        'shell4':    ('typed-gdn2', ['--head_type_logits', L({'gdn2_recall': nh - ns, 'gdn2_nonlin_shell': ns}),
                                     '--shell_state_nonlin', 'tanh']),
    }


def make_shared(scale):
    sc = SCALES[scale]
    return ['--dim', str(sc['dim']), '--n_heads', str(sc['n_heads']), '--n_state', '32',
            '--expansion', '1.0', '--mlp_ratio', '2.0', '--depth', '4',
            '--gdn_allow_neg_eigval', '1', '--optimizer', 'schedulefree']

# (task, K, steps, eval_lengths, corner)
BATTERY = [
    ('modular_quadratic',      64, 6000, ['128', '256', '512'], 'step_growth'),
    ('iterated_nonlinear_map', 0,  6000, ['128', '256', '512'], 'step_growth'),
    ('s5_permutation',         0,  6000, ['128', '256', '512'], 'track'),
    ('modular_counter',        5,  6000, ['128', '256', '512'], 'counting'),
    ('mqar_recall',            0,  6000, ['128', '256', '512'], 'recall'),
]
SEEDS = [42, 123, 456]
BASE_LR = 5e-4


@dataclass
class Job:
    task: str; arm: str; seed: int; K: int; steps: int
    eval_lengths: list; level: str; extra: list = field(default_factory=list)

    @property
    def label(self):
        ktag = f"_K{self.K}" if self.K > 0 else ""
        return f"emc_{self.task}{ktag}__{self.arm}__seed{self.seed}"


def build_jobs(steps_scale, scale):
    jobs = []
    for arm, (level, extra) in make_arms(scale).items():
        for task, K, steps, evals, corner in BATTERY:
            for seed in SEEDS:
                jobs.append(Job(task=task, arm=arm, seed=seed, K=K,
                                steps=max(50, int(steps * steps_scale)),
                                eval_lengths=list(evals), level=level, extra=list(extra)))
    return jobs


def build_cmd(job, args, out_dir):
    pattern = [job.level] * args.depth
    cmd = ['python', str(TRAIN), '--task', job.task, '--layer_pattern', *pattern,
           *make_shared(args.scale), *job.extra, '--steps', str(job.steps), '--seq_len', '128',
           '--batch_size', str(args.batch_size), '--lr', str(BASE_LR), '--seed', str(job.seed),
           '--label', job.label, '--output_dir', str(out_dir),
           '--eval_lengths', *job.eval_lengths,
           '--eval_lengths_n_batches', str(args.eval_n_batches)]
    if job.K > 0:
        cmd += ['--K', str(job.K)]
    return cmd


def leased_gpus():
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit("lease GPUs first: eval \"$(scripts/gpu_lease.sh 8)\"")
    return [g for g in cvd.split(',') if g.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps_scale', type=float, default=1.0)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--slots_per_gpu', type=int, default=2)
    ap.add_argument('--only_arms', type=str, default=None)
    ap.add_argument('--only_tasks', type=str, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--scale', choices=list(SCALES.keys()), default='large')
    ap.add_argument('--poll', type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir or (THIS / f'results_cap_{args.scale}'))
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = leased_gpus()
    jobs = build_jobs(args.steps_scale, args.scale)
    if args.only_arms:
        keep = set(args.only_arms.split(',')); jobs = [j for j in jobs if j.arm in keep]
    if args.only_tasks:
        keep = set(args.only_tasks.split(',')); jobs = [j for j in jobs if j.task in keep]

    pending = [j for j in jobs if args.force or not (out_dir / f'{j.label}.json').exists()]
    for j in jobs:
        if j not in pending:
            print(f"[skip] {j.label}", flush=True)
    slots = [g for g in gpus for _ in range(args.slots_per_gpu)]
    print(f"GPUs {gpus} slots/gpu={args.slots_per_gpu}  {len(pending)}/{len(jobs)} pending",
          flush=True)

    running, queue = {}, list(pending)
    t0 = time.time(); done = 0

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
            proc, job, ts, logf = running[si]
            if proc.poll() is not None:
                logf.close(); dt = time.time() - ts
                ok = (proc.returncode in (0, -15)) and (out_dir / f'{job.label}.json').exists()
                done += 1
                print(f"[{'ok' if ok else f'FAIL rc={proc.returncode}'} {dt:.0f}s] {job.label} "
                      f"({done}/{len(pending)}, {time.time()-t0:.0f}s)", flush=True)
                del running[si]
    print(f"\nBattery complete: {done} runs in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == '__main__':
    main()
