#!/usr/bin/env python3
"""emender-real-1p3b — expressivity coverage at the 1.3B shape vs GDN-2.

Runs the capability battery on the 1.3B-shaped cell for the EMENDER (CMA-found mixture)
vs two GDN-2 controls — fla-gdn (the incumbent) AND gdn2typed (all-recall on the SAME
typed path, the decisive substrate control from emender-real-cap §2) — scoring the named
separation tasks (S5 track / modular_quadratic step-growth cliff / counting) plus mqar
(recall must NOT break) with length-extrapolation eval lengths (128/256/512).

The gdn2typed control is essential: emender-real-cap showed the eye-catching modquad
"win" vs fla-gdn (+0.314) is the TYPED PLUMBING, not the emendment heads — against
gdn2typed it vanishes. Separation = (emender − gdn2typed).

  emender    typed-gdn2 x22  d3072 h64 ns32 e2.0  CMA-found 58 gdn2_recall/2 track/4 e97_delta
  gdn2typed  typed-gdn2 x22  d3072 h64 ns32 e2.0  64 gdn2_recall (substrate / attribution ctrl)
  cma_gdn2   fla-gdn   x21    d2688 h44 ns64 e2.0  (= CMA-best GDN-2, the incumbent B)

bf16 UNIFORM (autocast on — NO --disable_autocast; the fused e97_delta kernel is
bf16-only). FUSED Triton (typed use_triton_e97). REAL deterministic generators, REAL
training, schedule-free AdamW. Broker-leased GPUs only. Resumable: existing JSON skipped.

  eval "$(scripts/gpu_lease.sh 8)" && python run_capability.py --seeds 42 --steps_scale 1.0
"""
import os, sys, time, math, argparse, subprocess
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
os.environ.setdefault('XMA_PATH', '/home/erikg/xma')


def emender_logits(n_heads):
    """CMA-FOUND Emender mixture (emender-real-cap): 29 gdn2_recall + 1 e97_track + 2
    e97_delta @ nh32, scaled to nh64 at the same fractions = 58/2/4. NOT hand-picked."""
    assert n_heads % 32 == 0
    k = n_heads // 32
    g, tr, de = 29 * k, 1 * k, 2 * k
    n = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin', 'gdn2_nonlin_shell',
         'e97_raw', 'e97_delta', 'refit']
    c = {'gdn2_recall': g, 'e97_track': tr, 'e97_delta': de}
    return ','.join(f"{math.log(c[t]):.8f}" if c.get(t, 0) > 0 else '-30' for t in n)


def allrecall_logits(n_heads):
    n = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin', 'gdn2_nonlin_shell',
         'e97_raw', 'e97_delta', 'refit']
    return ','.join(f"{math.log(n_heads):.8f}" if t == 'gdn2_recall' else '-30' for t in n)


# Per-arm 1.3B geometry + the level-specific flags (matches lm_runner.py geometries).
# The emender + gdn2typed share the EXACT typed geometry so separation isolates the
# emendment heads (e97_delta) from the typed substrate.
ARMS = {
    'emender': dict(
        pattern=['typed-gdn2'], depth=22, dim=3072, n_heads=64, n_state=32, expansion=2.0,
        extra=['--head_type_logits', emender_logits(64), '--gdn_allow_neg_eigval', '1',
               '--e97_state_nonlin', 'tanh', '--use_chunked_e97_delta', '0',
               '--lam_max', '1.585', '--beta_max', '2.747']),
    'gdn2typed': dict(
        pattern=['typed-gdn2'], depth=22, dim=3072, n_heads=64, n_state=32, expansion=2.0,
        extra=['--head_type_logits', allrecall_logits(64), '--gdn_allow_neg_eigval', '1']),
    'cma_gdn2': dict(
        pattern=['fla-gdn'], depth=21, dim=2688, n_heads=44, n_state=64, expansion=2.0,
        extra=['--gdn_allow_neg_eigval', '1']),
}

# (task, K, steps, eval_lengths, corner) — one task per JCC corner, length-extrap eval.
# Step counts are reduced from the small-scale battery (8k-16k @ dim256): at 1.3B the
# cell has far more capacity and converges on these toy tasks in far fewer steps. We
# verified (per-step ~0.46s @ 1.3B) and pick budgets that reach the accuracy plateau
# while keeping the 12-run matrix tractable across 8 leased GPUs.
# Steps reduced from the small-scale battery: at 1.31B the cell has far more capacity and
# reaches the accuracy plateau in fewer steps; the e97_delta seq path is latency-bound so
# we keep the 12-run matrix tractable. (modular_quadratic = THE cliff, the primary
# separation; s5 = track; counter = counting; mqar guards recall must-not-break.)
BATTERY = [
    ('modular_quadratic', 64, 1500, ['128', '256', '512'], 'step_growth'),
    ('s5_permutation',    0,  1200, ['128', '256', '512'], 'track'),
    ('modular_counter',   5,  1500, ['128', '256', '512'], 'counting'),
    ('mqar_recall',       0,  1200, ['128', '256', '512'], 'recall'),
]
BASE_LR = 5e-4


@dataclass
class Job:
    arm: str
    task: str
    seed: int
    K: int
    steps: int
    evals: list
    corner: str

    @property
    def label(self):
        kt = f"_K{self.K}" if self.K > 0 else ""
        return f"cap_{self.task}{kt}__{self.arm}__seed{self.seed}"


def build_cmd(job, args, out_dir):
    a = ARMS[job.arm]
    cmd = ['python', str(ROOT / 'experiments/expressivity_tasks/train_hybrid.py'),
           '--task', job.task,
           '--layer_pattern', *(a['pattern'] * a['depth']),
           '--dim', str(a['dim']), '--depth', str(a['depth']),
           '--n_heads', str(a['n_heads']), '--n_state', str(a['n_state']),
           '--expansion', str(a['expansion']),
           '--optimizer', 'schedulefree',  # bf16 autocast ON (uniform; fused e97 is bf16-only)
           *a['extra'],
           '--steps', str(max(50, int(job.steps * args.steps_scale))),
           '--seq_len', '128', '--batch_size', str(args.batch_size), '--lr', str(BASE_LR),
           '--seed', str(job.seed), '--label', job.label, '--output_dir', str(out_dir),
           '--eval_lengths', *job.evals, '--eval_lengths_n_batches', str(args.eval_n_batches)]
    if job.K > 0:
        cmd += ['--K', str(job.K)]
    return cmd


def leased_gpus():
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit('lease first: eval "$(scripts/gpu_lease.sh 8)"')
    return [g for g in cvd.split(',') if g.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='42')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--steps_scale', type=float, default=1.0)
    ap.add_argument('--only_arms', type=str, default=None)
    ap.add_argument('--only_tasks', type=str, default=None)
    ap.add_argument('--slots_per_gpu', type=int, default=1)
    ap.add_argument('--output_dir', default=str(THIS / 'results_cap'))
    ap.add_argument('--poll', type=float, default=8.0)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    arms = set(args.only_arms.split(',')) if args.only_arms else set(ARMS)
    tasks = set(args.only_tasks.split(',')) if args.only_tasks else None

    jobs = []
    for arm in ARMS:
        if arm not in arms:
            continue
        for task, K, steps, evals, corner in BATTERY:
            if tasks and task not in tasks:
                continue
            for s in seeds:
                jobs.append(Job(arm, task, s, K, steps, list(evals), corner))

    pending = [j for j in jobs if args.force or not (out / f'{j.label}.json').exists()]
    gpus = leased_gpus()
    slots = [g for g in gpus for _ in range(args.slots_per_gpu)]
    print(f"Leased {gpus} slots={len(slots)}  {len(pending)}/{len(jobs)} pending", flush=True)

    running, queue, t0 = {}, list(pending), time.time()
    free = list(range(len(slots)))

    def launch(si, job):
        env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(slots[si])
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        logf = open(out / f'{job.label}.log', 'w')
        p = subprocess.Popen(build_cmd(job, args, out), stdout=logf,
                             stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        running[si] = (p, job, time.time(), logf)
        print(f"[launch gpu{slots[si]}] {job.label} ({job.steps} steps)", flush=True)

    while queue or running:
        while queue and free:
            launch(free.pop(0), queue.pop(0))
        time.sleep(args.poll)
        for si in list(running.keys()):
            p, job, ts, logf = running[si]
            if p.poll() is not None:
                logf.close()
                has = (out / f'{job.label}.json').exists()
                print(f"[{'ok' if (p.returncode in (0,-15) and has) else f'FAIL(rc={p.returncode},json={has})'} "
                      f"{time.time()-ts:.0f}s] {job.label} ({time.time()-t0:.0f}s elapsed)", flush=True)
                del running[si]
                free.append(si)
    print(f"\nCapability battery complete in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == '__main__':
    main()
