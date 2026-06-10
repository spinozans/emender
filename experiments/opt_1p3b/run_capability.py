#!/usr/bin/env python3
"""opt-1p3b — capability coverage at the 1.3B shape (OPT_SYNTHESIS §4.5.1).

Runs the §3 capability battery on the 1.3B-shaped cell for each of the three arms
at ITS 1.3B geometry, scoring one task per JCC corner (recall / counting /
step_growth / track) with length-extrapolation eval lengths (128/256/512) — the
axis the synthesis flags as where the levers' worst-corner / extrapolation value
is best detected. FUSED Triton (typed use_triton_e97; m2rnn XMA).

  rstar     typed-gdn2 x22  d3072 h64 ns32 e2.0  house mixture + head_lr_compute_mult=5
  cma_gdn2  fla-gdn   x21    d2688 h44 ns64 e2.0  (= CMA-best GDN-2, the incumbent B)
  cma_m2rnn m2rnn     x21    d1920 h370 ns16 e1.0 (= CMA-best m2rnn foil)

Each run uses train_hybrid.py (the same harness the four small-scale probes + the
R* re-run used). Broker-leased GPUs only. Resumable: existing JSON is skipped.

  eval "$(scripts/gpu_lease.sh 8)" && python run_capability.py --seeds 42 --steps_scale 1.0
"""
import os, sys, time, math, argparse, subprocess
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
os.environ.setdefault('XMA_PATH', '/home/erikg/xma')


def house_logits(n_heads):
    g = n_heads // 2; nl = n_heads // 4; rf = n_heads - g - nl
    return f"{math.log(g):.8f},-30,-30,-30,{math.log(nl):.8f},-30,-30,-30,{math.log(rf):.8f}"


# Per-arm 1.3B geometry + the level-specific flags (matches lm_runner.py geometries).
ARMS = {
    'rstar': dict(
        pattern=['typed-gdn2'], depth=22, dim=3072, n_heads=64, n_state=32, expansion=2.0,
        extra=['--head_type_logits', house_logits(64), '--refit_has_mom', '0',
               '--gdn_allow_neg_eigval', '1',
               '--head_lr_recall_mult', '1.0', '--head_lr_compute_mult', '5.0']),
    'cma_gdn2': dict(
        pattern=['fla-gdn'], depth=21, dim=2688, n_heads=44, n_state=64, expansion=2.0,
        extra=['--gdn_allow_neg_eigval', '1']),
    'cma_m2rnn': dict(
        pattern=['m2rnn'], depth=21, dim=1920, n_heads=370, n_state=16, expansion=1.0,
        extra=[]),
}

# (task, K, steps, eval_lengths, corner) — one task per JCC corner, length-extrap eval.
# Step counts are reduced from the small-scale battery (8k-16k @ dim256): at 1.3B the
# cell has far more capacity and converges on these toy tasks in far fewer steps. We
# verified (per-step ~0.46s @ 1.3B) and pick budgets that reach the accuracy plateau
# while keeping the 12-run matrix tractable across 8 leased GPUs.
BATTERY = [
    ('mqar_recall',       0,  2000, ['128', '256', '512'], 'recall'),
    ('modular_counter',   5,  3000, ['128', '256', '512'], 'counting'),
    ('modular_quadratic', 64, 3000, ['128', '256', '512'], 'step_growth'),
    ('s5_permutation',    0,  2000, ['128', '256', '512'], 'track'),
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
           '--optimizer', 'schedulefree', '--disable_autocast',
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
