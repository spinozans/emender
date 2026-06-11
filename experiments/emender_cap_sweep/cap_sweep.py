"""emender-cap-sweep CAPACITY SWEEP — find the dim where Emender's separation closes.

Trains, to convergence, Emender(CMA-found mixture) vs best-vs-best GDN-2 controls
across model dims {256,384,512,768,1024} on the documented separation tasks:
  modular_quadratic (THE step-growth cliff, length-extrapolation)
  s5_permutation    (state-tracking S5)
  modular_counter   (counting)
Eval at T in {128,256,512} (train 128) = length extrapolation. 3 seeds.

Arms (head_dim fixed at 8 -> n_heads = dim/8; n_state=32; bf16; fused e97 asserted):
  gdn2        fla-gdn neg-eig                 = the incumbent GDN-2 control
  gdn2typed   all gdn2_recall (typed path)    = clean substrate control (isolates
                                                the emendment HEADS from typed plumbing)
  emender_cma CMA-found e97_delta/e97_track fraction (the discovered mixture)
  emender_fix documented separating sprinkle (12.5% e97_delta-tanh = the 8/64 large
              fraction) -- guarantees a measurable boundary even if CMA picks f~0

The capacity BOUNDARY = the smallest dim at which (emender - gdn2typed) accuracy
separation collapses into noise on the modquad cliff / S5. Measured, per dim/seed.

  eval "$(scripts/gpu_lease.sh 8)" && python experiments/emender_cap_sweep/cap_sweep.py
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
TRAIN = ROOT / 'experiments' / 'expressivity_tasks' / 'train_hybrid.py'
sys.path.insert(0, str(THIS.parent / 'emender_real_cap'))
import emender_common as EC  # noqa: E402

DIMS = [256, 384, 512, 768, 1024]   # head_dim 8 -> nh = dim/8 in {32,48,64,96,128}
N_STATE = 32
DEPTH = 4
SEEDS = [42, 123, 456]
BASE_LR = 5e-4
# (task, K, steps, eval_lengths, corner)
BATTERY = [
    ('modular_quadratic', 64, 6000, ['128', '256', '512'], 'step_growth'),
    ('s5_permutation',     0, 6000, ['128', '256', '512'], 'track'),
    ('modular_counter',    5, 6000, ['128', '256', '512'], 'counting'),
]


def L_from_counts(counts):
    names = EC.head_counts  # not used; keep explicit name list
    names = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin',
             'gdn2_nonlin_shell', 'e97_raw', 'e97_delta', 'refit']
    return ','.join(f'{math.log(counts[t]):.6f}' if counts.get(t, 0) > 0 else '-30'
                    for t in names)


def fractions_to_logits(f_delta, f_track):
    """Reproduce target e97_delta/e97_track fractions as composition logits (gdn=0)."""
    f_gdn = max(1e-9, 1.0 - f_delta - f_track)
    a_delta = math.log(max(f_delta, 1e-12) / f_gdn) if f_delta > 0 else EC.LOG0
    a_track = math.log(max(f_track, 1e-12) / f_gdn) if f_track > 0 else EC.LOG0
    return EC.composition_logits(a_delta, a_track)


def make_arms(dim, f_delta, f_track):
    nh = dim // 8
    logits_cma = fractions_to_logits(f_delta, f_track)
    counts_cma = EC.head_counts(logits_cma, nh)
    # documented separating sprinkle: 12.5% e97_delta (the 8/64 large fraction)
    n_fix = max(1, round(0.125 * nh))
    counts_fix = {'gdn2_recall': nh - n_fix, 'e97_delta': n_fix}
    arms = {
        'gdn2':      ('fla-gdn',    [], {'gdn2_recall': nh}),
        'gdn2typed': ('typed-gdn2', ['--head_type_logits', L_from_counts({'gdn2_recall': nh})],
                      {'gdn2_recall': nh}),
        'emender_fix': ('typed-gdn2', ['--head_type_logits', L_from_counts(counts_fix),
                                       '--e97_state_nonlin', 'tanh', '--use_chunked_e97_delta', '0'],
                        counts_fix),
    }
    # only add a distinct CMA arm if the CMA mixture differs from pure gdn2typed
    if counts_cma.get('e97_delta', 0) > 0 or counts_cma.get('e97_track', 0) > 0:
        arms['emender_cma'] = ('typed-gdn2',
                               ['--head_type_logits', L_from_counts(counts_cma),
                                '--e97_state_nonlin', 'tanh', '--use_chunked_e97_delta', '0'],
                               counts_cma)
    return arms


@dataclass
class Job:
    task: str; arm: str; dim: int; seed: int; K: int; steps: int
    level: str; eval_lengths: list; extra: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    @property
    def label(self):
        ktag = f"_K{self.K}" if self.K > 0 else ""
        return f"caps_{self.task}{ktag}__d{self.dim}__{self.arm}__seed{self.seed}"


def build_jobs(steps_scale, f_delta, f_track, only_dims, only_arms, only_tasks):
    jobs = []
    for dim in DIMS:
        if only_dims and dim not in only_dims:
            continue
        for arm, (level, extra, counts) in make_arms(dim, f_delta, f_track).items():
            if only_arms and arm not in only_arms:
                continue
            for task, K, steps, evals, corner in BATTERY:
                if only_tasks and task not in only_tasks:
                    continue
                for seed in SEEDS:
                    jobs.append(Job(task=task, arm=arm, dim=dim, seed=seed, K=K,
                                    steps=max(50, int(steps * steps_scale)),
                                    level=level, eval_lengths=list(evals),
                                    extra=list(extra), counts=dict(counts)))
    return jobs


def build_cmd(job, args, out_dir):
    nh = job.dim // 8
    pattern = [job.level] * DEPTH
    cmd = ['python', str(TRAIN), '--task', job.task, '--layer_pattern', *pattern,
           '--dim', str(job.dim), '--n_heads', str(nh), '--n_state', str(N_STATE),
           '--expansion', '1.0', '--mlp_ratio', '2.0', '--depth', str(DEPTH),
           '--gdn_allow_neg_eigval', '1', '--optimizer', 'schedulefree',
           *job.extra, '--steps', str(job.steps), '--seq_len', '128',
           '--batch_size', str(args.batch_size), '--lr', str(BASE_LR),
           '--seed', str(job.seed), '--label', job.label, '--output_dir', str(out_dir),
           '--eval_lengths', *job.eval_lengths,
           '--eval_lengths_n_batches', str(args.eval_n_batches)]
    if job.K > 0:
        cmd += ['--K', str(job.K)]
    return cmd


def leased_gpus():
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit('lease GPUs first: eval "$(scripts/gpu_lease.sh 8)"')
    return [g for g in cvd.split(',') if g.strip()]


def slots_for_dim_default(dim):
    # large models -> 1 job/GPU (avoid OOM); small -> 2/GPU for throughput
    return 1 if dim >= 768 else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps_scale', type=float, default=1.0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--f_delta', type=float, default=None,
                    help='CMA-found e97_delta fraction (else read cma_result.json)')
    ap.add_argument('--f_track', type=float, default=None)
    ap.add_argument('--cma_result', default=str(THIS / 'results_cma' / 'cma_result.json'))
    ap.add_argument('--only_dims', default=None)
    ap.add_argument('--only_arms', default=None)
    ap.add_argument('--only_tasks', default=None)
    ap.add_argument('--output_dir', default=str(THIS / 'results_sweep'))
    ap.add_argument('--poll', type=float, default=5.0)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    f_delta, f_track = args.f_delta, args.f_track
    if f_delta is None:
        with open(args.cma_result) as f:
            cr = json.load(f)
        fm = cr['found_mixture']
        f_delta = float(fm.get('f_delta', 0.0))
        f_track = float(fm.get('f_track', 0.0))
        print(f"[cma] found_mixture f_delta={f_delta:.4f} f_track={f_track:.4f} "
              f"verdict={fm.get('verdict')}", flush=True)
    f_track = f_track or 0.0

    only_dims = set(int(x) for x in args.only_dims.split(',')) if args.only_dims else None
    only_arms = set(args.only_arms.split(',')) if args.only_arms else None
    only_tasks = set(args.only_tasks.split(',')) if args.only_tasks else None

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(dict(f_delta=f_delta, f_track=f_track, dims=DIMS, depth=DEPTH,
                   n_state=N_STATE, seeds=SEEDS, battery=[b[0] for b in BATTERY]),
              open(out_dir / 'sweep_config.json', 'w'), indent=2)

    gpus = leased_gpus()
    jobs = build_jobs(args.steps_scale, f_delta, f_track, only_dims, only_arms, only_tasks)
    pending = [j for j in jobs if args.force or not (out_dir / f'{j.label}.json').exists()]
    for j in jobs:
        if j not in pending:
            print(f"[skip] {j.label}", flush=True)
    # build slot list: each entry is a (gpu, max_dim_ok) — simpler: fixed 2/gpu, but
    # serialize big dims by capping concurrent big jobs. Use per-gpu 2 slots, and a
    # global cap on dim>=768 concurrency = len(gpus) (1 per gpu).
    slots = [g for g in gpus for _ in range(2)]
    big_running = 0
    print(f"GPUs {gpus} slots={len(slots)}  {len(pending)}/{len(jobs)} pending", flush=True)

    running, queue = {}, list(pending)
    t0 = time.time(); done = 0

    def launch(si, job):
        nonlocal big_running
        env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(slots[si])
        logf = open(out_dir / f'{job.label}.log', 'w')
        proc = subprocess.Popen(build_cmd(job, args, out_dir), stdout=logf,
                                stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        running[si] = (proc, job, time.time(), logf)
        if job.dim >= 768:
            big_running += 1
        print(f"[launch gpu{slots[si]} slot{si}] {job.label} ({job.steps} steps)", flush=True)

    def can_take(job, si):
        # 1 big job per gpu: don't put a big job on a gpu already running anything,
        # and don't run 2 big jobs total per gpu-slot pairing
        if job.dim >= 768:
            gpu = slots[si]
            for s2, (_, j2, _, _) in running.items():
                if slots[s2] == gpu:
                    return False  # keep big dims alone on their gpu
        return True

    while queue or running:
        for si in range(len(slots)):
            if si in running or not queue:
                continue
            # find a job this slot can take
            for qi, job in enumerate(queue):
                if can_take(job, si):
                    queue.pop(qi); launch(si, job); break
        time.sleep(args.poll)
        for si in list(running.keys()):
            proc, job, ts, logf = running[si]
            if proc.poll() is not None:
                logf.close(); dt = time.time() - ts
                if job.dim >= 768:
                    big_running -= 1
                ok = (proc.returncode in (0, -15)) and (out_dir / f'{job.label}.json').exists()
                done += 1
                print(f"[{'ok' if ok else f'FAIL rc={proc.returncode}'} {dt:.0f}s] {job.label} "
                      f"({done}/{len(pending)}, {time.time()-t0:.0f}s)", flush=True)
                del running[si]
    print(f"\nSweep complete: {done} runs in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == '__main__':
    main()
