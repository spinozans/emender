"""typed-gdn-2-head CMA: search the TYPED-HEAD MIXTURE over native recurrent head
types at the ~8M param band.

FORM (this experiment): one `typed-gdn2` layer per depth holds a horizontal
population of NATIVE head types -- real FLA Gated-DeltaNet-2 delta-memory recall
heads (allow_neg_eigval=True; N=32 matrix state) alongside the four E98 corner
specialists (E97-split-gate track, count integrator, latch, nonlinear-state),
each with FROZEN personality. This is the alternative to e98-cma's single
unified cell: maximum expressivity from a PLACED heterogeneous population of
update rules, not from operating points of one cell.

SEARCH (this driver, reusing cma_capability.py's two-phase LHS->CMA + idle-GPU
scheduler): the search variables are ONLY
    * 5 unconstrained head-type logits [gdn2_recall,e97_track,count,latch,nonlin]
      (softmax -> fractions -> largest-remainder integer head counts; a type may
      get zero heads, reported honestly), and
    * the shared learning rate (log).
Head personalities are frozen (fixed_pop buffers + native GDN kernel); lam_max /
beta_max are pinned to the cma-capability winner so placed corners sit at the
validated operating points; depth / n_heads / n_state are fixed and `dim` is
DERIVED (binary search) so every candidate sits at TARGET_PARAMS (8M) -- the
param-matched comparison.

OBJECTIVE (worst-case-aware, NOT a plain average so collapse is visible):
    per-probe p_i = mean over T in {128,256,512,1024} of length-extrap acc.
    fitness = 0.5 * mean_i(p_i) + 0.5 * min_i(p_i)
The min term is the floor: dropping ANY single capability (recall to win
specialists, OR specialists to win recall) halves the marginal gain. CMA
minimizes cost = 1 - fitness. Six probes span recall + every specialist + mixed.

REAL training only (fp32, schedule-free AdamW, disable_autocast), idle-GPU-only
(used mem < 2GB, never preempt). Resumable per (config,probe,seed) JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT))

from ndm.models.hybrid_ladder import HybridLadderLM  # noqa: E402
from ndm.models.typed_head_mixture import allocate_types, TYPE_NAMES  # noqa: E402
from experiments.expressivity_tasks.tasks import ALL_TASKS  # noqa: E402

try:
    import cma
except ImportError:
    print("Please install cma: pip install cma", file=sys.stderr)
    sys.exit(1)

FREE_MEM_MIB = 2000  # a GPU is "idle" iff used memory < this (task rule: <2GB)

# Six probes: recall (the GDN target) + the four specialist corners + the mixed
# all-at-once task. flag_hold_recall/mixed_probe take --K 4 like cma_capability.
PROBES = {
    'mqar_recall': [],
    's5_permutation': [],
    'anbncn_viability': [],
    'flag_hold_recall': ['--K', '4'],
    'iterated_nonlinear_map': [],
    'mixed_probe': ['--K', '4'],
}
PROBE_LIST = list(PROBES.keys())
REF_VOCAB_PROBE = 'mixed_probe'  # canonical vocab for the param-budget dim solve
EVAL_TS = [128, 256, 512, 1024]

# Frozen architecture (first clean experiment: only logits + lr are searched).
DEPTH = 4
N_HEADS = 48
N_STATE = 32
LAM_MAX = 1.585   # cma-capability winner -> latch corner (1.3) representable
BETA_MAX = 2.747  # cma-capability winner
LR_LO, LR_HI = 1e-4, 1e-3
# 5 logits in a bounded box (softmax is shift-invariant; box just bounds spread).
LOGIT_LO, LOGIT_HI = -4.0, 4.0
SEARCH_DIM = 6  # 5 logits + lr


def _lin(v, lo, hi):
    return lo + float(np.clip(v, 0, 1)) * (hi - lo)


def _log(v, lo, hi):
    return float(10 ** (np.log10(lo) + float(np.clip(v, 0, 1)) * (np.log10(hi) - np.log10(lo))))


def count_params(dim, head_type_logits, vocab):
    """Exact param count of the typed-gdn2 HybridLadderLM (CPU build, no kernels)."""
    m = HybridLadderLM(
        vocab_size=vocab, dim=dim, depth=DEPTH,
        layer_pattern=['typed-gdn2'],
        layer_kwargs=[{'head_type_logits': list(head_type_logits),
                       'lam_max': LAM_MAX, 'beta_max': BETA_MAX}],
        n_state=N_STATE, n_heads=N_HEADS, expansion=1.0,
    )
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


def derive_dim(head_type_logits, vocab, target_params):
    """Binary-search dim (multiple of 8) so total params ~= target_params."""
    lo, hi = 16, 4096
    best = None
    while lo <= hi:
        mid = max(8, ((lo + hi) // 2 // 8) * 8)
        p = count_params(mid, head_type_logits, vocab)
        if best is None or abs(p - target_params) < abs(best[1] - target_params):
            best = (mid, p)
        if p < target_params:
            lo = mid + 8
        else:
            hi = mid - 8
    return best  # (dim, actual_params)


def decode(x, vocab, target_params):
    """CMA vector -> concrete typed-gdn2 meta-config. dim derived to hit target."""
    logits = [round(_lin(x[i], LOGIT_LO, LOGIT_HI), 4) for i in range(5)]
    lr = _log(x[5], LR_LO, LR_HI)
    dim, actual = derive_dim(logits, vocab, target_params)
    alloc = allocate_types(N_HEADS, logits)
    return {
        'dim': dim, 'depth': DEPTH, 'n_heads': N_HEADS, 'n_state': N_STATE,
        'lr': lr, 'lam_max': LAM_MAX, 'beta_max': BETA_MAX,
        'head_type_logits': logits,
        'type_counts': alloc['counts'],
        'actual_params': int(actual),
    }


def config_hash(cfg):
    key = json.dumps({k: cfg[k] for k in (
        'dim', 'depth', 'n_heads', 'n_state', 'lr', 'lam_max', 'beta_max',
        'head_type_logits')}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


@dataclass
class Job:
    cfg_hash: str
    cfg: dict
    probe: str
    seed: int
    steps: int

    @property
    def label(self):
        return f"tgdn2_{self.cfg_hash}__{self.probe}__seed{self.seed}__s{self.steps}"


def build_cmd(job: Job, out_dir: Path):
    c = job.cfg
    cmd = [
        sys.executable, str(THIS / 'train_hybrid.py'),
        '--task', job.probe,
        '--layer_pattern', 'typed-gdn2',
        *PROBES[job.probe],
        '--dim', str(c['dim']),
        '--depth', str(c['depth']),
        '--n_heads', str(c['n_heads']),
        '--n_state', str(c['n_state']),
        '--lr', str(c['lr']),
        '--lam_max', str(c['lam_max']),
        '--beta_max', str(c['beta_max']),
        # '=' form: logits can be negative, which argparse would otherwise read as
        # a flag if passed as a separate argv token.
        '--head_type_logits=' + ','.join(str(f) for f in c['head_type_logits']),
        '--steps', str(job.steps),
        '--seq_len', '128',
        '--batch_size', '32',
        '--optimizer', 'schedulefree',
        '--disable_autocast',
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', *[str(t) for t in EVAL_TS],
        '--eval_lengths_n_batches', '8',
    ]
    return cmd


def gpu_used_mib():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def probe_score(out_dir: Path, label: str):
    """Mean length-extrap accuracy across EVAL_TS for one finished run; None if absent."""
    p = out_dir / f'{label}.json'
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None
    le = d.get('length_extrap')
    if not le:
        return None
    accs = []
    for t in EVAL_TS:
        e = le.get(str(t))
        if isinstance(e, dict) and 'acc' in e:
            accs.append(float(e['acc']))
    return float(np.mean(accs)) if accs else None


def run_jobs(jobs, out_dir: Path, max_gpus: int, poll: float):
    """Idle-GPU-only scheduler (never preempt). Skips jobs whose JSON exists."""
    pending = [j for j in jobs if not (out_dir / f'{j.label}.json').exists()]
    skipped = len(jobs) - len(pending)
    if skipped:
        print(f"[sched] {skipped} cached, {len(pending)} to run", flush=True)
    running = {}
    while pending or running:
        for gpu in list(running):
            job, proc, logf = running[gpu]
            if proc.poll() is not None:
                logf.close()
                st = 'ok' if proc.returncode == 0 else f'FAIL({proc.returncode})'
                print(f"[done] gpu{gpu} {job.label} -> {st}", flush=True)
                del running[gpu]
        if pending:
            used = gpu_used_mib()
            for gpu in range(max_gpus):
                if not pending:
                    break
                if gpu in running:
                    continue
                if used.get(gpu, 10**9) >= FREE_MEM_MIB:
                    continue
                job = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(out_dir / f'{job.label}.log', 'w')
                proc = subprocess.Popen(build_cmd(job, out_dir), cwd=str(ROOT),
                                        env=env, stdout=logf, stderr=subprocess.STDOUT)
                running[gpu] = (job, proc, logf)
                print(f"[run ] gpu{gpu} {job.label}", flush=True)
                time.sleep(3)
        time.sleep(poll)


def fitness_of(cfg, out_dir, seeds, steps):
    """Worst-case-aware fitness = 0.5*mean + 0.5*min over the six probes.
    Missing/failed probe -> 0.0 (penalize)."""
    h = config_hash(cfg)
    per_probe = {}
    for probe in PROBE_LIST:
        ss = []
        for seed in seeds:
            label = f"tgdn2_{h}__{probe}__seed{seed}__s{steps}"
            s = probe_score(out_dir, label)
            if s is not None:
                ss.append(s)
        per_probe[probe] = float(np.mean(ss)) if ss else 0.0
    vals = [per_probe[p] for p in PROBE_LIST]
    mean_s = float(np.mean(vals))
    min_s = float(np.min(vals))
    fitness = 0.5 * mean_s + 0.5 * min_s
    return fitness, per_probe, mean_s, min_s


def evaluate_population(configs, out_dir, seeds, steps, max_gpus, poll, trace, trace_path):
    jobs = []
    for cfg in configs:
        h = config_hash(cfg)
        for probe in PROBE_LIST:
            for seed in seeds:
                jobs.append(Job(h, cfg, probe, seed, steps))
    run_jobs(jobs, out_dir, max_gpus, poll)
    results = []
    for cfg in configs:
        fit, per_probe, mean_s, min_s = fitness_of(cfg, out_dir, seeds, steps)
        rec = {'hash': config_hash(cfg), 'cfg': cfg, 'fitness': fit,
               'mean': mean_s, 'min': min_s, 'per_probe': per_probe}
        results.append(rec)
        trace.append(rec)
        print(f"  [eval] {rec['hash']} fit={fit:.4f} (mean={mean_s:.3f} min={min_s:.3f}) "
              f"params={cfg['actual_params']:,} dim={cfg['dim']} "
              f"counts={cfg['type_counts']}", flush=True)
    json.dump(trace, open(trace_path, 'w'), indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_params', type=float, default=8.0e6)
    ap.add_argument('--lhs_samples', type=int, default=16)
    ap.add_argument('--popsize', type=int, default=10)
    ap.add_argument('--min_gens', type=int, default=8)
    ap.add_argument('--max_gens', type=int, default=12)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--sigma0', type=float, default=0.30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_gpus', type=int, default=8)
    ap.add_argument('--poll', type=float, default=12.0)
    ap.add_argument('--output_dir', default=str(THIS / 'results' / 'typed_gdn2_cma'))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / 'cma_trace.json'
    best_path = out_dir / 'cma_best.json'

    ref_vocab = ALL_TASKS[REF_VOCAB_PROBE](n_keys=4).vocab_size
    print(f"[cma] target={args.target_params:,.0f} params, ref_vocab={ref_vocab}, "
          f"pop={args.popsize}, min_gens={args.min_gens}, steps={args.steps}, "
          f"seeds={args.seeds}; DEPTH={DEPTH} N_HEADS={N_HEADS} N_STATE={N_STATE}",
          flush=True)

    trace = []
    if trace_path.exists():
        try:
            trace = json.load(open(trace_path))
            print(f"[cma] resumed trace with {len(trace)} prior evals", flush=True)
        except (json.JSONDecodeError, OSError):
            trace = []

    rng = np.random.default_rng(args.seed)

    print("\n=== PHASE 1: LHS exploration ===", flush=True)
    n = args.lhs_samples
    lhs = np.zeros((n, SEARCH_DIM))
    for d in range(SEARCH_DIM):
        edges = np.linspace(0, 1, n + 1)
        col = np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(n)])
        rng.shuffle(col)
        lhs[:, d] = col
    seen, uniq, vec_by_hash = set(), [], {}
    for i in range(n):
        c = decode(lhs[i], ref_vocab, args.target_params)
        hh = config_hash(c)
        if hh not in seen:
            seen.add(hh); uniq.append(c); vec_by_hash[hh] = np.clip(lhs[i], 0, 1)
    lhs_res = evaluate_population(uniq, out_dir, args.seeds, args.steps,
                                 args.max_gpus, args.poll, trace, trace_path)
    lhs_res.sort(key=lambda r: -r['fitness'])
    best = lhs_res[0]
    print(f"[lhs] best fit={best['fitness']:.4f} {best['hash']}", flush=True)

    x0 = vec_by_hash[best['hash']]

    print("\n=== PHASE 2: CMA-ES refinement ===", flush=True)
    es = cma.CMAEvolutionStrategy(
        list(x0), args.sigma0,
        {'popsize': args.popsize, 'bounds': [0, 1], 'seed': args.seed + 1,
         'verbose': -9})
    best_fit_hist = [best['fitness']]
    gen = 0
    while gen < args.max_gens:
        xs = es.ask()
        cfgs = [decode(np.array(x), ref_vocab, args.target_params) for x in xs]
        res = evaluate_population(cfgs, out_dir, args.seeds, args.steps,
                                  args.max_gpus, args.poll, trace, trace_path)
        fit_by_hash = {r['hash']: r['fitness'] for r in res}
        costs = [1.0 - fit_by_hash[config_hash(c)] for c in cfgs]
        es.tell(xs, costs)
        gen += 1
        gen_best = max(res, key=lambda r: r['fitness'])
        if gen_best['fitness'] > best['fitness']:
            best = gen_best
        best_fit_hist.append(best['fitness'])
        json.dump({'best': best, 'gen': gen, 'best_fit_hist': best_fit_hist},
                  open(best_path, 'w'), indent=2)
        print(f"[gen {gen}] gen_best={gen_best['fitness']:.4f} "
              f"overall_best={best['fitness']:.4f} {best['hash']} "
              f"counts={best['cfg']['type_counts']}", flush=True)
        if gen >= args.min_gens:
            recent = best_fit_hist[-(args.patience + 1):]
            if len(recent) > args.patience and (recent[-1] - recent[0]) < 1e-4:
                print(f"[cma] converged (no >1e-4 improvement in {args.patience} gens)", flush=True)
                break

    print("\n=== DONE ===", flush=True)
    print(f"BEST fitness={best['fitness']:.4f} hash={best['hash']}", flush=True)
    print(json.dumps(best['cfg'], indent=2), flush=True)
    json.dump({'best': best, 'gens_run': gen, 'best_fit_hist': best_fit_hist,
               'objective': 'fitness = 0.5*mean + 0.5*min over 6 probes',
               'search': {'target_params': args.target_params, 'steps': args.steps,
                          'seeds': args.seeds, 'popsize': args.popsize,
                          'lhs_samples': args.lhs_samples, 'depth': DEPTH,
                          'n_heads': N_HEADS, 'n_state': N_STATE,
                          'lam_max': LAM_MAX, 'beta_max': BETA_MAX,
                          'type_order': TYPE_NAMES}},
              open(best_path, 'w'), indent=2)
    print(f"[cma] best -> {best_path}", flush=True)
    print(f"[cma] trace ({len(trace)} evals) -> {trace_path}", flush=True)


if __name__ == '__main__':
    main()
