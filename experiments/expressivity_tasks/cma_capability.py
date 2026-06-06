"""CMA STAGE 1 (cma-capability): tune the unified-cell (E98) ARCHITECTURE
META-CONFIG on a MULTI-CAPABILITY fitness, at the ~8M param band.

FORM (decided upstream): the winning horizontal-hybrid form from
SPECIALIZATION_STUDY is *placement* -> spread-init + knob-specific LR (NOT the
specialization regularizer, NOT the type-dictionary, both of which FAILED to
cover the four corners from a generic center). E98_ON_E97 then showed the E97
split-gate is a strict superset that wins at length extrapolation. So the form we
CMA-tune is the `e98-cma` layer = learned spread-init knobs + split-gate +
gamma_mix phi (== e98-learned-spread), trained with --knob_lr_mult.

SEARCH (this driver, reusing the scripts/cmaes_search_v2.py two-phase LHS -> CMA
pattern + the run_e98_on_e97.py idle-GPU scheduler):
  meta-config = corner-mixture fractions [track,count,latch,nonlin]
              + knob_lr_mult + lam_max + beta_max
              + shape (depth, n_heads, n_state, lr);
  `dim` is DERIVED (binary search) so every candidate sits at TARGET_PARAMS (8M)
  -- this is the param-matched comparison E98_ON_E97 deferred to the CMA stage
  (it trades dim/heads/state against the +3.1M split-gate cost).

FITNESS = multi-capability score: train the ONE cell separately on each of the
four corner probes (s5 / a^n b^n c^n / iterated-nonlinear-map / flag-hold) AND
the MixedProbeTask, then take the mean LENGTH-EXTRAPOLATION accuracy across
T in {128,256,512,1024} over all five probes. CMA MINIMIZES cost = 1 - fitness.

REAL training only (fp32, schedule-free AdamW, disable_autocast), idle-GPU-only
(used mem < 2GB, never preempt). Resumable: each (config,probe,seed) writes a
JSON the scheduler skips if present. Cheap stage: 1 seed, reduced steps; the
winner is re-validated at full seeds/steps by run_cma_capability_validate.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT))

from ndm.models.hybrid_ladder import HybridLadderLM  # noqa: E402
from experiments.expressivity_tasks.tasks import ALL_TASKS  # noqa: E402

try:
    import cma
except ImportError:
    print("Please install cma: pip install cma", file=sys.stderr)
    sys.exit(1)

FREE_MEM_MIB = 2000  # a GPU is "idle" iff used memory < this (task rule: <2GB)

# Five probes: the four capability corners + the mixed (all-at-once) task.
PROBES = {
    's5_permutation': [],
    'anbncn_viability': [],
    'iterated_nonlinear_map': [],
    'flag_hold_recall': ['--K', '4'],
    'mixed_probe': ['--K', '4'],
}
PROBE_LIST = list(PROBES.keys())
REF_VOCAB_PROBE = 'mixed_probe'  # canonical vocab for the param-budget dim solve
EVAL_TS = [128, 256, 512, 1024]

# ---- search space: continuous [0,1]^D -> meta-config (dim derived) ----
#   0 frac_track   1 frac_count   2 frac_latch   3 frac_nonlin   (softmax-normalized)
#   4 depth        5 n_heads      6 n_state(bin) 7 lr(log)
#   8 knob_lr_mult(log)           9 lam_max      10 beta_max
DEPTH_LO, DEPTH_HI = 3, 6
NHEADS_LO, NHEADS_HI = 16, 64
LR_LO, LR_HI = 1e-4, 1e-3
KLR_LO, KLR_HI = 5.0, 40.0
LAMMAX_LO, LAMMAX_HI = 1.35, 1.80
BETAMAX_LO, BETAMAX_HI = 1.50, 2.80
SEARCH_DIM = 11


def _lin(v, lo, hi):
    return lo + float(np.clip(v, 0, 1)) * (hi - lo)


def _log(v, lo, hi):
    return float(10 ** (np.log10(lo) + float(np.clip(v, 0, 1)) * (np.log10(hi) - np.log10(lo))))


def count_params(dim, depth, n_heads, n_state, vocab):
    """Exact param count of the e98-cma HybridLadderLM (CPU build, no kernels run)."""
    m = HybridLadderLM(
        vocab_size=vocab, dim=dim, depth=depth,
        layer_pattern=['e98-cma'], layer_kwargs=[{}],
        n_state=n_state, n_heads=n_heads, expansion=1.0,
    )
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


def derive_dim(depth, n_heads, n_state, vocab, target_params):
    """Binary-search dim (multiple of 8) so total params ~= target_params."""
    lo, hi = 16, 4096
    best = None
    while lo <= hi:
        mid = ((lo + hi) // 2 // 8) * 8
        mid = max(8, mid)
        p = count_params(mid, depth, n_heads, n_state, vocab)
        if best is None or abs(p - target_params) < abs(best[1] - target_params):
            best = (mid, p)
        if p < target_params:
            lo = mid + 8
        else:
            hi = mid - 8
    return best  # (dim, actual_params)


def decode(x, vocab, target_params):
    """CMA vector -> concrete meta-config (dict). dim derived to hit target."""
    fr = np.array([max(1e-3, float(np.clip(x[i], 0, 1))) for i in range(4)], dtype=float)
    fr = fr / fr.sum()
    depth = int(round(_lin(x[4], DEPTH_LO, DEPTH_HI)))
    n_heads = int(round(_lin(x[5], NHEADS_LO, NHEADS_HI) / 4) * 4)  # multiple of 4
    n_heads = max(8, n_heads)
    n_state = 32 if x[6] >= 0.5 else 16
    lr = _log(x[7], LR_LO, LR_HI)
    knob_lr_mult = _log(x[8], KLR_LO, KLR_HI)
    lam_max = _lin(x[9], LAMMAX_LO, LAMMAX_HI)
    beta_max = _lin(x[10], BETAMAX_LO, BETAMAX_HI)
    dim, actual = derive_dim(depth, n_heads, n_state, vocab, target_params)
    return {
        'dim': dim, 'depth': depth, 'n_heads': n_heads, 'n_state': n_state,
        'lr': lr, 'knob_lr_mult': knob_lr_mult, 'lam_max': lam_max, 'beta_max': beta_max,
        'corner_mixture': [round(float(f), 4) for f in fr],
        'actual_params': int(actual),
    }


def config_hash(cfg):
    key = json.dumps({k: cfg[k] for k in (
        'dim', 'depth', 'n_heads', 'n_state', 'lr', 'knob_lr_mult',
        'lam_max', 'beta_max', 'corner_mixture')}, sort_keys=True)
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
        return f"cmacap_{self.cfg_hash}__{self.probe}__seed{self.seed}__s{self.steps}"


def build_cmd(job: Job, out_dir: Path):
    c = job.cfg
    cmd = [
        sys.executable, str(THIS / 'train_hybrid.py'),
        '--task', job.probe,
        '--layer_pattern', 'e98-cma',
        *PROBES[job.probe],
        '--dim', str(c['dim']),
        '--depth', str(c['depth']),
        '--n_heads', str(c['n_heads']),
        '--n_state', str(c['n_state']),
        '--lr', str(c['lr']),
        '--knob_lr_mult', str(c['knob_lr_mult']),
        '--lam_max', str(c['lam_max']),
        '--beta_max', str(c['beta_max']),
        '--corner_mixture', ','.join(str(f) for f in c['corner_mixture']),
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
    running = {}  # gpu -> (job, proc, logf)
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
    """Multi-capability fitness = mean over probes of mean-over-T length-extrap acc.
    Missing/failed probe -> 0.0 (penalize)."""
    h = config_hash(cfg)
    per_probe = {}
    for probe in PROBE_LIST:
        ss = []
        for seed in seeds:
            label = f"cmacap_{h}__{probe}__seed{seed}__s{steps}"
            s = probe_score(out_dir, label)
            if s is not None:
                ss.append(s)
        per_probe[probe] = float(np.mean(ss)) if ss else 0.0
    fitness = float(np.mean([per_probe[p] for p in PROBE_LIST]))
    return fitness, per_probe


def evaluate_population(configs, out_dir, seeds, steps, max_gpus, poll, trace, trace_path):
    """Train every (config,probe,seed) on idle GPUs, then score each config."""
    jobs = []
    for cfg in configs:
        h = config_hash(cfg)
        for probe in PROBE_LIST:
            for seed in seeds:
                jobs.append(Job(h, cfg, probe, seed, steps))
    run_jobs(jobs, out_dir, max_gpus, poll)
    results = []
    for cfg in configs:
        fit, per_probe = fitness_of(cfg, out_dir, seeds, steps)
        rec = {'hash': config_hash(cfg), 'cfg': cfg, 'fitness': fit, 'per_probe': per_probe}
        results.append(rec)
        trace.append(rec)
        print(f"  [eval] {rec['hash']} fit={fit:.4f} "
              f"params={cfg['actual_params']:,} dim={cfg['dim']} d{cfg['depth']} "
              f"H{cfg['n_heads']} N{cfg['n_state']} klr={cfg['knob_lr_mult']:.1f} "
              f"lam={cfg['lam_max']:.2f} mix={cfg['corner_mixture']}", flush=True)
    json.dump(trace, open(trace_path, 'w'), indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_params', type=float, default=8.0e6)
    ap.add_argument('--lhs_samples', type=int, default=16)
    ap.add_argument('--popsize', type=int, default=10)
    ap.add_argument('--min_gens', type=int, default=8)
    ap.add_argument('--max_gens', type=int, default=12)
    ap.add_argument('--patience', type=int, default=3, help='stop after min_gens if no improvement for this many gens')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--sigma0', type=float, default=0.30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_gpus', type=int, default=8)
    ap.add_argument('--poll', type=float, default=12.0)
    ap.add_argument('--output_dir', default=str(THIS / 'results' / 'cma_capability'))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / 'cma_trace.json'
    best_path = out_dir / 'cma_best.json'

    ref_vocab = ALL_TASKS[REF_VOCAB_PROBE](n_keys=4).vocab_size
    print(f"[cma] target={args.target_params:,.0f} params, ref_vocab={ref_vocab}, "
          f"pop={args.popsize}, min_gens={args.min_gens}, steps={args.steps}, "
          f"seeds={args.seeds}", flush=True)

    trace = []
    if trace_path.exists():
        try:
            trace = json.load(open(trace_path))
            print(f"[cma] resumed trace with {len(trace)} prior evals", flush=True)
        except (json.JSONDecodeError, OSError):
            trace = []

    rng = np.random.default_rng(args.seed)

    # ---- Phase 1: Latin Hypercube exploration ----
    print("\n=== PHASE 1: LHS exploration ===", flush=True)
    n = args.lhs_samples
    lhs = np.zeros((n, SEARCH_DIM))
    for d in range(SEARCH_DIM):
        edges = np.linspace(0, 1, n + 1)
        col = np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(n)])
        rng.shuffle(col)
        lhs[:, d] = col
    # de-dup identical decoded configs, keeping the originating vector for warm-start
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

    # Warm-start CMA at the best LHS vector.
    x0 = vec_by_hash[best['hash']]

    # ---- Phase 2: CMA-ES refinement ----
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
        # CMA minimizes cost = 1 - fitness; map by config hash (de-dup safe).
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
              f"overall_best={best['fitness']:.4f} {best['hash']}", flush=True)
        if gen >= args.min_gens:
            recent = best_fit_hist[-(args.patience + 1):]
            if len(recent) > args.patience and (recent[-1] - recent[0]) < 1e-4:
                print(f"[cma] converged (no >1e-4 improvement in {args.patience} gens)", flush=True)
                break

    print("\n=== DONE ===", flush=True)
    print(f"BEST fitness={best['fitness']:.4f} hash={best['hash']}", flush=True)
    print(json.dumps(best['cfg'], indent=2), flush=True)
    json.dump({'best': best, 'gens_run': gen, 'best_fit_hist': best_fit_hist,
               'search': {'target_params': args.target_params, 'steps': args.steps,
                          'seeds': args.seeds, 'popsize': args.popsize,
                          'lhs_samples': args.lhs_samples}},
              open(best_path, 'w'), indent=2)
    print(f"[cma] best -> {best_path}", flush=True)
    print(f"[cma] trace ({len(trace)} evals) -> {trace_path}", flush=True)


if __name__ == '__main__':
    main()
