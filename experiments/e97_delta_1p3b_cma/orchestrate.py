"""e97delta-1p3b: CMA-ES orchestrator over the WITHIN-LAYER 1.3B parameterization.

SEARCH (binding): head-type fractions over the 6 fused single-launch types
  {gdn2_recall(=gdn-neg), e97_delta, e97_track, count, latch, nonlin}
  (shell + e97_raw pinned OFF, never instantiated) + knob_lr_mult + lam_max +
  beta_max. dim is DERIVED per candidate to hold counted LadderLM params at the
  1.27B target (±2%), so the mixture search is not an accidental capacity search.
  Both arms carry the SwiGLU MLP (gdn2-mlp upgrade). Fitness = held-out BPB from a
  time-bounded FUSED screen on REAL Pile (lower is better).

ANCHORS (fixed, reported, NOT CMA-ranked as a head type):
  A_gdn2_mlp     : 100% gdn-neg + MLP  == the gdn2-mlp 1.3B BASELINE.
  A_delta_only   : 100% e97_delta + MLP.
  A_seed         : the CMA seed mixture (e97_delta + gdn-neg).

IDLE-GPU-ONLY / NO-PREEMPT: a GPU is usable iff nvidia-smi used-mem < 2GB and it
is not running one of THIS driver's workers. One screen per GPU, all in parallel.
REAL DATA. No mocks.
"""
import os, sys, json, time, math, subprocess, argparse, datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _THIS)
sys.path.insert(0, _ROOT)

import numpy as np
from shapes import fracs_to_logits8, derive_dim, allocate, BASE, LOG0, TYPE_NAMES

RESULTS = os.path.join(_THIS, 'results')
WORKER = os.path.join(_THIS, 'screen.py')
FREE_MEM_MIB = 2000
SEARCH_TYPES = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin', 'e97_delta']
# x layout: 6 head logits (SEARCH_TYPES order) + log_knob + lam_max + beta_max
LR = 1.0e-3
BATCH = 2


def log(m):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f'[{ts}] {m}', flush=True)


def gpu_used_mib():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        timeout=15).decode()
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def decode(x):
    """search vector -> cfg dict (head logits 8-vec, knob, lam, beta, derived dim)."""
    head = {SEARCH_TYPES[i]: x[i] for i in range(6)}
    logits8 = [0.0] * 8
    name_to_idx = {n: i for i, n in enumerate(TYPE_NAMES)}
    for n in TYPE_NAMES:
        logits8[name_to_idx[n]] = LOG0
    for n, v in head.items():
        logits8[name_to_idx[n]] = float(v)
    knob = float(np.clip(math.exp(x[6]), 1.0, 20.0))
    lam = float(np.clip(x[7], 1.0, 2.5))
    beta = float(np.clip(x[8], 1.0, 4.0))
    d = derive_dim(logits8, knob=dict(lam_max=lam, beta_max=beta))
    cfg = dict(dim=d['dim'], head_type_logits=logits8, lr=LR, knob_lr_mult=knob,
               batch_size=BATCH, bf16=True, lam_max=lam, beta_max=beta,
               params_b=d['params_b'], within_tol=d['within_tol'], counts=d['counts'])
    return cfg


def launch(cfg, tag, gpu, wall_seconds, seed, roundtrip=0):
    cfgp = os.path.join(RESULTS, f'cfg_{tag}.json')
    outp = os.path.join(RESULTS, f'out_{tag}.json')
    logp = os.path.join(RESULTS, f'log_{tag}.log')
    json.dump(cfg, open(cfgp, 'w'), indent=2)
    fh = open(logp, 'w')
    proc = subprocess.Popen(
        [sys.executable, WORKER, '--config_json', cfgp, '--out', outp,
         '--wall_seconds', str(wall_seconds), '--seed', str(seed),
         '--roundtrip', str(roundtrip)],
        stdout=fh, stderr=subprocess.STDOUT,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
    return dict(proc=proc, tag=tag, gpu=gpu, out=outp, fh=fh, t0=time.time())


def run_batch(configs, gpus, wall_seconds, seed, roundtrip, outer_timeout_s):
    """Run a list of (tag,cfg) across idle GPUs, one per GPU, return {tag: result}."""
    queue = list(configs)
    running = {}     # gpu -> job
    results = {}
    while queue or running:
        used = gpu_used_mib()
        busy_gpus = set(running.keys())
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in busy_gpus]
        while queue and free:
            tag, cfg = queue.pop(0)
            gpu = free.pop(0)
            running[gpu] = launch(cfg, tag, gpu, wall_seconds, seed, roundtrip)
            log(f'LAUNCH {tag} on GPU {gpu} (pid {running[gpu]["proc"].pid}) '
                f'dim={cfg["dim"]} pb={cfg.get("params_b")} '
                f'counts={ {k:v for k,v in cfg["counts"].items() if v} }')
        time.sleep(8)
        for gpu, job in list(running.items()):
            ret = job['proc'].poll()
            timed_out = (time.time() - job['t0']) > outer_timeout_s
            if ret is None and not timed_out:
                continue
            if ret is None and timed_out:
                job['proc'].kill(); log(f'TIMEOUT {job["tag"]} on GPU {gpu}, killed')
            job['fh'].close()
            res = None
            if os.path.exists(job['out']):
                try:
                    res = json.load(open(job['out']))
                except Exception as e:
                    res = dict(error=f'parse:{e}')
            if res is None:
                res = dict(error='no_output')
            results[job['tag']] = res
            h = (res.get('heldout') or {}) if isinstance(res, dict) else {}
            log(f'DONE {job["tag"]} bpb={h.get("heldout_bpb")} '
                f'avg_loss={res.get("avg_loss")} tokens={res.get("tokens")} '
                f'tok/s={res.get("sustained_tok_s")} stop={res.get("stop_reason")} '
                f'err={res.get("error")}')
            del running[gpu]
    return results


def fitness_of(res):
    if not isinstance(res, dict):
        return 99.0
    if res.get('error') or res.get('nan_seen') or res.get('nonfinite_grad'):
        return 99.0
    h = res.get('heldout') or {}
    bpb = h.get('heldout_bpb')
    return float(bpb) if bpb is not None and math.isfinite(bpb) else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--wall_seconds', type=float, default=420.0)
    ap.add_argument('--popsize', type=int, default=8)
    ap.add_argument('--ngen', type=int, default=3)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outer_timeout_s', type=float, default=900.0)
    ap.add_argument('--output', default=os.path.join(RESULTS, 'cma_all_results.json'))
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    os.makedirs(RESULTS, exist_ok=True)
    t_start = time.time()

    # CMA seed = e97_delta 0.5 / gdn-neg 0.5 (knob 4x, lam/beta at prior winner)
    x_seed = [math.log(0.5), LOG0, LOG0, LOG0, LOG0, math.log(0.5),
              math.log(4.0), 1.585, 2.747]

    # --- WARMUP: populate the shared Triton autotune cache for the e97_delta
    # chunked kernel at the exact training shape ONCE, serially, before the
    # parallel batches. A COLD autotune takes minutes; without this every worker
    # would re-pay it simultaneously. Blocking, generous timeout. ---
    warm_cfg = decode(x_seed)  # e97_delta + gdn-neg mix exercises both kernels
    log('WARMUP: compiling e97_delta + gdn kernels at training shape (cold autotune)...')
    wr = run_batch([('warmup', warm_cfg)], [gpus[0]], wall_seconds=20,
                   seed=args.seed, roundtrip=0, outer_timeout_s=900.0)
    log(f'WARMUP done: {("ok" if not (wr.get("warmup") or {}).get("error") else wr)}'
        f' (Triton cache now warm for all workers)')

    # --- anchors ---
    anchors = {}
    anchors['A_gdn2_mlp'] = decode([math.log(1.0)] + [LOG0]*4 + [LOG0, math.log(4.0), 1.585, 2.747])
    anchors['A_delta_only'] = decode([LOG0]*5 + [math.log(1.0), math.log(4.0), 1.585, 2.747])
    anchors['A_seed'] = decode(x_seed)

    anchor_cfgs = [(k, v) for k, v in anchors.items()]
    log(f'Running {len(anchor_cfgs)} anchors...')
    anchor_res = run_batch(anchor_cfgs, gpus, args.wall_seconds, args.seed, 1,
                           args.outer_timeout_s)

    # --- CMA over the 9-dim vector ---
    import cma
    x0 = x_seed
    stds = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.6, 0.4, 0.6]
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {
        'popsize': args.popsize, 'seed': 42, 'verbose': -1,
        'CMA_stds': stds,
        'bounds': [[LOG0, LOG0, LOG0, LOG0, LOG0, LOG0, 0.0, 1.0, 1.0],
                   [6, 6, 6, 6, 6, 6, 3.0, 2.5, 4.0]]})
    cma_evals = {}
    gen_best = []
    for g in range(args.ngen):
        sols = es.ask()
        cfgs = []
        for i, x in enumerate(sols):
            tag = f'cma_g{g}_e{i}'
            cfgs.append((tag, decode(x)))
        res = run_batch(cfgs, gpus, args.wall_seconds, args.seed, 0,
                        args.outer_timeout_s)
        fits = []
        for i, (tag, cfg) in enumerate(cfgs):
            r = res.get(tag, {})
            f = fitness_of(r)
            fits.append(f)
            cma_evals[tag] = dict(cfg=cfg, fitness=f, result=r)
        es.tell(sols, fits)
        bi = int(np.argmin(fits))
        gen_best.append(dict(gen=g, best_tag=cfgs[bi][0], best_bpb=fits[bi],
                             best_counts=cfgs[bi][1]['counts'], sigma=float(es.sigma)))
        log(f'GEN {g} best={cfgs[bi][0]} bpb={fits[bi]:.5f} '
            f'counts={ {k:v for k,v in cfgs[bi][1]["counts"].items() if v} } '
            f'sigma={es.sigma:.3f}')

    all_out = dict(
        task='e97delta-1p3b', kind='within-layer typed-gdn2-lm + MLP @1.3B',
        search_types=SEARCH_TYPES, param_target=1_270_000_000,
        base_axes=BASE, wall_seconds=args.wall_seconds, popsize=args.popsize,
        ngen=args.ngen, gpus=gpus,
        anchors={k: dict(cfg=anchors[k], fitness=fitness_of(anchor_res.get(k, {})),
                         result=anchor_res.get(k, {})) for k in anchors},
        cma_evals=cma_evals, gen_best=gen_best,
        wallclock_minutes=round((time.time() - t_start) / 60.0, 1),
        aggregate_gpu_minutes=round((time.time() - t_start) / 60.0 * len(gpus), 1),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(all_out, open(args.output, 'w'), indent=2)
    log(f'WROTE {args.output} ({all_out["wallclock_minutes"]} min wall, '
        f'{all_out["aggregate_gpu_minutes"]} GPU-min)')


if __name__ == '__main__':
    main()
