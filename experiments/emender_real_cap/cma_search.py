"""emender-real-cap CORE DELIVERABLE: CMA-ES SEARCHES the Emender mixture.

The revised task: do NOT hand-pick the nonlinear fraction. Run CMA-ES over the
head-type COMPOSITION and let it FIND the mixture. If CMA drives the nonlinear
fraction -> 0 that IS GDN-2 (convergent-loss null); if it keeps a nonlinear
fraction, that is the real Emender, DISCOVERED not assumed.

Search space (2-D): x = [a_delta, a_track] -> 9-type head_type_logits via
  composition_logits: softmax over [gdn2_recall=0 (the SEA), e97_delta=a_delta,
  e97_track=a_track]. Both E97 emendment-family heads (e97_delta = split-edit
  bounded-nonlinearity tanh = the named emendment cell; e97_track = E97 split-gated
  reflection/tracking). gdn2_recall fills the rest. Param/FLOP-LOCKED: dim is
  DERIVED per candidate so every model hits TARGET_PARAMS (the mixture search is
  not an accidental capacity search).

Fitness = held-out BPB on the fixed real-Pile slice (the task's primary fitness),
bf16 UNIFORM on every arm, fused split-edit asserted (lm_worker raises if eager).
PRECISION: fp16 is IMPOSSIBLE for the fused emendment kernel (verified, see
emender_common) -> bf16 uniform is the matched-precision fix to the opt-1p3b fp32
strawman. tok/s recorded but NEVER used for selection.

Idle-GPU-only / NO-PREEMPT (coexists with other agents' jobs): a GPU is usable
iff used-mem < FREE_MEM_MIB and not already running one of THIS driver's workers.
Resumable: a candidate whose result json exists is reused.
"""
import os, sys, json, time, subprocess, argparse, datetime, csv

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _THIS)
sys.path.insert(0, _ROOT)

import numpy as np
import emender_common as EC

FREE_MEM_MIB = 2500
WORKER = os.path.join(_THIS, 'lm_worker.py')
VOCAB = 50281

# --- proxy LM shape for the search (small, fast, param-matched) ---
DEPTH, N_HEADS, N_STATE = 8, 32, 32
TARGET_PARAMS = 40e6
BATCH_SIZE = 8
CHUNK = 512
MAX_TOKENS = 6_000_000      # token-matched budget per candidate
WALL_CAP_MIN = 6.0          # hard wall cap (safety)
BPB_BATCHES = 25
LR = 1.2e-3

# CMA bounds: a in [-9, 2]; a=-9 => ~0 heads of that type (pure GDN-2 recovered).
BOUND_LO, BOUND_HI = -9.0, 2.0


def decode(x):
    a_delta = float(np.clip(x[0], BOUND_LO, BOUND_HI))
    a_track = float(np.clip(x[1], BOUND_LO, BOUND_HI))
    logits = EC.composition_logits(a_delta, a_track)
    dim, params = EC.derive_lm_dim_logits(logits, DEPTH, N_HEADS, N_STATE, VOCAB,
                                          TARGET_PARAMS, dim_multiple=64,
                                          dim_lo=256, dim_hi=1024)
    counts = EC.head_counts(logits, N_HEADS)
    return dict(a_delta=a_delta, a_track=a_track, logits=logits, dim=dim,
                model_params=params, counts=counts,
                f_delta=counts['e97_delta'] / N_HEADS,
                f_track=counts['e97_track'] / N_HEADS)


def cfg_for(name, dec):
    return dict(name=name, head_type_logits=dec['logits'], dim=dec['dim'],
                depth=DEPTH, n_heads=N_HEADS, n_state=N_STATE, lr=LR,
                batch_size=BATCH_SIZE, chunk=CHUNK, eval_chunk=CHUNK,
                max_tokens=MAX_TOKENS, wall_minutes=WALL_CAP_MIN, seed=0)


def gpu_used_mib():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def run_batch(named_decs, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results, pending = {}, []
    for name, dec in named_decs:
        out_path = os.path.join(out_dir, f'res_{name}.json')
        if os.path.exists(out_path):
            try:
                results[name] = json.load(open(out_path))
                print(f"[cached] {name} bpb={results[name].get('heldout',{}).get('heldout_bpb')}",
                      flush=True)
                continue
            except (json.JSONDecodeError, OSError):
                pass
        pending.append((name, dec))
    running = {}
    while pending or running:
        for gpu in list(running):
            name, proc, out_path, t0 = running[gpu]
            if proc.poll() is not None:
                del running[gpu]
                if os.path.exists(out_path):
                    results[name] = json.load(open(out_path))
                    r = results[name]
                    print(f"[done] gpu{gpu} {name} bpb={r.get('heldout',{}).get('heldout_bpb')} "
                          f"tok/s={r.get('tok_s')} counts={r.get('counts')} "
                          f"({(time.time()-t0)/60:.1f}m)", flush=True)
                else:
                    results[name] = dict(name=name, error='no_output', rc=proc.returncode)
                    print(f"[FAIL] gpu{gpu} {name} rc={proc.returncode}", flush=True)
        if pending:
            used = gpu_used_mib()
            for gpu in args.gpus:
                if not pending:
                    break
                if gpu in running or used.get(gpu, 10**9) >= FREE_MEM_MIB:
                    continue
                name, dec = pending.pop(0)
                cfg = cfg_for(name, dec)
                cfg_path = os.path.join(out_dir, f'cfg_{name}.json')
                out_path = os.path.join(out_dir, f'res_{name}.json')
                json.dump(cfg, open(cfg_path, 'w'), indent=2)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(os.path.join(out_dir, f'log_{name}.txt'), 'w')
                cmd = [sys.executable, WORKER, '--config_json', cfg_path,
                       '--out', out_path, '--wall_minutes', str(WALL_CAP_MIN),
                       '--bpb_batches', str(BPB_BATCHES)]
                proc = subprocess.Popen(cmd, cwd=_ROOT, env=env, stdout=logf,
                                        stderr=subprocess.STDOUT)
                running[gpu] = (name, proc, out_path, time.time())
                print(f"[run ] gpu{gpu} {name} dim={cfg['dim']} "
                      f"params={dec['model_params']:,} counts={dec['counts']}", flush=True)
                time.sleep(4)
        time.sleep(8)
    return results


def bpb_of(r):
    ho = (r or {}).get('heldout') or {}
    b = ho.get('heldout_bpb')
    if b is None or not np.isfinite(b) or r.get('nan_seen'):
        return 1e6
    return float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=os.path.join(_THIS, 'results_cma'))
    ap.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
    ap.add_argument('--popsize', type=int, default=5)
    ap.add_argument('--generations', type=int, default=4)
    ap.add_argument('--sigma', type=float, default=2.0)
    args = ap.parse_args()
    args.gpus = [int(g) for g in args.gpus.split(',')]
    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    # ---- anchors: pure GDN-2 (null) + a recall-heavy emender seed ----
    anchors = {
        'anchor_gdn2': decode([BOUND_LO, BOUND_LO]),       # pure GDN-2 (f->0)
        'anchor_emender_seed': decode([-2.5, -3.5]),       # recall-heavy sprinkle (CMA seed)
    }
    print("=== ANCHORS ===", flush=True)
    for n, d in anchors.items():
        print(f"  {n}: f_delta={d['f_delta']:.4f} f_track={d['f_track']:.4f} "
              f"dim={d['dim']} params={d['model_params']:,}", flush=True)
    a_res = run_batch(list(anchors.items()), args, os.path.join(args.output, 'anchors'))

    # ---- CMA over [a_delta, a_track] (bpb fitness) ----
    import cma
    x0 = [-2.5, -3.5]   # recall-heavy seed (small emendment fraction)
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {
        'popsize': args.popsize, 'seed': 42, 'verbose': -1,
        'bounds': [[BOUND_LO, BOUND_LO], [BOUND_HI, BOUND_HI]]})
    gens_path = os.path.join(args.output, 'generations.jsonl')
    all_cands = []
    for gen in range(args.generations):
        sols = es.ask()
        decs = [decode(s) for s in sols]
        named = [(f'cma_g{gen}_e{j}', decs[j]) for j in range(len(sols))]
        print(f"=== CMA gen {gen+1}/{args.generations} ({len(named)} candidates) ===", flush=True)
        res = run_batch(named, args, os.path.join(args.output, f'gen{gen}'))
        fits = []
        for (name, dec), sol in zip(named, sols):
            r = res.get(name, {})
            b = bpb_of(r)
            fits.append(b)
            rec = dict(name=name, gen=gen, a_delta=dec['a_delta'], a_track=dec['a_track'],
                       f_delta=dec['f_delta'], f_track=dec['f_track'], dim=dec['dim'],
                       model_params=dec['model_params'], counts=dec['counts'],
                       heldout_bpb=(None if b >= 1e6 else b),
                       tok_s=r.get('tok_s'), avg_loss=r.get('avg_loss'),
                       fused_asserted=r.get('fused_asserted'),
                       fused_typed_layers=r.get('fused_typed_layers'))
            all_cands.append(rec)
        es.tell(sols, fits)
        bi = int(np.argmin(fits))
        snap = dict(gen=gen, fits=[float(x) for x in fits], gen_best=float(min(fits)),
                    gen_best_name=named[bi][0], gen_best_f_delta=decs[bi]['f_delta'],
                    gen_best_f_track=decs[bi]['f_track'], gen_best_counts=decs[bi]['counts'],
                    sigma=float(es.sigma),
                    cum_min=round((time.time()-t0)/60.0, 1))
        with open(gens_path, 'a') as f:
            f.write(json.dumps(snap, default=str) + '\n')
        print(f"  gen_best bpb={min(fits):.5f} f_delta={decs[bi]['f_delta']:.4f} "
              f"f_track={decs[bi]['f_track']:.4f} counts={decs[bi]['counts']}", flush=True)

    # ---- pick the CMA-found mixture (best bpb over all real candidates + anchors) ----
    pool = []
    for n, r in a_res.items():
        pool.append((n, bpb_of(r), r.get('counts'),
                     r.get('counts', {}).get('e97_delta', 0) / N_HEADS,
                     r.get('counts', {}).get('e97_track', 0) / N_HEADS))
    for rec in all_cands:
        if rec['heldout_bpb'] is not None:
            pool.append((rec['name'], rec['heldout_bpb'], rec['counts'],
                         rec['f_delta'], rec['f_track']))
    pool.sort(key=lambda t: t[1])
    found = pool[0]
    found_spec = dict(
        name=found[0], heldout_bpb=found[1], counts=found[2],
        f_delta=found[3], f_track=found[4],
        nonlinear_fraction=found[3] + found[4],
        verdict=('GDN-2 (CMA drove nonlinear fraction -> ~0: convergent-loss null)'
                 if (found[3] + found[4]) < 0.02 else
                 'EMENDER (CMA kept a nonlinear emendment fraction on loss)'))

    out = dict(search='CMA-ES over [a_delta,a_track] -> {gdn2_recall sea, e97_delta, e97_track}',
               fitness='held-out BPB (real-Pile slice), bf16 UNIFORM, fused asserted',
               precision='bf16 uniform on ALL arms; fp16 impossible for fused emendment kernel',
               param_target=TARGET_PARAMS, proxy_shape=dict(depth=DEPTH, n_heads=N_HEADS,
               n_state=N_STATE, chunk=CHUNK, max_tokens=MAX_TOKENS),
               popsize=args.popsize, generations=args.generations,
               anchors={n: dict(bpb=bpb_of(r), counts=r.get('counts'), tok_s=r.get('tok_s'))
                        for n, r in a_res.items()},
               candidates=all_cands, found_mixture=found_spec,
               leaderboard=[dict(name=t[0], bpb=t[1], f_delta=t[3], f_track=t[4],
                                 counts=t[2]) for t in pool[:10]],
               wall_minutes=round((time.time()-t0)/60.0, 1),
               timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    json.dump(out, open(os.path.join(args.output, 'cma_result.json'), 'w'),
              indent=2, default=str)

    # candidates CSV
    rows = all_cands
    if rows:
        keys = ['name', 'gen', 'a_delta', 'a_track', 'f_delta', 'f_track', 'dim',
                'model_params', 'heldout_bpb', 'tok_s', 'avg_loss', 'fused_asserted']
        with open(os.path.join(args.output, 'candidates.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)

    print("\n=== CMA-FOUND MIXTURE ===", flush=True)
    print(json.dumps(found_spec, indent=2), flush=True)
    print(f"DONE {out['wall_minutes']}min -> {args.output}", flush=True)


if __name__ == '__main__':
    main()
