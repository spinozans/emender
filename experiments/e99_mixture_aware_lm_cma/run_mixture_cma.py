"""redo-e99-1-3b: mixture-aware LM-CMA driver (idle-GPU-only, parallel).

Searches the GDN-2 / nonlinear mixture as an EXPLICIT CMA variable on the real
1.3B-class LM path. The architectural axes that are not under test stay fixed
(depth17/n_heads102/n_state32, bf16, lr=1.471e-3), but dim is derived per
deterministic head allocation to a fixed counted-parameter target. This keeps the
mixture search from becoming an accidental capacity search.

Stage 1 (anchors): the deterministic anchor set from mixtures.py -- dense GDN-2
  control, prior-E99 5:1, matched-fraction (c) legacy-nonlin and (b) GDN-2-shell
  triples at f in {1/6,1/3,1/2}.
Stage 2 (CMA): popsize-P CMA over the 6 logits, seeded at the prior-E99 winner,
  fitness = 15-min bf16 AvgLoss (handoff convention). Deterministic head
  allocation + exact head counts + exact counted LadderLM params logged for every
  eval. Configs outside the target tolerance are rejected before launch.

Idle-GPU-only / NO-PREEMPT: a GPU is usable iff nvidia-smi used-mem < 2GB and it
is not already running one of THIS driver's workers. Never shares/preempts.
All idle GPUs are used in parallel (one screen per GPU). Records per-eval
wallclock, tok/s, tokens, GPU id, peak mem, BPB, stability + aggregate GPU-min.
"""
import os, sys, json, time, subprocess, argparse, datetime, csv, math

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _THIS)
sys.path.insert(0, _ROOT)

import numpy as np
from mixtures import build_anchors, head_counts, TYPE_NAMES

FREE_MEM_MIB = 2000  # idle iff used < 2GB (matches the upstream E99 convention)
WORKER = os.path.join(_THIS, 'screen_worker.py')

# Fixed non-capacity axes from the prior-E99 front-runner eval87. `dim` is
# intentionally absent: it is derived per deterministic mixture allocation.
BASE_SHAPE = dict(depth=17, n_heads=102, n_state=32, expansion=1.0,
                  bf16=True, lr=1.471e-3)
VOCAB_SIZE = 50281
PARAM_TARGET = 1_270_000_000
PARAM_TOLERANCE = 0.02
DIM_MULTIPLE = 64
DIM_MIN = 2560
DIM_MAX = 4352
SHELL_NONLIN = 'tanh'  # the bounded/centered nonlinear-in-time map (verified)

_COUNT_CACHE = {}
_DERIVE_CACHE = {}


def _counts_key(counts):
    return tuple((t, int(counts.get(t, 0))) for t in TYPE_NAMES)


def _layer_kwargs(logits, counts):
    lk = dict(head_type_logits=[float(x) for x in logits],
              gdn_allow_neg_eigval=True, lam_max=1.585, beta_max=2.747)
    if counts.get('gdn2_nonlin_shell', 0) > 0:
        lk['shell_state_nonlin'] = SHELL_NONLIN
        lk['shell_state_chunk'] = 64
    return lk


def count_ladder_params(dim, logits, vocab_size=VOCAB_SIZE):
    """Exact LadderLM parameter count for a fixed deterministic allocation.

    The count is cached by integer head counts because the model parameter shapes
    depend on counts, not on the continuous logits that produced them.
    """
    counts = head_counts(logits, BASE_SHAPE['n_heads'])
    key = (int(dim), int(vocab_size), BASE_SHAPE['depth'], BASE_SHAPE['n_heads'],
           BASE_SHAPE['n_state'], float(BASE_SHAPE['expansion']), _counts_key(counts))
    if key in _COUNT_CACHE:
        return _COUNT_CACHE[key]

    import torch
    import ndm.models.typed_head_mixture as typed_head_mixture
    from ndm.models.ladder_lm import LadderLM

    kwargs = dict(
        vocab_size=int(vocab_size),
        dim=int(dim),
        depth=BASE_SHAPE['depth'],
        level='typed-gdn2-lm',
        n_heads=BASE_SHAPE['n_heads'],
        n_state=BASE_SHAPE['n_state'],
        expansion=BASE_SHAPE['expansion'],
        layer_kwargs=_layer_kwargs(logits, counts),
    )

    old_allocate_types = typed_head_mixture.allocate_types

    def allocate_types_cpu(n_heads, head_type_logits):
        # The model allocator calls `.tolist()`, so it must stay on CPU even
        # when the surrounding module parameters are constructed on meta.
        with torch.device('cpu'):
            return old_allocate_types(n_heads, head_type_logits)

    typed_head_mixture.allocate_types = allocate_types_cpu
    try:
        with torch.device('meta'):
            model = LadderLM(**kwargs)
    except Exception:
        typed_head_mixture.allocate_types = old_allocate_types
        model = LadderLM(**kwargs)
    finally:
        typed_head_mixture.allocate_types = old_allocate_types
    n_params = int(sum(p.numel() for p in model.parameters()))
    del model
    _COUNT_CACHE[key] = n_params
    return n_params


def _dim_grid(dim_min, dim_max, multiple):
    start = int(math.ceil(dim_min / multiple) * multiple)
    stop = int(math.floor(dim_max / multiple) * multiple)
    return list(range(start, stop + 1, multiple))


def _param_row(name, logits, dim, target_params, tolerance, vocab_size):
    counts = head_counts(logits, BASE_SHAPE['n_heads'])
    n_params = count_ladder_params(dim, logits, vocab_size=vocab_size)
    rel = (n_params - int(target_params)) / float(target_params)
    return dict(
        name=name,
        dim=int(dim),
        model_params=int(n_params),
        params_b=round(n_params / 1e9, 6),
        param_target=int(target_params),
        param_tolerance=float(tolerance),
        param_target_rel_error=rel,
        param_target_error_pct=round(rel * 100.0, 4),
        param_target_within_tolerance=abs(rel) <= float(tolerance),
        counts=counts,
    )


def assert_param_target(row):
    if not row['param_target_within_tolerance']:
        raise AssertionError(
            f"{row['name']} dim={row['dim']} params={row['model_params']:,} "
            f"target={row['param_target']:,} "
            f"err={row['param_target_error_pct']:.3f}% exceeds "
            f"+/-{row['param_tolerance'] * 100:.1f}% counts={row['counts']}")


def derive_dim(logits, target_params=PARAM_TARGET, tolerance=PARAM_TOLERANCE,
               vocab_size=VOCAB_SIZE, dim_multiple=DIM_MULTIPLE,
               dim_min=DIM_MIN, dim_max=DIM_MAX, name='candidate'):
    """Derive dim for one deterministic mixture allocation, then exact-count it.

    For fixed head counts, LadderLM parameter count is a smooth quadratic in dim
    at expansion=1.0. Three real LadderLM counts give a good candidate; the final
    selected dim and its immediate neighbors are exact-counted and asserted.
    """
    counts = head_counts(logits, BASE_SHAPE['n_heads'])
    cache_key = (int(target_params), float(tolerance), int(vocab_size),
                 int(dim_multiple), int(dim_min), int(dim_max), _counts_key(counts))
    if cache_key in _DERIVE_CACHE:
        row = dict(_DERIVE_CACHE[cache_key])
        row['name'] = name
        return row

    grid = _dim_grid(dim_min, dim_max, dim_multiple)
    if len(grid) < 3:
        raise ValueError(f"dim grid too small: min={dim_min} max={dim_max} "
                         f"multiple={dim_multiple}")

    probe_dims = sorted({
        min(grid, key=lambda d: abs(d - 3072)),
        min(grid, key=lambda d: abs(d - 3328)),
        min(grid, key=lambda d: abs(d - 3840)),
    })
    if len(probe_dims) < 3:
        probe_dims = [grid[0], grid[len(grid) // 2], grid[-1]]
    probe_params = [count_ladder_params(d, logits, vocab_size=vocab_size)
                    for d in probe_dims]
    coef = np.polyfit(np.array(probe_dims, dtype=np.float64),
                      np.array(probe_params, dtype=np.float64), 2)
    estimated = min(grid, key=lambda d: abs(float(np.polyval(coef, d)) - target_params))

    local_dims = sorted({d for d in
                         (estimated + k * dim_multiple for k in range(-4, 5))
                         if dim_min <= d <= dim_max})
    rows = [_param_row(name, logits, d, target_params, tolerance, vocab_size)
            for d in local_dims]
    best = min(rows, key=lambda r: abs(r['model_params'] - target_params))

    if not best['param_target_within_tolerance']:
        # Rare fallback if the local quadratic estimate missed due to a future
        # architecture discontinuity. This is slower but keeps the hard gate exact.
        rows = [_param_row(name, logits, d, target_params, tolerance, vocab_size)
                for d in grid]
        best = min(rows, key=lambda r: abs(r['model_params'] - target_params))

    assert_param_target(best)
    cached = dict(best)
    cached['name'] = '<cached>'
    _DERIVE_CACHE[cache_key] = cached
    return best


def gpu_used_mib():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def cfg_for(name, logits, args=None):
    target_params = getattr(args, 'target_params', PARAM_TARGET)
    tolerance = getattr(args, 'param_tolerance', PARAM_TOLERANCE)
    vocab_size = getattr(args, 'vocab_size', VOCAB_SIZE)
    dim_multiple = getattr(args, 'dim_multiple', DIM_MULTIPLE)
    dim_min = getattr(args, 'dim_min', DIM_MIN)
    dim_max = getattr(args, 'dim_max', DIM_MAX)
    row = derive_dim(logits, target_params=target_params, tolerance=tolerance,
                     vocab_size=vocab_size, dim_multiple=dim_multiple,
                     dim_min=dim_min, dim_max=dim_max, name=name)
    assert_param_target(row)

    c = dict(BASE_SHAPE)
    c.update(
        name=name,
        dim=row['dim'],
        head_type_logits=[float(x) for x in logits],
        counts=row['counts'],
        model_params=row['model_params'],
        params_b=row['params_b'],
        param_target=row['param_target'],
        param_tolerance=row['param_tolerance'],
        param_target_rel_error=row['param_target_rel_error'],
        param_target_error_pct=row['param_target_error_pct'],
        param_target_within_tolerance=row['param_target_within_tolerance'],
        param_vocab_size=int(vocab_size),
    )
    # only attach a shell nonlinearity when the mixture actually allocates shell heads
    cnt = row['counts']
    if cnt.get('gdn2_nonlin_shell', 0) > 0:
        c['shell_state_nonlin'] = SHELL_NONLIN
        c['shell_state_chunk'] = 64
    return c


def _cached_result_ok(result, cfg):
    if not result or result.get('error'):
        return False
    if int(result.get('dim', -1)) != int(cfg['dim']):
        return False
    n_params = result.get('model_params')
    if n_params is None:
        return False
    rel = abs((int(n_params) - int(cfg['param_target'])) / float(cfg['param_target']))
    if rel > float(cfg['param_tolerance']):
        return False
    if result.get('param_target') and int(result['param_target']) != int(cfg['param_target']):
        return False
    return True


def run_batch(named_logits, args, out_subdir, gpu_log):
    """Evaluate a list of (name, logits) on idle GPUs in parallel. NO-PREEMPT.
    Returns name -> result dict."""
    out_dir = os.path.join(args.output, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    pending = []
    for name, logits in named_logits:  # resume: skip already-completed evals
        cfg = cfg_for(name, logits, args)
        print(f"[cfg ] {name} dim={cfg['dim']} params={cfg['model_params']:,} "
              f"err={cfg['param_target_error_pct']:+.3f}% "
              f"counts={cfg['counts']}", flush=True)
        out_path = os.path.join(out_dir, f'res_{name}.json')
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    cached = json.load(f)
                if _cached_result_ok(cached, cfg):
                    results[name] = cached
                    print(f"[cached] {name}", flush=True)
                    continue
                print(f"[stale] {name} cached result is not target-sized; rerun",
                      flush=True)
            except (json.JSONDecodeError, OSError):
                pass
        pending.append((name, cfg))
    running = {}  # gpu -> (name, proc, out_path, t0)
    while pending or running:
        # reap finished
        for gpu in list(running):
            name, proc, out_path, t0 = running[gpu]
            if proc.poll() is not None:
                del running[gpu]
                if os.path.exists(out_path):
                    with open(out_path) as f:
                        results[name] = json.load(f)
                    r = results[name]
                    print(f"[done] gpu{gpu} {name} avg={r.get('avg_loss')} "
                          f"bpb={(r.get('heldout') or {}).get('heldout_bpb')} "
                          f"tok/s={r.get('tok_s')} steps={r.get('steps')} "
                          f"({(time.time()-t0)/60:.1f}m)", flush=True)
                else:
                    results[name] = dict(name=name, error='no_output',
                                         returncode=proc.returncode)
                    print(f"[FAIL] gpu{gpu} {name} rc={proc.returncode}", flush=True)
        # launch on idle GPUs (never preempt)
        if pending:
            used = gpu_used_mib()
            for gpu in args.gpus:
                if not pending:
                    break
                if gpu in running:
                    continue
                if used.get(gpu, 10**9) >= FREE_MEM_MIB:
                    continue  # busy with someone else's job -> skip (no-preempt)
                name, cfg = pending.pop(0)
                cfg_path = os.path.join(out_dir, f'cfg_{name}.json')
                out_path = os.path.join(out_dir, f'res_{name}.json')
                with open(cfg_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(os.path.join(out_dir, f'log_{name}.txt'), 'w')
                cmd = [sys.executable, WORKER, '--config_json', cfg_path,
                       '--out', out_path, '--wall_minutes', str(args.wall_minutes),
                       '--bpb_batches', str(args.bpb_batches),
                       '--roundtrip', str(args.roundtrip), '--outdir', out_dir]
                proc = subprocess.Popen(cmd, cwd=_ROOT, env=env,
                                        stdout=logf, stderr=subprocess.STDOUT)
                running[gpu] = (name, proc, out_path, time.time())
                gpu_log.append(dict(name=name, gpu=gpu,
                                    dim=cfg['dim'],
                                    model_params=cfg['model_params'],
                                    counts=cfg['counts'],
                                    t=datetime.datetime.utcnow().isoformat() + 'Z'))
                print(f"[run ] gpu{gpu} {name} dim={cfg['dim']} "
                      f"params={cfg['model_params']:,} "
                      f"counts={cfg['counts']}", flush=True)
                time.sleep(4)  # stagger launches (autotune / data init)
        time.sleep(8)
    return results


def smoke_param_check(args):
    from mixtures import PRIOR_E99_LOGITS_5
    checks = [
        ('smoke_dense', [12.0, -12.0, -12.0, -12.0, -12.0]),
        ('smoke_5to1', list(PRIOR_E99_LOGITS_5)),
        ('smoke_all_nonlin', [-12.0, -12.0, -12.0, -12.0, 12.0]),
    ]
    cfgs = []
    for name, logits in checks:
        cfg = cfg_for(name, logits, args)
        cfgs.append(cfg)
        assert_param_target(cfg)
        print(json.dumps(dict(name=name, dim=cfg['dim'],
                              model_params=cfg['model_params'],
                              params_b=cfg['params_b'],
                              param_target_error_pct=cfg['param_target_error_pct'],
                              counts=cfg['counts']),
                         sort_keys=True), flush=True)
    dims = {cfg['dim'] for cfg in cfgs}
    if len(dims) <= 1:
        raise AssertionError(f"smoke dims did not vary per mixture: {sorted(dims)}")
    print(f"SMOKE_PARAM_CHECK OK target={int(args.target_params):,} "
          f"tol=+/-{args.param_tolerance * 100:.1f}% dims={sorted(dims)}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=os.path.join(_THIS, 'results', 'run1'))
    ap.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
    ap.add_argument('--wall_minutes', type=float, default=15.0)
    ap.add_argument('--bpb_batches', type=int, default=40)
    ap.add_argument('--roundtrip', type=int, default=0)
    ap.add_argument('--anchor_roundtrip', type=int, default=1,
                    help='round-trip the anchors (hard gate) even if --roundtrip 0')
    ap.add_argument('--cma_popsize', type=int, default=6)
    ap.add_argument('--cma_generations', type=int, default=4)
    ap.add_argument('--cma_sigma', type=float, default=1.5)
    ap.add_argument('--skip_cma', type=int, default=0)
    ap.add_argument('--target_params', type=float, default=float(PARAM_TARGET))
    ap.add_argument('--param_tolerance', type=float, default=PARAM_TOLERANCE)
    ap.add_argument('--vocab_size', type=int, default=VOCAB_SIZE)
    ap.add_argument('--dim_multiple', type=int, default=DIM_MULTIPLE)
    ap.add_argument('--dim_min', type=int, default=DIM_MIN)
    ap.add_argument('--dim_max', type=int, default=DIM_MAX)
    ap.add_argument('--smoke_param_check', action='store_true',
                    help='CPU-only 3-mixture parameter sizing check, then exit')
    args = ap.parse_args()
    args.target_params = int(args.target_params)
    if args.smoke_param_check:
        smoke_param_check(args)
        return
    args.gpus = [int(g) for g in args.gpus.split(',')]
    os.makedirs(args.output, exist_ok=True)
    gpu_log = []
    t0 = time.time()

    # ---- Stage 1: anchors ----
    anchors = build_anchors()
    anchor_jobs = [(name, spec['logits']) for name, spec in anchors.items()]
    a_args = argparse.Namespace(**vars(args))
    a_args.roundtrip = max(args.roundtrip, args.anchor_roundtrip)
    print(f"=== STAGE 1: {len(anchor_jobs)} anchors on GPUs {args.gpus} ===", flush=True)
    anchor_res = run_batch(anchor_jobs, a_args, 'anchors', gpu_log)
    with open(os.path.join(args.output, 'anchors_results.json'), 'w') as f:
        json.dump({n: anchor_res[n] for n in anchor_res}, f, indent=2, default=str)

    # ---- Stage 2: CMA over the 6 mixture logits (matched target params) ----
    cma_results = []
    if not args.skip_cma:
        import cma
        from mixtures import PRIOR_E99_LOGITS_5
        x0 = list(PRIOR_E99_LOGITS_5) + [-2.0]  # seed: prior winner + small shell logit
        es = cma.CMAEvolutionStrategy(x0, args.cma_sigma, {
            'popsize': args.cma_popsize, 'seed': 42, 'verbose': -1,
            'bounds': [[-6.0] * 6, [6.0] * 6]})
        eval_id = 0
        gens_path = os.path.join(args.output, 'generations.jsonl')
        for gen in range(args.cma_generations):
            sols = es.ask()
            jobs = []
            for j, sol in enumerate(sols):
                name = f'cma_g{gen}_e{j}'
                jobs.append((name, [float(x) for x in sol]))
            print(f"=== STAGE 2 CMA gen {gen+1}/{args.cma_generations}: "
                  f"{len(jobs)} candidates ===", flush=True)
            res = run_batch(jobs, args, f'cma_gen{gen}', gpu_log)
            fits = []
            for (name, logits), sol in zip(jobs, sols):
                r = res.get(name, {})
                loss = r.get('avg_loss', float('inf'))
                if loss is None or not np.isfinite(loss):
                    loss = 1e6
                fits.append(loss)
                r['eval_id'] = eval_id
                r['logits'] = logits
                cma_results.append(r)
                eval_id += 1
            es.tell(sols, fits)
            best_i = int(np.argmin(fits))
            snap = dict(gen=gen, fits=[float(x) for x in fits],
                        gen_best=float(min(fits)),
                        gen_best_logits=jobs[best_i][1],
                        gen_best_counts=head_counts(jobs[best_i][1], BASE_SHAPE['n_heads']),
                        gen_best_dim=res.get(jobs[best_i][0], {}).get('dim'),
                        gen_best_model_params=res.get(jobs[best_i][0], {}).get('model_params'),
                        param_target=args.target_params,
                        param_tolerance=args.param_tolerance,
                        sigma=float(es.sigma),
                        cum_gpu_minutes=round((time.time()-t0)/60.0 * len(args.gpus), 1))
            with open(gens_path, 'a') as f:
                f.write(json.dumps(snap, default=str) + '\n')
            print(f"  gen_best={min(fits):.4f} counts={snap['gen_best_counts']}", flush=True)

    # ---- aggregate ----
    all_results = dict(anchors=anchor_res,
                       cma=cma_results,
                       base_shape=BASE_SHAPE, shell_nonlin=SHELL_NONLIN,
                       param_target=args.target_params,
                       param_tolerance=args.param_tolerance,
                       dim_multiple=args.dim_multiple,
                       wall_minutes_per_eval=args.wall_minutes,
                       gpus_used=sorted(set(g['gpu'] for g in gpu_log)),
                       n_evals=len(anchor_res) + len(cma_results),
                       gpu_log=gpu_log,
                       wallclock_minutes=round((time.time()-t0)/60.0, 1),
                       aggregate_gpu_minutes=round((time.time()-t0)/60.0 * len(args.gpus), 1),
                       timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    with open(os.path.join(args.output, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # CSV of every eval (anchors + cma) for the report
    rows = []
    def _row(r):
        hd = r.get('heldout') or {}
        c = r.get('counts') or {}
        return dict(name=r.get('name'), gpu=r.get('gpu'),
                    dim=r.get('dim'), model_params=r.get('model_params'),
                    param_target=r.get('param_target'),
                    param_target_error_pct=r.get('param_target_error_pct'),
                    n_gdn=c.get('gdn2_recall'), n_nonlin=c.get('nonlin'),
                    n_shell=c.get('gdn2_nonlin_shell'),
                    avg_loss=r.get('avg_loss'), final_loss=r.get('final_loss'),
                    heldout_bpb=hd.get('heldout_bpb'), tok_s=r.get('tok_s'),
                    steps=r.get('steps'), tokens=r.get('tokens'),
                    wall_minutes=r.get('wall_minutes'), params_b=r.get('params_b'),
                    nan_seen=r.get('nan_seen'),
                    roundtrip_ok=(r.get('roundtrip') or {}).get('ok'))
    for n, r in anchor_res.items():
        rows.append(_row(r))
    for r in cma_results:
        rows.append(_row(r))
    csv_path = os.path.join(args.output, 'candidates.csv')
    if rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nDONE: {all_results['n_evals']} evals, "
          f"{all_results['wallclock_minutes']}min wall, "
          f"{all_results['aggregate_gpu_minutes']} GPU-min -> {args.output}", flush=True)


if __name__ == '__main__':
    main()
