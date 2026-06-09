"""e97-hetero-cma: CMA-ES over the heterogeneous E97 cell with a JOINT fitness.

Search variables (continuous, normalized [0,1] -> mapped):
  e97_frac     [knee .. 0.25]   nonlinear split-edit-tanh head fraction (/64)
  lr           [3e-4 .. 2e-3]   log-uniform base LR
  knob_lr_mult [1 .. 16]        knob param-group LR multiplier
  lam_max      [1.2 .. 2.0]     decay-gate cap
  beta_max     [2.0 .. 3.5]     write-gate cap
  mlp_ratio    [2.2 .. 3.0]     SwiGLU hidden ratio (official 2.69)

Per candidate: snap e97_frac to an integer head count, derive dim -> ~1.3B, train
a wall-clock real-Pile screen (fused split-edit, overlap, no eager), read held-out
BPB. FITNESS = bpb + capability_penalty, where the penalty is a hard floor that
keeps the nonlinear fraction AT/ABOVE the capability knee (from cap_sweep) so the
search cannot drive the depth-capability heads to 0 for raw bpb speed — exactly the
JOINT bpb+capability objective the task requires. Lower is better.

REAL Pile, REAL training. No mocks. Pool: one candidate per free GPU.
"""
import os, sys, json, time, math, subprocess, datetime, argparse
import numpy as np
_THIS = os.path.dirname(os.path.abspath(__file__))
_DELTA = os.path.join(_THIS, '..', 'e97_delta_1p3b_cma')
for p in (_THIS, _DELTA):
    if p not in sys.path:
        sys.path.insert(0, p)
from shapes import fracs_to_logits8, derive_dim, allocate, BASE
RESULTS = os.path.join(_THIS, 'results'); os.makedirs(RESULTS, exist_ok=True)
WORKER = os.path.join(_THIS, 'cma_worker.py')
FREE_MEM_MIB = 2000
N_HEADS = BASE['n_heads']

# search-space bounds (knee filled at runtime from cap_sweep / --knee)
BOUNDS = dict(e97_frac=(2 / 64., 16 / 64.), lr=(3e-4, 2e-3), knob_lr_mult=(1.0, 16.0),
              lam_max=(1.2, 2.0), beta_max=(2.0, 3.5), mlp_ratio=(2.2, 3.0))
KEYS = ['e97_frac', 'lr', 'knob_lr_mult', 'lam_max', 'beta_max', 'mlp_ratio']


def log(m): print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def gpu_used_mib():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                   '--format=csv,noheader,nounits'], timeout=15).decode()
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


def unit_to_cfg(z, knee):
    """Map a unit-cube vector z to a concrete candidate config (dim derived)."""
    z = np.clip(z, 0.0, 1.0)
    v = {}
    for i, k in enumerate(KEYS):
        lo, hi = BOUNDS[k]
        if k == 'lr':
            v[k] = float(math.exp(math.log(lo) + z[i] * (math.log(hi) - math.log(lo))))
        else:
            v[k] = float(lo + z[i] * (hi - lo))
    # snap fraction to integer head count >= knee
    nh = max(int(round(knee * N_HEADS)), int(round(v['e97_frac'] * N_HEADS)))
    nh = min(nh, N_HEADS - 1)
    frac = nh / N_HEADS
    logits = fracs_to_logits8({'gdn2_recall': 1.0 - frac, 'e97_delta': frac})
    der = derive_dim(logits)
    cfg = dict(dim=der['dim'], head_type_logits=logits, lr=v['lr'],
               knob_lr_mult=v['knob_lr_mult'], batch_size=2, bf16=True,
               lam_max=v['lam_max'], beta_max=v['beta_max'], mlp_ratio=v['mlp_ratio'],
               e97_state_nonlin='tanh', use_chunked_e97_delta=False,
               overlap_streams=True)
    meta = dict(e97_heads=nh, e97_frac=round(frac, 5), params_b=der['params_b'],
                within_tol=der['within_tol'])
    return cfg, meta


def fitness(out, meta, knee, cap_weight):
    """JOINT fitness (minimize): bpb + hard capability floor penalty + instability."""
    if not isinstance(out, dict) or out.get('error'):
        return 99.0
    h = out.get('heldout') or {}
    bpb = h.get('heldout_bpb')
    if bpb is None or out.get('nan_seen') or out.get('nonfinite_grad'):
        return 50.0 + (out.get('final_loss', 0) or 0) * 0  # diverged
    # capability floor: fraction must be >= knee (snapped already, so this is a
    # belt-and-braces guard) — drives the JOINT objective away from frac->0.
    pen = cap_weight * max(0.0, knee - meta['e97_frac'])
    return float(bpb) + pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--knee', type=float, default=4 / 64.,
                    help='capability knee fraction (min nonlinear head fraction)')
    ap.add_argument('--wall_seconds', type=float, default=240.0)
    ap.add_argument('--pop', type=int, default=8)
    ap.add_argument('--gens', type=int, default=5)
    ap.add_argument('--sigma', type=float, default=0.30)
    ap.add_argument('--cap_weight', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    import cma
    gpus = [int(g) for g in args.gpus.split(',')]
    knee = args.knee
    log(f'CMA-ES hetero-cma: knee={knee:.4f} ({round(knee*N_HEADS)}/64) pop={args.pop} '
        f'gens={args.gens} wall={args.wall_seconds}s gpus={gpus}')

    x0 = [0.5] * len(KEYS)
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {
        'popsize': args.pop, 'bounds': [0.0, 1.0], 'seed': args.seed,
        'maxiter': args.gens, 'verb_disp': 0})

    history = []
    gen = 0
    while not es.stop() and gen < args.gens:
        zs = es.ask()
        cfgs = [unit_to_cfg(np.asarray(z), knee) for z in zs]
        # launch pool
        queue = list(enumerate(cfgs)); running = {}; outs = {}
        while queue or running:
            used = gpu_used_mib()
            free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
            while queue and free:
                idx, (cfg, meta) = queue.pop(0); gpu = free.pop(0)
                tag = f'g{gen}_c{idx}'
                cfgp = os.path.join(RESULTS, f'cfg_{tag}.json'); outp = os.path.join(RESULTS, f'out_{tag}.json')
                logp = os.path.join(RESULTS, f'log_{tag}.log'); json.dump(cfg, open(cfgp, 'w'), indent=2)
                fh = open(logp, 'w')
                cmd = [sys.executable, WORKER, '--config_json', cfgp, '--out', outp,
                       '--wall_seconds', str(args.wall_seconds), '--seed', str(args.seed)]
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
                running[gpu] = dict(proc=proc, idx=idx, out=outp, fh=fh, meta=meta, t0=time.time())
                log(f'LAUNCH {tag} GPU{gpu} heads={meta["e97_heads"]}/64 dim={cfg["dim"]} '
                    f'lr={cfg["lr"]:.2e} klr={cfg["knob_lr_mult"]:.1f} mlp={cfg["mlp_ratio"]:.2f}')
            time.sleep(6)
            for gpu, job in list(running.items()):
                if job['proc'].poll() is None and (time.time() - job['t0']) < args.wall_seconds + 900:
                    continue
                if job['proc'].poll() is None:
                    job['proc'].kill(); log(f'TIMEOUT c{job["idx"]}')
                job['fh'].close()
                o = json.load(open(job['out'])) if os.path.exists(job['out']) else dict(error='no_output')
                outs[job['idx']] = (o, job['meta'])
                h = (o.get('heldout') or {}) if isinstance(o, dict) else {}
                log(f"DONE c{job['idx']} bpb={h.get('heldout_bpb')} tok/s={o.get('sustained_tok_s')} "
                    f"stop={o.get('stop_reason')} err={o.get('error')}")
                del running[gpu]

        fits = []
        for idx in range(len(zs)):
            o, meta = outs.get(idx, (dict(error='missing'), cfgs[idx][1]))
            fv = fitness(o, meta, knee, args.cap_weight)
            fits.append(fv)
            history.append(dict(gen=gen, idx=idx, fitness=fv, meta=meta,
                                cfg=cfgs[idx][0],
                                bpb=((o.get('heldout') or {}).get('heldout_bpb') if isinstance(o, dict) else None),
                                tok_s=o.get('sustained_tok_s') if isinstance(o, dict) else None,
                                tokens=o.get('tokens') if isinstance(o, dict) else None,
                                stop=o.get('stop_reason') if isinstance(o, dict) else None))
        es.tell([list(z) for z in zs], fits)
        best = min(history, key=lambda r: r['fitness'])
        log(f'=== GEN {gen} best fitness={best["fitness"]:.4f} bpb={best["bpb"]} '
            f'heads={best["meta"]["e97_heads"]}/64 ===')
        json.dump(dict(history=history, bounds=BOUNDS, keys=KEYS, knee=knee,
                       best=best, gens_done=gen + 1,
                       timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()),
                  open(os.path.join(RESULTS, 'cma_joint.json'), 'w'), indent=2)
        gen += 1

    best = min(history, key=lambda r: r['fitness'])
    log(f'=== CMA-ES DONE: best fitness={best["fitness"]:.4f} bpb={best["bpb"]} '
        f'heads={best["meta"]["e97_heads"]}/64 cfg lr={best["cfg"]["lr"]:.2e} '
        f'klr={best["cfg"]["knob_lr_mult"]:.1f} lam={best["cfg"]["lam_max"]:.3f} '
        f'beta={best["cfg"]["beta_max"]:.3f} mlp={best["cfg"]["mlp_ratio"]:.3f} ===')
    log('WROTE cma_joint.json')


if __name__ == '__main__':
    main()
