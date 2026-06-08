"""e97delta-1p3b: FINAL head-to-head — e97_delta+gdn-neg (CMA winner) vs gdn2-mlp,
at 1.3B, TOKEN-MATCHED *and* WALL-CLOCK-MATCHED on REAL Pile.

Reads the CMA results, takes the best SEARCHED candidate (lowest held-out BPB that
actually instantiates e97_delta heads) and the A_gdn2_mlp baseline (100% gdn-neg +
MLP). Runs, at a longer wall budget (default 12 min) with checkpoint round-trip:

  W1  e97_delta_best   @ T_wall                      -> tokens N_d, BPB_d
  W2  gdn2_mlp         @ T_wall  (WALL-CLOCK-matched) -> tokens N_g, BPB_g_wall
  W3  gdn2_mlp         capped at N_d tokens (TOKEN-matched) -> BPB_g_tok

Verdict (lower BPB = better):
  WALL-CLOCK:  BPB_d  vs  BPB_g_wall
  TOKEN:       BPB_d  vs  BPB_g_tok

Each arm run with >=2 seeds so the gap is read against within-arm seed variance.
REAL DATA. No mocks. Idle-GPU-only.
"""
import os, sys, json, time, subprocess, argparse, datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from orchestrate import run_batch, gpu_used_mib, fitness_of  # reuse pool manager
RESULTS = os.path.join(_THIS, 'results')


def log(m):
    print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def pick_best_searched(cma):
    """Best CMA candidate that actually allocates >=1 e97_delta head (the cell
    under test), by held-out BPB. Falls back to overall best if none."""
    best = None
    for tag, ev in cma.get('cma_evals', {}).items():
        cfg = ev['cfg']; f = ev['fitness']
        has_delta = cfg['counts'].get('e97_delta', 0) > 0
        if f >= 99:
            continue
        cand = (f, tag, cfg, has_delta)
        if has_delta and (best is None or f < best[0]):
            best = cand
    if best is None:  # no delta-bearing finite candidate -> overall best searched
        for tag, ev in cma.get('cma_evals', {}).items():
            if ev['fitness'] < 99 and (best is None or ev['fitness'] < best[0]):
                best = (ev['fitness'], tag, ev['cfg'], ev['cfg']['counts'].get('e97_delta', 0) > 0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cma', default=os.path.join(RESULTS, 'cma_all_results.json'))
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--wall_seconds', type=float, default=720.0)
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--outer_timeout_s', type=float, default=1200.0)
    ap.add_argument('--output', default=os.path.join(RESULTS, 'headtohead_results.json'))
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    cma = json.load(open(args.cma))

    best = pick_best_searched(cma)
    if best is None:
        raise SystemExit('no finite CMA candidate to compare')
    bpb_d_screen, best_tag, delta_cfg, has_delta = best
    base_cfg = cma['anchors']['A_gdn2_mlp']['cfg']
    log(f'BEST searched = {best_tag} screen-bpb={bpb_d_screen} '
        f'counts={ {k:v for k,v in delta_cfg["counts"].items() if v} } has_delta={has_delta}')
    log(f'BASELINE A_gdn2_mlp counts={ {k:v for k,v in base_cfg["counts"].items() if v} } '
        f'dim={base_cfg["dim"]}')

    t0 = time.time()
    # Phase 1: W1 (e97_delta_best) + W2 (gdn2_mlp) at equal wall, all seeds, roundtrip on.
    p1 = []
    for s in seeds:
        p1.append((f'W1_delta_s{s}', dict(delta_cfg, _seed=s)))
        p1.append((f'W2_gdn_s{s}', dict(base_cfg, _seed=s)))
    # run each with its own seed (run_batch takes a single seed; launch per-seed groups)
    res = {}
    for s in seeds:
        grp = [(f'W1_delta_s{s}', delta_cfg), (f'W2_gdn_s{s}', base_cfg)]
        r = run_batch(grp, gpus, args.wall_seconds, s, roundtrip=1,
                      outer_timeout_s=args.outer_timeout_s)
        res.update(r)

    # token-matched budget = min tokens reached by the e97_delta arm across seeds
    delta_tokens = [res[f'W1_delta_s{s}'].get('tokens', 0) for s in seeds
                    if isinstance(res.get(f'W1_delta_s{s}'), dict)]
    N_d = min([t for t in delta_tokens if t]) if any(delta_tokens) else None
    log(f'e97_delta tokens per seed = {delta_tokens}; token-matched budget N_d = {N_d}')

    # Phase 2: W3 gdn2_mlp token-capped at N_d (token-matched), all seeds.
    if N_d:
        for s in seeds:
            grp = [(f'W3_gdn_tokcap_s{s}', base_cfg)]
            r = run_batch_tokcap(grp, gpus, args.wall_seconds, s, N_d,
                                 args.outer_timeout_s)
            res.update(r)

    out = dict(task='e97delta-1p3b-headtohead', best_tag=best_tag,
               delta_cfg=delta_cfg, base_cfg=base_cfg, seeds=seeds,
               wall_seconds=args.wall_seconds, token_matched_budget=N_d,
               results=res, wallclock_minutes=round((time.time() - t0) / 60, 1),
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(args.output, 'w'), indent=2)
    # summary
    def bpb(tag):
        r = res.get(tag, {})
        return (r.get('heldout') or {}).get('heldout_bpb') if isinstance(r, dict) else None
    log('=== SUMMARY (held-out BPB) ===')
    for s in seeds:
        log(f' seed{s}: W1_delta={bpb(f"W1_delta_s{s}")} '
            f'W2_gdn_wall={bpb(f"W2_gdn_s{s}")} W3_gdn_tok={bpb(f"W3_gdn_tokcap_s{s}")}')
    log(f'WROTE {args.output}')


def run_batch_tokcap(configs, gpus, wall_seconds, seed, token_cap, outer_timeout_s):
    """Like run_batch but passes --token_cap to the worker (token-matched arm)."""
    from orchestrate import WORKER, FREE_MEM_MIB
    queue = list(configs); running = {}; results = {}
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
        while queue and free:
            tag, cfg = queue.pop(0); gpu = free.pop(0)
            cfgp = os.path.join(RESULTS, f'cfg_{tag}.json')
            outp = os.path.join(RESULTS, f'out_{tag}.json')
            logp = os.path.join(RESULTS, f'log_{tag}.log')
            json.dump(cfg, open(cfgp, 'w'), indent=2)
            fh = open(logp, 'w')
            proc = subprocess.Popen(
                [sys.executable, WORKER, '--config_json', cfgp, '--out', outp,
                 '--wall_seconds', str(wall_seconds), '--token_cap', str(token_cap),
                 '--seed', str(seed), '--roundtrip', '1'],
                stdout=fh, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = dict(proc=proc, tag=tag, out=outp, fh=fh, t0=time.time())
            log(f'LAUNCH(tokcap={token_cap}) {tag} on GPU {gpu}')
        time.sleep(8)
        for gpu, job in list(running.items()):
            ret = job['proc'].poll()
            to = (time.time() - job['t0']) > outer_timeout_s
            if ret is None and not to:
                continue
            if ret is None:
                job['proc'].kill()
            job['fh'].close()
            results[job['tag']] = (json.load(open(job['out']))
                                   if os.path.exists(job['out']) else dict(error='no_output'))
            del running[gpu]
    return results


if __name__ == '__main__':
    main()
