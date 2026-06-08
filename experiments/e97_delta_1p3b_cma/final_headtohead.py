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
    # Phase 1: W1 (e97_delta_best) + W2 (gdn2_mlp) at equal wall, ALL seeds in
    # PARALLEL across the 8 GPUs (per-job seed), roundtrip on.
    jobs = []
    for s in seeds:
        jobs.append((f'W1_delta_s{s}', delta_cfg, s))
        jobs.append((f'W2_gdn_s{s}', base_cfg, s))
    res = run_jobs_perseed(jobs, gpus, args.wall_seconds, None, 1, args.outer_timeout_s)

    # token-matched budget = min tokens reached by the e97_delta arm across seeds
    delta_tokens = [res[f'W1_delta_s{s}'].get('tokens', 0) for s in seeds
                    if isinstance(res.get(f'W1_delta_s{s}'), dict)]
    N_d = min([t for t in delta_tokens if t]) if any(delta_tokens) else None
    log(f'e97_delta tokens per seed = {delta_tokens}; token-matched budget N_d = {N_d}')

    # Phase 2: W3 gdn2_mlp token-capped at N_d (token-matched), all seeds PARALLEL.
    if N_d:
        jobs2 = [(f'W3_gdn_tokcap_s{s}', base_cfg, s) for s in seeds]
        r = run_jobs_perseed(jobs2, gpus, args.wall_seconds, int(N_d), 1,
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


def run_jobs_perseed(jobs, gpus, wall_seconds, token_cap, roundtrip, outer_timeout_s):
    """Run (tag, cfg, seed) jobs across idle GPUs in PARALLEL, one per GPU, each
    with its OWN seed and an optional shared token_cap. Returns {tag: result}."""
    from orchestrate import WORKER, FREE_MEM_MIB
    queue = list(jobs); running = {}; results = {}
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
        while queue and free:
            tag, cfg, seed = queue.pop(0); gpu = free.pop(0)
            cfgp = os.path.join(RESULTS, f'cfg_{tag}.json')
            outp = os.path.join(RESULTS, f'out_{tag}.json')
            logp = os.path.join(RESULTS, f'log_{tag}.log')
            json.dump(cfg, open(cfgp, 'w'), indent=2)
            fh = open(logp, 'w')
            cmd = [sys.executable, WORKER, '--config_json', cfgp, '--out', outp,
                   '--wall_seconds', str(wall_seconds), '--seed', str(seed),
                   '--roundtrip', str(roundtrip)]
            if token_cap is not None:
                cmd += ['--token_cap', str(token_cap)]
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = dict(proc=proc, tag=tag, out=outp, fh=fh, t0=time.time())
            log(f'LAUNCH {tag} seed{seed} on GPU {gpu}'
                + (f' tokcap={token_cap}' if token_cap else ''))
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
            h = (results[job['tag']].get('heldout') or {})
            log(f'DONE {job["tag"]} bpb={h.get("heldout_bpb")} '
                f'tokens={results[job["tag"]].get("tokens")} '
                f'tok/s={results[job["tag"]].get("sustained_tok_s")}')
            del running[gpu]
    return results


if __name__ == '__main__':
    main()
