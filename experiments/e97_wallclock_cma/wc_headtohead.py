"""e97-wallclock-cma: longer-budget head-to-head, winner-C shell-tanh vs gdn-mlp.

Takes the wall-clock-best chunk-size C (fused boundary-phi, the kernel that
dominates per throughput_sweep.json) and runs, at a LONGER wall budget with >=2
seeds on REAL Pile:
  W1  shell-tanh @C (fused) @ T_wall                       -> tokens N_s, BPB_s
  W2  gdn-mlp baseline      @ T_wall (WALL-CLOCK matched)  -> BPB_g_wall
  W3  gdn-mlp baseline      capped at N_s (TOKEN matched)  -> BPB_g_tok

Verdict (lower BPB better):  WALL: BPB_s vs BPB_g_wall ;  TOKEN: BPB_s vs BPB_g_tok.
Compared against the 5M-token sweep edge to see if the sample-efficiency edge
GROWS with more tokens (could flip wall-clock at longer training). No mocks.
"""
import os, sys, json, time, argparse, datetime
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _THIS)
from bpb_sweep import base_cfg, run_pool, gpu_used_mib, RESULTS


def log(m): print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def run_seeded(jobs, gpus, wall_seconds, token_cap, outer_timeout_s):
    """jobs = list of (tag, cfg, seed). Pool one-per-GPU with per-job seed."""
    import subprocess
    WORKER = os.path.join(_THIS, 'wc_screen.py')
    queue = list(jobs); running = {}; results = {}
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < 2000 and g not in running]
        while queue and free:
            tag, cfg, seed = queue.pop(0); gpu = free.pop(0)
            cfgp = os.path.join(RESULTS, f'cfg_{tag}.json'); outp = os.path.join(RESULTS, f'out_{tag}.json')
            logp = os.path.join(RESULTS, f'log_{tag}.log'); json.dump(cfg, open(cfgp, 'w'), indent=2)
            fh = open(logp, 'w')
            cmd = [sys.executable, WORKER, '--config_json', cfgp, '--out', outp,
                   '--wall_seconds', str(wall_seconds), '--seed', str(seed), '--roundtrip', '0']
            if token_cap is not None:
                cmd += ['--token_cap', str(token_cap)]
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = dict(proc=proc, tag=tag, out=outp, fh=fh, t0=time.time())
            log(f'LAUNCH {tag} seed{seed} GPU{gpu}')
        time.sleep(8)
        for gpu, job in list(running.items()):
            ret = job['proc'].poll(); to = (time.time() - job['t0']) > outer_timeout_s
            if ret is None and not to:
                continue
            if ret is None:
                job['proc'].kill()
            job['fh'].close()
            results[job['tag']] = (json.load(open(job['out'])) if os.path.exists(job['out']) else dict(error='no_output'))
            r = results[job['tag']]; h = (r.get('heldout') or {}) if isinstance(r, dict) else {}
            log(f'DONE {job["tag"]} bpb={h.get("heldout_bpb")} tok={r.get("tokens")} tok/s={r.get("sustained_tok_s")}')
            del running[gpu]
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--C', type=int, required=True)
    ap.add_argument('--ratio', type=float, default=0.5)
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--wall_seconds', type=float, default=1500.0)
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--outer_timeout_s', type=float, default=2400.0)
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    shell = base_cfg(args.ratio, args.C, 'tanh', True)
    gdn = base_cfg(0.0, 64, 'identity', True)
    t0 = time.time()
    jobs = []
    for s in seeds:
        jobs.append((f'HT_shell_C{args.C}_s{s}', shell, s))
        jobs.append((f'HT_gdn_s{s}', gdn, s))
    res = run_seeded(jobs, gpus, args.wall_seconds, None, args.outer_timeout_s)
    shell_tokens = [res[f'HT_shell_C{args.C}_s{s}'].get('tokens', 0) for s in seeds
                    if isinstance(res.get(f'HT_shell_C{args.C}_s{s}'), dict)]
    N_s = min([t for t in shell_tokens if t]) if any(shell_tokens) else None
    log(f'shell tokens/seed={shell_tokens}; token-matched N_s={N_s}')
    if N_s:
        jobs2 = [(f'HT_gdn_tok_s{s}', gdn, s) for s in seeds]
        res.update(run_seeded(jobs2, gpus, args.wall_seconds, int(N_s), args.outer_timeout_s))
    out = dict(task='e97-wallclock-cma', phase='longer-budget-headtohead', C=args.C,
               ratio_shell=args.ratio, wall_seconds=args.wall_seconds, seeds=seeds,
               token_matched_budget=N_s, results=res,
               wallclock_minutes=round((time.time() - t0) / 60, 1),
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(os.path.join(RESULTS, f'headtohead_C{args.C}.json'), 'w'), indent=2)

    def bpb(tag):
        r = res.get(tag, {}); return (r.get('heldout') or {}).get('heldout_bpb') if isinstance(r, dict) else None
    log('=== HEAD-TO-HEAD SUMMARY (held-out BPB) ===')
    for s in seeds:
        log(f' seed{s}: shell_C{args.C}={bpb(f"HT_shell_C{args.C}_s{s}")} '
            f'gdn_wall={bpb(f"HT_gdn_s{s}")} gdn_tok={bpb(f"HT_gdn_tok_s{s}")}')
    log(f'WROTE headtohead_C{args.C}.json')


if __name__ == '__main__':
    main()
