"""e97-wallclock-cma: BPB-vs-(C, ratio) landscape on REAL Pile at 1.3B.

Two fitness modes (selected by --mode):
  token : TOKEN-MATCHED capability curve. All shell heads use fused=True (the
          exact single-launch boundary-phi model: phi(S)=tanh every C steps,
          linear within). Equal token_cap for every config. Answers: does the
          bounded-state BPB edge SURVIVE as C grows, or collapse toward linear?
  wall  : WALL-CLOCK fitness. Each config trains for equal wall_seconds with the
          kernel it would use in production at that operating point (identity ->
          1 FLA call; intermediate C -> fused=False matmul-within-chunk; small C
          -> fused=True sequential). Answers the decision question directly.

Param-matched dim=2240 (64 heads). gdn-neg baseline = 64 gdn2_recall + MLP.
Shell mix = 32 gdn-neg + 32 gdn2_nonlin_shell. Idle-GPU pool, one job per GPU.
REAL DATA. No mocks.
"""
import os, sys, json, time, math, subprocess, argparse, datetime
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.join(_THIS, '..', 'e97_delta_1p3b_cma'))
from shapes import derive_dim, fracs_to_logits8
RESULTS = os.path.join(_THIS, 'results')
WORKER = os.path.join(_THIS, 'wc_screen.py')
FREE_MEM_MIB = 2000
DIM = 2240


def log(m): print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def gpu_used_mib():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                   '--format=csv,noheader,nounits'], timeout=15).decode()
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


def base_cfg(ratio_shell, C, nonlin, fused):
    """ratio_shell in [0,1]; 0 -> pure gdn-neg baseline."""
    if ratio_shell <= 0:
        logits = fracs_to_logits8({'gdn2_recall': 1.0})
    else:
        logits = fracs_to_logits8({'gdn2_recall': 1.0 - ratio_shell,
                                   'gdn2_nonlin_shell': ratio_shell})
    return dict(dim=DIM, head_type_logits=logits, lr=1.0e-3, knob_lr_mult=1.0,
                batch_size=2, bf16=True, lam_max=1.585, beta_max=2.747,
                shell_state_nonlin=nonlin, shell_state_chunk=int(C), shell_fused=bool(fused))


def run_pool(jobs, gpus, wall_seconds, token_cap, outer_timeout_s):
    queue = list(jobs); running = {}; results = {}
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
        while queue and free:
            tag, cfg = queue.pop(0); gpu = free.pop(0)
            cfgp = os.path.join(RESULTS, f'cfg_{tag}.json'); outp = os.path.join(RESULTS, f'out_{tag}.json')
            logp = os.path.join(RESULTS, f'log_{tag}.log'); json.dump(cfg, open(cfgp, 'w'), indent=2)
            fh = open(logp, 'w')
            cmd = [sys.executable, WORKER, '--config_json', cfgp, '--out', outp,
                   '--wall_seconds', str(wall_seconds), '--seed', '0', '--roundtrip', '0']
            if token_cap is not None:
                cmd += ['--token_cap', str(token_cap)]
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = dict(proc=proc, tag=tag, out=outp, fh=fh, t0=time.time())
            log(f'LAUNCH {tag} on GPU {gpu}  C={cfg["shell_state_chunk"]} nl={cfg["shell_state_nonlin"]} fused={cfg["shell_fused"]}')
        time.sleep(8)
        for gpu, job in list(running.items()):
            ret = job['proc'].poll(); to = (time.time() - job['t0']) > outer_timeout_s
            if ret is None and not to:
                continue
            if ret is None:
                job['proc'].kill(); log(f'TIMEOUT {job["tag"]}')
            job['fh'].close()
            results[job['tag']] = (json.load(open(job['out'])) if os.path.exists(job['out']) else dict(error='no_output'))
            r = results[job['tag']]; h = (r.get('heldout') or {}) if isinstance(r, dict) else {}
            log(f'DONE {job["tag"]} bpb={h.get("heldout_bpb")} tok={r.get("tokens")} tok/s={r.get("sustained_tok_s")} stop={r.get("stop_reason")} err={r.get("error")}')
            del running[gpu]
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['token', 'wall'], required=True)
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--wall_seconds', type=float, default=720.0)
    ap.add_argument('--token_cap', type=int, default=5_000_000)
    ap.add_argument('--ratio', type=float, default=0.5)
    ap.add_argument('--outer_timeout_s', type=float, default=2400.0)
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    jobs = []
    # gdn-neg baseline (always, both modes)
    jobs.append(('base_gdn', base_cfg(0.0, 64, 'identity', True)))
    Cs = [1, 8, 16, 32, 64, 128, 256, 2048]
    if args.mode == 'token':
        # exact boundary-phi model, fused=True, token-matched. C=2048 ~ pure linear.
        for C in Cs:
            jobs.append((f'tok_C{C}', base_cfg(args.ratio, C, 'tanh', True)))
        # pure-linear shell control (identity, 1 FLA call) -- capability floor
        jobs.append(('tok_shell_identity', base_cfg(args.ratio, 64, 'identity', True)))
        res = run_pool(jobs, gpus, args.wall_seconds, args.token_cap, args.outer_timeout_s)
    else:
        # wall-clock: each C uses its FASTEST realization. throughput_sweep.json
        # proved the single-launch fused SEQUENTIAL boundary-phi kernel (0.88x mix,
        # ~const in C) DOMINATES the FLA chunk-reference matmul-loop at every C
        # (chunkref 0.07x@C16 .. 0.95x@C512 -- launch-bound). So fused=True is the
        # production kernel for every bounded C. C=2048 ~ pure-linear endpoint.
        for C in [1, 8, 16, 32, 64, 128, 256, 2048]:
            jobs.append((f'wall_C{C}', base_cfg(args.ratio, C, 'tanh', True)))
        # documented control: the matmul-within chunk-ref path at its best C (512)
        jobs.append(('wall_C512_chunkref', base_cfg(args.ratio, 512, 'tanh', False)))
        res = run_pool(jobs, gpus, args.wall_seconds, None, args.outer_timeout_s)

    out = dict(task='e97-wallclock-cma', mode=args.mode, dim=DIM, ratio_shell=args.ratio,
               wall_seconds=args.wall_seconds, token_cap=(args.token_cap if args.mode == 'token' else None),
               results=res, wallclock_minutes=round((time.time() - t0) / 60, 1),
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    outp = os.path.join(RESULTS, f'bpb_sweep_{args.mode}.json')
    json.dump(out, open(outp, 'w'), indent=2)

    def bpb(tag):
        r = res.get(tag, {}); return (r.get('heldout') or {}).get('heldout_bpb') if isinstance(r, dict) else None
    def tok(tag):
        r = res.get(tag, {}); return r.get('tokens') if isinstance(r, dict) else None
    def ts(tag):
        r = res.get(tag, {}); return r.get('sustained_tok_s') if isinstance(r, dict) else None
    log(f'=== {args.mode.upper()} SWEEP (held-out BPB) ===')
    for tag in res:
        log(f'  {tag:24s} bpb={bpb(tag)} tok={tok(tag)} tok/s={ts(tag)}')
    log(f'WROTE {outp}')


if __name__ == '__main__':
    main()
