"""emender-real-cap metric 2: convergent-loss TIE, token-matched AND wall-matched.

Compares pure GDN-2 (f=0) vs the CMA-FOUND Emender mixture on held-out BPB
(real-Pile slice), bf16 UNIFORM, fused asserted -- now FAIR (matched precision, the
fix vs the opt-1p3b fp32 strawman):
  * token-matched: both arms train on the SAME token budget (--max_tokens).
  * wall-matched : both arms train for the SAME wall minutes (--wall_minutes); the
    slower (emendment) arm does fewer tokens -> the honest wall-fair comparison.

Param/FLOP-locked: dim derived per mixture to a shared target. Idle-GPU no-preempt.

  python lm_tie.py --found_logits '<json 9-vector>' --seeds 0 1 [--gpus ...]
"""
import os, sys, json, time, subprocess, argparse, datetime
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _THIS); sys.path.insert(0, _ROOT)
import numpy as np
import emender_common as EC

FREE_MEM_MIB = 2500
WORKER = os.path.join(_THIS, 'lm_worker.py')
VOCAB = 50281
DEPTH, N_HEADS, N_STATE = 12, 32, 32   # a touch deeper than the CMA proxy for the tie
TARGET_PARAMS = 60e6
BATCH_SIZE, CHUNK = 8, 512
TOKEN_BUDGET = 12_000_000
WALL_MIN = 8.0
BPB_BATCHES = 40
LR = 5.0e-4


def gpu_used_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                          '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


def cfg_for(name, logits, mode, seed):
    dim, params = EC.derive_lm_dim_logits(logits, DEPTH, N_HEADS, N_STATE, VOCAB,
                                          TARGET_PARAMS, dim_multiple=64, dim_lo=256, dim_hi=1280)
    c = dict(name=name, head_type_logits=logits, dim=dim, depth=DEPTH, n_heads=N_HEADS,
             n_state=N_STATE, lr=LR, batch_size=BATCH_SIZE, chunk=CHUNK, eval_chunk=CHUNK,
             seed=seed, model_params=params)
    if mode == 'token':
        c['max_tokens'] = TOKEN_BUDGET; c['wall_minutes'] = 30.0
    else:
        c['wall_minutes'] = WALL_MIN
    return c


def run_batch(jobs, gpus, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results, pending = {}, []
    for name, cfg in jobs:
        op = os.path.join(out_dir, f'res_{name}.json')
        if os.path.exists(op):
            results[name] = json.load(open(op)); print(f"[cached] {name}", flush=True); continue
        pending.append((name, cfg))
    running = {}
    while pending or running:
        for gpu in list(running):
            name, proc, op, t0 = running[gpu]
            if proc.poll() is not None:
                del running[gpu]
                if os.path.exists(op):
                    results[name] = json.load(open(op)); r = results[name]
                    print(f"[done] gpu{gpu} {name} bpb={r.get('heldout',{}).get('heldout_bpb')} "
                          f"tok={r.get('tokens')} tok/s={r.get('tok_s')} ({(time.time()-t0)/60:.1f}m)", flush=True)
                else:
                    results[name] = dict(name=name, error='no_output', rc=proc.returncode)
                    print(f"[FAIL] gpu{gpu} {name} rc={proc.returncode}", flush=True)
        if pending:
            used = gpu_used_mib()
            for gpu in gpus:
                if not pending: break
                if gpu in running or used.get(gpu, 10**9) >= FREE_MEM_MIB: continue
                name, cfg = pending.pop(0)
                cp = os.path.join(out_dir, f'cfg_{name}.json'); op = os.path.join(out_dir, f'res_{name}.json')
                json.dump(cfg, open(cp, 'w'), indent=2)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(os.path.join(out_dir, f'log_{name}.txt'), 'w')
                cmd = [sys.executable, WORKER, '--config_json', cp, '--out', op,
                       '--wall_minutes', str(cfg.get('wall_minutes', WALL_MIN)),
                       '--bpb_batches', str(BPB_BATCHES)]
                proc = subprocess.Popen(cmd, cwd=_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
                running[gpu] = (name, proc, op, time.time())
                print(f"[run ] gpu{gpu} {name} dim={cfg['dim']} params={cfg['model_params']:,}", flush=True)
                time.sleep(4)
        time.sleep(8)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--found_logits', required=True, help='JSON 9-vector head_type_logits of CMA-found mixture')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
    ap.add_argument('--output', default=os.path.join(_THIS, 'results_lm_tie'))
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    found = json.loads(args.found_logits)
    arms = {'gdn2': EC.delta_logits(0.0), 'emender_found': found}
    jobs = []
    for seed in args.seeds:
        for mode in ('token', 'wall'):
            for arm, logits in arms.items():
                nm = f'{arm}_{mode}_s{seed}'
                jobs.append((nm, cfg_for(nm, logits, mode, seed)))
    res = run_batch(jobs, gpus, args.output)

    # aggregate
    agg = {}
    for (nm, cfg) in jobs:
        r = res.get(nm, {})
        ho = (r or {}).get('heldout') or {}
        parts = nm.rsplit('_', 1)[0]  # arm_mode
        agg.setdefault(parts, []).append(dict(seed=cfg['seed'],
            bpb=ho.get('heldout_bpb'), tokens=r.get('tokens'), tok_s=r.get('tok_s'),
            steps=r.get('steps'), nan=r.get('nan_seen'),
            fused=r.get('fused_asserted'), params=r.get('model_params')))
    summary = {}
    for k, v in agg.items():
        bpbs = [x['bpb'] for x in v if x['bpb'] is not None]
        toks = [x['tokens'] for x in v if x['tokens'] is not None]
        summary[k] = dict(bpb_mean=(round(float(np.mean(bpbs)), 5) if bpbs else None),
                          bpb_sd=(round(float(np.std(bpbs)), 5) if len(bpbs) > 1 else 0.0),
                          tokens_mean=(int(np.mean(toks)) if toks else None),
                          tok_s_mean=(round(float(np.mean([x['tok_s'] for x in v if x['tok_s']])), 1)
                                      if any(x['tok_s'] for x in v) else None),
                          n=len(bpbs), runs=v)
    out = dict(found_logits=found, found_counts=EC.head_counts(found, N_HEADS),
               shape=dict(depth=DEPTH, n_heads=N_HEADS, n_state=N_STATE, target_params=TARGET_PARAMS),
               token_budget=TOKEN_BUDGET, wall_minutes=WALL_MIN, dtype='bf16 uniform (fused)',
               summary=summary, timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    json.dump(out, open(os.path.join(args.output, 'tie_result.json'), 'w'), indent=2, default=str)
    print("\n=== CONVERGENT-LOSS TIE ===", flush=True)
    for mode in ('token', 'wall'):
        g = summary.get(f'gdn2_{mode}', {}); e = summary.get(f'emender_found_{mode}', {})
        if g.get('bpb_mean') is not None and e.get('bpb_mean') is not None:
            d = e['bpb_mean'] - g['bpb_mean']
            print(f"  {mode}-matched: gdn2={g['bpb_mean']:.5f} emender={e['bpb_mean']:.5f} "
                  f"Δ={d:+.5f} (gdn2 tok={g.get('tokens_mean')} emender tok={e.get('tokens_mean')})",
                  flush=True)
    print(f"-> {args.output}/tie_result.json", flush=True)


if __name__ == '__main__':
    main()
