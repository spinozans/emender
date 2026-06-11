"""emender-cap-sweep EFFICIENCY — CMA-found Emender vs GDN-2, measured.

Three measured comparisons, all bf16 uniform, fused split-edit asserted:
  (1) iso-param TOKEN-matched held-out BPB  (sample efficiency at equal params+tokens)
  (2) iso-param WALL-matched  held-out BPB  (wall efficiency at equal params+wallclock)
  (3) throughput tok/s ratio                (delegated to throughput.py, 1.3B head shape)

Arms at a shared param target (dim derived per arm so params match):
  gdn2_pure   pure gdn2_recall                 = GDN-2 incumbent
  emender     CMA-found {gdn2_recall, e97_delta, e97_track}  (read cma_result.json)

Runs each arm twice on its leased GPU: once token-capped (max_tokens), once wall-capped
(wall_minutes, no token cap). Writes efficiency.json.

  eval "$(scripts/gpu_lease.sh 4)" && python experiments/emender_cap_sweep/efficiency.py
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
RC = THIS.parent / 'emender_real_cap'
WORKER = RC / 'lm_worker.py'
sys.path.insert(0, str(RC))
import emender_common as EC  # noqa: E402

VOCAB = 50281
DEPTH, N_HEADS, N_STATE = 8, 32, 32
TARGET_PARAMS = 40e6
BATCH, CHUNK = 8, 512
TOKEN_BUDGET = 8_000_000
WALL_MIN = 6.0
LR = 1.2e-3


def found_logits(cma_result):
    """Reconstruct CMA-found 9-type logits from the found-mixture counts."""
    cr = json.load(open(cma_result))
    fm = cr['found_mixture']
    counts = fm['counts']
    import math
    names = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin',
             'gdn2_nonlin_shell', 'e97_raw', 'e97_delta', 'refit']
    logits = [math.log(counts[t]) if counts.get(t, 0) > 0 else EC.LOG0 for t in names]
    return logits, fm


def cfg_for(name, logits, mode):
    dim, params = EC.derive_lm_dim_logits(logits, DEPTH, N_HEADS, N_STATE, VOCAB,
                                           TARGET_PARAMS, dim_multiple=64,
                                           dim_lo=256, dim_hi=1024)
    c = dict(name=name, head_type_logits=logits, dim=dim, depth=DEPTH,
             n_heads=N_HEADS, n_state=N_STATE, lr=LR, batch_size=BATCH,
             chunk=CHUNK, eval_chunk=CHUNK, seed=0)
    if mode == 'token':
        c['max_tokens'] = TOKEN_BUDGET
        c['wall_minutes'] = WALL_MIN * 3  # generous cap; token budget is the binding limit
    else:
        c['wall_minutes'] = WALL_MIN      # wall is the binding limit (no token cap)
    return c, dim, params


def run_worker(cfg, out_path, gpu, log_path):
    cfg_path = out_path.replace('.json', '.cfg.json')
    json.dump(cfg, open(cfg_path, 'w'), indent=2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(log_path, 'w')
    cmd = [sys.executable, str(WORKER), '--config_json', cfg_path, '--out', out_path,
           '--wall_minutes', str(cfg['wall_minutes']), '--bpb_batches', '40']
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=logf,
                            stderr=subprocess.STDOUT), logf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cma_result', default=str(THIS / 'results_cma' / 'cma_result.json'))
    ap.add_argument('--output_dir', default=str(THIS / 'results_efficiency'))
    ap.add_argument('--gpus', default=None, help='csv; default=leased CUDA_VISIBLE_DEVICES')
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gpus = (args.gpus.split(',') if args.gpus
            else [g for g in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if g])
    if not gpus:
        raise SystemExit('lease GPUs: eval "$(scripts/gpu_lease.sh 4)"')

    em_logits, fm = found_logits(args.cma_result)
    gd_logits = EC.delta_logits(0.0)  # pure gdn2_recall
    arms = {'gdn2_pure': gd_logits, 'emender': em_logits}

    jobs = []  # (name, mode, cfg, out_path, log_path)
    for mode in ('token', 'wall'):
        for name, logits in arms.items():
            cfg, dim, params = cfg_for(f'{name}_{mode}', logits, mode)
            op = str(out_dir / f'res_{name}_{mode}.json')
            lp = str(out_dir / f'log_{name}_{mode}.txt')
            jobs.append((f'{name}_{mode}', mode, cfg, op, lp, dim, params))

    print(f"CMA-found mixture: counts={fm['counts']} nonlin_frac={fm.get('nonlinear_fraction')}", flush=True)
    running, queue = {}, list(jobs)
    free = list(gpus)
    while queue or running:
        while queue and free:
            name, mode, cfg, op, lp, dim, params = queue.pop(0)
            if os.path.exists(op):
                print(f"[cached] {name}", flush=True); continue
            gpu = free.pop(0)
            proc, logf = run_worker(cfg, op, gpu, lp)
            running[name] = (proc, gpu, logf, op, time.time())
            print(f"[run gpu{gpu}] {name} dim={dim} params={params:,}", flush=True)
            time.sleep(3)
        time.sleep(5)
        for name in list(running):
            proc, gpu, logf, op, t0 = running[name]
            if proc.poll() is not None:
                logf.close(); free.append(gpu)
                ok = os.path.exists(op)
                print(f"[{'ok' if ok else 'FAIL'} {(time.time()-t0)/60:.1f}m] {name}", flush=True)
                del running[name]

    # collect
    summary = {'cma_found_mixture': fm, 'token_budget': TOKEN_BUDGET,
               'wall_minutes': WALL_MIN, 'target_params': TARGET_PARAMS, 'arms': {}}
    for name, _ in arms.items():
        summary['arms'][name] = {}
        for mode in ('token', 'wall'):
            p = out_dir / f'res_{name}_{mode}.json'
            if p.exists():
                r = json.load(open(p))
                summary['arms'][name][mode] = dict(
                    bpb=(r.get('heldout') or {}).get('heldout_bpb'),
                    tok_s=r.get('tok_s'), tokens=r.get('tokens'),
                    wall_min=r.get('wall_minutes'), params_m=r.get('params_m'),
                    counts=r.get('counts'), fused=r.get('fused_asserted'),
                    nan=r.get('nan_seen'))
    # deltas
    try:
        for mode in ('token', 'wall'):
            be = summary['arms']['emender'][mode]['bpb']
            bg = summary['arms']['gdn2_pure'][mode]['bpb']
            summary.setdefault('deltas', {})[f'{mode}_bpb_emender_minus_gdn2'] = round(be - bg, 5)
        te = summary['arms']['emender']['token']['tok_s']
        tg = summary['arms']['gdn2_pure']['token']['tok_s']
        summary['deltas']['throughput_ratio_emender_over_gdn2'] = round(te / tg, 4)
    except (KeyError, TypeError):
        pass
    json.dump(summary, open(out_dir / 'efficiency.json', 'w'), indent=2, default=str)
    print('\n=== EFFICIENCY ===', flush=True)
    print(json.dumps(summary.get('deltas', {}), indent=2), flush=True)
    print(f"wrote {out_dir / 'efficiency.json'}", flush=True)


if __name__ == '__main__':
    main()
