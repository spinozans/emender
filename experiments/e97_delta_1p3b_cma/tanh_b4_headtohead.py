"""tanh-e97-1p3b: B=4 head-to-head — does the batch-parallel launch fix flip the
wall-clock verdict?

The launch sweep showed the tanh fused-checkpointed kernel improves from 0.784x
(B=2) to 0.843x (B=4) gdn2-mlp throughput. This runs the REAL 1.3B head-to-head
at micro-batch B=4 (both arms), 720s wall, 2 seeds, real Pile, with checkpoint
round-trip, then gdn2-mlp token-capped at the tanh token budget:

  W1 tanh-e97_delta+gdn-neg (LINEAR_STATE=False, sequential) @ B=4, T_wall
  W2 gdn2-mlp                                               @ B=4, T_wall  (wall-matched)
  W3 gdn2-mlp token-capped at tanh tokens                   @ B=4          (token-matched)

Verdict: WALL-CLOCK BPB_tanh vs BPB_gdn_wall ; TOKEN BPB_tanh vs BPB_gdn_tok.
REAL DATA. No mocks. Idle-GPU-only.
"""
import os, sys, json, time, datetime
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _THIS)
RESULTS = os.path.join(_THIS, 'results')
from final_headtohead import run_jobs_perseed


def log(m): print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)

if __name__ == '__main__':
    gpus = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0,1,2,3,4,5,6,7').split(',')]
    wall = float(sys.argv[2]) if len(sys.argv) > 2 else 720.0
    hh = json.load(open(os.path.join(RESULTS, 'headtohead_results.json')))
    delta = dict(hh['delta_cfg']); base = dict(hh['base_cfg'])
    # Force B=4 + explicit tanh (LINEAR_STATE=False), sequential kernel.
    delta['batch_size'] = 4; delta['e97_state_nonlin'] = 'tanh'; delta['use_chunked_e97_delta'] = False
    base['batch_size'] = 4
    for c in (delta, base):
        c.pop('params_b', None); c.pop('counts', None); c.pop('within_tol', None)
    seeds = [0, 1]

    t0 = time.time()
    jobs = []
    for s in seeds:
        jobs.append((f'W1_tanh_b4_s{s}', delta, s))
        jobs.append((f'W2_gdn_b4_s{s}', base, s))
    res = run_jobs_perseed(jobs, gpus, wall, None, 1, 1400.0)

    tanh_tokens = [res[f'W1_tanh_b4_s{s}'].get('tokens', 0) for s in seeds
                   if isinstance(res.get(f'W1_tanh_b4_s{s}'), dict)]
    N_t = min([t for t in tanh_tokens if t]) if any(tanh_tokens) else None
    log(f'tanh tokens per seed = {tanh_tokens}; token-matched budget = {N_t}')
    if N_t:
        jobs2 = [(f'W3_gdn_b4_tokcap_s{s}', base, s) for s in seeds]
        res.update(run_jobs_perseed(jobs2, gpus, wall, int(N_t), 1, 1400.0))

    out = dict(task='tanh-e97-1p3b-b4-headtohead', delta_cfg=delta, base_cfg=base,
               seeds=seeds, wall_seconds=wall, token_matched_budget=N_t, results=res,
               wallclock_minutes=round((time.time() - t0) / 60, 1),
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(os.path.join(RESULTS, 'tanh_b4_headtohead.json'), 'w'), indent=2)

    def bpb(t):
        r = res.get(t, {}); return (r.get('heldout') or {}).get('heldout_bpb') if isinstance(r, dict) else None
    def tk(t):
        r = res.get(t, {}); return r.get('tokens') if isinstance(r, dict) else None
    def ts(t):
        r = res.get(t, {}); return r.get('sustained_tok_s') if isinstance(r, dict) else None
    log('=== SUMMARY (B=4, held-out BPB) ===')
    for s in seeds:
        log(f' seed{s}: tanh={bpb(f"W1_tanh_b4_s{s}")} (tok={tk(f"W1_tanh_b4_s{s}")} '
            f'tok/s={ts(f"W1_tanh_b4_s{s}")}) | gdn_wall={bpb(f"W2_gdn_b4_s{s}")} '
            f'(tok={tk(f"W2_gdn_b4_s{s}")} tok/s={ts(f"W2_gdn_b4_s{s}")}) | '
            f'gdn_tok={bpb(f"W3_gdn_b4_tokcap_s{s}")}')
    log('WROTE results/tanh_b4_headtohead.json')
