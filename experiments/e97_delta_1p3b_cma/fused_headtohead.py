"""fuse-2kernel: DECISIVE head-to-head — e97_delta+gdn-neg with the e97_delta
heads ACTUALLY ON THE CHUNKED KERNEL (linear state) vs gdn2-mlp, at 1.3B,
TOKEN-MATCHED *and* WALL-CLOCK-MATCHED on REAL Pile.

Why this re-run exists: the prior e97delta-1p3b head-to-head built the candidate
with the DEFAULT e97_state_nonlin='tanh', which makes the e97_delta heads
NONLINEAR-state. The chunked-parallel fused Triton kernel only engages for
LINEAR-state e97_delta (the chunked guard requires linear_state=True), so the
prior run silently routed every e97_delta head to the SEQUENTIAL T-scan
(instrumented: 18/18 layers sequential, 0 chunked). That sequential kernel is the
real source of the ~23-40% wall-clock gap, NOT the within-layer two-kernel split.

Here the candidate carries e97_state_nonlin='identity' so the 43 e97_delta heads
run the chunked fused kernel (instrumented: 18/18 layers chunked). The gdn-neg
heads stay on FLA. Everything else (dim, params, lr, knobs, seeds, wall budget,
roundtrip) is identical to the prior decisive run so the ONLY change is the
kernel the e97_delta heads execute on.

REAL DATA. No mocks. Idle-GPU-only. Same worker (screen.py) as the prior batch.
"""
import os, sys, json, time, datetime, argparse

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
RESULTS = os.path.join(_THIS, 'results')

from final_headtohead import run_jobs_perseed  # reuse the per-seed GPU pool


def log(m):
    print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


# Prior decisive configs (headtohead_results.json), candidate gets linear state.
DELTA_CFG = {
    "dim": 2112,
    "head_type_logits": [-1.2426827908909686, -28.456765498541962, -29.97388828036681,
                         -29.289780094203703, -28.52456483818494, -30.0, -30.0,
                         -0.5059263781557628],
    "lr": 0.001, "knob_lr_mult": 1.1470526133942434, "batch_size": 2, "bf16": True,
    "lam_max": 2.42267247635037, "beta_max": 2.4506232209686507,
    "e97_state_nonlin": "identity",   # <-- engages the chunked kernel (the fix)
    "params_b": 1.2532,
    "counts": {"gdn2_recall": 21, "e97_delta": 43},
}
BASE_CFG = {
    "dim": 2240,
    "head_type_logits": [0.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0],
    "lr": 0.001, "knob_lr_mult": 4.0, "batch_size": 2, "bf16": True,
    "lam_max": 1.585, "beta_max": 2.747, "params_b": 1.2589,
    "counts": {"gdn2_recall": 64},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--wall_seconds', type=float, default=720.0)
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--outer_timeout_s', type=float, default=1200.0)
    ap.add_argument('--output', default=os.path.join(RESULTS, 'fused_headtohead_results.json'))
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    log(f'CANDIDATE (chunked-linear) counts={DELTA_CFG["counts"]} dim={DELTA_CFG["dim"]} '
        f'state=identity')
    log(f'BASELINE  A_gdn2_mlp counts={BASE_CFG["counts"]} dim={BASE_CFG["dim"]}')

    t0 = time.time()
    jobs = []
    for s in seeds:
        jobs.append((f'W1_deltaC_s{s}', DELTA_CFG, s))
        jobs.append((f'W2_gdn_s{s}', BASE_CFG, s))
    res = run_jobs_perseed(jobs, gpus, args.wall_seconds, None, 1, args.outer_timeout_s)

    delta_tokens = [res[f'W1_deltaC_s{s}'].get('tokens', 0) for s in seeds
                    if isinstance(res.get(f'W1_deltaC_s{s}'), dict)]
    N_d = min([t for t in delta_tokens if t]) if any(delta_tokens) else None
    log(f'candidate tokens per seed = {delta_tokens}; token-matched budget N_d = {N_d}')

    if N_d:
        jobs2 = [(f'W3_gdn_tokcap_s{s}', BASE_CFG, s) for s in seeds]
        r = run_jobs_perseed(jobs2, gpus, args.wall_seconds, int(N_d), 1, args.outer_timeout_s)
        res.update(r)

    out = dict(task='fuse-2kernel-headtohead', delta_cfg=DELTA_CFG, base_cfg=BASE_CFG,
               seeds=seeds, wall_seconds=args.wall_seconds, token_matched_budget=N_d,
               results=res, wallclock_minutes=round((time.time() - t0) / 60, 1),
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(args.output, 'w'), indent=2)

    def bpb(tag):
        r = res.get(tag, {})
        return (r.get('heldout') or {}).get('heldout_bpb') if isinstance(r, dict) else None

    def toks(tag):
        r = res.get(tag, {})
        return r.get('tokens') if isinstance(r, dict) else None

    def toks_s(tag):
        r = res.get(tag, {})
        return r.get('sustained_tok_s') if isinstance(r, dict) else None

    log('=== SUMMARY (held-out BPB / tokens / tok-s) ===')
    for s in seeds:
        log(f' seed{s}: W1_deltaC bpb={bpb(f"W1_deltaC_s{s}")} tok={toks(f"W1_deltaC_s{s}")} '
            f'tok/s={toks_s(f"W1_deltaC_s{s}")} | W2_gdn_wall bpb={bpb(f"W2_gdn_s{s}")} '
            f'tok={toks(f"W2_gdn_s{s}")} tok/s={toks_s(f"W2_gdn_s{s}")} | '
            f'W3_gdn_tok bpb={bpb(f"W3_gdn_tokcap_s{s}")} tok={toks(f"W3_gdn_tokcap_s{s}")}')
    log(f'WROTE {args.output}')


if __name__ == '__main__':
    main()
