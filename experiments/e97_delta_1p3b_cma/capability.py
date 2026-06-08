"""e97delta-1p3b: capability spot-check — recall / track / count.

Compares the within-layer head MIXTURE of the two arms on the canonical
length-extrapolation probes (the same train_hybrid harness the prior typed-gdn2
batches used), so the BPB verdict is read alongside what each cell can actually
*do*:
    recall = mqar_recall    track = s5_permutation    count = anbncn_viability

Arms (head_type_logits over the canonical 8 types; gdn-neg == gdn2_recall with
gdn_allow_neg_eigval=1):
    gdn2_mlp        : 100% gdn-neg     (the baseline cell)
    e97delta_mix    : the CMA-winner e97_delta + gdn-neg mixture

These probes are architectural capability tests (small dim, depth 4) — they test
whether the cell CAN learn the task and length-extrapolate, independent of the LM
param budget. REAL tasks, real training. No mocks.
"""
import os, sys, json, time, subprocess, argparse, datetime
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _THIS); sys.path.insert(0, _ROOT)
from shapes import fracs_to_logits8, TYPE_NAMES, LOG0

TRAIN_HYBRID = os.path.join(_ROOT, 'experiments', 'expressivity_tasks', 'train_hybrid.py')
RESULTS = os.path.join(_THIS, 'results')
PROBES = {'recall': 'mqar_recall', 'track': 's5_permutation', 'count': 'anbncn_viability'}
PROBE_EXTRA = {'mqar_recall': [], 's5_permutation': [], 'anbncn_viability': []}
EVAL_TS = [128, 256, 512, 1024]
DEPTH, N_HEADS, N_STATE = 4, 64, 32
FREE_MEM_MIB = 2000


def log(m):
    print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def gpu_used_mib():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                   '--format=csv,noheader,nounits'], timeout=15).decode()
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


def count_params(dim, logits, vocab):
    from ndm.models.hybrid_ladder import HybridLadderLM
    lk = {'head_type_logits': list(logits), 'lam_max': 1.585, 'beta_max': 2.747,
          'gdn_allow_neg_eigval': True}
    m = HybridLadderLM(vocab_size=vocab, dim=dim, depth=DEPTH,
                       layer_pattern=['typed-gdn2'], layer_kwargs=[lk],
                       n_state=N_STATE, n_heads=N_HEADS, expansion=1.0)
    n = sum(p.numel() for p in m.parameters()); del m
    return n


def derive_small_dim(logits, vocab=512, target=4_000_000):
    lo, hi, best = 16, 1024, None
    while lo <= hi:
        mid = max(8, ((lo + hi) // 2 // 8) * 8)
        p = count_params(mid, logits, vocab)
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (mid, p)
        if p < target:
            lo = mid + 8
        else:
            hi = mid - 8
    return best[0]


def build_cmd(name, logits, probe, seed, steps, dim, out_dir):
    label = f'cap_{name}__{probe}__seed{seed}'
    return [sys.executable, TRAIN_HYBRID, '--task', probe, '--layer_pattern', 'typed-gdn2',
            *PROBE_EXTRA[probe], '--dim', str(dim), '--depth', str(DEPTH),
            '--n_heads', str(N_HEADS), '--n_state', str(N_STATE), '--lr', '3e-4',
            '--lam_max', '1.585', '--beta_max', '2.747', '--gdn_allow_neg_eigval', '1',
            '--head_type_logits=' + ','.join(str(f) for f in logits),
            '--steps', str(steps), '--seq_len', '128', '--batch_size', '32',
            '--optimizer', 'schedulefree', '--seed', str(seed),
            '--label', label, '--output_dir', str(out_dir),
            '--eval_lengths', *[str(t) for t in EVAL_TS], '--eval_lengths_n_batches', '8'], label


def probe_score(out_dir, label):
    p = os.path.join(out_dir, f'{label}.json')
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    le = d.get('length_extrap')
    if not le:
        return None
    accs = [float(le[str(t)]['acc']) for t in EVAL_TS
            if isinstance(le.get(str(t)), dict) and 'acc' in le[str(t)]]
    return dict(mean_acc=float(np.mean(accs)) if accs else None,
                per_len={str(t): le.get(str(t), {}).get('acc') for t in EVAL_TS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cma', default=os.path.join(RESULTS, 'cma_all_results.json'))
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--output', default=os.path.join(RESULTS, 'capability_results.json'))
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    out_dir = os.path.join(RESULTS, 'cap_runs'); os.makedirs(out_dir, exist_ok=True)

    # arm logits (8-vector). gdn2_mlp = 100% gdn-neg; e97delta_mix = CMA winner.
    arms = {'gdn2_mlp': fracs_to_logits8({'gdn2_recall': 1.0})}
    if os.path.exists(args.cma):
        from final_headtohead import pick_best_searched
        cma = json.load(open(args.cma))
        best = pick_best_searched(cma)
        if best is not None:
            arms['e97delta_mix'] = best[2]['head_type_logits']
            log(f'e97delta_mix from CMA winner {best[1]} '
                f'counts={ {k:v for k,v in best[2]["counts"].items() if v} }')
    if 'e97delta_mix' not in arms:  # fallback: 50/50 seed
        arms['e97delta_mix'] = fracs_to_logits8({'gdn2_recall': 0.5, 'e97_delta': 0.5})

    dims = {name: derive_small_dim(lg) for name, lg in arms.items()}
    log(f'small dims (≈4M param matched): {dims}')

    jobs = []
    for name, lg in arms.items():
        for key, probe in PROBES.items():
            for s in seeds:
                jobs.append((name, key, probe, s, lg))
    log(f'{len(jobs)} capability jobs ({len(arms)} arms x {len(PROBES)} probes x {len(seeds)} seeds)')

    queue = list(jobs); running = {}; raw = {}
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
        while queue and free:
            name, key, probe, s, lg = queue.pop(0); gpu = free.pop(0)
            cmd, label = build_cmd(name, lg, probe, s, args.steps, dims[name], out_dir)
            lf = open(os.path.join(out_dir, label + '.log'), 'w')
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = (proc, name, key, probe, s, label, lf, time.time())
            log(f'LAUNCH {label} on GPU {gpu}')
        time.sleep(8)
        for gpu, job in list(running.items()):
            proc, name, key, probe, s, label, lf, t0 = job
            if proc.poll() is None and (time.time() - t0) < 1200:
                continue
            if proc.poll() is None:
                proc.kill()
            lf.close()
            sc = probe_score(out_dir, label)
            raw.setdefault(name, {}).setdefault(key, {})[s] = sc
            log(f'DONE {label} mean_acc={sc.get("mean_acc") if sc else None}')
            del running[gpu]

    # aggregate: mean over seeds per arm/probe
    summary = {}
    for name in arms:
        summary[name] = {}
        for key in PROBES:
            vals = [raw[name][key][s]['mean_acc'] for s in seeds
                    if raw.get(name, {}).get(key, {}).get(s) and raw[name][key][s].get('mean_acc') is not None]
            summary[name][key] = round(float(np.mean(vals)), 4) if vals else None
    out = dict(task='e97delta-1p3b-capability', arms=list(arms.keys()),
               probes=PROBES, depth=DEPTH, n_heads=N_HEADS, n_state=N_STATE,
               dims=dims, seeds=seeds, steps=args.steps, summary=summary, raw=raw,
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(args.output, 'w'), indent=2)
    log('=== CAPABILITY SUMMARY (mean length-extrap acc) ===')
    for name in arms:
        log(f'  {name}: ' + ' '.join(f'{k}={summary[name][k]}' for k in PROBES))
    log(f'WROTE {args.output}')


if __name__ == '__main__':
    main()
