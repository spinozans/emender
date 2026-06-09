"""e97-hetero-cma: clean confirmatory throughput at the capability KNEE.

Repeated (3x) blended tok/s for the decision-relevant configs, pooled one per GPU
(GPUs 0-6; GPU7 left free). Includes the same-harness gated-delta SHELL head
(hetero-kernel's 0.954x head) at the same fractions so the contrast is airtight:
the split-edit capability head vs the capability-weak shell head, same machine,
same TypedHeadMixtureLayer, same overlap. Reports mean ratio vs the f=0 GDN-2 base.
REAL models. No mocks.
"""
import os, sys, json, time, subprocess, datetime, argparse
import numpy as np
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _THIS)
RESULTS = os.path.join(_THIS, 'results'); WORKER = os.path.join(_THIS, 'tput_worker.py')
FREE_MEM_MIB = 2000


def log(m): print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def gpu_used_mib():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                   '--format=csv,noheader,nounits'], timeout=15).decode()
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6')
    ap.add_argument('--reps', type=int, default=3)
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    # (label, frac/64, head_kind, overlap)
    specs = [
        ('gdn2_base', 0, 'split', 1),
        ('split4_ov', 4, 'split', 1),
        ('split16_ov', 16, 'split', 1),
        ('split16_seq', 16, 'split', 0),
        ('shell4_ov', 4, 'shell', 1),
        ('shell16_ov', 16, 'shell', 1),
    ]
    jobs = [(lbl, fh, hk, ov, rep) for (lbl, fh, hk, ov) in specs for rep in range(args.reps)]
    log(f'{len(jobs)} throughput-confirm jobs ({len(specs)} configs x {args.reps} reps)')

    queue = list(jobs); running = {}; raw = {}
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
        while queue and free:
            lbl, fh, hk, ov, rep = queue.pop(0); gpu = free.pop(0)
            tag = f'{lbl}_r{rep}'
            outp = os.path.join(RESULTS, f'tc_{tag}.json'); logp = os.path.join(RESULTS, f'tc_{tag}.log')
            fh_ = open(logp, 'w')
            cmd = [sys.executable, WORKER, '--e97_frac', str(fh / 64.0), '--overlap', str(ov),
                   '--head_kind', hk, '--out', outp, '--n_iter', '25']
            proc = subprocess.Popen(cmd, stdout=fh_, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = dict(proc=proc, lbl=lbl, out=outp, fh=fh_, t0=time.time())
            log(f'LAUNCH {tag} GPU{gpu}')
        time.sleep(5)
        for gpu, job in list(running.items()):
            if job['proc'].poll() is None and (time.time() - job['t0']) < 900:
                continue
            if job['proc'].poll() is None:
                job['proc'].kill()
            job['fh'].close()
            if os.path.exists(job['out']):
                r = json.load(open(job['out']))
                if 'tok_s' in r:
                    raw.setdefault(job['lbl'], []).append(r['tok_s'])
            del running[gpu]

    base = float(np.mean(raw.get('gdn2_base', [1.0])))
    summary = {}
    for lbl, fh, hk, ov in specs:
        ts = raw.get(lbl, [])
        summary[lbl] = dict(frac_heads=fh, head_kind=hk, overlap=bool(ov),
                            tok_s_mean=round(float(np.mean(ts)), 1) if ts else None,
                            tok_s_std=round(float(np.std(ts)), 1) if ts else None,
                            ratio_vs_gdn2=round(float(np.mean(ts)) / base, 4) if ts else None,
                            n=len(ts))
    out = dict(task='e97-hetero-cma-throughput-confirm', gdn2_baseline_tok_s=round(base, 1),
               reps=args.reps, summary=summary,
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(os.path.join(RESULTS, 'throughput_confirm.json'), 'w'), indent=2)
    log('=== THROUGHPUT CONFIRM (mean of reps) vs GDN-2 ===')
    for lbl, fh, hk, ov in specs:
        s = summary[lbl]
        log(f"  {lbl:13s} f={fh:2d}/64 {hk:5s} ov={int(ov)}: "
            f"tok/s={s['tok_s_mean']}+-{s['tok_s_std']} ratio={s['ratio_vs_gdn2']} n={s['n']}")
    log('WROTE throughput_confirm.json')


if __name__ == '__main__':
    main()
