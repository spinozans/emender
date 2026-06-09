"""e97-hetero-cma: blended throughput vs GDN-2 across the nonlinear-head FRACTION.

Pools one tput_worker per free GPU over the fraction grid {0,1,2,4,8,16}/64 at
overlap on AND off, computing tok/s ratio vs the frac=0 GDN-2 baseline. This is
the throughput half of the throughput/capability tradeoff curve — the wall-clock
cost of the depth-capability split-edit head as a function of how many heads carry
it, with the side-stream overlap on vs off. REAL models. No mocks.
"""
import os, sys, json, time, subprocess, datetime, argparse
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _THIS)
RESULTS = os.path.join(_THIS, 'results'); os.makedirs(RESULTS, exist_ok=True)
WORKER = os.path.join(_THIS, 'tput_worker.py')
FREE_MEM_MIB = 2000


def log(m): print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def gpu_used_mib():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                   '--format=csv,noheader,nounits'], timeout=15).decode()
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--fracs', default='0,1,2,4,8,16')  # heads-over-64
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    fracs = [int(x) / 64.0 for x in args.fracs.split(',')]

    jobs = []
    for f in fracs:
        if f == 0:
            jobs.append((f, 1))  # baseline (overlap irrelevant; no seq head)
        else:
            jobs.append((f, 1)); jobs.append((f, 0))
    log(f'{len(jobs)} throughput jobs over fracs {args.fracs} (overlap on/off)')

    queue = list(jobs); running = {}; results = []
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
        while queue and free:
            f, ov = queue.pop(0); gpu = free.pop(0)
            tag = f'f{int(round(f*64))}_ov{ov}'
            outp = os.path.join(RESULTS, f'tput_{tag}.json')
            logp = os.path.join(RESULTS, f'tput_{tag}.log'); fh = open(logp, 'w')
            cmd = [sys.executable, WORKER, '--e97_frac', str(f), '--overlap', str(ov),
                   '--out', outp]
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = dict(proc=proc, tag=tag, out=outp, fh=fh, t0=time.time())
            log(f'LAUNCH {tag} on GPU {gpu}')
        time.sleep(5)
        for gpu, job in list(running.items()):
            if job['proc'].poll() is None and (time.time() - job['t0']) < 900:
                continue
            if job['proc'].poll() is None:
                job['proc'].kill(); log(f'TIMEOUT {job["tag"]}')
            job['fh'].close()
            if os.path.exists(job['out']):
                results.append(json.load(open(job['out'])))
                r = results[-1]
                log(f'DONE {job["tag"]} tok/s={r.get("tok_s")} dim={r.get("dim")} err={r.get("error")}')
            del running[gpu]

    base = next((r['tok_s'] for r in results
                 if r.get('e97_frac') == 0 and 'tok_s' in r), None)
    for r in results:
        if base and 'tok_s' in r:
            r['ratio_vs_gdn2'] = round(r['tok_s'] / base, 4)
    out = dict(task='e97-hetero-cma-throughput', gdn2_baseline_tok_s=base,
               rows=sorted(results, key=lambda r: (r.get('e97_frac', 0), not r.get('overlap', True))),
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(os.path.join(RESULTS, 'throughput.json'), 'w'), indent=2)
    log('=== BLENDED THROUGHPUT vs GDN-2 ===')
    for r in out['rows']:
        if 'tok_s' in r:
            log(f"  frac={int(round(r['e97_frac']*64))}/64 overlap={r['overlap']} "
                f"tok/s={r['tok_s']} ratio={r.get('ratio_vs_gdn2')}")
    log('WROTE throughput.json')


if __name__ == '__main__':
    main()
