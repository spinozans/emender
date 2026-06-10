#!/usr/bin/env python3
"""opt-1p3b — launch the matched-compute LM/BPB matrix across leased GPUs.

3 arms x N seeds, each WALLCLOCK-MATCHED to --train_minutes, one GPU per run, on
broker-leased GPUs only. Resumable: a run whose result JSON exists is skipped.

  eval "$(scripts/gpu_lease.sh 6)" && python run_lm_matrix.py --train_minutes 25 --seeds 0,1
"""
import os, sys, time, json, argparse, subprocess
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
ARMS = ['rstar', 'cma_gdn2', 'cma_m2rnn']


def leased_gpus():
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit('CUDA_VISIBLE_DEVICES empty — lease first: eval "$(scripts/gpu_lease.sh 6)"')
    return [g for g in cvd.split(',') if g.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_minutes', type=float, default=25.0)
    ap.add_argument('--seeds', type=str, default='0,1')
    ap.add_argument('--outdir', type=str, default=str(THIS / 'results'))
    ap.add_argument('--poll', type=float, default=10.0)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    gpus = leased_gpus()

    jobs = [(arm, s) for s in seeds for arm in ARMS]
    pending = [(a, s) for (a, s) in jobs
               if args.force or not (out / f'{a}_s{s}_result.json').exists()]
    print(f"Leased GPUs: {gpus}  jobs {len(pending)}/{len(jobs)} pending  "
          f"({args.train_minutes} min each)", flush=True)

    running, queue, t0 = {}, list(pending), time.time()

    def launch(gpu, job):
        arm, s = job
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        logf = open(out / f'{arm}_s{s}.log', 'w')
        cmd = [sys.executable, str(THIS / 'lm_runner.py'), '--arm', arm, '--seed', str(s),
               '--train_minutes', str(args.train_minutes), '--outdir', str(out)]
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        running[gpu] = (p, job, time.time(), logf)
        print(f"[launch gpu{gpu}] {arm} s{s}", flush=True)

    free = list(gpus)
    while queue or running:
        while queue and free:
            launch(free.pop(0), queue.pop(0))
        time.sleep(args.poll)
        for gpu in list(running.keys()):
            p, job, ts, logf = running[gpu]
            if p.poll() is not None:
                logf.close()
                arm, s = job
                ok = (out / f'{arm}_s{s}_result.json').exists()
                print(f"[done gpu{gpu} {time.time()-ts:.0f}s] {arm} s{s} "
                      f"rc={p.returncode} json={ok} ({time.time()-t0:.0f}s elapsed)", flush=True)
                del running[gpu]
                free.append(gpu)
    print(f"\nLM matrix complete in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == '__main__':
    main()
