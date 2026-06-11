"""emender-real-cap — LM convergent-loss cohort driver.

Runs every (arm, seed) of the convergent-loss tie ONE-PER-GPU (no slot sharing) so all
arms experience IDENTICAL contention -> the wall-matched BPB comparison is fair. Resumable.

  eval "$(scripts/gpu_lease.sh 6)"
  python experiments/emender_real_cap/run_lm_cohort.py --train_minutes 12 --seeds 0 1
"""
import argparse, os, subprocess, time
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
RUNNER = THIS / 'lm_convergent.py'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['gdn2', 'gdn2typed', 'emender4'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--train_minutes', type=float, default=12.0)
    ap.add_argument('--outdir', default=str(THIS / 'results_lm'))
    args = ap.parse_args()

    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit('lease GPUs first: eval "$(scripts/gpu_lease.sh 6)"')
    gpus = [g for g in cvd.split(',') if g.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    jobs = [(a, s) for a in args.arms for s in args.seeds
            if not (Path(args.outdir) / f'{a}_s{s}_result.json').exists()]
    print(f"GPUs {gpus}  jobs {len(jobs)}: {jobs}", flush=True)

    running, queue = {}, list(jobs)
    t0 = time.time(); done = 0

    def launch(gpu, job):
        a, s = job
        env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        logf = open(Path(args.outdir) / f'{a}_s{s}.log', 'w')
        cmd = ['python', str(RUNNER), '--arm', a, '--seed', str(s),
               '--train_minutes', str(args.train_minutes), '--outdir', args.outdir]
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        running[gpu] = (proc, job, logf)
        print(f"[launch gpu{gpu}] {a} s{s}", flush=True)

    while queue or running:
        for gpu in gpus:
            if gpu not in running and queue:
                launch(gpu, queue.pop(0))
        time.sleep(5)
        for gpu in list(running.keys()):
            proc, job, logf = running[gpu]
            if proc.poll() is not None:
                logf.close(); done += 1
                a, s = job
                ok = (Path(args.outdir) / f'{a}_s{s}_result.json').exists()
                print(f"[{'ok' if ok else 'FAIL'}] {a} s{s} ({done}/{len(jobs)}, "
                      f"{time.time()-t0:.0f}s)", flush=True)
                del running[gpu]
    print(f"\nLM cohort complete: {done} runs in {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
