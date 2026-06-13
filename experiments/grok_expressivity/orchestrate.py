"""Fan a grok-expressivity sweep across the 8-GPU box.

Grid: tasks x arms x weight_decay x seeds. Each cell is one train_grok.py run
pinned to one GPU. GPUs are taken from CUDA_VISIBLE_DEVICES (set by the lease
one-liner). Round-robin scheduling with one worker per visible GPU.

REAL runs only.
"""
import os, sys, json, time, argparse, itertools, subprocess
from pathlib import Path
import concurrent.futures as cf

THIS = Path(__file__).resolve().parent
TRAIN = str(THIS / 'train_grok.py')
PY = sys.executable


def build_cmd(task, arm, wd, seed, ntr, a, outdir):
    label = f"{task}__{arm}__wd{wd}__nt{ntr}__s{seed}"
    cmd = [PY, TRAIN, '--task', task, '--arm', arm,
           '--dim', str(a.dim), '--depth', str(a.depth),
           '--n_state', str(a.n_state), '--n_heads', str(a.n_heads),
           '--mlp_ratio', str(a.mlp_ratio), '--seq_len', str(a.seq_len),
           '--batch_size', str(a.batch_size), '--n_train', str(ntr),
           '--n_test', str(a.n_test), '--lr', str(a.lr),
           '--weight_decay', str(wd), '--steps', str(a.steps),
           '--eval_interval', str(a.eval_interval), '--K', str(a.K),
           '--seed', str(seed), '--grok_acc', str(a.grok_acc),
           '--patience_evals', str(a.patience_evals),
           '--label', label, '--output_dir', str(outdir)]
    return label, cmd


def run_one(task, arm, wd, seed, ntr, gpu, a, outdir):
    label, cmd = build_cmd(task, arm, wd, seed, ntr, a, outdir)
    jpath = outdir / f"{label}.json"
    if jpath.exists() and not a.force:
        print(f"SKIP {label} (exists)", flush=True)
        return dict(label=label, skipped=True)
    logf = outdir / f"{label}.log"
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    t0 = time.time()
    with open(logf, 'w') as lf:
        lf.write('CMD: ' + ' '.join(cmd) + '\n\n'); lf.flush()
        p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    out = dict(label=label, gpu=str(gpu), rc=p.returncode, wall_s=round(wall, 1))
    if jpath.exists():
        try:
            d = json.load(open(jpath))
            out.update(grokked=d.get('grokked'), grok_step=d.get('grok_step'),
                       final_test_acc=d.get('final_test_acc'),
                       final_train_acc=d.get('final_train_acc'))
        except Exception as e:
            out['parse_error'] = str(e)
    else:
        out['no_json'] = True
        out['tail'] = logf.read_text(errors='replace')[-400:]
    print('DONE ' + json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', required=True, help='csv gpu ids')
    ap.add_argument('--tasks', default='s5_permutation,modular_quadratic,anbncn_viability')
    ap.add_argument('--arms', default='e97,gdn2')
    ap.add_argument('--wds', default='0.01,0.1,0.3,1.0')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_state', type=int, default=32)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--mlp_ratio', type=float, default=4.0)
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--n_trains', default='1024', help='csv of train-set sizes')
    ap.add_argument('--n_test', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--eval_interval', type=int, default=500)
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--grok_acc', type=float, default=0.9)
    ap.add_argument('--patience_evals', type=int, default=40)
    ap.add_argument('--outdir', default=str(THIS / 'runs'))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    gpus = [g.strip() for g in a.gpus.split(',') if g.strip()]
    tasks = [t for t in a.tasks.split(',') if t]
    arms = [x for x in a.arms.split(',') if x]
    wds = [w for w in a.wds.split(',') if w]
    seeds = [int(s) for s in a.seeds.split(',')]
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ntrains = [int(x) for x in a.n_trains.split(',') if x]
    jobs = list(itertools.product(tasks, arms, wds, seeds, ntrains))
    print(f"{len(jobs)} jobs on {len(gpus)} GPUs", flush=True)
    results = []
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {}
        for i, (task, arm, wd, seed, ntr) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(run_one, task, arm, wd, seed, ntr, gpu, a, outdir)] = (task, arm, wd, seed, ntr)
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    summ = outdir / 'orchestrate_summary.json'
    summ.write_text(json.dumps(results, indent=2))
    print('WROTE', summ, flush=True)


if __name__ == '__main__':
    main()
