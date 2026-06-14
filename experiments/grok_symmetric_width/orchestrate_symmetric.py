"""SYMMETRIC width control + FAR length-extrapolation (task grok-symmetric-width).

The predecessor (grok-highp-temporal) ruled the nonlinear-in-time thesis a NO via
a width control -- but that control widened ONLY the LINEAR arms (e97-lin, gdn2)
and measured final accuracy at the TRAIN length (T=128); it never widened the
NONLINEAR e97, and it never pushed length-extrapolation past T=1024. A linear cell
that "groks" at the train length may be MEMORIZING the train length and collapse
when asked to extrapolate. This run closes that gap:

  * WIDTH applied SYMMETRICALLY to ALL THREE arms (e97, e97-lin, gdn2) at
    dim {256, 512, 1024} -- the nonlinear arm is widened too.
  * FAR length-extrapolation: eval T {128,256,512,1024,2048,4096} -- pushed until
    (if ever) the wide linear arms collapse.
  * HIGH p {64, 256, 512} (K >> L), small depth L=2 (O(L) decisively exceeded).
  * weight-decay {0.1, 1.0}, seeds {0,1,2,3} (pin the bimodal grok variance).
  * throughput tok/s recorded per run (PI: wide-e97 runs well).

The prediction under test (PI): wide-e97 extrapolates / "kicks ass"; wide-LINEAR
fits short T but COLLAPSES at long T DESPITE width (memorization != extrapolation);
there is a high-p / long-T regime where linear collapses and e97 holds.

Full grid = dim{3} x arm{3} x p{3} x wd{2} x seed{4} = 216 runs. Jobs are ordered
so the DECISIVE cells (high p, wide, wd=1.0) run first -- partial completion still
answers the core question. REAL runs only. bf16 + fused asserted in train_grok.py.
"""
import os, sys, json, time, argparse, subprocess
from pathlib import Path
import concurrent.futures as cf

THIS = Path(__file__).resolve().parent
TRAIN = str(THIS / 'train_grok.py')
PY = sys.executable

BASE = dict(task='modular_quadratic', n_state=32, n_heads=8, mlp_ratio=4.0,
            seq_len=128, batch_size=64, n_train=128, n_test=512, lr=1e-3,
            depth=2, grok_acc=0.9)
ARMS = ['e97', 'e97-lin', 'gdn2']
PS = [64, 256, 512]
DIMS = [256, 512, 1024]
WDS = [1.0, 0.1]
SEEDS = [0, 1, 2, 3]


def jobspecs():
    jobs = {}

    def add(arm, p, dim, wd, seed):
        label = f"sym__mq_p{p}__{arm}__L2__d{dim}__wd{wd}__s{seed}"
        jobs.setdefault(label, dict(label=label, arm=arm, p=p, dim=dim,
                                    wd=wd, seed=seed))

    for wd in WDS:
        for p in PS:
            for dim in DIMS:
                for arm in ARMS:
                    for seed in SEEDS:
                        add(arm, p, dim, wd, seed)

    jl = list(jobs.values())
    # priority: wd=1.0 first (the grok-effective decay), then high p, then wide
    # dim, then arm, then seed -> decisive high-p/wide cells complete first.
    def key(j):
        return (WDS.index(j['wd']), -j['p'], -j['dim'],
                ARMS.index(j['arm']), j['seed'])
    jl.sort(key=key)
    return jl


def build_cmd(j, a, outdir):
    return [PY, TRAIN,
            '--task', BASE['task'], '--arm', j['arm'],
            '--dim', str(j['dim']), '--depth', str(BASE['depth']),
            '--n_state', str(BASE['n_state']), '--n_heads', str(BASE['n_heads']),
            '--mlp_ratio', str(BASE['mlp_ratio']), '--seq_len', str(BASE['seq_len']),
            '--batch_size', str(BASE['batch_size']), '--n_train', str(BASE['n_train']),
            '--n_test', str(BASE['n_test']), '--lr', str(BASE['lr']),
            '--weight_decay', str(j['wd']), '--steps', str(a.steps),
            '--eval_interval', str(a.eval_interval), '--K', str(j['p']),
            '--seed', str(j['seed']), '--grok_acc', str(BASE['grok_acc']),
            '--patience_evals', str(a.patience_evals),
            '--eval_lengths', '128', '256', '512', '1024', '2048', '4096',
            '--label', j['label'], '--output_dir', str(outdir)]


def run_one(j, gpu, a, outdir):
    jpath = outdir / f"{j['label']}.json"
    if jpath.exists() and not a.force:
        print(f"SKIP {j['label']} (exists)", flush=True)
        return dict(label=j['label'], skipped=True)
    cmd = build_cmd(j, a, outdir)
    logf = outdir / f"{j['label']}.log"
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    t0 = time.time()
    with open(logf, 'w') as lf:
        lf.write('CMD: ' + ' '.join(cmd) + '\n\n'); lf.flush()
        p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    out = dict(label=j['label'], gpu=str(gpu), rc=p.returncode, wall_s=round(wall, 1))
    if jpath.exists():
        try:
            d = json.load(open(jpath))
            out.update(grokked=d.get('grokked'), grok_step=d.get('grok_step'),
                       final_test_acc=d.get('final_test_acc'),
                       final_train_acc=d.get('final_train_acc'),
                       throughput=d.get('throughput_toks_per_s'))
        except Exception as e:
            out['parse_error'] = str(e)
    else:
        out['no_json'] = True
        out['tail'] = logf.read_text(errors='replace')[-600:]
    print('DONE ' + json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', required=True, help='csv gpu ids')
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--eval_interval', type=int, default=500)
    ap.add_argument('--patience_evals', type=int, default=40)
    ap.add_argument('--outdir', default=str(THIS / 'runs'))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()

    gpus = [g.strip() for g in a.gpus.split(',') if g.strip()]
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jobs = jobspecs()
    print(f"{len(jobs)} jobs on {len(gpus)} GPUs", flush=True)
    if a.dry:
        for j in jobs:
            print('  ', j['label'])
        return

    results = []
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {}
        for i, j in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(run_one, j, gpu, a, outdir)] = j['label']
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    summ = outdir / 'orchestrate_symmetric_summary.json'
    summ.write_text(json.dumps(results, indent=2))
    print('WROTE', summ, flush=True)


if __name__ == '__main__':
    main()
