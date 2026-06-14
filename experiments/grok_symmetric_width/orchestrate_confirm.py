"""CONFIRM the temporal class separation (task grok-confirm).

Two confirmations on top of grok-symmetric-width's 1-4-seed result:

  PART 1 -- SEED EXPANSION at the decisive cells. The predecessor measured the
  extrapolation curves over seeds {0,1,2,3} (n=1..4 grokked per cell), which is
  thin for a paper claim. We add seeds {4,5,6,7} at the DECISIVE cells
    arm{e97,e97-lin,gdn2} x dim{512,1024} x p{256,512} x wd=1.0
  so every decisive cell has 8 seeds. Existing seed-0..3 JSONs are reused
  (skip-if-exists); only the 4 new seeds per cell actually run. The aggregate
  then re-derives per-cell grok counts and the test-acc-vs-T curve over 8 seeds.

  PART 2 -- SECOND TASK FAMILY. We replicate the e97-extrapolates /
  linear-collapses signature on a DIFFERENT temporal-composition task:
  iterated_nonlinear_map (the input-driven logistic map h_t = a_t h_{t-1}(1-h_t),
  a genuine state-quadratic). Same three arms, same train-to-grok protocol
  (AdamW + wd=1.0, fixed finite split, long training), same far length-extrap
  (T up to 4096). This is mechanism-matched to the first task (modular_quadratic
  is itself a mod-p iterated nonlinear map) but a genuinely different surface
  (continuous binned logistic vs modular arithmetic, different vocab/baseline).

Everything else is byte-identical to grok_symmetric_width's harness: same
train_grok.py, same geometry (n_state=32, n_heads=8, mlp_ratio=4, L=2,
seq_len=128, n_train=128, n_test=512, lr=1e-3, steps=50000), bf16 + fused
asserted per-arm inside train_grok.assert_kernel. REAL runs only.
"""
import os, sys, json, time, argparse, subprocess
from pathlib import Path
import concurrent.futures as cf

THIS = Path(__file__).resolve().parent
TRAIN = str(THIS / 'train_grok.py')
PY = sys.executable

BASE = dict(n_state=32, n_heads=8, mlp_ratio=4.0, seq_len=128, batch_size=64,
            n_train=128, n_test=512, lr=1e-3, depth=2, grok_acc=0.9)
ARMS = ['e97', 'e97-lin', 'gdn2']

# ---- Part 1: decisive-cell seed expansion (modular_quadratic) ----
P1_DIMS = [512, 1024]
P1_PS = [256, 512]
P1_WD = 1.0
P1_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]   # 0-3 reused (skip-exists), 4-7 are new

# ---- Part 2: second task family (iterated_nonlinear_map) ----
P2_DIMS = [512, 1024]
P2_SEEDS = [0, 1, 2, 3]
P2_WD = 1.0
P2_K = 2        # K<=2 => task default n_bins=10 (baseline 0.1)

# ---- Part 3: second task family (anbncn_viability) ----
# A NON-CONTRACTIVE long-memory task: per-position viability of a^n b^n c^n,
# decided by count comparisons whose magnitude scales with T. Unlike the
# (contractive, fading-memory) logistic map, length-extrapolation here is a
# genuine memorization-vs-rule test -- the right regime for the signature.
P3_DIMS = [512, 1024]
P3_SEEDS = [0, 1, 2, 3]
P3_WD = 1.0


def jobspecs():
    jobs = {}

    def add(task, arm, p, dim, wd, seed, label):
        jobs.setdefault(label, dict(label=label, task=task, arm=arm, p=p,
                                    dim=dim, wd=wd, seed=seed))

    # Part 1 -- reuse the predecessor's exact label scheme so seed-0..3 JSONs
    # already on disk are picked up unchanged.
    for p in P1_PS:
        for dim in P1_DIMS:
            for arm in ARMS:
                for seed in P1_SEEDS:
                    label = f"sym__mq_p{p}__{arm}__L2__d{dim}__wd{P1_WD}__s{seed}"
                    add('modular_quadratic', arm, p, dim, P1_WD, seed, label)

    # Part 2 -- second task family (logistic map).
    for dim in P2_DIMS:
        for arm in ARMS:
            for seed in P2_SEEDS:
                label = f"inm__b10__{arm}__L2__d{dim}__wd{P2_WD}__s{seed}"
                add('iterated_nonlinear_map', arm, P2_K, dim, P2_WD, seed, label)

    # Part 3 -- second task family (a^n b^n c^n viability, long-memory).
    for dim in P3_DIMS:
        for arm in ARMS:
            for seed in P3_SEEDS:
                label = f"abc__{arm}__L2__d{dim}__wd{P3_WD}__s{seed}"
                add('anbncn_viability', arm, 2, dim, P3_WD, seed, label)

    jl = list(jobs.values())
    # decisive cells first: high p, wide dim, e97 before linear arms.
    order = {'modular_quadratic': 0, 'anbncn_viability': 1,
             'iterated_nonlinear_map': 2}
    def key(j):
        return (order.get(j['task'], 9),
                -j['p'], -j['dim'], ARMS.index(j['arm']), j['seed'])
    jl.sort(key=key)
    return jl


def build_cmd(j, a, outdir):
    return [PY, TRAIN,
            '--task', j['task'], '--arm', j['arm'],
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
    ap.add_argument('--gpus', required=True, help='csv gpu ids (broker-leased)')
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
    n_new = sum(1 for j in jobs if not (outdir / f"{j['label']}.json").exists())
    print(f"{len(jobs)} jobs ({n_new} new, rest skip-exists) on {len(gpus)} GPUs",
          flush=True)
    if a.dry:
        for j in jobs:
            exists = (outdir / f"{j['label']}.json").exists()
            print(f"  {'SKIP' if exists else 'RUN '} {j['label']}")
        return

    results = []
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {}
        for i, j in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(run_one, j, gpu, a, outdir)] = j['label']
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    summ = outdir / 'orchestrate_confirm_summary.json'
    summ.write_text(json.dumps(results, indent=2))
    print('WROTE', summ, flush=True)


if __name__ == '__main__':
    main()
