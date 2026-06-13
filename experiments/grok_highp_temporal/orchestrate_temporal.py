"""High-p temporal-composition separation orchestrator (task grok-highp-temporal).

THE DECIDING test for the nonlinear-in-time thesis. modular_quadratic is an
iterated nonlinear map x_t=(x_{t-1}^2+c_t) mod p, so the composition depth K the
solver must realise scales with p. Separation between a per-step-NONLINEAR cell
(e97) and the matched per-step-LINEAR control (e97-lin) only appears when the
required temporal composition K exceeds the network's O(depth) nonlinear budget,
i.e. K > L.  The grok-expressivity calib found NO separation -- but at the EASY
control p=7 (default), where K < L=4 so everyone solves it.  This run pushes p
HIGH and L SMALL so K >> L and looks for separation IN THE TEMPORAL DOMAIN.

Each cell is one train_grok.py run pinned to one leased GPU.  Unlike the stock
orchestrate.py (which fixes dim/depth/K globally) this one varies p (=--K),
depth, and dim PER JOB, because the whole experiment is a p x L x width sweep.

Groups (explicit, deduped):
  MAIN     L=2, wd=1.0, seeds{0,1}: p{32,64,128,256} x {e97,e97-lin,gdn2}
           -> the headline separation(p,T) signature.
  LDEPTH   L=4, wd=1.0, seed 0:     p{64,256} x {e97,e97-lin,gdn2}
           -> does the gap shrink as L grows (K-vs-L mechanism)?
  WIDTH    L=2, wd=1.0, seed 0, p=128: {e97-lin,gdn2} x dim{512,1024}
           -> does MORE WIDTH on the linear arms close the gap? (capacity vs depth)
  WDSWEEP  L=2, seed 0, p=64: {e97,e97-lin} x wd{0.01,0.1,0.3}
           -> weight-decay axis (grokking is wd-driven); wd=1.0 already in MAIN.

REAL runs only.  bf16 + fused asserted inside train_grok.py.
"""
import os, sys, json, time, argparse, subprocess
from pathlib import Path
import concurrent.futures as cf

THIS = Path(__file__).resolve().parent
TRAIN = str(THIS / 'train_grok.py')
PY = sys.executable

# fixed small geometry (matched across arms)
BASE = dict(task='modular_quadratic', n_state=32, n_heads=8, mlp_ratio=4.0,
            seq_len=128, batch_size=64, n_train=128, n_test=512, lr=1e-3,
            grok_acc=0.9)
ARMS = ['e97', 'e97-lin', 'gdn2']


def jobspecs():
    jobs = {}  # label -> dict, dedup by label

    def add(group, arm, p, depth, dim, wd, seed):
        label = f"{group}__mq_p{p}__{arm}__L{depth}__d{dim}__wd{wd}__s{seed}"
        jobs.setdefault(label, dict(label=label, group=group, arm=arm, p=p,
                                    depth=depth, dim=dim, wd=wd, seed=seed))

    # MAIN: p x arm x seed  (L=2, dim=256, wd=1.0)
    for p in (32, 64, 128, 256):
        for arm in ARMS:
            for seed in (0, 1):
                add('main', arm, p, 2, 256, 1.0, seed)
    # LDEPTH: L=4 control at p{64,256}
    for p in (64, 256):
        for arm in ARMS:
            add('ldepth', arm, p, 4, 256, 1.0, 0)
    # WIDTH: sweep dim UP on the linear arms at p=128, L=2
    for arm in ('e97-lin', 'gdn2'):
        for dim in (512, 1024):
            add('width', arm, 128, 2, dim, 1.0, 0)
    # WDSWEEP: wd axis at p=64, L=2 (wd=1.0 already in main)
    for arm in ('e97', 'e97-lin'):
        for wd in (0.01, 0.1, 0.3):
            add('wdsweep', arm, 64, 2, 256, wd, 0)

    # CONFIRM: resolve the three confounds the first 40-run pass surfaced.
    #  (a) WIDTH AT p=256 (the clean-gap regime; the original width control was at
    #      p=128, the noisy regime) -> does more width close the p=256 gap? sweep
    #      dim UP on e97-lin AND gdn2 (and e97 itself as a width-robustness check).
    for arm in ('e97', 'e97-lin', 'gdn2'):
        for dim in (512, 1024):
            add('confirm', arm, 256, 2, dim, 1.0, 0)
    #  (b) MORE SEEDS at p=256 and p=128 (the first pass was bimodal: grok-or-stuck)
    #      to test whether the 2/2-vs-0/2 split at p=256 is real or seed-luck.
    for p in (128, 256):
        for arm in ('e97', 'e97-lin', 'gdn2'):
            for seed in (2, 3):
                add('confirm', arm, p, 2, 256, 1.0, seed)

    return list(jobs.values())


def build_cmd(j, a, outdir):
    cmd = [PY, TRAIN,
           '--task', BASE['task'], '--arm', j['arm'],
           '--dim', str(j['dim']), '--depth', str(j['depth']),
           '--n_state', str(BASE['n_state']), '--n_heads', str(BASE['n_heads']),
           '--mlp_ratio', str(BASE['mlp_ratio']), '--seq_len', str(BASE['seq_len']),
           '--batch_size', str(BASE['batch_size']), '--n_train', str(BASE['n_train']),
           '--n_test', str(BASE['n_test']), '--lr', str(BASE['lr']),
           '--weight_decay', str(j['wd']), '--steps', str(a.steps),
           '--eval_interval', str(a.eval_interval), '--K', str(j['p']),
           '--seed', str(j['seed']), '--grok_acc', str(BASE['grok_acc']),
           '--patience_evals', str(a.patience_evals),
           '--eval_lengths', '128', '256', '512', '1024',
           '--label', j['label'], '--output_dir', str(outdir)]
    return cmd


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
    out = dict(label=j['label'], group=j['group'], gpu=str(gpu),
               rc=p.returncode, wall_s=round(wall, 1))
    if jpath.exists():
        try:
            d = json.load(open(jpath))
            out.update(grokked=d.get('grokked'), grok_step=d.get('grok_step'),
                       final_test_acc=d.get('final_test_acc'),
                       final_train_acc=d.get('final_train_acc'),
                       length_extrap=d.get('length_extrap'))
        except Exception as e:
            out['parse_error'] = str(e)
    else:
        out['no_json'] = True
        out['tail'] = logf.read_text(errors='replace')[-600:]
    print('DONE ' + json.dumps({k: out[k] for k in out if k != 'length_extrap'}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', required=True, help='csv gpu ids')
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--eval_interval', type=int, default=500)
    ap.add_argument('--patience_evals', type=int, default=40)
    ap.add_argument('--groups', default='main,ldepth,width,wdsweep',
                    help='csv subset of groups to run')
    ap.add_argument('--outdir', default=str(THIS / 'runs'))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()

    gpus = [g.strip() for g in a.gpus.split(',') if g.strip()]
    want = set(g for g in a.groups.split(',') if g)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    jobs = [j for j in jobspecs() if j['group'] in want]
    print(f"{len(jobs)} jobs on {len(gpus)} GPUs (groups={sorted(want)})", flush=True)
    for j in jobs:
        print('  ', j['label'])
    if a.dry:
        return

    results = []
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {}
        for i, j in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(run_one, j, gpu, a, outdir)] = j['label']
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    summ = outdir / 'orchestrate_temporal_summary.json'
    summ.write_text(json.dumps(results, indent=2))
    print('WROTE', summ, flush=True)


if __name__ == '__main__':
    main()
