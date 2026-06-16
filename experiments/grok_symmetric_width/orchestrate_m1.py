"""M1 state-aware MLP — cheap falsification probe (task improve-mlp-integration).

Tests the M1 thesis (STATE_AWARE_MLP_DESIGN.md §5): let the post-mixer SwiGLU MLP
mix the per-head readouts NONLINEARLY *before* the linear o_proj collapse, by
concatenating a down-projected + RMSNorm'd copy of the SAME pre-o_proj readout the
o_proj already consumes. M1 touches NO recurrence dynamics — the E97 state update
stays on the FUSED Triton kernel (NON-NEGOTIABLE #1; [fused-guard] asserted in
train_grok.assert_kernel). The probe varies ONLY the per-block MLP across 3 arms,
holding the fused E97 cell + all geometry fixed (iso-config separator).

ARMS (all fused E97, iso-param within <0.02%):
  baseline : plain SwiGLU MLP, hidden 1024 (mlp_ratio 4.0), state_summary_dim 0.
  m1b      : state-aware — readout summary value_dim(256)->m=128 + RMSNorm, concat
             to MLP input; hidden shrunk to 736 to restore iso-param (mirrors the
             1.3B M1b_m512 -30% MLP-hidden shrink). The mechanism arm.
  control  : plain-wider SwiGLU MLP (state_summary_dim 0), hidden re-spent to the
             SAME iso-param budget -> 1024. At iso-param this coincides with the
             baseline hidden (PROVEN by state_aware_mlp_param_check.py --build-arms:
             the plain-MLP iso allocation == baseline). It is run on a DISJOINT seed
             set so it is an INDEPENDENT plain-MLP replicate (not a bit-identical
             duplicate), giving an independent capacity estimate to isolate the
             mechanism from capacity.

DECISION RULE (task): if m1b TIES the plain-MLP arms (baseline/control) — no extrap
or in-dist gain — declare NULL and STOP (do not escalate to 1.3B). Only if m1b beats
BOTH baseline and control do we plan a 1.3B A/B.

REAL data, REAL fused kernel. No mocks. Mirrors orchestrate_symmetric.py conventions.
"""
import os, sys, json, time, argparse, subprocess
from pathlib import Path
import concurrent.futures as cf

THIS = Path(__file__).resolve().parent
TRAIN = str(THIS / 'train_grok.py')
PY = sys.executable

# Iso-config geometry (matches the grok-symmetric-width separator regime):
#   dim 256, depth 2, n_state 32, n_heads 8 -> value_dim = n_heads*n_state = 256.
# Iso-param MLP hidden per arm (depth-independent per-layer relation; certified by
# the trainer's printed param count + assert below):
#   baseline/control: extra_in 0 -> hidden 1024.
#   m1b: R = value_dim*m + m = 256*128+128 = 32,896/layer; w1,w2 widen by m=128;
#        iso hidden = (3*256*1024 - 32,896)/1024 = 735.875 -> 736.
BASE = dict(task='modular_quadratic', dim=256, depth=2, n_state=32, n_heads=8,
            mlp_ratio=4.0, seq_len=128, batch_size=64, n_train=128, n_test=512,
            lr=1e-3, weight_decay=0.1, grok_acc=0.9)
M = 128                      # state_summary_dim for m1b (= value_dim/2, generous)
H_BASE, H_M1B = 1024, 736    # iso-param SwiGLU hidden
PS = [48, 256]               # mod 48 (design §6 spec) + high-p (headroom / separator regime)
SEEDS = [0, 1, 2]
CONTROL_SEED_OFFSET = 100     # disjoint seeds so control is an independent plain-MLP replicate

# (label-suffix, state_summary_dim, mlp_hidden, seed_offset)
ARMS = [
    ('baseline', 0,   H_BASE, 0),
    ('m1b',      M,   H_M1B,  0),
    ('control',  0,   H_BASE, CONTROL_SEED_OFFSET),
]


def jobspecs():
    jobs = []
    for p in PS:
        for arm, ssd, mh, soff in ARMS:
            for seed in SEEDS:
                label = f"m1__mq_p{p}__{arm}__d{BASE['dim']}__L{BASE['depth']}__s{seed}"
                jobs.append(dict(label=label, arm_tag=arm, ssd=ssd, mlp_hidden=mh,
                                 p=p, seed=seed + soff))
    return jobs


def build_cmd(j, a, outdir):
    return [PY, TRAIN,
            '--task', BASE['task'], '--arm', 'e97',
            '--dim', str(BASE['dim']), '--depth', str(BASE['depth']),
            '--n_state', str(BASE['n_state']), '--n_heads', str(BASE['n_heads']),
            '--mlp_ratio', str(BASE['mlp_ratio']), '--mlp_hidden', str(j['mlp_hidden']),
            '--state_summary_dim', str(j['ssd']),
            '--seq_len', str(BASE['seq_len']), '--batch_size', str(BASE['batch_size']),
            '--n_train', str(BASE['n_train']), '--n_test', str(BASE['n_test']),
            '--lr', str(BASE['lr']), '--weight_decay', str(BASE['weight_decay']),
            '--steps', str(a.steps), '--eval_interval', str(a.eval_interval),
            '--K', str(j['p']), '--seed', str(j['seed']), '--grok_acc', str(BASE['grok_acc']),
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
            out.update(params=d.get('params'), grokked=d.get('grokked'),
                       grok_step=d.get('grok_step'), final_test_acc=d.get('final_test_acc'),
                       final_train_acc=d.get('final_train_acc'),
                       length_extrap=d.get('length_extrap'),
                       throughput=d.get('throughput_toks_per_s'))
        except Exception as e:
            out['parse_error'] = str(e)
    else:
        out['no_json'] = True
        out['tail'] = logf.read_text(errors='replace')[-800:]
    print('DONE ' + json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', required=True, help='csv gpu ids (lease via scripts/gpu_lease.sh)')
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--eval_interval', type=int, default=500)
    ap.add_argument('--patience_evals', type=int, default=40)
    ap.add_argument('--outdir', default=str(THIS / 'runs_m1'))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()

    gpus = [g.strip() for g in a.gpus.split(',') if g.strip()]
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jobs = jobspecs()
    print(f"{len(jobs)} jobs ({len(PS)} p x {len(ARMS)} arms x {len(SEEDS)} seeds) "
          f"on {len(gpus)} GPUs", flush=True)
    if a.dry:
        for j in jobs:
            print('  ', j['label'], 'ssd=', j['ssd'], 'h=', j['mlp_hidden'], 'seed=', j['seed'])
        return

    results = []
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {}
        for i, j in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(run_one, j, gpu, a, outdir)] = j['label']
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    summ = outdir / 'orchestrate_m1_summary.json'
    summ.write_text(json.dumps(results, indent=2))
    print('WROTE', summ, flush=True)


if __name__ == '__main__':
    main()
