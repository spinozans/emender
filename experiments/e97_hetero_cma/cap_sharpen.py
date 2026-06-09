"""e97-hetero-cma: SHARPENED depth-cliff sweep — 4 seeds + within-substrate
identity control to cut grokking-seed variance and isolate the substrate.

modular_quadratic (mod 48) is a grokking task; the 2-seed first pass was noisy.
This runs 4 seeds on the decision-relevant arms and adds split8_id (split-edit
LINEAR/identity state) — phi-explore's clean within-substrate control: if the
split-edit SUBSTRATE (not the gdn-neg bulk) carries the capability, split8_tanh
should beat split8_id; if it's the per-step BOUNDED phi specifically, the gap is
the phi effect. shell8 (gated-delta + tanh) is the cross-substrate control.

Reports the CLIFF metric: T2048 acc (16x extrapolation) + retention T2048/T128.
REAL tasks, real training. No mocks.
"""
import os, sys, json, time, subprocess, datetime, argparse
import numpy as np
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from cap_sweep import (arm_spec, build_cmd, log, OUT_DIR, MODQUAD_P)
from capability import gpu_used_mib, FREE_MEM_MIB
RESULTS = os.path.join(_THIS, 'results')

# (name, kind, frac_heads, extra_state_nonlin)
ARMS = [
    ('linear', 'linear', 0),
    ('split4', 'split_tanh', 4),
    ('split8', 'split_tanh', 8),
    ('split16', 'split_tanh', 16),
    ('split8id', 'split_identity', 8),  # within-substrate linear control
    ('shell8', 'shell_tanh', 8),        # cross-substrate (gated-delta) control
]
LENS = ['128', '256', '512', '1024', '2048']


def arm_spec2(kind, f):
    if kind == 'split_identity':
        from shapes import fracs_to_logits8
        lg = fracs_to_logits8({'gdn2_recall': 1.0 - f, 'e97_delta': f})
        return lg, ['--e97_state_nonlin', 'identity', '--use_chunked_e97_delta', '0']
    return arm_spec(kind, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='4,5,6,7')
    ap.add_argument('--seeds', default='0,1,2,3')
    ap.add_argument('--steps', type=int, default=4000)
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    from capability import derive_small_dim
    os.makedirs(OUT_DIR, exist_ok=True)
    dims = {}
    for name, kind, fh in ARMS:
        lg, _ = arm_spec2(kind, fh / 64.0)
        dims[name] = derive_small_dim(lg)
    log(f'sharpen dims={dims} seeds={seeds}')

    jobs = [(name, kind, fh, s) for (name, kind, fh) in ARMS for s in seeds]
    log(f'{len(jobs)} sharpen jobs')
    queue = list(jobs); running = {}
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
        while queue and free:
            name, kind, fh, s = queue.pop(0); gpu = free.pop(0)
            lg, extra = arm_spec2(kind, fh / 64.0)
            cmd, label = build_cmd(f'shp_{name}', lg, extra, 'modular_quadratic', s, args.steps, dims[name])
            lf = open(os.path.join(OUT_DIR, label + '.log'), 'w')
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = (proc, label, lf, time.time())
            log(f'LAUNCH {label} GPU{gpu}')
        time.sleep(8)
        for gpu, job in list(running.items()):
            proc, label, lf, t0 = job
            if proc.poll() is None and (time.time() - t0) < 2400:
                continue
            if proc.poll() is None:
                proc.kill(); log(f'TIMEOUT {label}')
            lf.close(); log(f'DONE {label}'); del running[gpu]

    # aggregate cliff metric
    import glob
    summary = {}
    for name, kind, fh in ARMS:
        perlen = {l: [] for l in LENS}; means = []
        for f in glob.glob(os.path.join(OUT_DIR, f'cap_shp_{name}__modular_quadratic__seed*.json')):
            d = json.load(open(f)); le = d.get('length_extrap', {})
            accs = [le[l]['acc'] for l in LENS if le.get(l)]
            if accs:
                means.append(float(np.mean(accs)))
            for l in LENS:
                if le.get(l):
                    perlen[l].append(le[l]['acc'])
        pl = {l: (round(float(np.mean(perlen[l])), 4) if perlen[l] else None) for l in LENS}
        t128, t2048 = pl.get('128'), pl.get('2048')
        summary[name] = dict(kind=kind, frac=fh, n_seed=len(means),
                             mean_acc=round(float(np.mean(means)), 4) if means else None,
                             per_len=pl, cliff_T2048=t2048,
                             retention=(round(t2048 / t128, 4) if t128 and t2048 else None))
    out = dict(task='e97-hetero-cma-cap-sharpen', modquad_p=MODQUAD_P, eval_lengths=LENS,
               seeds=seeds, steps=args.steps, dims=dims, summary=summary,
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(os.path.join(RESULTS, 'capability_sharpen.json'), 'w'), indent=2)
    log('=== SHARPENED DEPTH CLIFF (4 seeds) ===')
    for name, kind, fh in ARMS:
        s = summary[name]
        pls = ' '.join(str(s['per_len'][l]) for l in LENS)
        log(f"  {name:9s} f={fh:2d}/64 ({kind:14s}): mean={s['mean_acc']} "
            f"T2048={s['cliff_T2048']} retention={s['retention']} n={s['n_seed']} "
            f"| per-len {pls}")
    log('WROTE capability_sharpen.json')


if __name__ == '__main__':
    main()
