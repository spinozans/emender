"""e97-hetero-cma: CAPABILITY vs nonlinear-head FRACTION (the knee).

Sweeps the depth-growing capability against how many heads carry the SPLIT-EDIT
per-step-tanh nonlinear state map, on the SEPARATING substrate from phi-explore
(modular_quadratic, length-extrapolation to 16x). Finds the MINIMAL nonlinear
fraction that maxes the depth capability — the capability half of the
throughput/capability tradeoff curve.

Probes (length-extrap mean acc, train T=128 -> eval up to 2048):
  depth  = modular_quadratic   (mod 48; THE depth cliff — phi-explore separator)
  count  = anbncn_viability    (bounded-state counting; secondary)
  recall = mqar_recall         (delta memory in gdn-neg heads; CONTROL, ~flat)

Arms:
  linear           : 100% gdn-neg (frac 0) — the depth-cliff floor.
  split_tanh @ f   : (1-f) gdn-neg + f e97_delta(split-edit, per-step tanh),
                     f in {1,2,4,8,16}/64 — the capability-vs-fraction curve.
  shell_tanh @ 8/64: (1-f) gdn-neg + f gdn2_nonlin_shell(tanh, per-step chunk=1)
                     — SUBSTRATE CONTROL: phi-explore says bounded phi is INERT on
                     gated-delta, so this should stay near the linear floor while
                     split_tanh climbs. Proves the substrate (not just "a
                     nonlinear head") is what unlocks the depth capability.

REAL tasks, real training (train_hybrid.py). No mocks.
"""
import os, sys, json, time, subprocess, datetime, argparse
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
_DELTA = os.path.join(_ROOT, 'experiments', 'e97_delta_1p3b_cma')
for p in (_THIS, _DELTA, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
from shapes import fracs_to_logits8
from capability import (derive_small_dim, probe_score, gpu_used_mib, EVAL_TS,
                        DEPTH, N_HEADS, N_STATE, TRAIN_HYBRID, FREE_MEM_MIB)

RESULTS = os.path.join(_THIS, 'results')
OUT_DIR = os.path.join(RESULTS, 'cap_runs')
PROBES = {'depth': 'modular_quadratic', 'count': 'anbncn_viability',
          'recall': 'mqar_recall'}
EVAL_LENS = [128, 256, 512, 1024, 2048]   # 16x length extrapolation (phi-explore)
MODQUAD_P = 48


def log(m): print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def arm_spec(kind, f):
    """Return (head_type_logits, extra_args) for an arm."""
    if kind == 'linear':
        return fracs_to_logits8({'gdn2_recall': 1.0}), []
    if kind == 'split_tanh':
        lg = fracs_to_logits8({'gdn2_recall': 1.0 - f, 'e97_delta': f})
        return lg, ['--e97_state_nonlin', 'tanh', '--use_chunked_e97_delta', '0']
    if kind == 'shell_tanh':
        lg = fracs_to_logits8({'gdn2_recall': 1.0 - f, 'gdn2_nonlin_shell': f})
        return lg, ['--shell_state_nonlin', 'tanh', '--shell_state_chunk', '1']
    raise ValueError(kind)


def build_cmd(name, logits, extra, probe, seed, steps, dim):
    label = f'cap_{name}__{probe}__seed{seed}'
    cmd = [sys.executable, TRAIN_HYBRID, '--task', probe,
           '--layer_pattern', 'typed-gdn2', '--dim', str(dim), '--depth', str(DEPTH),
           '--n_heads', str(N_HEADS), '--n_state', str(N_STATE), '--lr', '3e-4',
           '--lam_max', '1.585', '--beta_max', '2.747', '--gdn_allow_neg_eigval', '1',
           '--head_type_logits=' + ','.join(str(x) for x in logits),
           '--steps', str(steps), '--seq_len', '128', '--batch_size', '32',
           '--optimizer', 'schedulefree', '--seed', str(seed),
           '--label', label, '--output_dir', OUT_DIR,
           '--eval_lengths', *[str(t) for t in EVAL_LENS], '--eval_lengths_n_batches', '8']
    if probe == 'modular_quadratic':
        cmd += ['--K', str(MODQUAD_P)]
    return cmd + extra, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='4,5,6,7')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--fracs', default='1,2,4,8,16')  # split_tanh head counts /64
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    fracs = [int(x) / 64.0 for x in args.fracs.split(',')]
    os.makedirs(OUT_DIR, exist_ok=True)

    # arms: (name, kind, frac). depth cliff = the full fraction curve; count/recall
    # are spot controls at frac 0 and 8/64.
    arms = [('linear', 'linear', 0.0)]
    for f in fracs:
        arms.append((f'split{int(round(f*64))}', 'split_tanh', f))
    arms.append(('shell8', 'shell_tanh', 8 / 64.0))   # substrate control

    dims = {}
    for name, kind, f in arms:
        lg, _ = arm_spec(kind, f)
        dims[name] = derive_small_dim(lg)
    log(f'arms={[a[0] for a in arms]} small dims={dims}')

    # job matrix: depth cliff over ALL arms x seeds; count/recall only on
    # linear + split8 (controls).
    jobs = []
    for name, kind, f in arms:
        for pkey, probe in PROBES.items():
            if pkey != 'depth' and name not in ('linear', 'split8'):
                continue
            seedset = seeds if pkey == 'depth' else seeds[:1]
            for s in seedset:
                jobs.append((name, kind, f, pkey, probe, s))
    log(f'{len(jobs)} capability jobs')

    queue = list(jobs); running = {}; raw = {}
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
        while queue and free:
            name, kind, f, pkey, probe, s = queue.pop(0); gpu = free.pop(0)
            lg, extra = arm_spec(kind, f)
            cmd, label = build_cmd(name, lg, extra, probe, s, args.steps, dims[name])
            lf = open(os.path.join(OUT_DIR, label + '.log'), 'w')
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = (proc, name, pkey, s, label, lf, time.time())
            log(f'LAUNCH {label} on GPU {gpu}')
        time.sleep(8)
        for gpu, job in list(running.items()):
            proc, name, pkey, s, label, lf, t0 = job
            if proc.poll() is None and (time.time() - t0) < 2400:
                continue
            if proc.poll() is None:
                proc.kill(); log(f'TIMEOUT {label}')
            lf.close()
            sc = probe_score(OUT_DIR, label)
            raw.setdefault(name, {}).setdefault(pkey, {})[s] = sc
            log(f'DONE {label} mean_acc={sc.get("mean_acc") if sc else None}')
            del running[gpu]

    summary = {}
    for name, kind, f in arms:
        summary[name] = dict(kind=kind, frac=int(round(f * 64)))
        for pkey in PROBES:
            vals = [raw[name][pkey][s]['mean_acc'] for s in seeds
                    if raw.get(name, {}).get(pkey, {}).get(s)
                    and raw[name][pkey][s].get('mean_acc') is not None]
            summary[name][pkey] = round(float(np.mean(vals)), 4) if vals else None
    out = dict(task='e97-hetero-cma-capability', probes=PROBES, modquad_p=MODQUAD_P,
               eval_lengths=EVAL_LENS, depth=DEPTH, n_heads=N_HEADS, n_state=N_STATE,
               dims=dims, seeds=seeds, steps=args.steps, summary=summary, raw=raw,
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(os.path.join(RESULTS, 'capability.json'), 'w'), indent=2)
    log('=== CAPABILITY vs FRACTION (mean length-extrap acc) ===')
    for name, kind, f in arms:
        log(f'  {name:10s} ({kind} f={int(round(f*64))}/64): ' +
            ' '.join(f'{k}={summary[name].get(k)}' for k in PROBES))
    log('WROTE capability.json')


if __name__ == '__main__':
    main()
