"""e97-wallclock-cma: CAPABILITY GUARD (Erik's refinement 2026-06-08).

Chunk-length C is EXPRESSIVITY-RISKY: a larger C bounds the state with tanh less
often, so a pure wall-clock-BPB fitness would push C up for speed and SILENTLY
KILL the bounded-state capability (count / nonlinear-state) that is the whole
reason to run the tanh shell. This guard makes that trade-off VISIBLE: it runs
the canonical length-extrapolation probes on the within-layer mixture
(0.5 gdn-neg + 0.5 gdn2_nonlin_shell) at C in {1, 64, 2048} and reports, per C:

  count  = anbncn_viability        (bounded-state / counting -- C-SENSITIVE)
  nonlin = iterated_nonlinear_map  (nonlinear state         -- C-SENSITIVE)
  recall = mqar_recall             (delta memory, gdn-neg arm -- C-CONTROL, flat)

Reference arm gdn2_mlp = 100% gdn-neg (the baseline cell, no shell).

Claim under test: as C -> large, count/nonlin collapse toward the gdn2_mlp /
linear floor (the tanh win is bounded LESS often), while recall is ~flat
(it lives in the gdn-neg heads). If true, any "fast" large-C config is
capability-dead and must be REJECTED by the guarded fitness.

REAL tasks, real training (reuses expressivity_tasks/train_hybrid.py). No mocks.
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
OUT_DIR = os.path.join(RESULTS, 'cap_guard_runs')
# C-sensitive bounded-state probes + one delta-memory control
PROBES = {'count': 'anbncn_viability',
          'nonlin': 'iterated_nonlinear_map',
          'recall': 'mqar_recall'}


def log(m):
    print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def build_cmd(name, logits, probe, C, nonlin, seed, steps, dim):
    label = f'capg_{name}__{probe}__seed{seed}'
    cmd = [sys.executable, TRAIN_HYBRID, '--task', probe,
           '--layer_pattern', 'typed-gdn2', '--dim', str(dim), '--depth', str(DEPTH),
           '--n_heads', str(N_HEADS), '--n_state', str(N_STATE), '--lr', '3e-4',
           '--lam_max', '1.585', '--beta_max', '2.747', '--gdn_allow_neg_eigval', '1',
           '--head_type_logits=' + ','.join(str(f) for f in logits),
           '--steps', str(steps), '--seq_len', '128', '--batch_size', '32',
           '--optimizer', 'schedulefree', '--seed', str(seed),
           '--label', label, '--output_dir', OUT_DIR,
           '--eval_lengths', *[str(t) for t in EVAL_TS], '--eval_lengths_n_batches', '8']
    if nonlin is not None:
        cmd += ['--shell_state_nonlin', str(nonlin), '--shell_state_chunk', str(C)]
    return cmd, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--Cs', default='1,64,2048')
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    Cs = [int(c) for c in args.Cs.split(',')]
    os.makedirs(OUT_DIR, exist_ok=True)

    # arms: gdn2_mlp reference (no shell), then the 0.5/0.5 shell mix at each C
    mix_logits = fracs_to_logits8({'gdn2_recall': 0.5, 'gdn2_nonlin_shell': 0.5})
    arms = {'gdn2_mlp': (fracs_to_logits8({'gdn2_recall': 1.0}), None, None)}
    for C in Cs:
        arms[f'shell_C{C}'] = (mix_logits, C, 'tanh')
    dims = {name: derive_small_dim(lg) for name, (lg, _, _) in arms.items()}
    log(f'small dims (~4M param matched): {dims}')

    jobs = []
    for name, (lg, C, nl) in arms.items():
        for key, probe in PROBES.items():
            jobs.append((name, key, probe, lg, C, nl))
    log(f'{len(jobs)} guard jobs ({len(arms)} arms x {len(PROBES)} probes, seed {args.seed})')

    queue = list(jobs); running = {}; raw = {}
    while queue or running:
        used = gpu_used_mib()
        free = [g for g in gpus if used.get(g, 0) < FREE_MEM_MIB and g not in running]
        while queue and free:
            name, key, probe, lg, C, nl = queue.pop(0); gpu = free.pop(0)
            cmd, label = build_cmd(name, lg, probe, C, nl, args.seed, args.steps, dims[name])
            lf = open(os.path.join(OUT_DIR, label + '.log'), 'w')
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[gpu] = (proc, name, key, label, lf, time.time())
            log(f'LAUNCH {label} on GPU {gpu} (C={C})')
        time.sleep(8)
        for gpu, job in list(running.items()):
            proc, name, key, label, lf, t0 = job
            if proc.poll() is None and (time.time() - t0) < 1200:
                continue
            if proc.poll() is None:
                proc.kill()
            lf.close()
            sc = probe_score(OUT_DIR, label)
            raw.setdefault(name, {})[key] = sc
            log(f'DONE {label} mean_acc={sc.get("mean_acc") if sc else None}')
            del running[gpu]

    summary = {}
    for name in arms:
        summary[name] = {k: (round(raw[name][k]['mean_acc'], 4)
                             if raw.get(name, {}).get(k) and raw[name][k].get('mean_acc') is not None
                             else None) for k in PROBES}
    out = dict(task='e97-wallclock-cma-capability-guard', arms=list(arms.keys()),
               probes=PROBES, Cs=Cs, dims=dims, seed=args.seed, steps=args.steps,
               summary=summary, raw=raw,
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(os.path.join(RESULTS, 'capability_guard.json'), 'w'), indent=2)
    log('=== CAPABILITY GUARD (mean length-extrap acc; count/nonlin = C-sensitive) ===')
    for name in arms:
        log(f'  {name:14s}: ' + ' '.join(f'{k}={summary[name][k]}' for k in PROBES))
    log('WROTE capability_guard.json')


if __name__ == '__main__':
    main()
