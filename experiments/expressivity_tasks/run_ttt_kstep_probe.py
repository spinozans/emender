"""K-step inner-optimization probe for `refit` (task ttt-capability).

The main battery toggles MOMENTUM (refit-mom vs refit-del) and finds a null.
This probe closes the other half of the "K-step/momentum inner optimization"
question: does the inner-optimizer STEP COUNT K (newton_steps) — i.e. solving the
per-chunk inner reconstruction more accurately — unlock anything? Run on the two
tasks with the most capability spread in the main battery (modular_counter K=5,
modular_quadratic p=64), momentum ON, K in {1, 5, exact}, 3 seeds.

Broker-aware (reads leased CUDA_VISIBLE_DEVICES). Resumable.
"""
from __future__ import annotations
import os, subprocess, time
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
OUT = THIS / 'results_ttt_capability'

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0',
          '--mlp_ratio', '2.0', '--depth', '4', '--disable_autocast',
          '--gdn_allow_neg_eigval', '1', '--optimizer', 'schedulefree']
REFIT = '0,0,0,0,0,0,0,0,30'

# (task, K-task-arg, steps, eval_lengths)
TASKS = [
    ('modular_counter', 5, 8000, ['128', '256', '512']),
    ('modular_quadratic', 64, 8000, ['128', '256', '512']),
]
# inner-step arm: label-tag -> newton_steps value ('exact' => omit flag)
KSTEPS = {'k1': 1, 'k5': 5, 'kexact': None}
SEEDS = [42, 123, 456]
LR = '5e-4'


def cmd(task, ktask, steps, evals, ktag, ksteps, seed):
    label = f"ttt_{task}_K{ktask}__refit-mom-{ktag}__seed{seed}"
    c = ['python', str(THIS / 'train_hybrid.py'), '--task', task,
         '--layer_pattern', 'typed-gdn2', 'typed-gdn2', 'typed-gdn2', 'typed-gdn2',
         *SHARED, '--head_type_logits', REFIT, '--refit_has_mom', '1',
         '--steps', str(steps), '--seq_len', '128', '--batch_size', '32', '--lr', LR,
         '--seed', str(seed), '--label', label, '--output_dir', str(OUT),
         '--eval_lengths', *evals, '--eval_lengths_n_batches', '8', '--K', str(ktask)]
    if ksteps is not None:
        c += ['--refit_newton_steps', str(ksteps)]
    return label, c


def main():
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit("lease GPUs first: eval \"$(scripts/gpu_lease.sh N)\"")
    gpus = [g for g in cvd.split(',') if g]
    slots = [g for g in gpus for _ in range(2)]
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    for task, ktask, steps, evals in TASKS:
        for ktag, ksteps in KSTEPS.items():
            for seed in SEEDS:
                label, c = cmd(task, ktask, steps, evals, ktag, ksteps, seed)
                if (OUT / f'{label}.json').exists():
                    print(f"[skip] {label}"); continue
                jobs.append((label, c))
    print(f"GPUs {gpus} slots {len(slots)} jobs {len(jobs)}", flush=True)

    running, t0 = {}, time.time()
    while jobs or running:
        for si in range(len(slots)):
            if si not in running and jobs:
                label, c = jobs.pop(0)
                env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(slots[si])
                lf = open(OUT / f'{label}.log', 'w')
                running[si] = (subprocess.Popen(c, stdout=lf, stderr=subprocess.STDOUT,
                                                env=env, cwd=str(ROOT)), label, lf)
                print(f"[launch gpu{slots[si]}] {label}", flush=True)
        time.sleep(5)
        for si in list(running):
            p, label, lf = running[si]
            if p.poll() is not None:
                lf.close()
                ok = (OUT / f'{label}.json').exists()
                print(f"[{'ok' if ok else 'FAIL'}] {label} ({time.time()-t0:.0f}s)", flush=True)
                del running[si]
    print(f"K-probe complete in {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
