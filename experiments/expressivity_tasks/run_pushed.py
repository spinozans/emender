"""Pushed task suite — harder variants where theory predicts E88 should win
more decisively than at the easy K/seq settings.

Designed to chain after the basic run_suite.py once the easier configs
have produced data.

Variants:
  - parity at seq_len 512, 1024, 2048 (longer = harder for non-nonlinear)
  - modular_counter at K=20 (forces real counter)
  - fsm_tracking at n_states=8, 16 (more states than dim/seq can table-lookup)
"""
import os, sys, json, argparse, subprocess, time

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS))

MODELS = {
    'E88_n16':  ['--model', 'E88', '--n_heads', '4', '--n_state', '16'],
    'E88_n32':  ['--model', 'E88', '--n_heads', '4', '--n_state', '32'],
    'fla-gdn':  ['--model', 'fla-gdn', '--n_heads', '4', '--expansion', '2'],
    'llama':    ['--model', 'llama', '--n_heads', '4', '--expansion', '4'],
}

# (label, base_task, K, seq_len, steps)
PUSHED = [
    ('parity_T512',     'parity',          2,   512, 3000),
    ('parity_T1024',    'parity',          2,  1024, 3000),
    ('parity_T2048',    'parity',          2,  2048, 3000),
    ('modcount_K20',    'modular_counter', 20,  256, 4000),
    ('modcount_K50',    'modular_counter', 50,  256, 4000),
    ('fsm_S8',          'fsm_tracking',    8,   256, 3000),
    ('fsm_S16',         'fsm_tracking',    16,  256, 3000),
]


def run_one(variant_label, task, K, seq_len, steps, model_label, model_args, seed, output_dir):
    label = f'{variant_label}__{model_label}__seed{seed}'
    out_path = os.path.join(output_dir, f'{label}.json')
    if os.path.exists(out_path):
        print(f"[skip] {label}", flush=True)
        return None
    cmd = ['python', os.path.join(THIS, 'train_task.py'),
           '--task', task,
           *model_args,
           '--dim', '128', '--depth', '4',
           '--steps', str(steps), '--seq_len', str(seq_len),
           '--batch_size', '32', '--lr', '3e-4',
           '--K', str(K), '--seed', str(seed),
           '--label', label, '--output_dir', output_dir]
    t0 = time.time()
    print(f"[run ] {label}  T={seq_len} K={K}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL] {label}  ({elapsed:.0f}s)\n  {proc.stderr[-300:]}", flush=True)
        return None
    try:
        result = json.load(open(out_path))
        print(f"[done] {label}  acc={result['final_acc']:.4f}  ({elapsed:.0f}s)", flush=True)
        return result
    except Exception as e:
        print(f"[FAIL] {label} read error: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--output_dir', default=os.path.join(THIS, 'results'))
    ap.add_argument('--variants', nargs='+', default=None)
    args = ap.parse_args()

    variants = PUSHED if args.variants is None else [v for v in PUSHED if v[0] in args.variants]
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    for variant_label, base_task, K, seq_len, steps in variants:
        for model_label, model_args in MODELS.items():
            for seed in args.seeds:
                run_one(variant_label, base_task, K, seq_len, steps,
                        model_label, model_args, seed, args.output_dir)
    print(f"\n=== Pushed suite complete ({time.time()-t0:.0f}s) ===")


if __name__ == '__main__':
    main()
