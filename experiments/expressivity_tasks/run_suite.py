"""Run the full (task × model × seed) suite serially on a single GPU,
or pass a GPU id explicitly. Records each run as a JSON file in results/.

Usage:
    CUDA_VISIBLE_DEVICES=7 python run_suite.py --tasks all --models all --seeds 42 123 456
    CUDA_VISIBLE_DEVICES=7 python run_suite.py --tasks parity dyck --models E88 fla-gdn
"""
import os, sys, json, argparse, subprocess, time

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS))

ALL_TASKS = [
    'parity', 'modular_counter', 'dyck', 'fsm_tracking', 'selective_copy',
    'assoc_recall', 'overwrite_recall', 'reset_recall', 'keyed_fsm_memory',
]

# Model → (level_name, model-specific kwargs)
MODELS = {
    'E88_n16':       {'level': 'E88',           'kwargs': {'n_heads': 4, 'n_state': 16}},
    'E88_n32':       {'level': 'E88',           'kwargs': {'n_heads': 4, 'n_state': 32}},
    'E93':           {'level': 'E93',           'kwargs': {'n_state': 16}},
    'E93_no_decay':  {'level': 'E93a_no_decay', 'kwargs': {'n_state': 16}},
    'm2rnn_tied':    {'level': 'm2rnn',         'kwargs': {'n_heads': 4, 'n_state': 16}},
    'm2rnn_paper':   {'level': 'm2rnn-paper',   'kwargs': {'n_heads': 4, 'n_state': 16}},
    'fla-gdn':       {'level': 'fla-gdn',       'kwargs': {'n_heads': 4, 'expansion': 2}},
    'llama':         {'level': 'llama',         'kwargs': {'n_heads': 4, 'expansion': 4}},
    'mamba2':        {'level': 'mamba2',        'kwargs': {}},
}

# Per-task config (steps, seq_len, K, lr); tuned for "should-be-solvable in K steps"
TASK_CONFIG = {
    'parity':           {'steps': 2000, 'seq_len': 128, 'K': 2,  'lr': 3e-4},
    'modular_counter':  {'steps': 3000, 'seq_len': 128, 'K': 5,  'lr': 3e-4},
    'dyck':             {'steps': 3000, 'seq_len': 256, 'K': 8,  'lr': 3e-4},
    'fsm_tracking':     {'steps': 3000, 'seq_len': 256, 'K': 4,  'lr': 3e-4},
    'selective_copy':   {'steps': 4000, 'seq_len': 256, 'K': 8,  'lr': 3e-4},
    'assoc_recall':     {'steps': 4000, 'seq_len': 64,  'K': 8,  'lr': 3e-4},
    'overwrite_recall': {'steps': 4000, 'seq_len': 64,  'K': 16, 'lr': 3e-4},
    'reset_recall':     {'steps': 4000, 'seq_len': 64,  'K': 16, 'lr': 3e-4},
    'keyed_fsm_memory': {'steps': 6000, 'seq_len': 128, 'K': 8,  'lr': 3e-4},
}


def run_one(task, model_label, seed, dim, depth, output_dir):
    cfg = TASK_CONFIG[task]
    model = MODELS[model_label]
    label = f'{task}__{model_label}__seed{seed}'
    cmd = [
        'python', os.path.join(THIS, 'train_task.py'),
        '--task', task,
        '--model', model['level'],
        '--dim', str(dim),
        '--depth', str(depth),
        '--steps', str(cfg['steps']),
        '--seq_len', str(cfg['seq_len']),
        '--batch_size', '32',
        '--lr', str(cfg['lr']),
        '--K', str(cfg['K']),
        '--seed', str(seed),
        '--label', label,
        '--output_dir', output_dir,
    ]
    for k, v in model['kwargs'].items():
        cmd.extend([f'--{k}', str(v)])

    out_path = os.path.join(output_dir, f'{label}.json')
    if os.path.exists(out_path):
        print(f"[skip] {label} exists", flush=True)
        return label, None

    t0 = time.time()
    print(f"[run ] {label}  ({' '.join(cmd[2:])})", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL] {label}  ({elapsed:.0f}s)\nstderr: {proc.stderr[-500:]}", flush=True)
        return label, None
    try:
        result = json.load(open(out_path))
        print(f"[done] {label}  acc={result['final_acc']:.4f}  ({elapsed:.0f}s)", flush=True)
        return label, result
    except Exception as e:
        print(f"[FAIL] {label} couldn't read result: {e}", flush=True)
        return label, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['all'])
    ap.add_argument('--models', nargs='+', default=['all'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--output_dir', default=os.path.join(THIS, 'results'))
    args = ap.parse_args()

    tasks = ALL_TASKS if args.tasks == ['all'] else args.tasks
    models = list(MODELS.keys()) if args.models == ['all'] else args.models

    os.makedirs(args.output_dir, exist_ok=True)

    summary = []
    t0 = time.time()
    for task in tasks:
        for model in models:
            for seed in args.seeds:
                label, result = run_one(task, model, seed, args.dim, args.depth, args.output_dir)
                if result:
                    summary.append({
                        'task': task, 'model': model, 'seed': seed,
                        'final_acc': result['final_acc'],
                        'final_loss': result['final_loss'],
                        'random_baseline': result.get('random_baseline_acc', 0),
                        'params': result['params'],
                    })

    print(f"\n=== SUITE COMPLETE ({time.time()-t0:.0f}s) ===")
    # Aggregate
    print(f"\n{'task':>18s}  {'model':>10s}  {'mean_acc':>9s}  {'min':>6s}  {'max':>6s}  {'baseline':>8s}")
    by_task_model = {}
    for s in summary:
        key = (s['task'], s['model'])
        by_task_model.setdefault(key, []).append(s['final_acc'])
    for (task, model), accs in sorted(by_task_model.items()):
        baseline = next((s['random_baseline'] for s in summary if s['task'] == task), 0)
        accs = sorted(accs)
        print(f"  {task:>18s}  {model:>10s}  {sum(accs)/len(accs):>9.4f}  {accs[0]:>6.4f}  {accs[-1]:>6.4f}  {baseline:>8.4f}")

    json.dump(summary, open(os.path.join(args.output_dir, 'suite_summary.json'), 'w'), indent=2)
    print(f"\nSummary saved to {os.path.join(args.output_dir, 'suite_summary.json')}")


if __name__ == '__main__':
    main()
