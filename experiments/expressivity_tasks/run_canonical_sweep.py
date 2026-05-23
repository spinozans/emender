"""Canonical paper sweep: dim=384 H=32 N=32 sf-AdamW 10K steps.

All 6 tasks × 3 patterns × 3 seeds = 54 jobs.
This is the configuration that gave clean grokking on modular_counter K=5
(0.97 by step 9000, smooth no oscillation).

Usage: python run_canonical_sweep.py --gpus 3 5 7
"""
import os, sys, json, argparse, subprocess, time
import multiprocessing as mp

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS))

PATTERNS = {
    'pure_E88':       (['E88'], {}),
    'pure_FLA':       (['fla-gdn'], {}),
    'hybrid_AABB':    (['E88', 'E88', 'fla-gdn', 'fla-gdn'], {}),
    'pure_M2RNN':     (['m2rnn'], {}),
    'pure_M2RNN_paper': (['m2rnn-paper'], {}),
    'hybrid_GDN_M2RNN_single': (['fla-gdn', 'fla-gdn', 'fla-gdn', 'm2rnn-paper'], {}),
    'hybrid_GDN_E88_single':   (['fla-gdn', 'fla-gdn', 'fla-gdn', 'E88'], {}),
    # Mamba-2 + Transformer not in HybridLadderLM (separate LM classes); cite Délétang/Grazzi numbers
}

# Per-task config (steps, seq_len, K, lr)
TASK_CONFIG = {
    'parity':           {'steps': 10000, 'seq_len': 128, 'K': 2,  'lr': 3e-4},
    'modular_counter':  {'steps': 10000, 'seq_len': 128, 'K': 5,  'lr': 3e-4},
    'fsm_tracking':     {'steps': 10000, 'seq_len': 256, 'K': 4,  'lr': 3e-4},
    'selective_copy':   {'steps': 10000, 'seq_len': 256, 'K': 8,  'lr': 3e-4},
    'assoc_recall':     {'steps': 10000, 'seq_len': 64,  'K': 8,  'lr': 3e-4},
    'overwrite_recall': {'steps': 10000, 'seq_len': 128, 'K': 16, 'lr': 3e-4},
    'reset_recall':     {'steps': 10000, 'seq_len': 128, 'K': 16, 'lr': 3e-4},
    'dyck':             {'steps': 10000, 'seq_len': 256, 'K': 8,  'lr': 3e-4},
    'keyed_fsm_memory': {'steps': 10000, 'seq_len': 128, 'K': 8,  'lr': 3e-4},
}


def run_one(args_tuple):
    gpu, label, layer_pattern, kwargs, task, seed, output_dir = args_tuple
    cfg = TASK_CONFIG[task]
    full_label = f'canon_{label}__{task}__seed{seed}'
    out_path = os.path.join(output_dir, f'{full_label}.json')
    if os.path.exists(out_path):
        return f"[skip] {full_label}"
    cmd = ['python', os.path.join(THIS, 'train_hybrid.py'),
           '--task', task, '--layer_pattern', *layer_pattern,
           '--dim', '384', '--depth', '4',
           '--n_heads', '32', '--n_state', '32',
           '--steps', str(cfg['steps']), '--seq_len', str(cfg['seq_len']),
           '--batch_size', '32', '--lr', str(cfg['lr']),
           '--K', str(cfg['K']), '--seed', str(seed),
           '--optimizer', 'schedulefree',
           '--label', full_label, '--output_dir', output_dir]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    dt = time.time() - t0
    if p.returncode != 0:
        return f"[fail gpu={gpu}] {full_label}: {p.stderr[-500:]}"
    try:
        with open(out_path) as f:
            d = json.load(f)
        acc = d.get('final_acc', d.get('final_accuracy'))
        return f"[done gpu={gpu}] {full_label}  acc={acc:.4f}  ({dt:.0f}s)" if acc is not None else f"[done gpu={gpu}] {full_label}  ({dt:.0f}s)"
    except Exception as e:
        return f"[done gpu={gpu}] {full_label}  ({dt:.0f}s)  parse_err={e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, nargs='+', default=[3, 5, 7])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--patterns', type=str, nargs='+', default=['pure_E88', 'pure_FLA', 'hybrid_AABB'])
    ap.add_argument('--tasks', type=str, nargs='+', default=list(TASK_CONFIG.keys()))
    ap.add_argument('--output_dir', type=str, default=os.path.join(THIS, 'results'))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Order: hardest tasks first so we get critical answers fastest
    PRIORITY = [
        'keyed_fsm_memory', 'overwrite_recall', 'reset_recall',
        'modular_counter', 'fsm_tracking', 'dyck', 'parity',
        'assoc_recall', 'selective_copy',
    ]
    task_order = [t for t in PRIORITY if t in args.tasks] + [t for t in args.tasks if t not in PRIORITY]

    jobs = []
    for task in task_order:
        for pattern in args.patterns:
            layer_pattern, kwargs = PATTERNS[pattern]
            for seed in args.seeds:
                jobs.append((pattern, layer_pattern, kwargs, task, seed))

    print(f"Total jobs: {len(jobs)}, GPUs: {args.gpus}, dim=384 H=32 N=32 sf-AdamW 10K steps")
    print(f"Task order: {task_order}")

    # Round-robin GPU assignment
    gpu_jobs = []
    for i, job in enumerate(jobs):
        gpu = args.gpus[i % len(args.gpus)]
        gpu_jobs.append((gpu, *job, args.output_dir))

    with mp.Pool(len(args.gpus)) as pool:
        for result in pool.imap_unordered(run_one, gpu_jobs):
            print(result, flush=True)

    print("=== Canonical sweep complete ===")


if __name__ == '__main__':
    main()
