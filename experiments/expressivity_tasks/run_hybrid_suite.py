"""Run hybrid (E88+FLA-GDN) and pure-architecture controls on the full task suite.

Spreads runs across multiple GPUs in parallel using subprocess.

Usage:
    python run_hybrid_suite.py --gpus 0 1 2 3 4 5 7 --seeds 42 123 456
"""
import os, sys, json, argparse, subprocess, time
import multiprocessing as mp

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS))

# Each entry: (label, layer_pattern, model_kwargs)
PATTERNS = {
    'pure_E88':       (['E88'], {}),
    'pure_FLA':       (['fla-gdn'], {}),
    'pure_M2RNN':     (['m2rnn'], {}),
    'pure_M2RNN_paper': (['m2rnn-paper'], {}),
    'hybrid_AB':      (['E88', 'fla-gdn'], {}),       # alternating
    'hybrid_AABB':    (['E88', 'E88', 'fla-gdn', 'fla-gdn'], {}),
    'hybrid_BABA':    (['fla-gdn', 'E88'], {}),       # alternating, FLA first
    'hybrid_GDN_M2RNN_single': (['fla-gdn', 'fla-gdn', 'fla-gdn', 'm2rnn-paper'], {}),
    'hybrid_GDN_E88_single':   (['fla-gdn', 'fla-gdn', 'fla-gdn', 'E88'], {}),
}

# Per-task config (steps, seq_len, K, lr)
TASK_CONFIG = {
    'parity':           {'steps': 2000, 'seq_len': 128, 'K': 2,  'lr': 3e-4},
    'modular_counter':  {'steps': 3000, 'seq_len': 128, 'K': 5,  'lr': 3e-4},
    'fsm_tracking':     {'steps': 3000, 'seq_len': 256, 'K': 4,  'lr': 3e-4},
    'selective_copy':   {'steps': 4000, 'seq_len': 256, 'K': 8,  'lr': 3e-4},
    'assoc_recall':     {'steps': 4000, 'seq_len': 64,  'K': 8,  'lr': 3e-4},
    'dyck':             {'steps': 3000, 'seq_len': 256, 'K': 8,  'lr': 3e-4},
}


def run_one(args_tuple):
    gpu, label, layer_pattern, kwargs, task, seed, output_dir = args_tuple
    cfg = TASK_CONFIG[task]
    full_label = f'hyb_{label}__{task}__seed{seed}'
    out_path = os.path.join(output_dir, f'{full_label}.json')
    if os.path.exists(out_path):
        return f"[skip] {full_label}"
    cmd = ['python', os.path.join(THIS, 'train_hybrid.py'),
           '--task', task, '--layer_pattern', *layer_pattern,
           '--dim', '128', '--depth', '4',
           '--n_heads', '4', '--n_state', '16',
           '--steps', str(cfg['steps']), '--seq_len', str(cfg['seq_len']),
           '--batch_size', '32', '--lr', str(cfg['lr']),
           '--K', str(cfg['K']), '--seed', str(seed),
           '--label', full_label, '--output_dir', output_dir]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return f"[FAIL gpu={gpu}] {full_label} ({elapsed:.0f}s) err: {proc.stderr[-200:]}"
    try:
        result = json.load(open(out_path))
        return f"[done gpu={gpu}] {full_label}  acc={result['final_acc']:.4f}  ({elapsed:.0f}s)"
    except Exception as e:
        return f"[ERR gpu={gpu}] {full_label} read err: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--patterns', nargs='+', default=None,
                    help='subset of pattern keys to run')
    ap.add_argument('--tasks', nargs='+', default=None,
                    help='subset of tasks to run (default: all)')
    ap.add_argument('--output_dir', default=os.path.join(THIS, 'results'))
    args = ap.parse_args()

    patterns = list(PATTERNS.keys()) if args.patterns is None else args.patterns
    tasks = list(TASK_CONFIG.keys()) if args.tasks is None else args.tasks
    os.makedirs(args.output_dir, exist_ok=True)

    # Build job list
    jobs = []
    for pattern_label in patterns:
        layer_pattern, kw = PATTERNS[pattern_label]
        for task in tasks:
            for seed in args.seeds:
                jobs.append((pattern_label, layer_pattern, kw, task, seed))

    print(f"Total jobs: {len(jobs)}, GPUs: {args.gpus}, seeds: {args.seeds}")

    # Round-robin GPU assignment, run in parallel pool of size len(gpus)
    pool_size = len(args.gpus)
    job_args = []
    for i, (pl, lp, kw, task, seed) in enumerate(jobs):
        gpu = args.gpus[i % pool_size]
        job_args.append((gpu, pl, lp, kw, task, seed, args.output_dir))

    t0 = time.time()
    with mp.Pool(pool_size) as pool:
        for result in pool.imap_unordered(run_one, job_args):
            print(result, flush=True)

    print(f"\n=== ALL DONE ({time.time()-t0:.0f}s) ===")


if __name__ == '__main__':
    main()
