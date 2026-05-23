"""Head-count sweep on expressivity tasks.

Tests whether modular_counter K=5, fsm_tracking, etc. are head-count-bottlenecked
by sweeping n_heads = {4, 8, 16, 32} for pure E88, pure FLA, hybrid AABB.

For a K-state task, theory predicts E88 needs n_heads >= K to express the K-cycle.

Usage:
    python run_heads_sweep.py --gpus 3 5 7 --seeds 42 123 456 \
        --heads 8 16 --patterns pure_E88 pure_FLA hybrid_AABB
"""
import os, sys, json, argparse, subprocess, time
import multiprocessing as mp

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS))

PATTERNS = {
    'pure_E88':       (['E88'], {}),
    'pure_FLA':       (['fla-gdn'], {}),
    'hybrid_AB':      (['E88', 'fla-gdn'], {}),
    'hybrid_AABB':    (['E88', 'E88', 'fla-gdn', 'fla-gdn'], {}),
}

TASK_CONFIG = {
    'parity':           {'steps': 2000, 'seq_len': 128, 'K': 2,  'lr': 3e-4},
    'modular_counter':  {'steps': 3000, 'seq_len': 128, 'K': 5,  'lr': 3e-4},
    'fsm_tracking':     {'steps': 3000, 'seq_len': 256, 'K': 4,  'lr': 3e-4},
    'selective_copy':   {'steps': 4000, 'seq_len': 256, 'K': 8,  'lr': 3e-4},
    'assoc_recall':     {'steps': 4000, 'seq_len': 64,  'K': 8,  'lr': 3e-4},
    'dyck':             {'steps': 3000, 'seq_len': 256, 'K': 8,  'lr': 3e-4},
}


def run_one(args_tuple):
    gpu, label, layer_pattern, kwargs, n_heads, n_state, task, seed, output_dir = args_tuple
    cfg = TASK_CONFIG[task]
    full_label = f'heads_{label}_H{n_heads}_N{n_state}__{task}__seed{seed}'
    out_path = os.path.join(output_dir, f'{full_label}.json')
    if os.path.exists(out_path):
        return f"[skip] {full_label}"
    cmd = ['python', os.path.join(THIS, 'train_hybrid.py'),
           '--task', task, '--layer_pattern', *layer_pattern,
           '--dim', '128', '--depth', '4',
           '--n_heads', str(n_heads), '--n_state', str(n_state),
           '--steps', str(cfg['steps']), '--seq_len', str(cfg['seq_len']),
           '--batch_size', '32', '--lr', str(cfg['lr']),
           '--K', str(cfg['K']), '--seed', str(seed),
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
        acc = d.get('final_accuracy', d.get('accuracy', d.get('best_accuracy', None)))
        return f"[done gpu={gpu}] {full_label}  acc={acc:.4f}  ({dt:.0f}s)" if acc is not None else f"[done gpu={gpu}] {full_label}  (no acc)"
    except Exception as e:
        return f"[done gpu={gpu}] {full_label}  ({dt:.0f}s)  parse_err={e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 7])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--heads', type=int, nargs='+', default=[8, 16, 32])
    ap.add_argument('--n_states', type=int, nargs='+', default=[16, 32],
                    help='List of n_state values (per-head matrix size)')
    ap.add_argument('--patterns', type=str, nargs='+', default=['pure_E88', 'pure_FLA', 'hybrid_AABB'])
    ap.add_argument('--tasks', type=str, nargs='+', default=list(TASK_CONFIG.keys()))
    ap.add_argument('--output_dir', type=str, default=os.path.join(THIS, 'results'))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Priority ordering: critical paper-test configs first.
    # Phase 1: smallest H value × all patterns × hardest tasks (modular_counter, fsm_tracking)
    # Phase 2: bigger H values for the hard tasks
    # Phase 3: all other tasks
    # This way an obvious story emerges in the first ~2 hours.
    PRIORITY_TASKS = ['modular_counter', 'fsm_tracking', 'parity', 'dyck', 'assoc_recall', 'selective_copy']
    jobs = []
    seen = set()
    # Phase 1: smallest H on hardest tasks
    smallest_H = min(args.heads)
    smallest_N = min(args.n_states)
    for task in PRIORITY_TASKS[:2]:  # modular_counter, fsm_tracking
        for pattern in args.patterns:
            layer_pattern, kwargs = PATTERNS[pattern]
            for seed in args.seeds:
                key = (pattern, smallest_H, smallest_N, task, seed)
                if key not in seen:
                    seen.add(key)
                    jobs.append((pattern, layer_pattern, kwargs, smallest_H, smallest_N, task, seed))
    # Phase 2: full sweep on hard tasks (modular_counter, fsm_tracking) at all H, N
    for task in PRIORITY_TASKS[:2]:
        for n_heads in args.heads:
            for n_state in args.n_states:
                for pattern in args.patterns:
                    layer_pattern, kwargs = PATTERNS[pattern]
                    for seed in args.seeds:
                        key = (pattern, n_heads, n_state, task, seed)
                        if key not in seen:
                            seen.add(key)
                            jobs.append((pattern, layer_pattern, kwargs, n_heads, n_state, task, seed))
    # Phase 3: remaining tasks at all H, N
    for task in PRIORITY_TASKS[2:]:
        for n_heads in args.heads:
            for n_state in args.n_states:
                for pattern in args.patterns:
                    layer_pattern, kwargs = PATTERNS[pattern]
                    for seed in args.seeds:
                        key = (pattern, n_heads, n_state, task, seed)
                        if key not in seen:
                            seen.add(key)
                            jobs.append((pattern, layer_pattern, kwargs, n_heads, n_state, task, seed))

    print(f"Total jobs: {len(jobs)}, GPUs: {args.gpus}, heads: {args.heads}, n_states: {args.n_states}, patterns: {args.patterns}")
    print(f"Phase 1 (~{len(args.patterns)*len(args.seeds)*2} jobs at smallest H): critical answers in ~2hr")
    print(f"Then Phase 2 hard-task full sweep, then Phase 3 other tasks")

    # Round-robin GPU assignment
    gpu_jobs = []
    for i, job in enumerate(jobs):
        gpu = args.gpus[i % len(args.gpus)]
        gpu_jobs.append((gpu, *job, args.output_dir))

    with mp.Pool(len(args.gpus)) as pool:
        for result in pool.imap_unordered(run_one, gpu_jobs):
            print(result, flush=True)

    print("=== Head sweep complete ===")


if __name__ == '__main__':
    main()
