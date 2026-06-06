"""redo-e99-1-3b: capability axis for the mixture anchors (idle-GPU-only).

Runs the SAME 6-probe length-extrapolation suite the typed-gdn-2-head batch used
(mqar_recall, s5_permutation, anbncn_viability, flag_hold_recall,
iterated_nonlinear_map, mixed_probe; depth4/n_heads48/n_state32, seq128, bs32,
schedule-free AdamW, eval T in {128,256,512,1024}) on EACH anchor mixture,
including the GDN-2-shell arms. dim is derived per-mixture to a matched param
budget so the capability comparison is fair.

This makes the capability axis directly comparable to TYPED_GDN2_MIXTURE_CMA and
gives the decisive three-way read on the *nonlinear-state* probe + recall:
    (a) dense GDN-2   M0  vs  (b) GDN-2-shell  S{1,2,3}  vs  (c) legacy nonlin  C{1,2,3}.

Idle-GPU-only / NO-PREEMPT: a GPU is usable iff used-mem < 2GB and not running one
of THIS driver's jobs.
"""
import os, sys, json, time, subprocess, argparse, datetime, hashlib

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _THIS)
sys.path.insert(0, _ROOT)

import numpy as np
from mixtures import build_anchors, head_counts, TYPE_NAMES

FREE_MEM_MIB = 2000
TRAIN_HYBRID = os.path.join(_ROOT, 'experiments', 'expressivity_tasks', 'train_hybrid.py')

PROBES = {
    'mqar_recall': [],
    's5_permutation': [],
    'anbncn_viability': [],
    'flag_hold_recall': ['--K', '4'],
    'iterated_nonlinear_map': [],
    'mixed_probe': ['--K', '4'],
}
PROBE_LIST = list(PROBES.keys())
EVAL_TS = [128, 256, 512, 1024]
DEPTH, N_HEADS, N_STATE = 4, 48, 32
LAM_MAX, BETA_MAX = 1.585, 2.747
SHELL_NONLIN = 'tanh'


def count_params(dim, logits, vocab):
    from ndm.models.hybrid_ladder import HybridLadderLM
    lk = {'head_type_logits': list(logits), 'lam_max': LAM_MAX, 'beta_max': BETA_MAX}
    if head_counts(logits, N_HEADS).get('gdn2_nonlin_shell', 0) > 0:
        lk['shell_state_nonlin'] = SHELL_NONLIN
    m = HybridLadderLM(vocab_size=vocab, dim=dim, depth=DEPTH,
                       layer_pattern=['typed-gdn2'], layer_kwargs=[lk],
                       n_state=N_STATE, n_heads=N_HEADS, expansion=1.0)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


def derive_dim(logits, vocab, target_params):
    lo, hi, best = 16, 1024, None
    while lo <= hi:
        mid = max(8, ((lo + hi) // 2 // 8) * 8)
        p = count_params(mid, logits, vocab)
        if best is None or abs(p - target_params) < abs(best[1] - target_params):
            best = (mid, p)
        if p < target_params:
            lo = mid + 8
        else:
            hi = mid - 8
    return best


def gpu_used_mib():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


def build_cmd(name, cfg, probe, seed, steps, out_dir):
    label = f"cap_{name}__{probe}__seed{seed}__s{steps}"
    has_shell = head_counts(cfg['logits'], N_HEADS).get('gdn2_nonlin_shell', 0) > 0
    cmd = [sys.executable, TRAIN_HYBRID, '--task', probe, '--layer_pattern', 'typed-gdn2',
           *PROBES[probe], '--dim', str(cfg['dim']), '--depth', str(DEPTH),
           '--n_heads', str(N_HEADS), '--n_state', str(N_STATE), '--lr', '3e-4',
           '--lam_max', str(LAM_MAX), '--beta_max', str(BETA_MAX),
           '--head_type_logits=' + ','.join(str(f) for f in cfg['logits']),
           '--steps', str(steps), '--seq_len', '128', '--batch_size', '32',
           '--optimizer', 'schedulefree', '--disable_autocast', '--seed', str(seed),
           '--label', label, '--output_dir', str(out_dir),
           '--eval_lengths', *[str(t) for t in EVAL_TS], '--eval_lengths_n_batches', '8']
    if has_shell:
        cmd += ['--shell_state_nonlin', SHELL_NONLIN, '--shell_state_chunk', '64']
    return cmd, label


def probe_score(out_dir, label):
    p = os.path.join(out_dir, f'{label}.json')
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None
    le = d.get('length_extrap')
    if not le:
        return None
    accs = [float(le[str(t)]['acc']) for t in EVAL_TS
            if isinstance(le.get(str(t)), dict) and 'acc' in le[str(t)]]
    return float(np.mean(accs)) if accs else None


def run_jobs(jobs, out_dir, gpus, poll=10.0):
    pending = [j for j in jobs if not os.path.exists(os.path.join(out_dir, f'{j[1]}.json'))]
    cached = len(jobs) - len(pending)
    if cached:
        print(f"[sched] {cached} cached, {len(pending)} to run", flush=True)
    running = {}
    while pending or running:
        for gpu in list(running):
            job, proc, logf = running[gpu]
            if proc.poll() is not None:
                logf.close()
                st = 'ok' if proc.returncode == 0 else f'FAIL({proc.returncode})'
                print(f"[done] gpu{gpu} {job[1]} -> {st}", flush=True)
                del running[gpu]
        if pending:
            used = gpu_used_mib()
            for gpu in gpus:
                if not pending:
                    break
                if gpu in running or used.get(gpu, 10**9) >= FREE_MEM_MIB:
                    continue
                job = pending.pop(0)
                cmd, label = job[0], job[1]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(os.path.join(out_dir, f'{label}.log'), 'w')
                proc = subprocess.Popen(cmd, cwd=_ROOT, env=env, stdout=logf,
                                        stderr=subprocess.STDOUT)
                running[gpu] = (job, proc, logf)
                print(f"[run ] gpu{gpu} {label}", flush=True)
                time.sleep(2)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=os.path.join(_THIS, 'results', 'capability'))
    ap.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--target_params', type=float, default=8.0e6)
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    from ndm.models.hybrid_ladder import HybridLadderLM  # noqa
    from experiments.expressivity_tasks.tasks import ALL_TASKS
    vocab = ALL_TASKS['mixed_probe'](n_keys=4).vocab_size

    anchors = build_anchors()
    # derive a matched-param dim per mixture
    cfgs = {}
    for name, spec in anchors.items():
        dim, actual = derive_dim(spec['logits'], vocab, args.target_params)
        cfgs[name] = dict(logits=spec['logits'], dim=dim, actual_params=int(actual),
                          counts=head_counts(spec['logits'], N_HEADS), role=spec['role'])
        print(f"{name:22s} dim={dim} params={actual:,} counts={cfgs[name]['counts']}", flush=True)
    json.dump(cfgs, open(os.path.join(out_dir, 'capability_cfgs.json'), 'w'),
              indent=2, default=str)

    jobs = []
    for name, cfg in cfgs.items():
        for probe in PROBE_LIST:
            for seed in args.seeds:
                jobs.append(build_cmd(name, cfg, probe, seed, args.steps, out_dir))
    print(f"=== CAPABILITY: {len(jobs)} probe-jobs ({len(cfgs)} mixtures x "
          f"{len(PROBE_LIST)} probes x {len(args.seeds)} seeds) on GPUs {gpus} ===", flush=True)
    run_jobs(jobs, out_dir, gpus)

    # aggregate per mixture
    summary = {}
    for name, cfg in cfgs.items():
        per_probe = {}
        for probe in PROBE_LIST:
            ss = [probe_score(out_dir, f"cap_{name}__{probe}__seed{seed}__s{args.steps}")
                  for seed in args.seeds]
            ss = [s for s in ss if s is not None]
            per_probe[probe] = round(float(np.mean(ss)), 4) if ss else None
        valid = [v for v in per_probe.values() if v is not None]
        summary[name] = dict(role=cfg['role'], counts=cfg['counts'],
                             actual_params=cfg['actual_params'], per_probe=per_probe,
                             mean=round(float(np.mean(valid)), 4) if valid else None,
                             min=round(float(np.min(valid)), 4) if valid else None)
    json.dump(dict(summary=summary, steps=args.steps, seeds=args.seeds,
                   depth=DEPTH, n_heads=N_HEADS, n_state=N_STATE,
                   shell_nonlin=SHELL_NONLIN, eval_ts=EVAL_TS,
                   timestamp=datetime.datetime.utcnow().isoformat() + 'Z'),
              open(os.path.join(out_dir, 'capability_summary.json'), 'w'),
              indent=2, default=str)
    print('\n=== CAPABILITY SUMMARY ===', flush=True)
    for name in cfgs:
        s = summary[name]
        pp = '  '.join(f"{k.split('_')[0]}={v}" for k, v in s['per_probe'].items())
        print(f"  {name:22s} mean={s['mean']} min={s['min']} | {pp}", flush=True)


if __name__ == '__main__':
    main()
