"""Validate the typed-gdn-2-head CMA winner against the references at full budget.

Three arms, six probes, REAL training (longer steps + multi-seed), idle-GPU-only:
  * typed-gdn2 winner  : the CMA-discovered typed-head mixture (fp32, the native
                         GDN-2 recall heads + frozen E98 corner specialists),
                         read from results/typed_gdn2_cma/cma_best.json.
  * e98-cma winner     : the current CMA E98 unified-cell winner from
                         cma-capability (cma_capability_best.json), fp32.
  * gdn native ref     : the native FLA Gated-DeltaNet baseline ('gdn' layer),
                         bf16 (its chunk kernel rejects fp32), param-matched ~8M.

Probes: mqar_recall, s5_permutation, anbncn_viability, flag_hold_recall,
iterated_nonlinear_map, mixed_probe. Each run writes a JSON the scheduler skips if
present (resumable). This is the *reduced/staged* validation the task allows when
full 3-seed validation would over-spend: configurable --seeds / --steps; default
is 2 seeds x 4000 steps.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT))

from ndm.models.hybrid_ladder import HybridLadderLM  # noqa: E402
from experiments.expressivity_tasks.tasks import ALL_TASKS  # noqa: E402

FREE_MEM_MIB = 2000
EVAL_TS = [128, 256, 512, 1024]
PROBES = {
    'mqar_recall': [],
    's5_permutation': [],
    'anbncn_viability': [],
    'flag_hold_recall': ['--K', '4'],
    'iterated_nonlinear_map': [],
    'mixed_probe': ['--K', '4'],
}
PROBE_LIST = list(PROBES.keys())


def gdn_ref_params(vocab, dim, n_heads, n_state, depth):
    """Exact params of the native 'gdn' reference HybridLadderLM (CPU build).

    Mirrors the e98-five-corner GDN reference SHARED config exactly (the known-good
    0.951-MQAR arm): the GatedDeltaNet wrapper self-configures head_dim/num_heads
    from `dim` (it ignores n_state/n_heads via HybridLadderLM's kwargs fallback);
    we just report the resulting param count."""
    m = HybridLadderLM(
        vocab_size=vocab, dim=dim, depth=depth,
        layer_pattern=['gdn'], layer_kwargs=[{}],
        n_state=n_state, n_heads=n_heads, expansion=1.0)
    n = sum(p.numel() for p in m.parameters()); del m
    return n


@dataclass
class Arm:
    name: str
    layer_pattern: str
    extra: list           # extra CLI args (config-specific)
    autocast: bool        # True -> bf16 autocast (gdn ref); False -> fp32


def build_arms(args):
    arms = []

    # --- typed-gdn2 winner ---
    tw = json.load(open(args.typed_best))
    tcfg = tw['best']['cfg'] if 'best' in tw else tw['cfg']
    typed_extra = [
        '--dim', str(tcfg['dim']),
        '--depth', str(tcfg['depth']),
        '--n_heads', str(tcfg['n_heads']),
        '--n_state', str(tcfg['n_state']),
        '--lr', str(tcfg['lr']),
        '--lam_max', str(tcfg['lam_max']),
        '--beta_max', str(tcfg['beta_max']),
        '--head_type_logits=' + ','.join(str(f) for f in tcfg['head_type_logits']),
    ]
    arms.append(Arm('typed-gdn2-winner', 'typed-gdn2', typed_extra, autocast=False))

    # --- e98-cma winner (cma-capability) ---
    e = json.load(open(args.e98_best))
    e98_extra = [
        '--dim', str(e['dim']),
        '--depth', str(e['depth']),
        '--n_heads', str(e['n_heads']),
        '--n_state', str(e['n_state']),
        '--lr', str(e['lr']),
        '--knob_lr_mult', str(e['knob_lr_mult']),
        '--lam_max', str(e['lam_max']),
        '--beta_max', str(e['beta_max']),
        '--corner_mixture', ','.join(str(f) for f in e['corner_mixture']),
    ]
    arms.append(Arm('e98-cma-winner', 'e98-cma', e98_extra, autocast=False))

    # --- native gdn reference (bf16): exact e98-five-corner SHARED config ---
    vocab = ALL_TASKS['mixed_probe'](n_keys=4).vocab_size
    gparams = gdn_ref_params(vocab, args.gdn_dim, args.gdn_heads,
                             args.gdn_n_state, args.depth)
    print(f"[gdn-ref] dim={args.gdn_dim} params={gparams:,} "
          f"(n_heads={args.gdn_heads} n_state={args.gdn_n_state}; "
          f"GatedDeltaNet self-configures head_dim/num_heads)", flush=True)
    gdn_extra = [
        '--dim', str(args.gdn_dim),
        '--depth', str(args.depth),
        '--n_heads', str(args.gdn_heads),
        '--n_state', str(args.gdn_n_state),
        '--expansion', '1.0',
    ]
    arms.append(Arm('gdn-native-ref', 'gdn', gdn_extra, autocast=True))
    return arms


def label(arm, probe, seed, steps):
    return f"tgdn2val_{arm.name}__{probe}__seed{seed}__s{steps}"


def build_cmd(arm, probe, seed, steps, out_dir):
    cmd = [
        sys.executable, str(THIS / 'train_hybrid.py'),
        '--task', probe,
        '--layer_pattern', arm.layer_pattern,
        *PROBES[probe],
        *arm.extra,
        '--steps', str(steps),
        '--seq_len', '128',
        '--batch_size', '32',
        '--optimizer', 'schedulefree',
        '--seed', str(seed),
        '--label', label(arm, probe, seed, steps),
        '--output_dir', str(out_dir),
        '--eval_lengths', *[str(t) for t in EVAL_TS],
        '--eval_lengths_n_batches', '8',
    ]
    if not arm.autocast:
        cmd.append('--disable_autocast')
    return cmd


def gpu_used_mib():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def run_jobs(jobs, out_dir, max_gpus, poll):
    pending = [j for j in jobs if not (out_dir / f'{j[4]}.json').exists()]
    skipped = len(jobs) - len(pending)
    if skipped:
        print(f"[sched] {skipped} cached, {len(pending)} to run", flush=True)
    running = {}
    while pending or running:
        for gpu in list(running):
            lbl, proc, logf = running[gpu]
            if proc.poll() is not None:
                logf.close()
                st = 'ok' if proc.returncode == 0 else f'FAIL({proc.returncode})'
                print(f"[done] gpu{gpu} {lbl} -> {st}", flush=True)
                del running[gpu]
        if pending:
            used = gpu_used_mib()
            for gpu in range(max_gpus):
                if not pending:
                    break
                if gpu in running or used.get(gpu, 10**9) >= FREE_MEM_MIB:
                    continue
                arm, probe, seed, steps, lbl, cmd = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(out_dir / f'{lbl}.log', 'w')
                proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                        stdout=logf, stderr=subprocess.STDOUT)
                running[gpu] = (lbl, proc, logf)
                print(f"[run ] gpu{gpu} {lbl}", flush=True)
                time.sleep(3)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--typed_best',
                    default=str(THIS / 'results' / 'typed_gdn2_cma' / 'cma_best.json'))
    ap.add_argument('--e98_best',
                    default=str(THIS / 'cma_capability_best.json'))
    ap.add_argument('--target_params', type=float, default=8.0e6)
    ap.add_argument('--depth', type=int, default=4)
    # native gdn reference: exact e98-five-corner SHARED config (dim 256, n_heads
    # 32, n_state 32) -- the known-good 0.951-MQAR arm.
    ap.add_argument('--gdn_dim', type=int, default=256)
    ap.add_argument('--gdn_heads', type=int, default=32)
    ap.add_argument('--gdn_n_state', type=int, default=32)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    ap.add_argument('--max_gpus', type=int, default=8)
    ap.add_argument('--poll', type=float, default=12.0)
    ap.add_argument('--output_dir',
                    default=str(THIS / 'results' / 'typed_gdn2_validate'))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = build_arms(args)
    print(f"[validate] arms={[a.name for a in arms]} probes={PROBE_LIST} "
          f"seeds={args.seeds} steps={args.steps}", flush=True)

    jobs = []
    for arm in arms:
        for probe in PROBE_LIST:
            for seed in args.seeds:
                lbl = label(arm, probe, seed, args.steps)
                cmd = build_cmd(arm, probe, seed, args.steps, out_dir)
                jobs.append((arm, probe, seed, args.steps, lbl, cmd))
    run_jobs(jobs, out_dir, args.max_gpus, args.poll)
    print("[validate] all jobs done", flush=True)


if __name__ == '__main__':
    main()
