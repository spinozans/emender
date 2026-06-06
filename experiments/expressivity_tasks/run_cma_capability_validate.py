"""Validate the cma-capability winner vs the hand-set baseline at FULL budget.

Runs (REAL training, 5000 steps, 3 seeds {42,123,456}, 5 probes) two arms:
  * WINNER   : the CMA best meta-config (cma_best.json), ~8M, e98-cma form.
  * BASE8M   : the HAND-SET form at the SAME 8M budget (equal 25/25/25/25 mixture,
               knob_lr_mult=20, lam_max=1.5, beta_max=2.0, depth4/H32/N32, dim
               derived to 8M) -- isolates the CMA TUNING gain at matched params.
A third reference (BASE11M = e98-learned-spread-klr20 @ its native 11.07M) is
NOT re-run: it is read from the existing results/e98_*.json files.

Reuses the cma_capability idle-GPU scheduler + builders. Output JSONs land in
results/cma_capability/ with the same naming so the aggregator picks them up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.expressivity_tasks.cma_capability as C

THIS = Path(__file__).resolve().parent


def base8m_cfg(target_params, ref_vocab):
    dim, actual = C.derive_dim(4, 32, 32, ref_vocab, target_params)
    return {
        'dim': dim, 'depth': 4, 'n_heads': 32, 'n_state': 32,
        'lr': 3e-4, 'knob_lr_mult': 20.0, 'lam_max': 1.5, 'beta_max': 2.0,
        'corner_mixture': [0.25, 0.25, 0.25, 0.25], 'actual_params': int(actual),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--best', default=str(THIS / 'results' / 'cma_capability' / 'cma_best.json'))
    ap.add_argument('--target_params', type=float, default=8.0e6)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--max_gpus', type=int, default=8)
    ap.add_argument('--poll', type=float, default=12.0)
    ap.add_argument('--output_dir', default=str(THIS / 'results' / 'cma_capability'))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_vocab = C.ALL_TASKS[C.REF_VOCAB_PROBE](n_keys=4).vocab_size

    winner = json.load(open(args.best))['best']['cfg']
    base = base8m_cfg(args.target_params, ref_vocab)
    arms = {'winner': winner, 'base8m': base}
    print(f"[validate] winner={C.config_hash(winner)} ({winner['actual_params']:,}p) "
          f"base8m={C.config_hash(base)} ({base['actual_params']:,}p), "
          f"steps={args.steps}, seeds={args.seeds}", flush=True)

    jobs = []
    for cfg in arms.values():
        h = C.config_hash(cfg)
        for probe in C.PROBE_LIST:
            for seed in args.seeds:
                jobs.append(C.Job(h, cfg, probe, seed, args.steps))
    C.run_jobs(jobs, out_dir, args.max_gpus, args.poll)

    summary = {}
    for name, cfg in arms.items():
        _, per_probe = C.fitness_of(cfg, out_dir, args.seeds, args.steps)
        summary[name] = {'hash': C.config_hash(cfg), 'cfg': cfg, 'per_probe': per_probe}
    json.dump({'arms': summary, 'steps': args.steps, 'seeds': args.seeds},
              open(out_dir / 'validate_summary.json', 'w'), indent=2)
    print(f"[validate] done -> {out_dir / 'validate_summary.json'}", flush=True)
    for name in arms:
        pp = summary[name]['per_probe']
        print(f"  {name:8s} mean={sum(pp.values())/len(pp):.4f}  " +
              "  ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in pp.items()), flush=True)


if __name__ == '__main__':
    main()
