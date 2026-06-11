#!/usr/bin/env python3
"""emender-1p3b-cma — aggregate the STANDARD-driver CMA-ES search over the typed-gdn2
Emender at 1.3B into the committed deliverables (convergence curve + found best).

REAL data only: every number traces to the committed search output under
search/<run>/ (generations.jsonl, results.json) — no paper numbers, no synthetic data.

Usage:
  python experiments/emender_1p3b_cma/aggregate.py
"""
import json
import os
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, REPO)


def find_run_dir():
    runs = sorted(glob.glob(os.path.join(HERE, 'search', 'emender_*')))
    if not runs:
        raise SystemExit("no search run dir found under search/")
    return runs[-1]


def main():
    run = find_run_dir()
    gens = [json.loads(l) for l in open(os.path.join(run, 'generations.jsonl'))]
    res = json.load(open(os.path.join(run, 'results.json')))

    # Recompute the EXACT param count + head allocation of the found best from the
    # committed driver (no hand numbers).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'cm', os.path.join(REPO, 'scripts', 'cmaes_search_v2.py'))
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    cm.PARAM_VOCAB_SIZE = cm.resolve_vocab_size('p50k_base')
    cm.estimate_emender_params._target = 1_300_000_000
    from ndm.models.typed_head_mixture import allocate_types

    bp = res['best_params']
    actual = cm.estimate_params_for_config(bp, 'emender')
    logits = cm.emender_head_type_logits(bp['mixture_nonlin'])
    counts = allocate_types(bp['n_heads'], logits)['counts']
    alloc = {k: v for k, v in counts.items() if v > 0}

    print("=" * 72)
    print("emender-1p3b-cma — CMA-ES convergence (best-fitness per generation)")
    print("=" * 72)
    print(f"{'gen':>3} {'best_so_far':>12} {'gen_best':>10} {'sigma':>8}")
    for g in gens:
        print(f"{g['gen']+1:>3} {g['best_loss_so_far']:>12.5f} "
              f"{g['gen_best_loss']:>10.5f} {g['sigma']:>8.4f}")
    print()
    print(f"FOUND BEST (CMA fitness = avg train CE, p50k): {res['best_loss']:.5f}")
    print(f"  geometry : dim={bp['dim']} n_heads={bp['n_heads']} depth={bp['depth']} "
          f"n_state=32 expansion=1.0")
    print(f"  mixture  : nonlin_frac={bp['mixture_nonlin']:.4f} -> {alloc}")
    print(f"  training : lr={bp['lr']:.3e} batch_size(probed)={bp['batch_size']}")
    print(f"  params   : {actual/1e9:.4f}B (target 1.300B, dev {(actual-1.3e9)/1.3e9*100:+.2f}%)")
    print(f"  total evaluations: {len(res.get('all_results', []))} | generations: {len(gens)}")

    # held-out BPB (measured separately on the CMA-best, leaderboard averaged-weights)
    ho = os.path.join(HERE, 'heldout', 'heldout_run.log')
    if os.path.exists(ho):
        for line in open(ho):
            if 'FINAL_HELDOUT_CE' in line or 'FINAL_HELDOUT_BPB' in line:
                print('  ' + line.strip())


if __name__ == '__main__':
    main()
