"""Aggregate cma-capability search + validation into CMA_CAPABILITY_RESULTS.md
tables. REAL JSONs only; no synthetic numbers.

Reads:
  results/cma_capability/cma_best.json   (search winner + fitness history)
  results/cma_capability/cma_trace.json  (every evaluated config)
  results/cma_capability/cmacap_<hash>__<probe>__seed<seed>__s5000.json  (winner, base8m)
  results/e98_<probe>__e98-learned-spread-klr20__seed<seed>.json         (base11m, reused)

Emits markdown tables to stdout (the report embeds them).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.expressivity_tasks.run_cma_capability_validate as V
import experiments.expressivity_tasks.cma_capability as C

THIS = Path(__file__).resolve().parent
RES = THIS / 'results'
CMADIR = RES / 'cma_capability'
TS = C.EVAL_TS
PROBES = C.PROBE_LIST
CORNER = {'s5_permutation': 'track', 'anbncn_viability': 'count',
          'iterated_nonlinear_map': 'nonlin', 'flag_hold_recall': 'latch',
          'mixed_probe': 'ALL'}
SEEDS = [42, 123, 456]


def _accs(path: Path):
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    le = d.get('length_extrap') or {}
    return {t: (le.get(str(t), {}) or {}).get('acc') for t in TS}


def winner_path(h, probe, seed):
    return CMADIR / f'cmacap_{h}__{probe}__seed{seed}__s5000.json'


def base11m_path(probe, seed):
    return RES / f'e98_{probe}__e98-learned-spread-klr20__seed{seed}.json'


def mean_over_seeds(path_fn, probe):
    """Return dict T-> (mean,std) over seeds for one probe, or None."""
    perT = {t: [] for t in TS}
    for s in SEEDS:
        a = _accs(path_fn(probe, s))
        if not a:
            continue
        for t in TS:
            if a[t] is not None:
                perT[t].append(a[t])
    out = {}
    for t in TS:
        out[t] = (float(np.mean(perT[t])), float(np.std(perT[t]))) if perT[t] else (None, None)
    return out


def overall_mean(path_fn):
    vals = []
    for probe in PROBES:
        m = mean_over_seeds(path_fn, probe)
        for t in TS:
            if m[t][0] is not None:
                vals.append(m[t][0])
    return float(np.mean(vals)) if vals else None


def fmt(m, s):
    return f"{m:.3f}±{s:.3f}" if m is not None else "--"


def main():
    best = json.load(open(CMADIR / 'cma_best.json'))
    winner = best['best']['cfg']
    wh = C.config_hash(winner)
    ref_vocab = C.ALL_TASKS[C.REF_VOCAB_PROBE](n_keys=4).vocab_size
    base = V.base8m_cfg(8.0e6, ref_vocab)
    bh = C.config_hash(base)

    arms = [
        ('WINNER (8M, CMA)', lambda p, s: winner_path(wh, p, s)),
        ('base8m hand-set', lambda p, s: winner_path(bh, p, s)),
        ('base11m e98-spread-klr20', base11m_path),
    ]

    print("## Winner vs baselines — length-extrapolation accuracy (mean±std, 3 seeds)\n")
    for T in TS:
        print(f"### T={T}")
        header = "| arm | " + " | ".join(f"{CORNER[p]}" for p in PROBES) + " | **mean** |"
        sep = "|" + "---|" * (len(PROBES) + 2)
        print(header); print(sep)
        for name, fn in arms:
            cells = []
            allm = []
            for p in PROBES:
                m, s = mean_over_seeds(fn, p)[T]
                cells.append(fmt(m, s))
                if m is not None:
                    allm.append(m)
            mean_cell = f"**{np.mean(allm):.3f}**" if allm else "--"
            print(f"| {name} | " + " | ".join(cells) + f" | {mean_cell} |")
        print()

    print("## Multi-capability fitness (mean length-extrap acc over all probes×T)\n")
    print("| arm | params | multi-cap fitness |")
    print("|---|---|---|")
    print(f"| WINNER (8M, CMA) | {winner['actual_params']:,} | {overall_mean(arms[0][1]):.4f} |")
    print(f"| base8m hand-set | {base['actual_params']:,} | {overall_mean(arms[1][1]):.4f} |")
    o11 = overall_mean(base11m_path)
    print(f"| base11m e98-spread-klr20 | 11,072,640 | {o11:.4f} |" if o11 else "| base11m | 11,072,640 | -- |")

    print("\n## Winning meta-config\n")
    print("```json")
    print(json.dumps(winner, indent=2))
    print("```")
    print(f"\nsearch: gens_run={best.get('gens_run')}, "
          f"best_fit_hist={[round(x,4) for x in best.get('best_fit_hist',[])]}")


if __name__ == '__main__':
    main()
