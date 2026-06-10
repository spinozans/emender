"""Freeze the per-corner specialist ceilings S_c for the opt-norm line
(OPT_SPEC.md §1.3, §6.1). Writes opt_ceilings.json — the SINGLE denominator file
every aggregate_*.py divides by, so all four probes' JCC are comparable.

Ceilings (best single specialist, seed-averaged, frozen):
  recall, track          <- GDN-2 = the best-LR fla-gdn control (B_gdn)
  counting, step_growth  <- the all-refit-del counting specialist (spec_refit)

The fla-gdn LR sanity {3e-4,5e-4,1e-3} picks B's best LR (reasonably tuned, NOT
hobbled; OPT_SPEC.md §4.1) by its recall+track mean — that LR is reported as B and
also defines S_recall / S_track.

Run AFTER the battery has produced the specialist arms:
    python experiments/expressivity_tasks/compute_opt_ceilings.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from opt_norm_common import (CORNERS, SCORED, TAU, CEILING_SPECIALIST,
                             load_runs, index_by_arm_task, corner_acc_for_seed)

THIS = Path(__file__).resolve().parent
B_LR_ARMS = ['B_gdn_lr3e4', 'B_gdn_lr5e4', 'B_gdn_lr1e3']


def seed_avg_corner(idx, arm, corner):
    seeds = set()
    for tk in CORNERS[corner]:
        for r in idx.get(arm, {}).get(tk, []):
            from opt_norm_common import parse_label
            seeds.add(parse_label(r['_file'])[2])
    vals = [corner_acc_for_seed(idx[arm], corner, s) for s in sorted(seeds)]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=str(THIS / 'results_opt_norm'))
    ap.add_argument('--out', default=str(THIS / 'opt_ceilings.json'))
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    runs = load_runs(out_dir)
    idx = index_by_arm_task(runs)
    print(f"Loaded {len(runs)} runs; arms present: {sorted(idx.keys())}")

    # --- pick B's best LR by recall+track mean (its specialty) ---
    lr_scores = {}
    for arm in B_LR_ARMS:
        if arm not in idx:
            continue
        rec = seed_avg_corner(idx, arm, 'recall')
        trk = seed_avg_corner(idx, arm, 'track')
        if rec is None and trk is None:
            continue
        sc = sum(v for v in (rec, trk) if v is not None) / sum(1 for v in (rec, trk) if v is not None)
        lr_scores[arm] = {'recall': rec, 'track': trk, 'score': sc}
    if not lr_scores:
        raise SystemExit("No B_gdn LR arms found — run the battery first.")
    best_lr_arm = max(lr_scores, key=lambda a: lr_scores[a]['score'])
    print(f"B_gdn LR sanity: {json.dumps(lr_scores, indent=2)}")
    print(f"-> best-LR B = {best_lr_arm}")

    # alias the chosen LR arm as the canonical 'B_gdn' for ceiling lookup
    idx['B_gdn'] = idx[best_lr_arm]

    ceilings, source = {}, {}
    for c in SCORED:
        spec = CEILING_SPECIALIST[c]
        s = seed_avg_corner(idx, spec, c)
        if s is None:
            print(f"  [warn] no specialist data for {c} (arm {spec}); ceiling unset")
            continue
        ceilings[c] = s
        source[c] = best_lr_arm if spec == 'B_gdn' else spec

    # battery hash so the synth agent can verify all probes used identical ceilings
    h = hashlib.sha256(json.dumps({'corners': CORNERS, 'ceilings': ceilings},
                                  sort_keys=True).encode()).hexdigest()[:16]
    doc = {'probe_origin': 'opt-norm', 'tau': TAU, 'ceilings': ceilings,
           'source': source, 'best_lr_arm': best_lr_arm, 'lr_sanity': lr_scores,
           'corners': CORNERS, 'ceilings_hash': h}
    json.dump(doc, open(args.out, 'w'), indent=2)
    print(f"\nFrozen ceilings -> {args.out}")
    print(json.dumps({c: round(v, 4) for c, v in ceilings.items()}, indent=2))


if __name__ == '__main__':
    main()
