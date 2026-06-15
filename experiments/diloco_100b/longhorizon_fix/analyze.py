#!/usr/bin/env python3
"""fix-long-horizon: tabulate corrected-recipe held-out BPB curves + DiLoCo gap.

Reads results.txt (BPB_RESULT <tag> <step> <bpb> <ce> <tokens> lines), prints a
per-tag step->BPB table, flags monotonicity (any mid-run rise from the running
minimum), and a matched-token DiLoCo-vs-DDP gap table. REAL numbers only.

  python3 analyze.py results.txt
"""
import sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else 'results.txt'
TOK_PER_STEP = 6 * 2048 * 7  # 86016

runs = collections.defaultdict(dict)  # tag -> {step: bpb}
for line in open(path):
    p = line.split()
    if len(p) >= 4 and p[0] == 'BPB_RESULT':
        tag, step, bpb = p[1], int(p[2]), float(p[3])
        runs[tag][step] = bpb

for tag in sorted(runs):
    steps = sorted(runs[tag])
    print(f"\n=== {tag} ===")
    print(f"{'step':>6} {'Mtok':>8} {'BPB':>8}  {'note':<24}")
    run_min = float('inf'); min_step = None; rose = False; peak_rise = 0.0
    for s in steps:
        b = runs[tag][s]
        note = ''
        if b < run_min:
            run_min = b; min_step = s
        else:
            rise = b - run_min
            peak_rise = max(peak_rise, rise)
            if rise > 0.02:  # tolerance for eval noise
                rose = True; note = f'ROSE +{rise:.3f} from min@{min_step}'
        print(f"{s:>6} {s*TOK_PER_STEP/1e6:>8.1f} {b:>8.4f}  {note:<24}")
    verdict = 'NON-MONOTONIC (mid-run rise)' if rose else 'MONOTONIC (no mid-run rise)'
    print(f"  -> min BPB {run_min:.4f} @ step {min_step} ({min_step*TOK_PER_STEP/1e6:.1f}M tok); "
          f"peak rise from min = +{peak_rise:.4f}; {verdict}")

# DiLoCo-vs-DDP matched-token gap. Prefer the HEALTHY DDP baseline (cosine/adamw)
# over any earlier diagnostic DDP arm so the gap is measured against a non-degrading
# reference (the whole point of fix-long-horizon).
ddp_tags = [t for t in runs if 'ddp' in t.lower()]
ddp_tag = next((t for t in ddp_tags if 'cos' in t.lower() or 'adamw' in t.lower()),
               ddp_tags[0] if ddp_tags else None)
dil_tags = [t for t in runs if t != ddp_tag and ('dil' in t.lower() or 'diloco' in t.lower())]
if ddp_tag and dil_tags:
    for dt in dil_tags:
        print(f"\n=== matched-token gap: {dt} - {ddp_tag} ===")
        print(f"{'step':>6} {'Mtok':>8} {ddp_tag[:10]:>10} {dt[:12]:>12} {'gap':>8}")
        for s in sorted(set(runs[dt]) & set(runs[ddp_tag])):
            g = runs[dt][s] - runs[ddp_tag][s]
            print(f"{s:>6} {s*TOK_PER_STEP/1e6:>8.1f} {runs[ddp_tag][s]:>10.4f} {runs[dt][s]:>12.4f} {g:>+8.3f}")
