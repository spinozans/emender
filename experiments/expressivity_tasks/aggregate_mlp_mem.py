"""Aggregate the mlp-mem capability battery (task nlmem-capability).

Reads results_mlp_mem/*.json (written by train_hybrid.py via run_mlp_mem_battery.py)
and prints, per task, the mean +/- std final/extrapolation accuracy for each arm
(mlpmem vs gdn2) and the mlpmem - gdn2 capability gap, at the matched compute used.

Usage:
    python experiments/expressivity_tasks/aggregate_mlp_mem.py [--dir results_mlp_mem]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float('nan'), float('nan')
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def parse_label(label):
    # mm_<task>[_K#][_mlp#]__<arm>__seed#
    body, arm, seed = label.rsplit('__', 2)
    assert body.startswith('mm_')
    body = body[3:]
    seed = int(seed.replace('seed', ''))
    mlp = 2.0
    if '_mlp' in body:
        body, mtag = body.rsplit('_mlp', 1)
        mlp = float(mtag)
    return body, arm, seed, mlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(THIS / 'results_mlp_mem'))
    ap.add_argument('--eval_len', default='128', help='which eval length is the headline')
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob('mm_*.json'))
    if not files:
        print(f"no result jsons in {d}")
        return

    # (taskkey, mlp_ratio) -> arm -> {eval_len -> [accs]}, plus params, baseline, elapsed
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta = defaultdict(lambda: defaultdict(lambda: {'params': [], 'elapsed': [], 'baseline': None}))
    eval_lens_seen = set()

    for f in files:
        try:
            log = json.load(open(f))
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            continue
        task, arm, seed, mlp = parse_label(f.stem)
        key = (task, mlp)
        le = log.get('length_extrap', {})
        for T, rec in le.items():
            if isinstance(rec, dict) and 'acc' in rec:
                acc[key][arm][T].append(rec['acc'])
                eval_lens_seen.add(T)
        meta[key][arm]['params'].append(log.get('params'))
        meta[key][arm]['elapsed'].append(log.get('elapsed_total_s'))
        meta[key][arm]['baseline'] = log.get('random_baseline_acc')

    eval_lens = sorted(eval_lens_seen, key=lambda x: int(x))
    headline = args.eval_len if args.eval_len in eval_lens else (eval_lens[0] if eval_lens else None)

    print("=" * 100)
    print("mlp-mem capability battery — mlpmem (nonlinear MLP-memory) vs gdn2 (linear matrix memory)")
    print("matched compute (same shell/dim/heads/depth/steps); fp32; 3 seeds; gap = mlpmem - gdn2")
    print("=" * 100)

    for mlp in sorted({k[1] for k in acc}, reverse=True):
        tag = f"mlp_ratio={mlp:g}" + ("  [headline: post-head SwiGLU readout PRESENT]" if mlp == 2.0
                                      else "  [ablation: post-head SwiGLU readout REMOVED]")
        print(f"\n### {tag}")
        header = f"{'task':<26}{'baseline':>9}"
        for T in eval_lens:
            header += f"  mlpmem@{T:<5} gdn2@{T:<6} gap@{T:<6}"
        print(header)
        print("-" * len(header))
        tasks = sorted({k[0] for k in acc if k[1] == mlp})
        for task in tasks:
            key = (task, mlp)
            base = meta[key].get('mlpmem', {}).get('baseline') or meta[key].get('gdn2', {}).get('baseline')
            row = f"{task:<26}{(base if base is not None else float('nan')):>9.3f}"
            for T in eval_lens:
                mm_m, mm_s = mean_std(acc[key]['mlpmem'].get(T, []))
                gd_m, gd_s = mean_std(acc[key]['gdn2'].get(T, []))
                gap = (mm_m - gd_m) if not (math.isnan(mm_m) or math.isnan(gd_m)) else float('nan')
                row += f"  {mm_m:5.3f}±{mm_s:4.3f} {gd_m:5.3f}±{gd_s:4.3f} {gap:+6.3f}"
            print(row)

    # compute / param accounting
    print("\n### compute & params (matched-compute audit)")
    print(f"{'task':<22}{'mlp':>5}  {'mlpmem params':>14} {'gdn2 params':>12}  "
          f"{'mm s/run':>9} {'gdn s/run':>9}  {'mm/gdn wall':>11}")
    for (task, mlp) in sorted(meta):
        mp = meta[(task, mlp)]
        mmp, _ = mean_std(mp.get('mlpmem', {}).get('params', []))
        gdp, _ = mean_std(mp.get('gdn2', {}).get('params', []))
        mme, _ = mean_std(mp.get('mlpmem', {}).get('elapsed', []))
        gde, _ = mean_std(mp.get('gdn2', {}).get('elapsed', []))
        ratio = (mme / gde) if (gde and not math.isnan(gde) and gde > 0) else float('nan')
        print(f"{task:<22}{mlp:>5g}  {mmp:>14,.0f} {gdp:>12,.0f}  "
              f"{mme:>9.1f} {gde:>9.1f}  {ratio:>10.2f}x")

    # headline verdict summary
    if headline:
        print(f"\n### verdict summary @ T={headline} (mlp_ratio=2.0 headline)")
        for task in sorted({k[0] for k in acc if k[1] == 2.0}):
            key = (task, 2.0)
            mm_m, _ = mean_std(acc[key]['mlpmem'].get(headline, []))
            gd_m, _ = mean_std(acc[key]['gdn2'].get(headline, []))
            if math.isnan(mm_m) or math.isnan(gd_m):
                continue
            gap = mm_m - gd_m
            verdict = 'TIE' if abs(gap) < 0.03 else ('mlpmem WINS' if gap > 0 else 'gdn2 WINS')
            print(f"  {task:<26} gap={gap:+6.3f}  -> {verdict}")


if __name__ == '__main__':
    main()
