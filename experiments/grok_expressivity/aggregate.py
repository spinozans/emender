"""Aggregate grok-expressivity runs into a grok-status table + verdict inputs.

Reads every runs/*.json (one per task x arm x wd x seed), reports per-cell:
  grokked (y/n), grok_step, memorize_step, final/best test-acc, final train-acc,
  length-extrapolation at the trained model.  Writes:
    - grok_table.md       human-readable table grouped by task
    - grok_summary.json   machine-readable rollup
No fabrication; only reads measured run logs.
"""
import json, sys, glob, os
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent


def load(runs_dir):
    rows = []
    for f in sorted(glob.glob(str(runs_dir / '*.json'))):
        if os.path.basename(f) in ('grok_summary.json', 'orchestrate_summary.json'):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if 'arm' not in d or 'task' not in d:
            continue
        rows.append(d)
    return rows


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main():
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else THIS / 'runs'
    rows = load(runs_dir)
    if not rows:
        print(f"no runs in {runs_dir}"); return

    by_task = defaultdict(list)
    for d in rows:
        by_task[d['task']].append(d)

    md = ["# grok-expressivity — grok-status table\n",
          f"Source: `{runs_dir}` ({len(rows)} runs)\n",
          "Arms: **e97** = E97 split-edit fused Triton, tanh state (nonlinear-in-time); "
          "**e97-lin** = same fused kernel, linear state (matched control); "
          "**gdn2** = FLA GatedDeltaNet (linear); **e97-ht** = phi-shell split-edit hardtanh.\n",
          "`grokked` = test-acc reached --grok_acc; `grok_step` = first step it did; "
          "`mem_step` = first step train-acc>=--train_sat_acc.\n"]

    summary = {}
    for task in sorted(by_task):
        ds = by_task[task]
        base = ds[0].get('random_baseline_acc')
        md.append(f"\n## {task}  (baseline acc {fmt(base)})\n")
        md.append("| arm | wd | n_train | seed | params | mem_step | grok_step | grokked | "
                  "final_train | final_test | best_test | extrap@128 | extrap@256 | extrap@512 | extrap@1024 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        def sortkey(d):
            return (d['arm'], d.get('n_train', 0), float(d.get('weight_decay', 0)), d.get('seed', 0))
        for d in sorted(ds, key=sortkey):
            le = d.get('length_extrap', {}) or {}
            md.append("| {arm} | {wd} | {nt} | {seed} | {params:,} | {mem} | {grok} | {gk} | "
                      "{ftr} | {fte} | {bte} | {e1} | {e2} | {e3} | {e4} |".format(
                          arm=d['arm'], wd=d.get('weight_decay'), nt=d.get('n_train'),
                          seed=d.get('seed'), params=d.get('params', 0),
                          mem=d.get('memorize_step'), grok=d.get('grok_step'),
                          gk='Y' if d.get('grokked') else 'n',
                          ftr=fmt(d.get('final_train_acc')),
                          fte=fmt(d.get('final_test_acc')),
                          bte=fmt(d.get('best_test_acc')),
                          e1=fmt(le.get('128')), e2=fmt(le.get('256')),
                          e3=fmt(le.get('512')), e4=fmt(le.get('1024'))))
        # per-arm best-over-wd/seed rollup
        roll = defaultdict(lambda: {'best_test': 0.0, 'grokked': False,
                                    'min_grok_step': None})
        for d in ds:
            r = roll[d['arm']]
            r['best_test'] = max(r['best_test'], d.get('best_test_acc') or 0.0)
            if d.get('grokked'):
                r['grokked'] = True
                gs = d.get('grok_step')
                if gs is not None and (r['min_grok_step'] is None or gs < r['min_grok_step']):
                    r['min_grok_step'] = gs
        summary[task] = {'baseline': base, 'arms': dict(roll)}

    md.append("\n## Per-arm rollup (best over wd x seed)\n")
    md.append("| task | arm | any_grokked | min_grok_step | best_test_acc |")
    md.append("|---|---|---|---|---|")
    for task in sorted(summary):
        for arm, r in sorted(summary[task]['arms'].items()):
            md.append(f"| {task} | {arm} | {'Y' if r['grokked'] else 'n'} | "
                      f"{r['min_grok_step']} | {fmt(r['best_test'])} |")

    (runs_dir / 'grok_table.md').write_text("\n".join(md) + "\n")
    (runs_dir / 'grok_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print("\n".join(md))
    print(f"\nWROTE {runs_dir/'grok_table.md'} and grok_summary.json")


if __name__ == '__main__':
    main()
