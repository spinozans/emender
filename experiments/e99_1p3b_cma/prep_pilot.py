#!/usr/bin/env python3
"""Extract top-K promoted configs + record the bounded pilot budget."""
import json, os, sys

OUT = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
_THIS = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.join(_THIS, 'pilot_results')
os.makedirs(pdir, exist_ok=True)

rows = json.load(open(os.path.join(OUT, 'candidates.json')))
top = [r for r in rows if r.get('success')][:K]
ref_steps = top[0].get('n_steps')
budget = dict(pilot_step_cap=1800, pilot_mult_approx='~3x short-run steps',
              short_run_ref_steps=ref_steps, pilot_wall_minutes=45.0,
              batch_size=2, chunk_size=2048, dtype='fp32',
              recorded_before_launch=True,
              note='Bounded rank-stability pilot: min(1800 steps, 45 wall-min) per '
                   'config; a fixed multiple of the short-run token budget AND a fixed '
                   'walltime ceiling. NOT a full run.')
json.dump(budget, open(os.path.join(pdir, 'pilot_budget.json'), 'w'), indent=2)
for i, r in enumerate(top):
    p = {k: r[k] for k in ['dim', 'n_heads', 'n_state', 'depth', 'lr']}
    p['batch_size'] = 2
    json.dump([p], open(os.path.join(pdir, f'cfg_{i}_eval{r["eval_id"]}.json'), 'w'), indent=2)
    print(f"top{i+1} eval{r['eval_id']}: AvgLoss={r['avg_loss']:.4f} "
          f"dim={p['dim']} nh={p['n_heads']} ns={p['n_state']} depth={p['depth']} lr={p['lr']:.4g}")
print('budget:', json.dumps(budget))
