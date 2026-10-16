"""Apply registered capacity checks to runs whose training loop already exited."""
import json
from pathlib import Path
from experiments.expressivity_tasks.run_invertible_stage2 import prune_checkpoints, selected

root = Path('experiments/expressivity_tasks/results_invertible_stage2b')
for result_path in sorted(root.glob('*/result.json')):
    run = result_path.parent
    history = json.loads((run / 'validation.json').read_text())
    best = None
    chain = set()
    for row in history:
        if selected(row, best):
            best = row
            chain.add(row['step'])
    checkpoints = json.loads((run / 'checkpoints.json').read_text())
    prune_checkpoints(run, checkpoints, chain, history[-1]['step'])
    print(run.name, 'registered capacity check; stop', history[-1]['step'],
          'pruning receipt present', (run / 'pruning.jsonl').exists())
