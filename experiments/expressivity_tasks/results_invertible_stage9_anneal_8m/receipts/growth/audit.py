"""Read-only X9 completion audit; writes its derived summary to stdout."""
import hashlib
import json
import math
from pathlib import Path

from experiments.expressivity_tasks.run_invertible_stage9_anneal_8m import learning_rate

root = Path('experiments/expressivity_tasks/results_invertible_stage9_anneal_8m')
name = 'e97_bf16_seed0'
out = root / name
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reg = load(root / 'registration.json')
for p, h in reg['sources'].items():
    assert sha(Path(p)) == h, p
config = load(out / 'config.json')
for key in ('source_sha256', 'fla_source_sha256'):
    for p, h in config[key].items():
        assert sha(Path(p)) == h, p
result = load(out / 'result.json')
terminal = load(root / 'receipts' / name / 'terminal.json')
assert terminal == load(root / 'markers' / (name + '.DONE'))
assert terminal['exit_code'] == 0 and terminal['result_sha256'] == sha(out / 'result.json')
assert result['status'] == 'complete' and result['total_steps'] == 500000
assert result['stop_reason'] == 'safety_ceiling'
rows = load(out / 'validation.json')
assert [r['step'] for r in rows] == list(range(0, 500001, 500))
for row in rows:
    assert len(row['positions']) == 16
    assert all(p['count'] == 512 for p in row['positions'])
    assert sum(p['correct'] for p in row['positions']) == row['aggregate']['correct']
    assert math.isfinite(row['aggregate']['loss'])
assert result['final'] == rows[-1]
best = next(r for r in rows if r['step'] == result['best']['step'])
assert best is min(rows, key=lambda r: (-r['aggregate']['accuracy'], r['aggregate']['loss']))
last = None
with (out / 'training.jsonl').open() as stream:
    for count, line in enumerate(stream, 1):
        last = json.loads(line)
        assert last['step'] == count and last['depth'] == 16
        assert last['lr'] == learning_rate(count)
        assert all(math.isfinite(last[k]) for k in ('loss', 'grad_norm', 'lr'))
assert count == 500000 and last['lr'] == 1e-5
checkpoints = load(out / 'checkpoints.json')
retained = []
for p, cp in checkpoints.items():
    if Path(p).exists():
        assert sha(Path(p)) == cp['sha256']
        retained.append(cp)
assert {rows[0]['checkpoint']['path'], best['checkpoint']['path'], rows[-1]['checkpoint']['path']} <= {c['path'] for c in retained}
x7 = Path('experiments/expressivity_tasks/results_invertible_stage7_deepfirst_8m') / name
assert load(out / 'data_receipt.json')['array_sha256'] == load(x7 / 'data_receipt.json')['array_sha256']
assert sha(out / 'frozen_validation.npz') == load(out / 'data_receipt.json')['sha256']
summary = {
    'status': 'passed', 'training_rows': count, 'validation_rows': len(rows),
    'finished_at': result['finished_at'], 'wall_seconds': result['wall_seconds'],
    'final': rows[-1], 'best': best,
    'first_512': [next((r['step'] for r in rows if r['positions'][i]['correct'] == 512), None) for i in range(16)],
    'first_material': [next((r['step'] for r in rows if r['positions'][i]['correct'] >= 50), None) for i in range(16)],
    'position_maxima': [max(r['positions'][i]['correct'] for r in rows) for i in range(16)],
    'lr_samples': {str(s): learning_rate(s) for s in (1,5000,62500,245500,347500,465000,500000)},
    'retained_checkpoints': retained,
    'artifact_sha256': {p: sha(out / p) for p in ('result.json','validation.json','training.jsonl','config.json','data_receipt.json')},
    'x7_final': load(x7 / 'result.json')['final'],
    'terminal': terminal,
}
print(json.dumps(summary, indent=2))
