"""CPU-only terminal arithmetic/coverage audit; run from repository root."""
import gzip
import hashlib
import json
import math
from pathlib import Path

r = Path(__file__).resolve().parent
load = lambda p: json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())
s = load(r / 'eval_resume.json.gz')
t = load(r / 'eval_table.json')
assert load(r / 'eval_terminal.json')['exit_code'] == 0
assert (s['status'], s['next_shard'], s['windows'], s['targets']) == ('complete', 26807, 3431242, 434204084)
assert len(t) == 66 and set(t) == set(s['sums'])
arms = load(r.parent / 'c14_b_v2_unbudgeted/triangle_tables.json')
for arm in arms.values():
    assert set(arm) == set(t)
    assert all(arm[k]['targets'] == v['targets'] for k, v in t.items())
for k, v in t.items():
    assert [v['total_nll'], v['targets']] == s['sums'][k]
    assert math.isclose(v['nll'], v['total_nll'] / v['targets'], abs_tol=1e-12)
    mode, split, stratum = k.split('/')
    if split == 'all':
        for field in ['targets', 'total_nll']:
            assert math.isclose(v[field], sum(t[f'{mode}/{x}/{stratum}'][field] for x in ['calibration', 'final_test']), rel_tol=1e-12)
for mode in ['B1_on', 'B1_off']:
    assert t[f'{mode}/final_test/total']['targets'] == 214097022
train = load(r / 'training.json.gz')
assert (train['steps'], train['actual_bytes_targets']) == (35419, 71677042)
for v in train['telemetry'].values():
    assert v['peak_occupancy'] < 257 and v['counts']['capacity_merges'] == 0
launch = load(r / 'EVAL_LAUNCH.json')
assert s['config']['checkpoint_sha256'] == train['checkpoint_sha256'] == launch['training_checkpoint_sha256']
summary = load(r / 'HARVEST_COMPLETE.json')
for name, digest in summary['sha256'].items():
    path = r / name
    raw = gzip.decompress((r / (name + '.gz')).read_bytes()) if name in ['eval_resume.json', 'training.json'] else path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
print(json.dumps({'status': 'passed', 'shards': 26807, 'windows': 3431242, 'targets': 434204084, 'rows': 66, 'frozen_denominators': 'identical', 'resume_sums': 'exact', 'training_cap': 'nonbinding'}))
