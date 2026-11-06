"""Recompute the rejected diagnostic candidate projection from retained samples."""
from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent
r=p/'receipts/scalability_phase5'
data=json.loads((r/'sweep_barrier.json').read_text())
assert len(data['rows'])==10
assert all(x['exact'] and x['repeat_exact'] for x in data['rows'])
for path,expected in data['sources'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
ms=data['rows'][-1]['median_ms']
projection=dict(status='rejected_speed_regression_not_qualified',
    native_head_ms=ms,compiler_spills=data['rows'][-1]['spills'],
    seconds_per_target=ms/1000*22*2*2/1024,
    native_only_days=ms/1000*22*2*2/1024*2e9/86400,
    required_targets_per_second=2e9/86400,
    max_head_ms_if_other_work_free=86400/2e9*1024/(22*2*2)*1000,
    scope='measured nonzero controlled native head, hypothetical 22-layer/two-head/two-pass raw-byte budget scenario; no full training rate or hardware spill transactions')
projection['targets_per_second']=1/projection['seconds_per_target']
projection['phase4_native_only_days']=1429.604
projection['phase4_head_ms']=718.648853
projection['timing_comparability']='Different values and five vs ten timing samples. No matched speedup claim; candidate is rejected on its own cost.'
(r/'projection.json').write_text(json.dumps(projection,indent=2)+'\n')
print(json.dumps(projection,indent=2))
print('\n'.join(f"|{x['batch']}|{x['tokens']}|{x['cap']}|{x['median_ms']:.6f}|{x['registers']}|{x['spills']}|" for x in data['rows']))
