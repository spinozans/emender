"""Integrity-checked rollup of the 100 isolated post-promotion reproductions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
result=json.loads((root/'nine_x.json').read_text())
assert result['repeats']==len(result['rows'])==100
counts=Counter();inputs=set();checkpoints=set();digests=set();missing_metadata=[]
for row in result['rows']:
    raw=(root/row['console']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==row['console_sha256']
    text=raw.decode(errors='replace');records=[]
    for line in text.splitlines():
        try:record=json.loads(line)
        except ValueError:continue
        if isinstance(record,dict):records.append(record)
    metadata=[r for r in records if 'checkpoint_sha256' in r]
    if metadata:
        meta=metadata[-1];checkpoints.add(meta['checkpoint_sha256']);inputs.add(meta['input_sha256'])
        assert meta['checkpoint_sha256']==result['checkpoint_sha256']
        assert meta['input_sha256']==row['input_sha256']
    else:missing_metadata.append(row['repeat'])
    if row['exit_code']==124:category='timeout'
    elif row['exit_code']==-15:category='attended_stalled_attempt_termination'
    elif 'int32 overflow detected for operation add' in text:category='int32_add_assert'
    elif 'nonfinite reproduction logits' in text:category='nonfinite_logits'
    elif row['exit_code']==0:
        category='finite_exit';assert row['logits_sha256'];digests.add(row['logits_sha256'])
    else:category='other_failure'
    counts[category]+=1
assert len(inputs)==len(checkpoints)==1
summary=dict(status=result['status'],attempts=100,outcomes=dict(counts),distinct_finite_logit_hashes=len(digests),matches_first_finite_hash=sum(r['status']=='passed' for r in result['rows']),checkpoint_sha256=result['checkpoint_sha256'],input_sha256=next(iter(inputs)),missing_terminal_metadata=missing_metadata,interpretation='FAIL: 9x remains blocked. Finite hashes are repeat-consistency checks against the first finite result, not an accuracy oracle. Device assertions/nonfinite outputs and hash variation coexist with the unresolved race inventory; no causal repair or race-free claim is made.',scope_decision='Operator part22 work order steps 3–4 explicitly places the existing races and post-promotion 9x work outside the uninitialized-access promotion blockers.')
(root/'nine_x_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
