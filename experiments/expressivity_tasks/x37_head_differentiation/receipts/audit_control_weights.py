"""Read-only audit of the reused X34 checkpoint; no GPU or training."""
import json
import hashlib
from pathlib import Path
import torch

R=Path(__file__).resolve().parent
P=R.parents[1]/'x34_lm_pilot/runs/pilot_tape_lm/latest.pt'
with P.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
checkpoint=torch.load(P,map_location='cpu',weights_only=False)
result=dict(checkpoint=str(P.relative_to(R.parents[1])),sha256=digest,
            step=checkpoint['step'],tokens=checkpoint['step']*4096,
            head_projection_distances={})
for name in ('W_k','W_q','W_v','W_write','W_pop','W_append'):
    rows=checkpoint['model'][f'blocks.2.cell.{name}.weight'].reshape(4,-1)
    result['head_projection_distances'][name]={f'0-{j}':dict(
        bit_equal=torch.equal(rows[0],rows[j]),l2=float((rows[0]-rows[j]).norm()))
        for j in (1,2,3)}
assert all(not item['bit_equal'] for group in result['head_projection_distances'].values()
           for item in group.values())
result['conclusion']='The trained X34 control has unequal head parameters in every audited projection, despite 16/16 identical peak-live stories.'
(R/'control_checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['conclusion'])
