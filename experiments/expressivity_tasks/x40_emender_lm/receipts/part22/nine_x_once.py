"""One isolated full 9x reproduction attempt."""
import hashlib
import io
import traceback
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from experiments.expressivity_tasks.x40_emender_lm import depth
from experiments.expressivity_tasks.x40_emender_lm.model import LM
assert sys.executable=='/home/erikg/emender/.venv/bin/python'
assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3','4')
os.environ.update(X40_STABLE_MAPPING='1',X40_FRESH_POOL='1',X40_FUSED_INIT='1',X40_BATCHED_READS='0')
torch.set_num_threads(1)
p=Path(os.environ['PART22_DEPTH_CHECKPOINT'])
blob=p.read_bytes();checkpoint_sha256=hashlib.sha256(blob).hexdigest()
s=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);del blob
probe=depth.Probe(LM(s['model']['lm.head.weight'].shape[0],ignition=True).cuda())
probe.load_state_dict(s['cursor']['weights']);probe.eval()
pool=depth.ladder.ladder_bin_pool('bin_9x')
ln=pool[2][80:88]
ids=torch.from_numpy(pool[0][80:88,:int(ln.max())].copy()).cuda()
lengths=torch.from_numpy(ln).cuda()
input_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
rows=[];reference=None;first_error=True
with torch.no_grad(),patch.object(depth.memory_model,'profile',depth.depth_profile):
    for repeat in range(1):
        try:
            out=probe(ids,lengths);torch.cuda.synchronize()
            assert bool(torch.isfinite(out).all()),"nonfinite reproduction logits"
            digest=hashlib.sha256(out.detach().cpu().numpy().tobytes()).hexdigest()
            if reference is None:reference=digest
            assert digest==reference,dict(reference=reference,observed=digest)
            row=dict(repeat=repeat,status='passed',logits_sha256=digest)
        except Exception as exc:
            row=dict(repeat=repeat,status='failed',error=repr(exc))
            if first_error:traceback.print_exc();first_error=False
        rows.append(row);print(json.dumps(row),flush=True)
result=dict(checkpoint_sha256=checkpoint_sha256,input_sha256=input_sha256,lengths=ln.tolist(),start=80,scope='full depth.Probe 9x reproduction batch 80:88, one isolated attempt; successful outputs also compared as byte hashes',status='failed' if any(r['status']=='failed' for r in rows) else 'passed',repeats=len(rows),rows=rows)

print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
raise SystemExit(1 if result['status']=='failed' else 0)
