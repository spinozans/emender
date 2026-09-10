#!/usr/bin/env python3
"""Synthetic CUDA CE precision measurement; no checkpoint or optimizer."""
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    root=Path(os.environ['PROBE_ROOT'])
    path=root/'result.json'
    assert not path.exists()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    reports=[]
    for rows,vocab in ((1,64),(1,50281),(128,50281),(1,50304)):
        target_id=7 if vocab==64 else 32750
        winner=9 if vocab==64 else 12502
        logits=torch.full((rows,vocab),-20.0,device='cuda',dtype=torch.bfloat16)
        logits[:,winner]=20;logits[:,target_id]=0.5625
        targets=torch.full((rows,),-100,device='cuda',dtype=torch.long);targets[-1]=target_id
        for enabled in (False,True):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
                loss=F.cross_entropy(logits,targets,reduction='sum')
                explicit=F.cross_entropy(logits.float(),targets,reduction='sum')
                manual=-logits[-1].float().log_softmax(-1)[target_id]
            reports.append({'rows':rows,'vocab':vocab,'autocast_enabled':enabled,
                'input_dtype':str(logits.dtype),'loss_dtype':str(loss.dtype),
                'loss':float(loss),'explicit_fp32':float(explicit),'manual_fp32':float(manual)})
    report={'status':'completed-diagnostic','torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(),'local_rank':rank,'current_device':torch.cuda.current_device(),
            'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'synthetic CE only; not an explanation of the observed model discrepancy without matching evidence',
            'cases':reports}
    with path.open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
    path.chmod(0o600)
    print('CE_PRECISION_PROBE_COMPLETE '+json.dumps(report),flush=True)


if __name__=='__main__':main()
