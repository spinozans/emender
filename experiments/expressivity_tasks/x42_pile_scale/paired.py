"""Fail-closed matched exposure plan; train both arms sequentially under one lease."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .data import digest
from .train import train


def plan(configs, data):
    keys=('seed','context','microbatch','accumulation','tokens','eval_every','eval_batches')
    protocols=[{k:c[k] for k in keys} for c in configs]
    if len(protocols)!=2 or protocols[0]!=protocols[1]:
        raise ValueError('paired arms must have identical data/steps/evaluation protocol')
    protocol=protocols[0]
    unit=protocol['microbatch']*protocol['context']*protocol['accumulation']
    protocol['steps']=math.ceil(protocol['tokens']/unit)
    protocol['actual_tokens']=protocol['steps']*unit
    protocol['manifest_sha256']=digest(Path(data)/'manifest.json')
    protocol['order']='affine packed-window permutation, identical seed, reset cursor=0'
    protocol['protocol_sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    return protocol

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--configs',nargs=2,required=True)
    for key in ('data','output','verdict'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args()
    configs=[json.loads(Path(f).read_text()) for f in a.configs]
    protocol=plan(configs,a.data)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'matched_protocol.json').write_text(json.dumps(protocol,indent=2))
    for i,cfg in enumerate(configs):
        try:
            train(cfg,a.data,out/f'arm{i}',a.verdict)
        except Exception as exc:
            (out/'failure.json').write_text(json.dumps(dict(status='failed',arm=i,error=repr(exc))))
            raise
