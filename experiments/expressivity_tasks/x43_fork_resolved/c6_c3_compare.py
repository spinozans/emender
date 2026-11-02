"""Additive preregistered C3-exact empty-support walk comparison."""
import json
import numpy as np
from .c6 import ROOT, OUT, digest, write


def score():
    protocol=ROOT/'PREREGISTRATION_C6_C3_COMPARISON.md'
    assert digest(protocol) in (ROOT/'PROGRAM.md').read_text()
    results=json.loads((OUT/'results.json').read_text())
    assert results['frozen_gate_sha256']==digest(OUT/'frozen_gates.json')
    fits=json.loads((OUT/'frozen_gates.json').read_text())
    z=np.load(OUT/'index.npz'); split=int(.9*len(z['fork']))
    positions=np.arange(split+1,len(z['fork'])); half=len(positions)//2
    test=np.arange(len(positions))>=half
    fork=z['fork'][positions]; s=z['support'][positions]; q=z['q'][positions]
    def summary(p):
        loss=-np.log(p)
        masks=dict(overall=np.ones(len(p),bool),fork=fork,forced=~fork,
                   covered=s>0,unsupported=s==0,zero_target=(s>0)&(q==0))
        return {name:dict(n=int((test&m).sum()),nll=float(loss[test&m].mean()) if (test&m).any() else None)
                for name,m in masks.items()}
    report=dict(protocol_sha256=digest(protocol),frozen_gate_sha256=results['frozen_gate_sha256'],
                scope='C3-exact walk coverage; corpus-selected contexts, train-prefix counts, transductive diagnostic',
                forced_empty_support_final=int((test&~fork&(s==0)).sum()),models={},c3=results['c3'])
    for name,f in fits.items():
        ev=np.load(OUT/(name+'_prediction.npz')); p=ev['p']
        x=np.column_stack([np.ones(len(p)),ev['confidence'],np.log1p(z['context_length'][positions]),np.log1p(s)])
        x[:,1:]=(x[:,1:]-f['center'])/f['scale']
        g=1/(1+np.exp(-np.clip(x@f['weights'],-35,35)))
        mixed=np.where(s>0,(1-g)*p+g*q,p)
        walk=(~fork)&(s>0)
        report['models'][name]=dict(c3_protocol_standalone=summary(np.where(walk,q,p)),
            c3_protocol_fusion=summary(np.where(walk,q,np.where(fork,mixed,p))))
    a=report['models']['emender_aware']; b=report['models']['transformer_aware']
    report['standalone_emender_minus_transformer']=a['c3_protocol_standalone']['overall']['nll']-b['c3_protocol_standalone']['overall']['nll']
    report['fusion_emender_minus_transformer']=a['c3_protocol_fusion']['overall']['nll']-b['c3_protocol_fusion']['overall']['nll']
    return report


if __name__=='__main__':
    report=score(); write('c3_comparison.json',report)
    print(json.dumps(report,indent=2))
