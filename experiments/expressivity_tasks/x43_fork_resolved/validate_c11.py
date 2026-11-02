"""Independent CPU replay of C11 receipts, strata and feature invariants."""
import json
import numpy as np
from .c11 import ROOT,OUT,digest

def main():
    training=json.loads((OUT/'training.json').read_text())
    results=json.loads((OUT/'results.json').read_text())
    device=json.loads((OUT/'device.json').read_text())
    assert training['protocol_sha256']==results['protocol_sha256']==digest(ROOT/'PREREGISTRATION_C11.md')
    assert training['code_sha256']==digest(ROOT/'c11.py')
    assert device['corpus_sha256']==digest(ROOT/'input.txt')
    assert device['device_sha256']==digest(OUT/'device.npz')
    assert device['index_sha256']==digest(ROOT/'receipts/c6/index.npz')
    x=np.load(OUT/'device.npz')['features']; z=np.load(ROOT/'receipts/c6/index.npz')
    assert np.allclose(x[:,:256].sum(-1),(z['support']>0).astype(float))
    data=np.frombuffer((ROOT/'input.txt').read_bytes(),dtype=np.uint8)
    assert np.array_equal(x[np.arange(len(data)),data],z['q'])
    assert np.array_equal((x[:,:256]>0).sum(-1),z['support'])
    assert np.allclose(x[:,256],z['features'][z['ids'],257])
    assert np.allclose(x[:,257],z['features'][z['ids'],259])
    assert np.array_equal(x[:,258],z['fork'])
    assert np.allclose(x[:,259],np.log1p(z['context_length']))
    split=int(.9*len(z['fork'])); pos=np.arange(split+1,len(z['fork'])); half=len(pos)//2
    f=z['fork'][pos]; s=z['support'][pos]; q=z['q'][pos]; test=np.arange(len(pos))>=half
    masks=dict(overall=np.ones(len(pos),bool),fork=f,forced=~f,empty=s==0,empty_fork=(s==0)&f,empty_forced=(s==0)&~f,zero_target=(s>0)&(q==0))
    gates=json.loads((OUT/'frozen_gates.json').read_text()); freeze=json.loads((OUT/'calibration_freeze.json').read_text())
    assert not freeze['final_test_scored']
    assert freeze['gate_sha256']==results['gate_sha256']==digest(OUT/'frozen_gates.json')
    checks=0
    for name,info in results['inputs'].items():
        path=ROOT/info['path']; assert digest(path)==info['sha256']
        pred=np.load(path); p=pred['p']; g=gates[name]
        xx=np.column_stack([np.ones(len(p)),pred['confidence'],np.log1p(z['context_length'][pos]),np.log1p(s)])
        xx[:,1:]=(xx[:,1:]-g['center'])/g['scale']
        index_weight=1/(1+np.exp(-np.clip(xx@np.array(g['weights']),-35,35)))
        fused=np.where((~f)&(s>0),q,np.where(f&(s>0),(1-index_weight)*p+index_weight*q,p))
        for kind,prob in [('neural',p),('common_c3',fused)]:
            for key,mask in masks.items():
                row=results['models'][name][kind][key]; m=test&mask
                assert row['n']==int(m.sum())
                assert abs(row['nll']-float(-np.log(prob[m]).mean()))<1e-10
                checks+=1
    assert not training['failures'] and len(training['models'])==2
    for row in training['models']:
        assert 180<=row['training_seconds']<200
        assert row['tokens']==row['steps']*16*128
        assert digest(OUT/(row['name']+'_prediction.npz'))==row['prediction_sha256']
    assert training['physical_gpu'] in ('2','4')
    print(json.dumps(dict(status='green',loss_denominator_checks=checks,feature_rows=len(x))))
if __name__=='__main__': main()
