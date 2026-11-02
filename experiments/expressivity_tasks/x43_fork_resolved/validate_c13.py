"""Independent CPU loss, denominator, fusion and frozen-recovery audit."""
import json
import numpy as np
from .c13 import ROOT, OUT, digest

def main():
    load=lambda p:json.loads(p.read_text())
    for name,expected in load(OUT/'dependencies.json').items():
        assert digest(ROOT/name)==expected,name
    report=load(OUT/'results.json');recovery=load(OUT/'recovery.json');gates=load(OUT/'frozen_gates.json')
    assert load(OUT/'terminal.json')['exit_code']==0
    assert recovery['physical_gpu'] in ('2','4') and (OUT/'lease.txt').read_text().strip()==recovery['physical_gpu']
    assert recovery['new_training_arms']==0 and recovery['source_sha256']==digest(ROOT/'c13.py')
    assert recovery['protocol_sha256']==report['protocol_sha256']==digest(ROOT/'PREREGISTRATION_C13.md')
    assert not load(OUT/'calibration_freeze.json')['final_test_scored']
    assert load(OUT/'calibration_freeze.json')['gate_sha256']==digest(OUT/'frozen_gates.json')
    z={k:v for k,v in np.load(ROOT/'receipts/c6/index.npz').items()};split=int(.9*len(z['fork']));pos=np.arange(split+1,len(z['fork']));test=np.arange(len(pos))>=len(pos)//2
    support=z['support'][pos];fork=z['fork'][pos];q=z['q'][pos]
    x=np.load(ROOT/'receipts/c12/disjoint.npz')['support'][pos]
    masks=dict(total=np.ones(len(pos),bool),fork=fork,forced=~fork,ood_self=support==0,ood_x=x==0,nonempty_zero_target=(support>0)&(q==0))
    oldmeta=load(ROOT/'receipts/c12/training.json');checks=0
    for row in recovery['models']:
        arm=row['arm'];assert row['steps']==next(r['steps'] for r in oldmeta['models'] if r['name']==arm+'_disjoint')
        assert all(v['passed'] and v['n']==len(pos) for v in row['parity'].values())
        path=OUT/(arm+'_self_prediction.npz');assert digest(path)==row['prediction_sha256']
        pred=np.load(path);p=pred['p'].astype(float);f=gates[arm]
        inputs=np.column_stack([np.ones(len(p)),pred['confidence'],np.log1p(z['context_length'][pos]),np.log1p(support)])
        inputs[:,1:]=(inputs[:,1:]-f['center'])/f['scale'];g=1/(1+np.exp(-np.clip(inputs@f['weights'],-35,35)))
        stack=p.copy();at=fork&(support>0);stack[at]=(1-g[at])*p[at]+g[at]*q[at];at=(~fork)&(support>0);stack[at]=q[at]
        old=np.load(ROOT/'receipts/c12'/(arm+'_disjoint_prediction.npz'))
        values=dict(dense=np.load(ROOT/'receipts/part2'/(arm+'_prediction.npz'))['p'],x_index=np.exp(-old['with']),no_features=np.exp(-old['without']),self_index=p,stacking=stack)
        for recipe,prob in values.items():
            assert np.isfinite(prob).all() and (prob>0).all() and (prob<=1+1e-6).all()
            for k,mask in masks.items():
                at=test&mask;actual=report['models'][arm][recipe][k]
                assert actual['n']==int(at.sum())
                assert np.isclose(actual['nll'],-np.log(prob[at]).mean(),rtol=0,atol=1e-10)
                checks+=1
        rows=report['models'][arm]
        for label,left,right,columns in [('P1','self_index','x_index',('total','fork')),('P2','self_index','dense_fusion',('total','fork','ood_self')),('P3','stacking','self_index',('total','fork','ood_self'))]:
            assert report['predictions'][arm][label]=={k:rows[left][k]['nll']<rows[right][k]['nll'] for k in columns}
        assert report['models'][arm]['self_index']['total']['n']==55770
        assert report['models'][arm]['self_index']['fork']['n']==32415
        assert report['models'][arm]['self_index']['ood_self']['n']==9049
        assert report['models'][arm]['self_index']['ood_x']['n']==41668
    print(json.dumps(dict(status='green',loss_denominator_checks=checks,parity_vectors=4,physical_gpu=recovery['physical_gpu'])))
if __name__=='__main__':main()
