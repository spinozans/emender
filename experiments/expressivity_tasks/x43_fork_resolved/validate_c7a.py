"""Independent CPU replay of C7a receipts and final-test probabilities."""
import json
import numpy as np
from .c7a import ROOT, OUT, digest


def main():
    train=json.loads((OUT/'training.json').read_text())
    result=json.loads((OUT/'results.json').read_text())
    fits=json.loads((OUT/'frozen_gates.json').read_text())
    freeze=json.loads((OUT/'calibration_freeze.json').read_text())
    assert freeze['final_test_scored'] is False
    assert result['gate_sha256']==freeze['gate_sha256']==digest(OUT/'frozen_gates.json')
    assert train['protocol_sha256']==result['protocol_sha256']==digest(ROOT/'PREREGISTRATION_C7a.md')
    assert train['code_sha256']==digest(ROOT/'c7a.py')
    assert train['index_sha256']==digest(ROOT/'receipts/c6/index.npz')
    assert train['physical_gpu'] in ('2','4') and not train['failures']
    assert json.loads((OUT/'terminal.json').read_text())['exit_code']==0
    for row in train['models']:
        assert 180 <= row['training_seconds'] < 190
        assert row['tokens']==row['steps']*16*128 and row['steps']>0
        assert row['prediction_sha256']==digest(OUT/(row['name']+'_prediction.npz'))
    assert [r['arm'] for r in train['models']]==['emender','transformer']
    z=np.load(ROOT/'receipts/c6/index.npz'); split=int(.9*len(z['fork']))
    pos=np.arange(split+1,len(z['fork'])); test=np.arange(len(pos))>=len(pos)//2
    assert test.sum()==55770
    fork=z['fork'][pos]; support=z['support'][pos]; q=z['q'][pos]
    checks=0
    for name,inp in result['inputs'].items():
        assert inp==freeze['inputs'][name]
        assert digest(ROOT/inp['path'])==inp['sha256']
        ev=np.load(ROOT/inp['path']); p=ev['p']; f=fits[name]
        x=np.column_stack([np.ones(len(p)),ev['confidence'],np.log1p(z['context_length'][pos]),np.log1p(support)])
        x[:,1:]=(x[:,1:]-f['center'])/f['scale']
        g=1-1/(1+np.exp(-np.clip(x@f['weights'],-35,35)))
        variants={'neural':p}
        def fuse(g):
            p_mix=np.where(support>0,g*p+(1-g)*q,p)
            return np.where((~fork)&(support>0),q,np.where(fork,p_mix,p))
        variants['common_c3']=fuse(g)
        if name.endswith('_boosted'):
            joint=ev['neural_gate']
            assert ((joint>=0)&(joint<=1)).all() and (joint[support==0]==1).all()
            variants['joint_gate']=fuse(joint)
        for variant,pred in variants.items():
            assert np.isfinite(pred).all() and (pred>0).all()
            for label,m in [('overall',np.ones(len(p),bool)),('fork',fork),('forced',~fork),('empty',support==0),('zero_target',(support>0)&(q==0))]:
                saved=result['models'][name][variant][label]; m=m&test
                assert saved['n']==m.sum()
                assert abs(saved['nll']-(-np.log(pred[m])).mean())<1e-10
                checks+=1
    # Historical dense reference numerical reproduction under common protocol.
    for arm,expected in [('emender',1.673012),('transformer',1.746996)]:
        assert abs(result['models'][arm+'_dense']['common_c3']['overall']['nll']-expected)<1e-6
    print(json.dumps(dict(status='green',replayed_loss_denominators=checks,training_arms=2,final_targets=55770)))

if __name__=='__main__': main()
