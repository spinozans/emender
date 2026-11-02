"""Independent CPU replay of C6 final scoring and hash contracts."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/c6'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate():
    training=json.loads((OUT/'training.json').read_text())
    result=json.loads((OUT/'results.json').read_text())
    index=json.loads((OUT/'index.json').read_text())
    comparison=json.loads((OUT/'c3_comparison.json').read_text())
    assert comparison['protocol_sha256']==sha(ROOT/'PREREGISTRATION_C6_C3_COMPARISON.md')
    assert comparison['frozen_gate_sha256']==result['frozen_gate_sha256']
    dependencies=json.loads((OUT/'dependencies.json').read_text())
    for path, receipt in dependencies['files'].items():
        assert sha(ROOT.parents[2]/path)==receipt['sha256']
        assert receipt['matches_execution_source_commit']
    assert training['code_sha256']==sha(ROOT/'c6.py')
    assert training['gdn2_code_sha256']==sha(ROOT/'c6_gdn2.py')
    assert training['gdn2_protocol_sha256']==sha(ROOT/'PREREGISTRATION_C6_GDN2.md')
    assert training['gdn2_physical_gpu'] in ('2','4')
    gdn2=json.loads((OUT/'gdn2_source.json').read_text())
    assert gdn2['architecture'].endswith('.GatedDeltaNet2')
    for path, digest in gdn2['files'].items():
        assert sha(Path(path))==digest
    assert training['protocol_sha256']==sha(ROOT/'PREREGISTRATION_C6.md')
    assert training['protocol_sha256'] in (ROOT/'PROGRAM.md').read_text()
    assert index['index_sha256']==sha(OUT/'index.npz')
    assert index['corpus_sha256']==sha(ROOT/'input.txt')
    assert result['frozen_gate_sha256']==sha(OUT/'frozen_gates.json')
    freeze=json.loads((OUT/'calibration_freeze.json').read_text())
    assert freeze==dict(sha256=result['frozen_gate_sha256'],final_test_scored=False)
    assert training['physical_gpu'] in ('2','4')
    gates=json.loads((OUT/'frozen_gates.json').read_text())
    z=np.load(OUT/'index.npz'); positions=np.arange(index['split']+1,index['n'])
    half=len(positions)//2; assert half==55769 and len(positions)-half==55770
    test=np.arange(len(positions))>=half
    fork=z['fork'][positions]; q=z['q'][positions]; s=z['support'][positions]
    masks=dict(overall=np.ones(len(test),bool),fork=fork,forced=~fork,
               covered=s>0,unsupported=s==0,zero_target=(s>0)&(q==0))
    checks=0
    for row in training['models']:
        name=row['name']; ev=np.load(OUT/(name+'_prediction.npz'))
        assert row['prediction_sha256']==sha(OUT/(name+'_prediction.npz'))==gates[name]['prediction_sha256']
        assert row['training_seconds']>=180 and row['steps']>0
        assert row['tokens']==row['steps']*16*128
        assert gates[name]['calibration_targets']==half
        p=ev['p']; fit=gates[name]
        x=np.column_stack([np.ones(len(p)),ev['confidence'],np.log1p(z['context_length'][positions]),np.log1p(s)])
        x[:,1:]=(x[:,1:]-fit['center'])/fit['scale']
        g=1/(1+np.exp(-np.clip(x@fit['weights'],-35,35)))
        mix=np.where(s>0,(1-g)*p+g*q,p)
        variants=dict(neural=p,standalone=np.where(fork,p,1) if row['fork_aware'] else p,
                      fusion=np.where(fork,mix,1))
        covered_walk=(~fork)&(s>0)
        c3variants=dict(c3_protocol_standalone=np.where(covered_walk,q,p),
                        c3_protocol_fusion=np.where(covered_walk,q,np.where(fork,mix,p)))
        for kind,prediction in c3variants.items():
            nll=-np.log(prediction)
            for stratum,m in masks.items():
                selected=test&m; measured=comparison['models'][name][kind][stratum]
                assert measured['n']==int(selected.sum())
                assert measured['nll'] is None if not selected.any() else np.isclose(measured['nll'],nll[selected].mean(),atol=1e-10,rtol=0)
                checks+=1
        for kind,prediction in variants.items():
            loss=-np.log(prediction)
            for stratum,m in masks.items():
                selected=test&m; measured=result['models'][name][kind][stratum]
                assert measured['n']==int(selected.sum())
                assert measured['nll'] is None if not selected.any() else np.isclose(measured['nll'],loss[selected].mean(),atol=1e-10,rtol=0)
                checks+=1
    a=result['models']['emender_aware']; b=result['models']['transformer_aware']
    delta=a['standalone']['overall']['nll']-b['standalone']['overall']['nll']
    assert result['decisive_emender_minus_transformer']==delta
    assert result['verdict']==('survives' if delta<0 else 'refuted')
    return dict(status='green',scoring_checks=checks,successful_models=len(training['models']),
                retained_failures=len(training['failures']),calibration_targets=half,
                final_targets=int(test.sum()),verdict=result['verdict'])


if __name__=='__main__':
    print(json.dumps(validate()))
