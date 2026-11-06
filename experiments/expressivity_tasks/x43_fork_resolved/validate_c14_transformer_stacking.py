"""Independent CPU replay of frozen P3 probabilities, strata and signs."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from .c14_documents import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/c14_transformer_stacking'


def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    start=time.monotonic()
    load=lambda name:json.loads((OUT/name).read_text())
    identity=load('prediction_identity.json'); frozen=load('calibration_freeze.json')
    gates=load('frozen_gates.json'); table=load('table.json'); verdict=load('verdicts.json')
    assert frozen['final_test_scored'] is False
    assert frozen['analysis_plan_sha256']==sha(OUT/'analysis_plan.json')
    assert frozen['gate_sha256']==sha(OUT/'frozen_gates.json')
    assert frozen['prediction_sha256']==identity['sha256']==sha(Path(identity['path']))
    assert frozen['scorer_sha256']==sha(ROOT/'c14_transformer_stacking_score.py')
    for path,receipt in load('protocol_inputs.json').items():
        assert sha(ROOT.parents[2]/path)==receipt['sha256'],path
    state=load('resume.json')
    assert state['status']=='complete' and state['targets']==434204084 and state['windows']==3431242
    assert state['config']['code_sha256']==sha(ROOT/'c14_transformer_stacking_predict.py')
    assert state['config']['gpu'] in ('2','4') and state['config']['modes']==['dense','v2_on']
    assert state['config']['deterministic_algorithms']
    # Verify frozen model artifacts remain precisely the original paired artifacts.
    paired_state=json.loads((ROOT/'receipts/c14_transformer_eval/resume.json').read_text())
    for name,h in state['config']['checkpoint_hashes'].items():
        assert h==paired_state['config']['checkpoint_hashes'][name]
    assert sha(Path('/home/erikg/c14_dense_bytes/transformer_dense.pt'))==state['config']['checkpoint_hashes']['dense']
    trained=json.loads((ROOT/'receipts/c14_transformer/training.json').read_text())
    assert sha(Path(trained['checkpoint']))==state['config']['checkpoint_hashes']['v2']
    assert sha(Path('/home/erikg/c14_consultation_features/registered_eval/journal.jsonl'))==state['config']['journal_sha256']
    assert sha(Path('/home/erikg/c14_dense_bytes/eval_windows.jsonl'))==state['config']['windows_sha256']
    paired=json.loads((ROOT/'receipts/c14_transformer_eval/table.json').read_text())
    parity=0
    for key,old in paired.items():
        if key in table:
            actual=table[key]
            assert actual['targets']==old['targets'] and actual['infinite_losses']==0
            assert abs(actual['nll']-old['nll'])<1e-8
            parity+=1
    a=np.load(identity['path'],mmap_mode='r'); assert list(a.shape)==identity['shape']==[434204084,10]
    sums={}; counts={}; infinities={}
    # Formula and strata are reconstructed here, without importing scoring helpers.
    for lo in range(0,len(a),1_000_000):
        b=np.array(a[lo:lo+1_000_000],dtype=np.float64)
        assert np.isfinite(b).all() and np.all((b[:,4]>=0)&(b[:,4]<=1))
        assert np.isin(b[:,7],[0,1]).all() and np.isin(b[:,8],[0,1]).all()
        support=b[:,5]>0; fork=b[:,7]==1; q=b[:,4]; positive=q>0
        k=np.rint(np.expm1(b[:,6])).astype(int)
        assert ((k>=0)&(k<=b[:,9]+1)).all()
        assert ((k>0)==support).all()
        assert np.all(q[support&~fork&positive]==1)
        masks=dict(total=np.ones(len(b),bool),wikipedia_support=support,empty_support=~support,
            supported_target=positive,nonempty_zero_target=support&~positive,zero_target=~positive,
            fork=fork,nonfork=~fork,forced=support&~fork,forced_zero_target=support&~fork&~positive,
            supported_fork=positive&fork,supported_nonfork=positive&~fork)
        for label,low,high in [('1-3',1,3),('4-12',4,12),('13-64',13,64),('65+',65,1024)]:
            masks['length_'+label+'/supported']=support&positive&(k>=low)&(k<=high)
        loss={'dense':b[:,0],'v2_on':b[:,1]}
        with np.errstate(divide='ignore'): loss['array_alone']=-np.log(q)
        for arm,name in enumerate(('dense','v2_on')):
            f=gates[name]
            cov=np.stack((b[:,arm+2],b[:,6],b[:,5]),axis=1)
            raw=((cov-np.array(f['center']))/np.array(f['scale']))@np.array(f['weights'][1:])+f['weights'][0]
            g=1/(1+np.exp(-np.clip(raw,-35,35)))
            probability=np.where(support,(1-g)*np.exp(-b[:,arm])+g*q,np.exp(-b[:,arm]))
            with np.errstate(divide='ignore'):
                loss[name+'_soft']=-np.log(probability)
                loss[name+'_c13']=-np.log(np.where(support&~fork,q,probability))
        for half,selection in [('calibration',b[:,8]==0),('final_test',b[:,8]==1)]:
            for stratum,mask in masks.items():
                m=mask&selection
                for name,l in loss.items():
                    key=f'{name}/{half}/{stratum}'; selected=l[m]
                    finite=np.isfinite(selected)
                    sums[key]=sums.get(key,0.)+float(selected[finite].sum())
                    counts[key]=counts.get(key,0)+len(selected)
                    infinities[key]=infinities.get(key,0)+int((~finite).sum())
        if lo%100_000_000==0: print('VALIDATE',lo,flush=True)
    max_error=0
    for key,s in sums.items():
        r=table[key]; error=abs(s-r['finite_nll_sum']); max_error=max(error,max_error)
        assert error<1e-4,(key,error)
        assert counts[key]==r['targets'] and infinities[key]==r['infinite_losses'],key
        expected='Infinity' if infinities[key] else (s/counts[key] if counts[key] else None)
        if isinstance(expected,str): assert r['nll']==expected
        elif expected is None: assert r['nll'] is None
        else: assert abs(expected-r['nll'])<1e-11
    for name,f in gates.items():
        r=table[name+'_soft/calibration/fork']
        assert r['targets']==f['calibration_targets']
        assert abs(r['nll']-f['calibration_fork_nll'])<1e-10
        assert np.all(np.abs(f['weights'])<=20)
    for name in ['dense','v2_on','dense_c13','v2_on_c13','dense_soft','v2_on_soft','array_alone']:
        for half in ['calibration','final_test']:
            for total,parts in [('total',['supported_target','zero_target']),('total',['fork','nonfork']),
                                ('wikipedia_support',['supported_target','nonempty_zero_target']),
                                ('supported_target',['supported_fork','supported_nonfork']),
                                ('supported_target',['length_'+k+'/supported' for k in ['1-3','4-12','13-64','65+']])]:
                r=table[f'{name}/{half}/{total}']; p=[table[f'{name}/{half}/{x}'] for x in parts]
                assert sum(x['targets'] for x in p)==r['targets']
                assert sum(x['infinite_losses'] for x in p)==r['infinite_losses']
                assert abs(sum(x['finite_nll_sum'] for x in p)-r['finite_nll_sum'])<1e-4
    primary=table['v2_on_c13/final_test/total']
    assert primary['nll']=='Infinity' and primary['infinite_losses']>0
    assert verdict['literal_C13_P3']=='FAIL' and not verdict['literal_C13_beats_consultation_alone'] and not verdict['literal_C13_beats_dense_calibration']
    assert primary['infinite_losses']==table['dense_c13/final_test/total']['infinite_losses']
    assert primary['infinite_losses']==table['v2_on/final_test/forced_zero_target']['targets']
    soft=table['v2_on_soft/final_test/total']['nll']
    comparisons=dict(soft_beats_consultation_alone=soft<table['v2_on/final_test/total']['nll'],
        soft_beats_dense_calibration=soft<table['dense_soft/final_test/total']['nll'],
        soft_beats_transformer=soft<1.404434206)
    for key,value in comparisons.items(): assert verdict[key]==value
    assert verdict['P3_without_walk']==('PASS' if comparisons['soft_beats_consultation_alone'] and comparisons['soft_beats_dense_calibration'] else 'FAIL')
    assert verdict['P3']==verdict['P3_without_walk']==verdict['primary_soft_P3']
    assert verdict['stacked_nll']==soft
    assert verdict['beats_consultation_alone']==comparisons['soft_beats_consultation_alone']
    assert verdict['beats_dense_calibration']==comparisons['soft_beats_dense_calibration']
    result=dict(status='green',independent_loss_denominator_replays=len(sums),
        paired_aggregate_checks=parity,max_finite_sum_error=max_error,
        exact_walk_failure_preserved=True,all_434204084_positions_checked=True,
        seconds=time.monotonic()-start,source_sha256=sha(__file__),
        scope='full CPU replay; frozen gate/checkpoint/cache identity; calibrated fork objective; partitions and verdict signs')
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
