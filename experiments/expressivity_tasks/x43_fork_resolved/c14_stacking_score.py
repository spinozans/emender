"""Frozen C13 gates and post-hoc C14 decomposition; no extraction or training.

Hard single-branch walks are scored literally, including infinite loss on foreign
array contradictions. Soft fusion is a separately labelled no-walk diagnostic.
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from .c14_documents import sha
from .c14_stacking_predict import ROOT, OUT, PRED, EXPECTED


def write(name, value):
    p=OUT/name
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)


def gate_features(rows, arm):
    """C13 feature order: intercept, max neural p, log1p k, log1p count."""
    return np.column_stack((np.ones(len(rows)),rows[:,2+arm],rows[:,6],rows[:,5])).astype(np.float64)


def objective(w, x, p, q):
    raw=x@w
    g=1/(1+np.exp(-np.clip(raw,-35,35)))
    prob=(1-g)*p+g*q
    return -np.log(prob).mean(),x.T@((p-q)*g*(1-g)/prob*(np.abs(raw)<35))/len(prob)


def fuse(p, q, support, fork, g, walk):
    mixed=np.where(support,(1-g)*p+g*q,p)
    return np.where(support&~fork,q,mixed) if walk else mixed


def masks_for(a):
    support=a[:,5]>0; fork=a[:,7]>0; positive=a[:,4]>0
    masks=dict(total=np.ones(len(a),bool),window_reset_1_16=a[:,9]<16,
        window_interior_17_128=a[:,9]>=16,wikipedia_support=support,
        empty_support=~support,supported_target=positive,agreement=positive,
        nonempty_zero_target=support&~positive,zero_target=~positive,
        fork=fork,nonfork=~fork,forced=support&~fork,
        forced_zero_target=support&~fork&~positive)
    # Decode log1p(k) using nearest-integer inverse; float32 metadata cannot
    # be compared directly to float64 logarithmic endpoints.
    k=np.rint(np.expm1(a[:,6].astype(np.float64))).astype(np.int32)
    for label,lo,hi in [('1-3',1,3),('4-12',4,12),('13-64',13,64),('65+',65,1024)]:
        length=support&(k>=lo)&(k<=hi)
        masks['length_'+label]=length
        for name,condition in [('supported',positive),('zero',~positive)]:
            masks['length_'+label+'/'+name]=length&condition
            for flabel,fcondition in [('fork',fork),('nonfork',~fork)]:
                masks['length_'+label+'/'+name+'/'+flabel]=length&condition&fcondition
    masks['supported_fork']=positive&fork
    masks['supported_nonfork']=positive&~fork
    return masks


def add(sums, key, loss):
    old=sums.setdefault(key,[0.,0,0])
    finite=np.isfinite(loss)
    assert not np.isnan(loss).any() and not np.isneginf(loss).any()
    old[0]+=float(loss[finite].sum(dtype=np.float64)); old[1]+=len(loss)
    old[2]+=int((~finite).sum())


def table(sums):
    return {key:dict(finite_nll_sum=s,targets=n,infinite_losses=i,
            total_nll='Infinity' if i else s,nll='Infinity' if i else (s/n if n else None))
            for key,(s,n,i) in sums.items()}


def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    start=time.monotonic()
    identity=json.loads((OUT/'prediction_identity.json').read_text())
    assert sha(PRED)==identity['sha256']
    resume=json.loads((OUT/'resume.json').read_text())
    assert resume['status']=='complete' and resume['targets']==EXPECTED[1]
    a=np.load(PRED,mmap_mode='r'); assert a.shape==(EXPECTED[1],10)
    if not (OUT/'calibration_freeze.json').exists():
        # Only calibration rows enter feature moments or the objective. The
        # final-test metrics are never computed until both gates are on disk.
        eligible=(a[:,8]==0)&(a[:,7]>0)&(a[:,5]>0)
        cal=np.array(a[eligible]); del eligible
        fits={}
        for arm,name in enumerate(('dense','v2_on')):
            x=gate_features(cal,arm); center=x[:,1:].mean(0); scale=x[:,1:].std(0)
            scale[scale==0]=1; x[:,1:]=(x[:,1:]-center)/scale
            p=np.exp(-cal[:,arm].astype(np.float64)); q=cal[:,4].astype(np.float64)
            fit=minimize(objective,np.zeros(4),args=(x,p,q),jac=True,method='L-BFGS-B',
                bounds=[(-20,20)]*4,options={'maxiter':1000,'ftol':1e-12})
            assert fit.success,fit.message
            fits[name]=dict(weights=fit.x.tolist(),center=center.tolist(),scale=scale.tolist(),
                calibration_fork_nll=float(fit.fun),calibration_targets=len(cal),
                iterations=int(fit.nit),message=str(fit.message))
            print('FIT',name,json.dumps(fits[name]),flush=True)
            del x,p,q
        del cal
        write('frozen_gates.json',fits)
        write('calibration_freeze.json',dict(final_test_scored=False,
            prediction_sha256=identity['sha256'],gate_sha256=sha(OUT/'frozen_gates.json'),
            scorer_sha256=sha(__file__),fit_rule='all calibration nonempty fork positions; C13 full four-feature logistic gate; unchanged bounds, optimizer and tolerances',
            primary='literal C13 forced nonempty single-byte walk; no target-conditioned abstention or smoothing',
            supplemental='same frozen gate on every nonempty support position, without forced walk; not a rescue of registered P3'))
    frozen=json.loads((OUT/'calibration_freeze.json').read_text())
    assert frozen['prediction_sha256']==identity['sha256'] and frozen['scorer_sha256']==sha(__file__)
    assert frozen['gate_sha256']==sha(OUT/'frozen_gates.json')
    fits=json.loads((OUT/'frozen_gates.json').read_text()); sums={}
    transformer=json.loads((OUT/'transformer_identity.json').read_text())
    assert sha(Path(transformer['path']))==transformer['sha256']
    control=np.load(transformer['path'],mmap_mode='r')
    assert control.shape==(len(a),)
    for lo in range(0,len(a),1_000_000):
        b=np.array(a[lo:lo+1_000_000]); masks=masks_for(b)
        q=b[:,4].astype(np.float64); support=b[:,5]>0; fork=b[:,7]>0
        losses={'dense':b[:,0].astype(np.float64),'v2_on':b[:,1].astype(np.float64),
            'transformer':np.asarray(control[lo:lo+len(b)],dtype=np.float64)}
        with np.errstate(divide='ignore'): losses['array_alone']=-np.log(q)
        for arm,name in enumerate(('dense','v2_on')):
            f=fits[name]; x=gate_features(b,arm); x[:,1:]=(x[:,1:]-f['center'])/f['scale']
            g=1/(1+np.exp(-np.clip(x@np.array(f['weights']),-35,35)))
            p=np.exp(-b[:,arm].astype(np.float64))
            for walk,label in [(True,'c13'),(False,'soft')]:
                with np.errstate(divide='ignore'):
                    losses[name+'_'+label]=-np.log(fuse(p,q,support,fork,g,walk))
        for half,selection in [('calibration',b[:,8]==0),('final_test',b[:,8]==1)]:
            for stratum,mask in masks.items():
                selected=mask&selection
                for name,loss in losses.items(): add(sums,'/'.join((name,half,stratum)),loss[selected])
        if lo%20_000_000==0: print('SCORE',lo,flush=True)
    result=table(sums)
    # Independent aggregate parity against the committed paired table.
    paired=json.loads((ROOT/'receipts/c14_paired_eval/table.json').read_text())
    parity=[]
    for key,old in paired.items():
        if key in result:
            new=result[key]; assert new['targets']==old['targets']
            error=abs(new['total_nll']-old['total_nll'])
            assert abs(new['nll']-old['nll'])<1e-8,(key,error)
            parity.append(dict(key=key,targets=new['targets'],sum_error=error,mean_error=new['nll']-old['nll']))
    write('table.json',result)
    primary=result['v2_on_c13/final_test/total']; dense=result['dense_c13/final_test/total']
    # Infinity ties do not beat each other. Finite diagnostics have separate signs.
    def number(x): return float(x)
    v=number(primary['nll']); baseline=number(result['v2_on/final_test/total']['nll'])
    verdict=dict(P3='PASS' if v<baseline and v<number(dense['nll']) else 'FAIL',
        beats_consultation_alone=v<baseline,beats_dense_calibration=v<number(dense['nll']),
        beats_dense_transformer=v<1.404434206,
        dense_calibration_nll=dense['nll'],stacked_nll=primary['nll'],
        soft_diagnostic_nll=result['v2_on_soft/final_test/total']['nll'],
        soft_dense_nll=result['dense_soft/final_test/total']['nll'],
        P3_without_walk='PASS' if number(result['v2_on_soft/final_test/total']['nll'])<baseline and number(result['v2_on_soft/final_test/total']['nll'])<number(result['dense_soft/final_test/total']['nll']) else 'FAIL',
        soft_beats_consultation_alone=number(result['v2_on_soft/final_test/total']['nll'])<baseline,
        soft_beats_dense_calibration=number(result['v2_on_soft/final_test/total']['nll'])<number(result['dense_soft/final_test/total']['nll']),
        soft_beats_transformer=number(result['v2_on_soft/final_test/total']['nll'])<1.404434206,
        prediction_sha256=identity['sha256'],gate_sha256=frozen['gate_sha256'])
    write('verdicts.json',verdict)
    write('scoring_validation.json',dict(status='passed',aggregate_parity=parity,
        paired_keys_checked=len(parity),seconds=time.monotonic()-start,
        final_test_targets=primary['targets'],predictions_sha256=identity['sha256'],
        exact_walk_infinite_losses=primary['infinite_losses'],
        no_final_test_fit=True,no_retraining=True,no_new_extraction=True))
    print(json.dumps(verdict),flush=True)


if __name__=='__main__': main()
