"""Common C3 calibration/forced-walk evaluation, with joint gate as separate row."""
import json
import numpy as np
from scipy.optimize import minimize
from .c7b import ROOT, OUT, write, digest


def main():
    z=np.load(ROOT/'receipts/c6/index.npz')
    split=int(.9*len(z['fork'])); pos=np.arange(split+1,len(z['fork']))
    fork=z['fork'][pos]; support=z['support'][pos]; q=z['q'][pos]
    lengths=z['context_length'][pos]; half=len(pos)//2
    cal=np.arange(len(pos))<half; test=~cal
    preds={}; fits={}; gates={}; inputs={}
    for arm in ('emender','transformer'):
        for recipe,folder,filename in [('dense','part2',arm),('starved','c7a',arm+'_boosted'),('fixed','c7b',arm+'_fixed'),('floor','c7b',arm+'_floor')]:
            name=arm+'_'+recipe; path=ROOT/'receipts'/folder/(filename+'_prediction.npz')
            ev=np.load(path); p=ev['p']; assert len(p)==len(pos) and (p>0).all()
            inputs[name]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
            preds[name]=p
            x=np.column_stack([np.ones(len(p)),ev['confidence'],np.log1p(lengths),np.log1p(support)])
            eligible=cal&fork&(support>0)
            center=x[eligible,1:].mean(0); scale=x[eligible,1:].std(0); scale[scale==0]=1
            x[:,1:]=(x[:,1:]-center)/scale
            def objective(w):
                raw=x[eligible]@w; g=1/(1+np.exp(-np.clip(raw,-35,35)))
                prob=(1-g)*p[eligible]+g*q[eligible]
                return -np.log(prob).mean(),x[eligible].T@((p[eligible]-q[eligible])*g*(1-g)/prob*(np.abs(raw)<35))/len(prob)
            fit=minimize(objective,np.zeros(4),jac=True,method='L-BFGS-B',bounds=[(-20,20)]*4,options={'maxiter':1000,'ftol':1e-12})
            assert fit.success,fit.message
            fits[name]=dict(weights=fit.x.tolist(),center=center.tolist(),scale=scale.tolist(),calibration_fork_nll=float(fit.fun))
            gates[name]=1-1/(1+np.exp(-np.clip(x@fit.x,-35,35)))
            if recipe=='floor': gates[arm+'_joint']=ev['neural_gate']
    write('frozen_gates.json',fits)
    write('calibration_freeze.json',dict(final_test_scored=False,gate_sha256=digest(OUT/'frozen_gates.json'),inputs=inputs))
    def summary(p):
        assert np.isfinite(p).all() and (p>0).all()
        loss=-np.log(p)
        return {name:dict(n=int((test&m).sum()),nll=float(loss[test&m].mean())) for name,m in
                [('overall',np.ones(len(p),bool)),('fork',fork),('forced',~fork),('empty',support==0),('empty_fork',(support==0)&fork),('empty_forced',(support==0)&~fork),('zero_target',(support>0)&(q==0))]}
    report=dict(protocol_sha256=digest(ROOT/'PREREGISTRATION_C7b.md'),models={},telemetry={},inputs=inputs,
                gate_sha256=digest(OUT/'frozen_gates.json'),scope='corpus-selected intervals; train-prefix counts; one seed, transductive diagnostic')
    def fused(p,g):
        mixed=np.where(support>0,g*p+(1-g)*q,p)
        return np.where((~fork)&(support>0),q,np.where(fork,mixed,p))
    for name,p in preds.items():
        report['models'][name]=dict(neural=summary(p),common_c3=summary(fused(p,gates[name])))
        if name.endswith('_floor'):
            arm=name.split('_')[0]
            report['models'][name]['joint_gate']=summary(fused(p,gates[arm+'_joint']))
    # Dimensions retain empty support and all final positions. Gate weights below are
    # pre-walk neural weights; forced walk overrides them during inference.
    totals=np.expm1(z['features'][z['ids'][pos],257])
    for arm in ('emender','transformer'):
        rows=[]
        for dim,values,bins in [('interval_count',totals,[0,.5,1.5,4.5,16.5,64.5,1e10]),
                                ('branch_count',support,[0,1,2,4,8,16,257]),
                                ('context_length',lengths,[0,1,2,4,8,16,32,128,1e10])]:
            for lo,hi in zip(bins[:-1],bins[1:]):
                m=test&(values>=lo)&(values<hi)
                if m.any():
                    rows.append(dict(dimension=dim,lo=lo,hi=hi,n=int(m.sum()),fork_n=int((m&fork).sum()),
                      joint_neural_weight=float(gates[arm+'_joint'][m].mean()),
                      c3_dense_neural_weight=float(np.where(support>0,gates[arm+'_dense'],1)[m].mean()),
                      c3_fixed_neural_weight=float(np.where(support>0,gates[arm+'_fixed'],1)[m].mean()),
                      c3_floor_neural_weight=float(np.where(support>0,gates[arm+'_floor'],1)[m].mean()),
                      c3_starved_neural_weight=float(np.where(support>0,gates[arm+'_starved'],1)[m].mean())))
        report['telemetry'][arm]=rows
    report['decisive']={}
    for arm in ('emender','transformer'):
        b=report['models'][arm+'_fixed']['common_c3']; d=report['models'][arm+'_dense']['common_c3']
        deltas={k:b[k]['nll']-d[k]['nll'] for k in ('overall','fork','empty')}
        report['decisive'][arm]=dict(deltas=deltas,strict_win=all(v<0 for v in deltas.values()))
    write('results.json',report)
    print(json.dumps(report['decisive'],indent=2))

if __name__=='__main__': main()
