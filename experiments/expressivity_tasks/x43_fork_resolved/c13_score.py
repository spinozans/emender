"""C3 calibration, frozen before final-test coverage comparison."""
import numpy as np
from scipy.optimize import minimize
from .c13 import ROOT, OUT, write, digest

def main():
    z={k:v for k,v in np.load(ROOT/'receipts/c6/index.npz').items()}
    split=int(.9*len(z['fork']));pos=np.arange(split+1,len(z['fork']));half=len(pos)//2
    fork=z['fork'][pos];support=z['support'][pos];q=z['q'][pos];length=z['context_length'][pos]
    cal=np.arange(len(pos))<half;test=~cal
    x_support=np.load(ROOT/'receipts/c12/disjoint.npz')['support'][pos]
    strata=dict(total=np.ones(len(pos),bool),fork=fork,forced=~fork,ood_self=support==0,ood_x=x_support==0,nonempty_zero_target=(support>0)&(q==0))
    def summary(p):
        assert p.shape==pos.shape and np.isfinite(p).all() and (p>0).all()
        loss=-np.log(p)
        return {k:dict(n=int((test&m).sum()),nll=float(loss[test&m].mean())) for k,m in strata.items()}
    fits={};preds={}
    for arm in ('emender','transformer'):
        ev=np.load(OUT/(arm+'_self_prediction.npz'));p=ev['p'].astype(float);preds[arm]=p
        x=np.column_stack([np.ones(len(p)),ev['confidence'],np.log1p(length),np.log1p(support)])
        eligible=cal&fork&(support>0);center=x[eligible,1:].mean(0);scale=x[eligible,1:].std(0);scale[scale==0]=1
        x[:,1:]=(x[:,1:]-center)/scale
        def objective(w):
            raw=x[eligible]@w;g=1/(1+np.exp(-np.clip(raw,-35,35)));prob=(1-g)*p[eligible]+g*q[eligible]
            return -np.log(prob).mean(),x[eligible].T@((p[eligible]-q[eligible])*g*(1-g)/prob*(np.abs(raw)<35))/len(prob)
        fit=minimize(objective,np.zeros(4),jac=True,method='L-BFGS-B',bounds=[(-20,20)]*4,options={'maxiter':1000,'ftol':1e-12})
        assert fit.success,fit.message
        neural=1-1/(1+np.exp(-np.clip(x@fit.x,-35,35)))
        fits[arm]=dict(weights=fit.x.tolist(),center=center.tolist(),scale=scale.tolist(),calibration_nll=float(fit.fun))
        preds[arm+'_stack']=np.where((~fork)&(support>0),q,np.where(fork&(support>0),neural*p+(1-neural)*q,p))
    write('frozen_gates.json',fits)
    write('calibration_freeze.json',dict(final_test_scored=False,gate_sha256=digest(OUT/'frozen_gates.json')))
    report=dict(models={},predictions={},protocol_sha256=digest(ROOT/'PREREGISTRATION_C13.md'))
    c11=__import__('json').loads((ROOT/'receipts/c11/results.json').read_text())
    for arm in ('emender','transformer'):
        dense=np.load(ROOT/'receipts/part2'/(arm+'_prediction.npz'))['p']
        old=np.load(ROOT/'receipts/c12'/(arm+'_disjoint_prediction.npz'))
        rows=dict(dense=summary(dense),x_index=summary(np.exp(-old['with'])),no_features=summary(np.exp(-old['without'])),self_index=summary(preds[arm]),stacking=summary(preds[arm+'_stack']))
        fused=c11['models'][arm+'_dense']['common_c3']
        rows['dense_fusion']={k:fused[v] for k,v in [('total','overall'),('fork','fork'),('ood_self','empty')]}
        for recipe in ('neural','common_c3'):
            c=c11['models'][arm+'_ops'][recipe]
            rows['c11_'+recipe]={k:c[v] for k,v in [('total','overall'),('fork','fork'),('ood_self','empty')]}
        report['models'][arm]=rows
        report['predictions'][arm]=dict(P1={k:rows['self_index'][k]['nll']<rows['x_index'][k]['nll'] for k in ('total','fork')},P2={k:rows['self_index'][k]['nll']<rows['dense_fusion'][k]['nll'] for k in ('total','fork','ood_self')},P3={k:rows['stacking'][k]['nll']<rows['self_index'][k]['nll'] for k in ('total','fork','ood_self')})
    write('results.json',report);print(__import__('json').dumps(report,indent=2))
if __name__=='__main__':main()
