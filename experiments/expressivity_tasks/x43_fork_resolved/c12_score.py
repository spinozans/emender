"""No fitting or selection: raw NLL on the frozen C1 final-test positions."""
import json
import numpy as np
from .c12 import ROOT, OUT, digest, write

def main():
    z=np.load(ROOT/'receipts/c6/index.npz');d=np.load(OUT/'disjoint.npz')
    pos=np.arange(int(.9*len(z['fork']))+1,len(z['fork']));pos=pos[len(pos)//2:]
    fork=z['fork'][pos];xs=d['support'][pos];ss=z['support'][pos]
    masks=dict(total=np.ones(len(pos),bool),fork=fork,ood_x=xs==0,ood_self=ss==0,
        x_fork=xs>1,x_nonempty_zero_target=(xs>0)&~d['target_supported'][pos],
        self_nonempty_zero_target=(ss>0)&(z['q'][pos]==0))
    def summary(nll):
        nll=nll[-len(pos):];assert np.isfinite(nll).all()
        return {k:dict(n=int(v.sum()),nll=float(nll[v].mean()) if v.any() else None) for k,v in masks.items()}
    raw=np.frombuffer((ROOT/'input.txt').read_bytes(),dtype=np.uint8)
    copy_q=d['features'][d['ids'][pos],raw[pos]]
    replay=json.loads((OUT/'control_replay.json').read_text())
    report=dict(protocol_sha256=digest(ROOT/'PREREGISTRATION_C12.md'),models={},inputs={},decisive={},
        final_position_first=int(pos[0]),final_position_last=int(pos[-1]),final_targets=len(pos),
        disjoint_copy_diagnostic=dict(empty_support=int((xs==0).sum()),target_zero=int((copy_q==0).sum()),
            nonempty_zero_target=int(((xs>0)&(copy_q==0)).sum()),
            exact_copy_nll='infinity' if (copy_q==0).any() else float(-np.log(copy_q).mean()),
            positive_target_nll=float(-np.log(copy_q[copy_q>0]).mean())),
        c11_without_features='adopt only exact-step recovered weights passing whole-vector reference parity; see control_replay.json')
    for arm in ('emender','transformer'):
        for recipe in ('dense','c11','disjoint','masked'):
            if recipe in ('dense','c11'):
                path=ROOT/'receipts'/('part2' if recipe=='dense' else 'c11')/(arm+('_ops' if recipe=='c11' else '')+'_prediction.npz')
                ev=np.load(path);preds={'without' if recipe=='dense' else 'with':-np.log(ev['p'])}
            else:
                path=OUT/(arm+'_'+recipe+'_prediction.npz');preds=dict(np.load(path))
            report['inputs'][arm+'_'+recipe]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
            report['models'][arm+'_'+recipe]={mode:summary(nll) for mode,nll in preds.items()}
        recovered=next(row for row in replay['models'] if row['arm']==arm)
        if recovered['parity']:
            path=OUT/(arm+'_c11_replay_prediction.npz');assert digest(path)==recovered['prediction_sha256']
            report['inputs'][arm+'_c11_replay']=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
            report['models'][arm+'_c11']['without']=summary(np.load(path)['without'])
        models=report['models'];dense=models[arm+'_dense']['without'];dis=models[arm+'_disjoint']['with'];masked=models[arm+'_masked'];old=models[arm+'_c11']['with']
        report['decisive'][arm]=dict(P1=dis['fork']['nll']<dense['fork']['nll'],P2=dis['ood_x']['nll']<=dense['ood_x']['nll'],
            P3={mode:row['ood_self']['nll']<old['ood_self']['nll'] for mode,row in masked.items()},
            disjoint_delta_fork=dis['fork']['nll']-dense['fork']['nll'],disjoint_delta_ood=dis['ood_x']['nll']-dense['ood_x']['nll'],
            masked_vs_dense_ood={mode:row['ood_self']['nll']-dense['ood_self']['nll'] for mode,row in masked.items()})
    write('results.json',report);print(json.dumps(report['decisive'],indent=2))
if __name__=='__main__':main()
