"""X43 follow-up A: frozen GPU predictions, exact counts, calibration-only gates."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # direct_sa retains its original local decompose import
OUT = ROOT / 'receipts/part2'


def verify_inputs():
    assert hashlib.sha256((ROOT/'input.txt').read_bytes()).hexdigest() == '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    manifest = json.loads((ROOT/'receipts/artifact_manifest.json').read_text())
    for name, meta in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == meta['sha256'], name
    manifest = json.loads((ROOT/'receipts/part1c/manifest.json').read_text())
    for name in ('direct_sa.py', 'receipts/part1c/labels.npz'):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == manifest[name]['sha256'], name



def count_distribution(data, occurrences, target, split=None):
    if split is not None:
        occurrences = occurrences[occurrences < split]
    counts = np.bincount(data[occurrences], minlength=256)
    total = int(counts.sum())
    return (float(counts[target] / total) if total else 0.,
            int(np.count_nonzero(counts)), total)


def mix(p, q, g, support):
    return np.where(support > 0, (1-g)*p + g*q, p)


def loss(p):
    with np.errstate(divide='ignore'):
        return -np.log(p)


def infer():
    import torch
    from .run import Candidate, Policy, Control, logits, count
    assert os.environ.get('GPU_LEASE_HELD') in ('2', '4')
    torch.set_num_threads(1)
    raw = (ROOT/'input.txt').read_bytes()
    data = torch.tensor(list(raw), dtype=torch.long, device='cuda')
    split = int(.9*len(raw))
    target = 2_000_000
    receipts = {}
    for arm in ('emender', 'transformer'):
        model = (Candidate(Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.),
                           vocab=256, d=256, layers=2, target=target)
                 if arm == 'emender' else Control(target)).cuda()
        if arm == 'emender':
            target = count(model)
        model.load_state_dict(torch.load(ROOT/f'{arm}.pt', weights_only=True, map_location='cuda'))
        model.eval()
        probs, confidence, correct = [], [], []
        with torch.no_grad():
            for lo in range(split, len(raw)-1, 128):
                hi = min(lo+128, len(raw)-1)
                pred = logits(model, data[lo:hi][None], arm)[0]
                logp = pred.log_softmax(-1)
                truth = data[lo+1:hi+1]
                probs.append(logp.gather(1, truth[:,None])[:,0].double().exp().cpu().numpy())
                confidence.append(logp.max(-1).values.exp().cpu().numpy())
                correct.append((pred.argmax(-1)==truth).cpu().numpy())
        p = np.concatenate(probs)
        retained = np.load(ROOT/f'{arm}_per_position.npz')
        error = float(np.max(np.abs(loss(p)-retained['nll'])))
        assert error < 2e-4, (arm,error)
        assert np.array_equal(np.concatenate(correct), retained['correct'])
        np.savez_compressed(OUT/f'{arm}_prediction.npz', p=p, confidence=np.concatenate(confidence))
        receipts[arm] = dict(max_nll_replay_error=error, accuracy_replayed=True,
                            checkpoint_sha256=hashlib.sha256((ROOT/f'{arm}.pt').read_bytes()).hexdigest())
        del model
        torch.cuda.empty_cache()
    (OUT/'inference.json').write_text(json.dumps(dict(physical_gpu=os.environ['GPU_LEASE_HELD'],models=receipts),indent=2)+'\n')


def tests():
    from .direct_sa import classify, brute, tests as sa_tests
    sa = sa_tests()
    for raw in (b'abac', b'aaaa', b'banana', bytes([0,255,0,1])):
        data = np.frombuffer(raw, dtype=np.uint8)
        pos, lengths, iv = classify(raw)
        for i, (_, occ) in enumerate(brute(raw)):
            lo,hi = iv[i]
            for split in (None, 1, len(raw)//2):
                a = count_distribution(data, pos[lo:hi], data[i], split)
                eligible = [j for j in occ if split is None or j < split]
                expected = sum(raw[j]==raw[i] for j in eligible)
                assert a == (expected/len(eligible) if eligible else 0.,len(set(raw[j] for j in eligible)),len(eligible))
    p=np.array([.2,.3,.4]); q=np.array([.8,0.,0.]); support=np.array([2,1,0])
    assert np.allclose(mix(p,q,0.,support),p)
    assert np.allclose(mix(p,q,1.,support),[.8,0.,.4])
    assert np.isinf(loss(np.array([0.]))[0])
    return dict(status='passed',sa=sa,checks=['exact train-prefix and corpus counts match exhaustive occurrences','mixture endpoints and empty-support abstention','zero probability produces infinite unsmoothed loss'])


def analyze():
    from .direct_sa import classify
    raw=(ROOT/'input.txt').read_bytes(); data=np.frombuffer(raw,dtype=np.uint8)
    split=int(.9*len(raw)); targets=np.arange(split+1,len(raw)); half=len(targets)//2
    pos,lengths,iv=classify(raw)
    labels=np.load(ROOT/'receipts/part1c/labels.npz')
    assert np.array_equal(lengths,labels['context_length'])
    fork=labels['fork'][targets]; lengths=lengths[targets]
    q={}; support={}; totals={}
    for scope, boundary in [('train',split),('corpus',None)]:
        cache={}; rows=[]
        for i in targets:
            lo,hi=iv[i]; key=(int(lo),int(hi))
            if key not in cache:
                occ=pos[lo:hi]
                if boundary is not None: occ=occ[occ<boundary]
                c=np.bincount(data[occ],minlength=256)
                cache[key]=(c,int(c.sum()),int(np.count_nonzero(c)))
            c,total,s=cache[key]
            rows.append((c[data[i]]/total if total else 0.,s,total))
        arr=np.array(rows); q[scope]=arr[:,0]; support[scope]=arr[:,1]; totals[scope]=arr[:,2]
    assert np.all(q['corpus']>0)
    assert np.all(q['corpus'][~fork]==1.)
    np.savez_compressed(OUT/'counts.npz',position=targets,fork=fork,context_length=lengths,
                        **{f'{s}_{k}':v for s,d in [('q',q),('support',support),('total',totals)] for k,v in d.items()})
    report=dict(corpus_sha256=hashlib.sha256(raw).hexdigest(), split=split,
                calibration_targets=half,final_test_targets=len(targets)-half,
                calibration_position_range=[int(targets[0]),int(targets[half-1])],
                final_test_position_range=[int(targets[half]),int(targets[-1])],models={})
    calibration=np.arange(len(targets))<half; test=~calibration
    def summary(pred):
        nll=loss(pred)
        return {name:float(nll[test&mask].mean()) for name,mask in [('overall',np.ones(len(test),bool)),('forced',~fork),('fork',fork)]}
    for arm in ('emender','transformer'):
        ev=np.load(OUT/f'{arm}_prediction.npz'); p=ev['p']; model=summary(p)
        armreport=dict(model=model,scopes={})
        for scope in ('train','corpus'):
            s=support[scope]; exact=q[scope]; eligible=calibration&fork&(s>0)
            grid=np.linspace(0,1,101)
            scores=[float(loss(mix(p,exact,g,s))[calibration&fork].mean()) for g in grid]
            fixed=float(grid[np.argmin(scores)])
            scalar=minimize(lambda a: loss(mix(p[eligible],exact[eligible],1/(1+np.exp(-a[0])),s[eligible])).mean(),[0.],method='BFGS')
            scalar_g=float(1/(1+np.exp(-scalar.x[0])))
            features=np.column_stack([np.ones(len(p)),ev['confidence'],np.log1p(lengths),np.log1p(s)])
            center=features[eligible,1:].mean(0); scale=features[eligible,1:].std(0); scale[scale==0]=1
            features[:,1:]=(features[:,1:]-center)/scale
            def objective(w):
                raw_z=features[eligible]@w
                z=np.clip(raw_z,-35,35); g=1/(1+np.exp(-z))
                prob=(1-g)*p[eligible]+g*exact[eligible]
                value=loss(prob).mean()
                grad=features[eligible].T@((p[eligible]-exact[eligible])*g*(1-g)/prob*(np.abs(raw_z)<35))/len(prob)
                return value,grad
            fit=minimize(objective,np.zeros(4),jac=True,method='L-BFGS-B',bounds=[(-20,20)]*4,options={'maxiter':1000,'ftol':1e-12})
            assert fit.success,fit.message
            gate=1/(1+np.exp(-np.clip(features@fit.x,-35,35)))
            def variant(g,walk=False):
                pred=np.where(fork,mix(p,exact,g,s),p)
                if walk: pred=np.where((~fork)&(s>0),exact,pred)
                return pred
            variants={}
            for name,g in [('fixed',fixed),('registered_scalar',scalar_g),('learned',gate)]:
                for walk in (False,True):
                    pred=variant(g,walk); row=summary(pred)
                    row['paired_delta_overall']=row['overall']-model['overall']
                    variants[name+('_walk' if walk else '')]=row
            np.savez_compressed(OUT/f'{arm}_{scope}_gates.npz',
                learned_index_weight=np.where((s>0)&fork,gate,0.),
                fixed_index_weight=np.where((s>0)&fork,fixed,0.),
                learned_target_probability=variant(gate),
                learned_walk_target_probability=variant(gate,True))
            oracle=np.where(s>0,exact,p)
            variants['walk_oracle_with_model_abstention']=summary(oracle)
            telemetry=[]
            for dimension,values,bins in [('support',s,[0,1,2,4,8,16,257]),('context_length',lengths,[0,1,2,4,8,16,32,128,10**9])]:
                for lo,hi in zip(bins[:-1],bins[1:]):
                    mask=test&fork&(values>=lo)&(values<hi)
                    if mask.any(): telemetry.append(dict(feature=dimension,lower=lo,upper_exclusive=hi,positions=int(mask.sum()),learned_index_weight=float(np.where(s>0,gate,0)[mask].mean()),fixed_index_weight=float(np.where(s>0,fixed,0)[mask].mean()),model_nll=float(loss(p[mask]).mean()),learned_nll=float(loss(variant(gate)[mask]).mean())))
            armreport['scopes'][scope]=dict(fixed_grid_lambda=fixed,grid=grid.tolist(),calibration_fork_nll=scores,
                registered_scalar_lambda=scalar_g,registered_scalar_optimizer_success=bool(scalar.success),
                learned_coefficients=fit.x.tolist(),feature_order=['intercept','max_model_probability','log1p_context_length','log1p_support'],
                feature_center=center.tolist(),feature_scale=scale.tolist(),optimizer_message=str(fit.message),
                variants=variants,telemetry=telemetry,
                final_test_forks=int((test&fork).sum()),empty_support_forks=int((test&fork&(s==0)).sum()),
                zero_target_count_nonempty_forks=int((test&fork&(s>0)&(exact==0)).sum()),
                empty_support_targets=int((test&(s==0)).sum()),
                empty_support_model_nll=float(loss(p[test&(s==0)]).mean()) if np.any(test&(s==0)) else None)
        report['models'][arm]=armreport
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list): return [clean(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value): return 'infinity'
        return value
    (OUT/'fusion.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({a:{s:r['variants'] for s,r in v['scopes'].items()} for a,v in report['models'].items()},indent=2))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    verify_inputs()
    if '--infer' in sys.argv: infer()
    elif '--test' in sys.argv: (OUT/'tests.json').write_text(json.dumps(tests(),indent=2)+'\n')
    else: analyze()
