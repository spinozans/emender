"""Posthoc diagnostics only; never used to choose weights, panels or gates."""
import json,sys
from pathlib import Path
import numpy as np
from .k15_coupled_summary import interval


def diagnostics(out):
    cfg=json.loads((out/'result.json').read_text())['config']; rows=[json.loads(l) for l in (out/'rows.jsonl').read_text().splitlines()]; report={}
    for h in ('1024','4096'):
        w=cfg[h]['weights']; bank,cheap=cfg[h]['pair']; loss=lambda m:np.array([r[h]['scores'][m][w[m]] for r in rows]); b=loss(bank); c=loss(cheap); base=loss('base'); report[h]={}
        for name,a in [('always_cheap_over_base',base-c),('hindsight_choose_bank_over_cheap',c-np.minimum(c,b)),('hindsight_abstain_over_cheap',c-np.minimum(c,base)),('hindsight_abstain_or_bank_over_cheap',c-np.minimum.reduce([c,b,base]))]:
            report[h][name]=interval(a.mean(1))
        diverse=np.array([[len(v)>1 for v in r[h]['dist']['aggregate']] for r in rows]); delta=np.array([np.array(r[h]['scores']['newest'][1])-r[h]['scores']['aggregate'][1] for r in rows]); report[h]['distinct_live_continuations']=dict(bytes=int(diverse.sum()),docs=int(diverse.any(1).sum()),fixed05_gain=float(delta[diverse].mean()) if diverse.any() else None)
        report[h]['transactions']={}
        for mode in (bank,cheap):
            ts=[r['transactions'][f'{h}:{mode}'] for r in rows if 'transactions' in r];
            if ts:
                warm=np.array([t['seconds'] for t in ts]); report[h]['transactions'][mode]=dict(docs=len(ts),targets=warm.size,mean_ms=warm.mean()*1000,p95_ms=np.percentile(warm,95)*1000,inclusive_mean_ms=np.mean([t['wall']/64 for t in ts])*1000,prefill_mean_ms=np.mean([t['prefill'] for t in ts])*1000)
    report['fixed_weight_horizon']={}
    for mode in ('depth8','backoff','current','newest','aggregate'):
        i=1; diffs=[]; counts={}; sums={}
        for r in rows:
            short,long=(r[h] for h in ('1024','4096')); d=np.array(short['scores'][mode][i])-long['scores'][mode][i]; diffs.append(d.mean())
            for l,m,delta in zip(short['dist'][mode],long['dist'][mode],d):
                for k,yes in [('overlap',bool(l and m)),('new',bool(m and not l)),('short_only',bool(l and not m)),('different_overlap',bool(l and m and l!=m))]:
                    if yes: counts[k]=counts.get(k,0)+1; sums[k]=sums.get(k,0)+delta
        report['fixed_weight_horizon'][mode]=dict(gain=interval(diffs),counts=counts,conditional_gain={k:v/counts[k] for k,v in sums.items()})
    return report

if __name__=='__main__': print(json.dumps(diagnostics(Path(sys.argv[1])),indent=2))
