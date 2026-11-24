"""Materialize preselected paired comparisons and deployment work from per-doc receipts."""
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from .k15_coupled_summary import interval


def summary(out):
    rows=[json.loads(l) for l in (out/'rows.jsonl').read_text().splitlines()]; result=json.loads((out/'result.json').read_text()); report={}; median=np.median([r['length'] for r in rows]); result['median_length']=median
    for h,cfg in result['config'].items():
        report[h]={}; bank,cheap=cfg['pair']; weights=cfg['weights']; gain=[]; buckets={}; work={}
        for mode,w in weights.items():
            loss=np.array([r[h]['scores'][mode][w] for r in rows]); mean=lambda x:float(np.mean(x)); cost=[r[h]['stats'].get(mode,{}) for r in rows]; neural=np.array([r[h]['neural'] for r in rows]); mem=np.array([v.get('latency',[0]*64) for v in cost]); total=np.array([sum(n)+v.get('total',0) for n,v in zip(neural,cost)])/64; timing=lambda x:[mean(x)*1000,float(np.percentile(x,95))*1000]; counts=Counter(); [counts.update(v.get('work',{})) for v in cost]; report[h][mode]=dict(nll=mean(loss),total_ms_per_byte_mean_p95doc=timing(total),warm_ms_mean_p95byte=timing(neural+mem),memory_ms_mean_p95byte=timing(mem),prefill_ms=timing([v.get('prefill',0) for v in cost]),cpu_ms_per_target=1000*mean([v.get('cpu',0) for v in cost])/64,native_peak_kib=max(v.get('native_peak_kib',0) for v in cost),work_total=counts)
        for r in rows:
            z=r[h]; loss=lambda m,w:np.array(z['scores'][m][w]); gain.append(float((loss(cheap,weights[cheap])-loss(bank,weights[bank])).mean()))
            for tag,b,a in [('calibrated',cheap,bank),('fixed05','current','newest'),('fixed05','newest','aggregate')]:
                v=z['dist'].get(b,[[]]*64); u=z['dist'][a]; d=loss(b,weights[b] if tag=='calibrated' else 1)-loss(a,weights[a] if tag=='calibrated' else 1)
                for k,(l,m) in enumerate(zip(v,u)):
                    for cond,yes in [('all',True),('incremental',bool(m and not l)),('overlap',bool(m and l)),('disagree',bool(m and l and m!=l)),('control_only',bool(l and not m))]:
                        if yes: buckets.setdefault(f'{tag}:{a}-{b}:{cond}',[]).append(d[k])
        report[h]['contrasts']={k:[len(v),float(np.mean(v))] for k,v in buckets.items()}; report[h]['by_length']={k:interval([g for r,g in zip(rows,gain) if (r['length']<=median)==short]) for k,short in [('shorter',True),('longer',False)]}
    return dict(**result,summary=report)

if __name__=='__main__': print(json.dumps(summary(Path(sys.argv[1])),indent=2))
