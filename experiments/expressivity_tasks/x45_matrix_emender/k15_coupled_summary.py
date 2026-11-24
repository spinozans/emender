"""Recompute stage I paired-document contrasts (nats per byte) from retained JSONL."""
import json
from pathlib import Path
import sys
import numpy as np


def interval(data):
    a=np.array([x for x in data if x is not None]); rng=np.random.default_rng(3411)
    lo,hi=np.quantile(rng.choice(a,(10000,len(a))).mean(1),[.025,.975]); return [float(a.mean()),float(lo),float(hi),len(a)]


def records(path): return [json.loads(l) for l in path.read_text().splitlines() if l.startswith('{')]


def summarize(out):
    train=records(out/'console'); diagnostic=records(out.with_name(out.name+'_diagnostic')/'console'); cal=records(out/'cal.jsonl')
    assert train.pop()==diagnostic.pop()=={'complete':True} and len(train)==192 and len(cal)==32; assert [r['doc'] for r in diagnostic]==[r['doc'] for r in cal]
    a=[r['adv'] for r in train if r['adv'] is not None]; eligible=sum(r['eligible'] for r in train)
    result=dict(train=dict(nll=interval([r['nll'] for r in train]),eligible=eligible,positive=sum(x>0 for x in a),negative=sum(x<0 for x in a),ties=a.count(0),advantage=interval(a),correct=sum(r['correct'] for r in train),echo=sum(r['echo'] for r in train),nonzero_grad=sum(r['grad']>0 for r in train)))
    result['cal']={mode:[interval([r['modes'][mode][k] for r in cal]) for k in range(3)] for mode in cal[0]['modes']}; result['contrasts']={}
    for mode,ref in [('off','learned'),('fixed','learned'),('skip','learned'),('echo','learned'),('skip','always'),('off','skip'),('off','fixed')]:
        result['contrasts'][mode+'-'+ref]=[interval([r['modes'][mode][k]-r['modes'][ref][k] for r in cal if r['modes'][ref][k] is not None]) for k in range(3)]
    result['diagnostic']={mode:[interval([r['changes'][mode][k] for r in diagnostic]) for k in range(3)] for mode in diagnostic[0]['changes']}
    ds=[r for r in diagnostic if r['adv'] is not None]; lens=[n for r in diagnostic for n in r['max_common']]; result['sites']=dict(count=len(lens),distinct=sum(r['distinct'] for r in diagnostic),length_histogram={str(k):lens.count(k) for k in sorted(set(lens))},correct=sum(r['correct'] for r in diagnostic),echo=sum(r['echo'] for r in diagnostic),writes=sum(r['writes'] for r in cal),coupled=len(ds),positives=sum(r['adv']>0 for r in ds),ties=sum(r['adv']==0 for r in ds),correct_decisions=sum((r['adv']>0)==r['write'] for r in ds),decision_regret=float(np.mean([max(0,(-1 if r['write'] else 1)*r['adv']) for r in ds])))
    return result

if __name__=='__main__':print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
