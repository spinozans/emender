"""CPU-only paired recovery receipt. Read only retained calibration rows."""
from collections import Counter
from pathlib import Path
import json
import sys
import numpy as np


def read(path):
    return list(map(json.loads,path.read_text().splitlines()))


def paired(a,b):
    assert len(a)==64 and a.keys()==b.keys()
    d=np.array([a[k]-b[k] for k in sorted(a)]); rng=np.random.default_rng(3407)
    return dict(mean=float(d.mean()),ci95=np.quantile(rng.choice(d,(10000,64)).mean(1),[.025,.975]).tolist())


def summarize(out):
    panel={}; result={}
    for row in read(out/'panel.jsonl'):
        for mode,loss in row['scores'].items():
            key=row['arm']+'_'+mode; panel.setdefault(key,{})[row['doc']]=loss
    assert len(panel)==6 and all(len(rows)==64 for rows in panel.values()); result['G_nll']={k:np.mean(list(v.values())) for k,v in panel.items()}
    result['G_paired']={a+'-'+b:paired(panel[a],panel[b]) for a,b in [('B_off','original_off'),('B_native','original_off'),('D_off','B_off'),('G_off','B_off'),('D_off','G_off'),('D_native','G_off'),('D_native','D_off')]}
    if not (out/'index_eval.jsonl').exists():return result
    streams=[]; cols={}; counts={}
    for arm in ('write','index'):
        rows=read(out/(arm+'_train.jsonl')); assert len(rows)==512 and len({r['doc'] for r in rows[:256]})==256
        streams.append([(r['doc'],r['start'],r['sha']) for r in rows]); assert all(r['grad']>=0 and np.isfinite(r['grad']) for r in rows)
        for r in read(out/(arm+'_eval.jsonl')):
            key=arm+'_'+r['mode']; counts.setdefault(key,Counter()).update({k:r[k] for k in ('miss','writes','extensions')})
            for m in ('nll','delayed'): cols.setdefault(key+'_'+m,{})[r['doc']]=r[m]
    assert streams[0]==streams[1] and cols['write_off_nll']==cols['index_off_nll']
    result['H_mean']={k:np.mean(list(v.values())) for k,v in cols.items()}; result['H_counts']=counts
    result['H_paired']={k+'-write_none_'+m:paired(cols[k+'_'+m],cols['write_none_'+m]) for m in ('nll','delayed') for k in ('index_clear','write_clear','write_address','write_echo','write_off')};return result

if __name__=='__main__': print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
