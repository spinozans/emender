"""Seal N after release. Recompute paired metrics from all atomic document caches."""
import json, shutil
from pathlib import Path
import numpy as np
from .k15_context_run import OUT, ARMS, interval, loss, put
from .k15_historical_training import j


def seal():
    dest=Path(__file__).parent/'receipts/k15_context_credit'
    launch=json.loads((OUT/'launch.json').read_text()); initial=dict(line.split('=') for line in (OUT/'initial').read_text().splitlines() if '=' in line)
    assert int(initial['pid'])==launch['pid'] and initial['starttime']==launch['starttime']
    proc=Path(f"/proc/{launch['pid']}/stat"); assert not proc.exists() or proc.read_text().split()[2]=='Z' or proc.read_text().split()[21]!=launch['starttime']
    assert not Path('.wg/gpu_leases',initial['gpu']).exists(); seconds=int((OUT/'released').read_text())-int(initial['created'])+1; assert seconds<=5400 and seconds+4238<=21600
    hb=(OUT/'heartbeat').read_text().splitlines(); renew=max(int(s.split('=')[1]) for s in hb if s.startswith('heartbeat=')); assert renew>int(initial['created'])
    p=json.loads((OUT/'panels.json').read_text()); docs=sum([p[s] for s in ('train','dev','locked')],[]); assert len({d['doc'] for d in docs})==len({d['sha256'] for d in docs})==192
    console=[json.loads(l) for l in (OUT/'console').read_text().splitlines() if l.startswith('{')]; proof={};
    ck=json.loads((OUT/'checkpoint.json').read_text()); assert ck['panels']==j.digest(OUT/'panels.json'); k={8:5,2:6}[ck['depth']]
    for stage in ('train','dev','locked'):
        cache=[]
        for row in (r for r in console if r.get('stage')==stage):
            i=len(cache); path=OUT/f'{stage}-{i:03}.npy'; assert row['step']==i+1 and row['doc']==p[stage][i]['doc'] and row['sha']==j.digest(path); cache.append(np.load(path))
        if not cache: continue
        x=np.stack(cache); s={arm:loss(x,k,arm,ck['weights'][arm]).mean(1) for arm in ARMS}; proof[stage]=dict(docs=len(x),targets=x.shape[0]*x.shape[1],base=interval(-np.log(x[...,-2]).mean(1)),vs_base={a:interval(-np.log(x[...,-2]).mean(1)-v) for a,v in s.items()},revisit=x[...,k+2].mean(),has_hist=(x[...,0]>0).mean(),vs_write={a:interval(v-s['writable']) for a,v in s.items()})
        summary=OUT/(stage+'.json');
        if summary.exists():
            assert len(cache)==len(p[stage]) and json.loads(summary.read_text())['gain']==proof[stage]['vs_write']; shutil.copy(summary,dest)
    if not (OUT/'dev.json').exists() or not json.loads((OUT/'dev.json').read_text())['passed']: assert 'locked' not in proof
    for name in ('checkpoint.json','heartbeat','released','console','timing.json'):
        if (OUT/name).exists(): shutil.copy(OUT/name,dest)
    put(dest/'verified.json',dict(stages=proof,upper_card_seconds=seconds,cumulative_upper=seconds+4238,released=True,complete=console[-1]=={'complete':True},checkpoint_sha=j.digest(OUT/'checkpoint.json')))

if __name__=='__main__': seal()
