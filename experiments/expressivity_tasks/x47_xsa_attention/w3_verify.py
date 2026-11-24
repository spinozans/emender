"""Offline audit of populations, per-doc NLLs, dev/confirmation gate and GPU accounting."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).with_name('receipts')/'w3'
N={'train':2048,'dev':128,'confirm':128}; T=2048; CONTROLS=('direct','fixed','blind')


def check(v,msg):
    if not v: raise ValueError(msg)


def load(name): return json.loads((ROOT/name).read_text())

def interval(x):
    v=np.asarray(x); ix=np.random.default_rng(4703).integers(0,len(v),(10000,len(v)))
    return [v.mean(),*np.quantile(v[ix].mean(1),[.025,.975])]


def audit():
    panel=load('panels.json'); ph=hashlib.sha256((ROOT/'panels.json').read_bytes()).hexdigest(); seen=set(); ids=set()
    for name,rows in panel['panels'].items():
        check(len(rows)==N[name],'document denominator'); prev=''
        for d in rows:
            h=d['sha256']; rank=hashlib.sha256(('W3-natural-v1:'+h).encode()).hexdigest()
            check(h not in seen and d['doc'] not in ids and rank>prev,'duplicate document/order'); seen.add(h); ids.add(d['doc']); prev=rank
            check(d['offset']>=100_600_000_000 and d['start']==int(h[16:32],16)%(d['length']-T) and
                  0<=d['start']<=d['length']-T-1,'stream span')
    tr=load('train.json'); pr=load('progress.json'); c=load('curve.json'); ck=pr['checkpoint_sha']
    check(tr['steps']==pr['step']==512 and tr['targets_per_arm']==len(panel['panels']['train'])*T,'training exposure')
    check([r['step'] for r in c]==[1,*range(16,513,16)],'training curve incomplete')
    q=load('qualify.json'); check(max(q.values())<1e-5,'frozen host causality')
    confirm_needed=load('dev.json')['pass_gate']; check((ROOT/'confirm.json').exists()==confirm_needed,'confirmation gate')
    stats={}
    for s in ('train','dev',*(('confirm',) if confirm_needed else ())):
        receipt=load(f'{s}-pages.json'); cached=receipt['shards']; n=N[s]*T
        check(receipt['panel_sha']==ph and sum(r['targets'] for r in cached)==n,'page identity/length'); active=sum(r['joins'] for r in cached)
        stats[s]=dict(targets=n,active=active,coverage=active/n,prep_s=receipt['seconds'])
        if s=='train': continue
        g=load(s+'.json'); docs=g['docs']; check(g['checkpoint_sha']==ck and g['panel_sha']==ph,'scorer binding')
        check([d['doc'] for d in docs]==[d['doc'] for d in panel['panels'][s]],'incorrect paired document IDs')
        check(all(d['targets']==T and 0<=d['hit']<=d['active']<=d['addressed']<=T for d in docs) and sum(d['active'] for d in docs)==active,'population')
        means={k:np.array([d['nll'][k] for d in docs]) for k in g['means']}
        for k,v in means.items():
            check(np.isfinite(v).all() and np.isclose(v.mean(),g['means'][k],atol=1e-12,rtol=0),'mean')
            if k!='full': check(np.allclose(interval(v-means['full']),g['control_minus_full'][k],atol=1e-12,rtol=0),'bootstrap')
        check(g['pass_gate']==all(g['control_minus_full'][k][0]>=.001 and g['control_minus_full'][k][1]>0 for k in CONTROLS),'decision')
        stats[s].update(means=g['means'],gain=g['control_minus_full'],passed=g['pass_gate'],
                        docs_improved={k:int((means[k]>means['full']).sum()) for k in CONTROLS},
                        candidate_hit_given_active=sum(d['hit'] for d in docs)/active,
                        mean_lag_given_active=sum(d['lag_sum'] for d in docs)/active,
                        mean_join_gate_given_active=sum(d['join_weight_sum']['full'] for d in docs)/active,
                        active_nll={k:sum(d['active_loss_sum'][k] for d in docs)/active for k in means})
    total=0; leases=[]
    for sub in sorted(ROOT.glob('lease*')):
        def f(name): return dict(line.split('=',1) for line in (sub/name).read_text().splitlines())
        i,e,r=f('initial'),f('last'),f('released'); survival=sub/'survival.jsonl'; seq=[json.loads(l) for l in survival.read_text().splitlines()] if survival.exists() else []
        check(i['gpu'] in ('0','1') and all(i[k]==e[k] for k in ('pid','starttime','host','created','gpu')) and
              int(e['heartbeat'])>int(i['heartbeat']) and r['rc']=='0','lease'); delta=int(r['released_epoch'])-int(i['created'])+1
        check(delta>0,'lease time'); total+=delta; leases.append(dict(pid=i['pid'],gpu=i['gpu'],card_s_upper=delta))
        if sub.name=='lease1': check(len(seq)>=2 and seq[-1]['step']>seq[0]['step'] and all(s['ppid']==1 and str(s['pid'])==i['pid'] and s['starttime']==i['starttime'] for s in seq),'detached progress')
    check(leases and total<=1400 and 66+total<=1800,'resource cap')
    return dict(stats=stats,leases=leases,stage_card_s_upper=total,x47_upper=66+total)


if __name__=='__main__': print(json.dumps(audit(),indent=2))
