"""Independent CPU accounting of every opened P2 target, checkpoint and paired contrast."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .k15_coupled_summary import interval


def records(p): return [json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]


def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def ledger(out):
    fields=dict(l.split('=',1) for l in (out/'initial').read_text().splitlines() if '=' in l); launch=json.loads((out/'launch.json').read_text()); assert fields['pid']==str(launch['pid']) and fields['starttime']==launch['starttime'] and fields['gpu'] in ('0','1')
    hb=[int(l.split('=')[1]) for l in (out/'heartbeat').read_text().splitlines() if l.startswith('heartbeat=')]; release=int((out/'released').read_text()); assert max(hb)>int(fields['created']) and release>=max(hb)
    assert (out/'console').read_text().endswith('rc=0\n') or 'rc=0\n' in (out/'console').read_text(); return release-int(fields['created'])+1


def summary():
    out=Path('output/k15P2'); log=records(out/'console'); p=json.loads((out/'panels.json').read_text()); tr=[r for r in log if 'arm' in r]; early=[r for r in log if 'detail' in r]; assert len(tr)==16 and len(early)==8 and {r['stage'] for r in log if 'stage' in r}=={'early'}
    ids=[d['doc'] for stage in ('train','early','dev','locked') for d in p[stage]]; assert len(set(ids))==len(ids)
    for a in ('compiled','random'):
        train=[r for r in tr if r['arm']==a]; assert [r['doc'] for r in train]==[d['doc'] for d in p['train'][:8]]
        for i,r in enumerate(train,1): assert digest(out/f'{a}-{i}.pt')==r['checkpoint_sha']
    assert [r['doc'] for r in early]==[d['doc'] for d in p['early']]; parts={}; metrics={}; stats={}
    for arm in early[0]['detail']:
        docs=[r['detail'][arm] for r in early]; rows=[v for d in docs for v in d['rows']]; assert len(rows)==64
        assert all(abs(d['nll']-np.mean([v['nll'] for v in d['rows']]))<1e-6 for d in docs) and all(abs(sum(v['ops'])-1)<1e-5 for v in rows)
        parts[arm]={}
        for k in ('start','continue','mismatch','no-copy'):
            nll=[v['nll'] for v in rows if v['part']==k]; parts[arm][k]=[len(nll),np.mean(nll) if nll else None]
        # modal_ages counts pre-observation modal COPY-edge ages, not an executed parse.
        cp=[v['decision'] for v in rows if v['decision'][0]]
        stats[arm]=dict(pre_entropy=np.mean([v['entropy'] for v in rows]),mean_post=np.mean([v['ops'] for v in rows],axis=0).tolist(),distance_range=[min(v[1] for v in cp),max(v[1] for v in cp)] if cp else None,modal_ages=dict(Counter(v[2] for v in cp)),wall_s=sum(d['seconds'] for d in docs))
    for a in ('off','indexed',*parts): metrics[a]=float(np.mean([r[a] for r in early]))
    gains={a:interval([r[a]-r['compiled'] for r in early]) for a in ('off','indexed','random','birth')}; assert gains==log[-1]['gains']; assert not log[-1]['passed'] and not any(r['changes'] for r in early)
    audit=out.with_name('k15P2-audit'); probes=records(audit/'console'); audits={}
    for name in ('birth','compiled-8'):
        rows=[v for v in probes if v.get('audit')==name]; assert len(rows)==8
        delays=Counter(); newborn=Counter(); work=Counter()
        for v in rows: delays.update(v['read_delay']); newborn.update(v['newborn_mass']); work.update(v['work'])
        audits[name]=dict(delay_mass=delays,newborn_mass=newborn,work=work,phase_grad_mean=np.mean([v['phase_grad'] for v in rows],axis=0).tolist(),transition_grad_mean=np.mean([v['transition_grad'] for v in rows],axis=0).tolist(),k15_grad_range=[min(v['k15_grad'] for v in rows),max(v['k15_grad'] for v in rows)],seconds={k:sum(v[k] for v in rows) for k in ('native_seconds','forward_seconds','total_seconds')})
    return dict(train_unique_targets=64,early_unique_targets=64,primary=metrics,gains=gains,parts=parts,usage=stats,audits=audits,ledger=[ledger(out),ledger(audit)])


if __name__=='__main__': print(json.dumps(summary(),indent=2))
