"""Verify released O lease, cache hashes, exact panel stopping and paired scores (CPU)."""
import json
from .k15_singleton_credit import ARMS, OUT, interval, j, loss, np


def read(name): return json.loads((OUT/name).read_text())


def verify():
    rows=[json.loads(line) for line in (OUT/'console').read_text().splitlines() if line.startswith('{')]
    assert rows.pop()=={'complete':True}
    panel=read('panels.json'); ck=read('checkpoint.json'); assert ck['panels']==j.digest(OUT/'panels.json')
    stages=['train','dev']+(['locked'] if read('dev.json')['passed'] else [])
    assert len(rows)==sum(len(panel[s]) for s in stages)
    for stage in stages:
        docs=[]
        for i,d in enumerate(panel[stage]):
            r=rows.pop(0); path=OUT/f'{stage}-{i:03}.npy'
            assert (r['stage'],r['step'],r['doc'],r['cache_sha'])==(stage,i+1,d['doc'],j.digest(path))
            arr=np.load(path,allow_pickle=False); assert arr.shape==(256,10); docs.append(arr)
        x=np.stack(docs); scores={a:loss(x,a,ck['weights'][a]).mean(1) for a in ARMS}; report=read(stage+'.json')
        assert report['nll']=={a:interval(s) for a,s in scores.items()}
        assert report['gain']=={a:interval(s-scores['writable']) for a,s in scores.items() if a!='writable'}
        assert report['passed']==all(v[0]>=.001 and v[1]>0 for v in report['gain'].values())
        parity=read(stage+'-parity.json'); assert set(parity)==set(ARMS) and all(r['max']<=.01 and r['mean']<=.0001 and r['normalization']<1e-5 for r in parity.values())
    assert len(list(OUT.glob('locked-*.npy')))==(48 if 'locked' in stages else 0)
    fields=dict(line.split('=',1) for line in (OUT/'initial').read_text().splitlines() if '=' in line); launch=read('launch.json')
    assert (int(fields['pid']),fields['starttime'])==(launch['pid'],launch['starttime'])
    release=int((OUT/'released').read_text()); elapsed=release-int(fields['created'])+1
    assert 0<elapsed<=5400 and elapsed+4402<=21600
    return dict(opened=stages,verified_caches=sum(len(panel[s]) for s in stages),card=int(fields['gpu']),lease_seconds_upper_bound=elapsed,cumulative_seconds_upper_bound=elapsed+4402)


if __name__=='__main__': print(json.dumps(verify()))
