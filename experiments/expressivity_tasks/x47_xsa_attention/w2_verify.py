"""Read-only W2 artifact audit. Stdlib; does not invoke model, GPU, or native XSA."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from .w1_check_receipts import check, vector

ROOT=Path(__file__).with_name('receipts')/'w2'


def load(path):
    return json.loads((ROOT/path).read_text())


def audit():
    cpu,gpu,timing=(load(path) for path in ('cpu/cpu.json','gpu/gpu.json','gpu/timing.json'))
    check(cpu['ref']==gpu['ref']==timing['ref'] and cpu['state']==gpu['state']==timing['state'],'state/ref')
    arms=('full','erase','blind','fixed'); ids=[(w,t,arm) for w in range(2) for t in b'MS' for arm in arms]
    check(cpu['trace']['cases']==gpu['trace']['cases'] and gpu['cpu_gpu_max_error']==0,'recorded device parity')
    cases=gpu['trace']['cases']; check([(c['world'],c['tag'],c['arm']) for c in cases]==ids,'case order')
    correct={a:0 for a in arms}; loss={a:[] for a in arms}; err=0.
    for c,(w,t,arm) in zip(cases,ids,strict=True):
        tags=b'SM' if w else b'MS'; codes=b'ab'; docs=[b'key='+bytes([code,tag]) for code,tag in zip(codes,tags)]
        version=hashlib.sha256(b''.join(len(d).to_bytes(8,'big')+d for d in docs)).hexdigest()
        check(c['archive_version']==version,'mount binding')
        stream=b'unit='+bytes([t])+b';0123456789;key='; n=len(stream); gold=codes[tags.index(t)]
        rs=c['requests']; check([(r['at'],bytes.fromhex(r['key'])) for r in rs]==[(n,k) for k in (b'key=',b'key=a',b'key=b')],'requests')
        for r in rs:
            expected=[0]*257; k=bytes.fromhex(r['key'])
            for d in docs:
                if d.startswith(k): expected[d[len(k)]]+=1
            check(expected==vector(r['h'],257),'raw answer')
        vs=c['views']; check([(v['at'],v['op']) for v in vs]==[(i,'STORE_EMIT' if i==6 else 'EMIT') for i in range(n)]
                       +[(n,'PARENT'),(n,'CHILDREN'),(n,'EMIT')],'op schedule')
        for v in vs:
            i=v['at']; check(bytes.fromhex(v['tail'])==stream[max(0,i-8):i] and v['written_at']==(6 if i>6 else None),'visibility')
        saved=vs[6]['write']; check((saved==t if arm=='fixed' else vector(saved,256)==[int(i==t) for i in range(256)]),'saved cue')
        for split,active in ((-2,False),(-1,arm in ('full','fixed'))):
            p=vector(vs[split]['prob'],256); check(all(v>0 and math.isfinite(v) for v in p) and abs(sum(p)-1)<3e-5,'mass')
            for i in range(256):
                expected=(1/256 + (4*int(i==gold) if active else 2*int(i in codes)))/5
                err=max(err,abs(expected-p[i]))
        check(c['reads']==3 and c['passes']==(0 if arm=='fixed' else n+3),'work mismatch')
        check(c['byte']==p.index(max(p)),'greedy emission'); correct[arm]+=int(c['byte']==gold); loss[arm].append(-math.log(p[gold]))
    check(err<=3e-5 and correct==dict(full=4,erase=2,blind=2,fixed=4),'unqualified join')
    rows=[json.loads(l) for l in (ROOT/'gpu/samples.jsonl').read_text().splitlines()]; expected=[]
    for block,num in (('warmup',1),('primary',8),('diagnostic',1)):
        for rep in range(num):
            for w in range(2):
                for j,t in enumerate(b'MS'):
                    for a in (('fixed','full') if (rep+j+w)%2 else ('full','fixed')):
                        expected.append((block,rep,w,t,a))
    check([(r['block'],r['rep'],r['world'],r['tag'],r['arm']) for r in rows]==expected,'timing population/order')
    for r in rows:
        check((r['prefill_passes'],r['current_passes'],r['current_rpcs'])==((22,3,3) if r['arm']=='full' else (0,0,3)),'timing counts')
        check(r['byte']==b'ab'[(b'SM' if r['world'] else b'MS').index(r['tag'])],'timed answer')
    walls={}
    for a in ('full','fixed'):
        times=[r['current_ns']/1e6 for r in rows if r['block']=='primary' and r['arm']==a]; walls[a]=median(times)
        check(abs(walls[a]-timing['arms'][a]['current']['p50_ms'])<1e-9,'median')
    def lease(path): return dict(s.split('=',1) for s in (ROOT/'gpu'/path).read_text().splitlines())
    ini,end,release=lease('initial'),lease('last'),lease('released')
    check(all(ini[k]==end[k]==gpu['lease'][k] for k in ('pid','starttime','gpu','host','created')) and
          release['rc']=='0' and ini['gpu'] in ('0','1') and int(end['heartbeat'])>int(ini['heartbeat']),'lease')
    charge=1+int(release['released_epoch'])-int(ini['created']); check(0<charge<=180,'cap')
    return dict(cases_per_arm=4,correct=correct,selected_byte_mean_nll={k:mean(v) for k,v in loss.items()},
                analytic_max_error=err,primary_predictions_per_arm=32,median_current_ms=walls,card_s_upper=charge)


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
