"""Run one precommitted horizon stage; never load a later stage's payloads."""
import hashlib,itertools,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from .k15_bank_pilot import setup
from .k15_historical_training import load_net,j,REACH
from .k15_horizon import serve,POLICIES
from .k15_horizon_decode import transactions
from .k15_coupled_summary import interval

OUT=Path('output/k15M'); H=(1024,4096); GRID=(0,.05,.2,.5,'count'); MODES=('base',)+POLICIES


def panels(stream):
    used=set(); root=j.B.parents[1]; limit={}; hashes=set()
    for p in [REACH]+[p for a in 'DEFGH' for p in (root/f'k15_bookmarks_{a}').glob('*.jsonl')]:
        with p.open() as f: used.update(json.loads(l)['doc'] for l in f)
    lpath=Path('output/k15L-pilot/panels.json'); assert j.digest(lpath)=='710a335c70b7e83b2691a4ab4b403e697bed8582a878db6650ca408fe120b2b3'
    for p in [root/a/'panels.json' for a in ('k15_coupled_I','k15_relation_J','k15_counterfactual_K')]+[lpath]:
        for docs in json.loads(p.read_text()).values():
            if isinstance(docs,list): used.update(d['doc'] for d in docs)
    for d in stream.documents_iter():
        if d['doc'] in used:
            hashes.add(d['sha256']); limit[d['split']]=max(limit.get(d['split'],0),d['offset'])
    def choose(split):
        for d in stream.documents(split):
            if d['offset']>limit[split] and d['length']>=4160 and d['sha256'] not in hashes:
                hashes.add(d['sha256']); yield d
    tr=choose('train'); cal=choose('calibration'); result={}
    for name,n,it in [('pilot',16,tr),('dev',64,cal),('locked',64,cal)]:
        result[name]=list(itertools.islice(it,n)); assert len(result[name])==n
    return result


def cpu():
    return sum(v for who in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN) for v in resource.getrusage(who)[:2])


@torch.inference_mode()
def measure(net,raw,h,audit):
    y=raw[4096:4160]; base=[]; neural=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for t in range(4096,4160):
            tick=time.perf_counter(); ids=torch.tensor(list(raw[t-128:t]),device='cuda')[None]; dist=net(ids,doors=False)[0][0,-1].float().softmax(-1).cpu().numpy(); neural.append(time.perf_counter()-tick); base.append(float(dist[raw[t]]))
    b=np.array(base); assert np.all(b>0); scores={'base':[(-np.log(b)).tolist()]}; stats={}; details={}
    for mode in POLICIES:
        r=raw[4096-h:4159]; tic=cpu(); ev,st=serve(j.EXE,mode,r,h,audit); st['cpu']=cpu()-tic; stats[mode]=st
        if audit: assert ev[17]==serve(j.EXE,mode,r[:h+17],h,True)[0][-1]
        support=np.array([v.total() for v in ev]); q=np.array([v[label]/max(v.total(),1) for v,label in zip(ev,y)]); scores[mode]=[]
        details[mode]=[sorted((int(k),float(v/z.total())) for k,v in z.items()) for z in ev]
        for w in GRID if mode in ('depth8','backoff') else GRID[:-1]:
            a=np.where(support>0,support/(support+1) if w=='count' else w,0); scores[mode].append((-np.log((1-a)*b+a*q)).tolist())
    return dict(scores=scores,stats=stats,dist=details,neural=neural)


def evaluate(rows,config):
    advantages=[]; costs=[]
    for h in map(str,H):
        bank,cheap=config[h]['pair']; weights=config[h]['weights']; advantage=[]; cost=[]
        for row in rows:
            r=row[h]; loss=lambda m:np.mean(r['scores'][m][weights[m]])
            advantage.append(loss(cheap)-loss(bank)); cost.append([sum(r['neural'])+r['stats'].get(m,{}).get('total',0) for m in (bank,cheap)])
        advantages.append(np.array(advantage)); costs.append(np.sum(cost,axis=0))
    long=interval(advantages[1]); interaction=interval(advantages[1]-advantages[0]); ratio=costs[1][0]/costs[1][1]; return dict(short=interval(advantages[0]),long=long,interaction=interaction,cost_ratio=ratio,passed=bool(long[0]>=.001 and long[1]>0 and interaction[1]>0 and ratio<=2))


def main(stage):
    assert stage in ('pilot','dev','locked'); start=time.monotonic(); stream,identity=setup(); out=OUT/stage; out.mkdir(exist_ok=True,parents=True); pfile=OUT/'panels.json'; config={}
    if stage=='pilot':
        with pfile.open('x') as f: json.dump(dict(identity=identity,**panels(stream)),f)
    else:
        pilot=json.loads((OUT/'pilot/result.json').read_text()); assert pilot['feasible'] and pilot['panels']==j.digest(pfile); config=pilot['config']
        if stage=='locked': assert json.loads((OUT/'dev/result.json').read_text())['gate']['passed']
    panels_read=json.loads(pfile.read_text()); assert panels_read['identity']==identity; net=load_net().eval(); rows=[]
    with stream.source.open('rb') as source,(out/'rows.jsonl').open('x') as f:
        for i,d in enumerate(panels_read[stage]):
            source.seek(d['offset']); raw=source.read(d['length']); assert hashlib.sha256(raw).hexdigest()==d['sha256']; row=dict(d)
            for h in H: row[str(h)]=measure(net,raw,h,stage=='pilot' and i==0)
            if stage=="dev" and i<2:
                row["transactions"]={}
                for h in H:
                    cfg=config[str(h)]
                    for mode in cfg["pair"]:
                        w=cfg["weights"][mode]; dist,st=transactions(net,j.EXE,mode,GRID[w],raw[4096-h:4159],h); loss=-np.log(dist[np.arange(64),list(raw[4096:4160])]); assert np.max(np.abs(loss-row[str(h)]["scores"][mode][w]))<2e-6 and np.max(np.abs(dist.sum(1)-1))<1e-5; row["transactions"][f"{h}:{mode}"]=st
            assert row['1024']['scores']['base']==row['4096']['scores']['base']; print(json.dumps(row),file=f,flush=True); rows.append(row); j.emit(step=i+1)
    if stage=='pilot':
        for h in map(str,H):
            means={m:np.mean([r[h]['scores'][m] for r in rows],axis=(0,2)) for m in MODES}; weights={m:int(v.argmin()) for m,v in means.items()}; best=lambda group:min(group,key=lambda m:means[m][weights[m]])
            config[h]=dict(weights=weights,pair=[best(['newest','aggregate']),best(['base','depth8','backoff','current'])])
    result=dict(config=config,gate=evaluate(rows,config),panels=j.digest(pfile),rows=j.digest(out/'rows.jsonl'),seconds=time.monotonic()-start); result['feasible']=4*result['seconds']<900
    (out/'result.json').write_text(json.dumps(result,indent=2)); j.emit(complete=True)

if __name__=='__main__': main(sys.argv[1])
