"""N staged extraction, frozen comparison and label-free single-request audit."""
import hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np
import torch
from .k15_context_credit import ARMS, OutcomeRows, evidence, design, mass
from .k15_coupled_summary import interval
from .k15_historical_training import j, load_net, REACH, ANCHOR_SHA
from .k15_bank_pilot import setup
from .k15_horizon import Resident

OUT=Path('output/k15N'); START=3840


def put(path,data):
    tmp=path.with_suffix('.partial'); tmp.write_text(json.dumps(data)); tmp.replace(path)


def panels(stream):
    ids=set(); hashes=set(); root=j.B.parents[1]
    for path in [REACH]+[p for a in 'DEFGH' for p in (root/f'k15_bookmarks_{a}').glob('*.jsonl')]:
        ids.update(json.loads(l)['doc'] for l in path.open())
    m=Path('output/k15M/panels.json'); assert j.digest(m)=='761ecd5002a8a1865b456545780a1ad96c0639b1a185e0195316144908cf5859'
    for path in [root/a/'panels.json' for a in ('k15_coupled_I','k15_relation_J','k15_counterfactual_K')]+[Path('output/k15L-pilot/panels.json'),m]:
        for docs in json.loads(path.read_text()).values():
            if isinstance(docs,list): ids.update(d['doc'] for d in docs)
    limit={}; found=set()
    for d in stream.documents_iter():
        if d['doc'] in ids:
            found.add(d['doc']); hashes.add(d['sha256']); limit[d['split']]=max(limit.get(d['split'],0),d['offset'])
    assert found==ids
    def choose(split):
        for d in stream.documents(split):
            if d['offset']>limit[split] and d['length']>=4096 and d['sha256'] not in hashes and d['doc'] not in ids:
                ids.add(d['doc']); hashes.add(d['sha256']); yield d
    train=choose('train'); cal=choose('calibration'); p={}
    for name,n,it in [('train',96,train),('dev',48,cal),('locked',48,cal)]:
        p[name]=list(itertools.islice(it,n)); assert len(p[name])==n
    return p


@torch.inference_mode()
def local(net, prefixes):
    x=torch.tensor([list(s) for s in prefixes],device='cuda'); assert x.shape[1]==128
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return net(x,doors=False)[0][:,-1].float().softmax(-1).cpu().numpy()


def prefill(raw,audit=False):
    state=Resident(j.EXE,'backoff',audit)
    for s in range(0,START,32): state.append(raw[s:s+32])
    return state


def extract(net,raw,audit):
    state=prefill(raw,audit); rows=OutcomeRows(); result=[]; dense=[]
    try:
        for t in range(START,4096,16):
            batch=local(net,[raw[i-128:i] for i in range(t,t+16)])
            for base in batch:
                h,f=evidence(base,state.read()); keys=[state.raw[-k:] for k in (8,2)]
                reads=[rows.read(k) for k in keys]; y=raw[len(state.raw)]
                result.append([*f,*[c for c,hit in reads],*[int(hit) for c,hit in reads],base[y],h[y]])
                if len(dense)<32: dense.append([base,h])
                for k in keys: rows.observe(k,base,h,y)
                state.append(bytes([y]))
    finally: state.session.close()
    return np.array(result),np.array(dense)


def fit(x,k):
    f=x[:,:5]; c=x[:,k]; p,q=x[:,-2:].T; weights={}
    for arm in ARMS[:-1]:
        a=design(f,c,arm)
        if arm in ARMS[:3]:
            grid=([i] if arm=='global' else [i,1.] for i in np.linspace(-6,3,128))
            w=min(grid,key=lambda w:loss(x,k,arm,w).mean())
        else:
            a=torch.from_numpy(a); w=torch.tensor([-1.5]+[0.]*6,dtype=a.dtype,requires_grad=True); opt=torch.optim.Adam([w],lr=.05)
            for _ in range(128):
                z=(a@w).sigmoid()*torch.tensor(f[:,0]>0); nll=-((1-z)*torch.tensor(p)+z*torch.tensor(q)).log().mean(); opt.zero_grad(); nll.backward(); opt.step()
            w=w.detach().tolist()
        weights[arm]=w
    weights['reset']=weights['writable']; return weights


def loss(x,k,arm,w):
    a=mass(x[...,:5],x[...,k],arm,w); return -np.log((1-a)*x[...,-2]+a*x[...,-1])


def replay(net,observed,arm,w,depth):
    # Only consumed bytes. Last prediction has no corresponding observation.
    tic=time.perf_counter(); state=prefill(observed); rows=OutcomeRows(); times=[]; dist=[]
    try:
        for t in range(START,len(observed)+1):
            start=time.perf_counter()
            if t>START:
                byte=observed[t-1]; rows.observe(key,base,h,byte); state.append(bytes([byte]))
            key=state.raw[-depth:]; base=local(net,[state.raw[-128:]])[0]; h,f=evidence(base,state.read()); a=mass(f,rows.read(key)[0],arm,w); dist.append((1-a)*base+a*h); times.append(time.perf_counter()-start)
    finally: state.session.close()
    return np.array(dist),dict(wall=time.perf_counter()-tic,mean=np.mean(times),p95=np.quantile(times,.95))


def payload(stream,d):
    with stream.source.open('rb') as f: f.seek(d['offset']); raw=f.read(d['length'])
    assert hashlib.sha256(raw).hexdigest()==d['sha256']; return raw[:4096]


def main():
    tic=time.monotonic(); stream,ident=setup(); p=panels(stream); put(OUT/'panels.json',dict(identity=ident,**p)); net=load_net().eval(); k=None
    for stage in ('train','dev','locked'):
        docs=[]; begin=time.monotonic()
        for i,d in enumerate(p[stage]):
            raw=payload(stream,d); x,dense=extract(net,raw,stage=='train' and i==0); docs.append(x)
            path=OUT/f'{stage}-{i:03}.npy'; tmp=path.with_suffix('.partial');
            with tmp.open('wb') as f: np.save(f,x)
            tmp.replace(path); j.emit(stage=stage,step=i+1,doc=d['doc'],sha=j.digest(path))
            if stage=='train' and i==7:
                hit=np.stack(docs)[:,:,7:9].mean(axis=(0,1)); k=5 if hit[0]>=.2 else 6; estimate=(time.monotonic()-begin)/8*(192-8)+120
                put(OUT/'pilot.json',dict(revisit=hit.tolist(),depth=[8,2][k-5],projected_remaining=estimate)); assert hit[k-5]>=.2 and estimate+time.monotonic()-tic<5100
            if stage=='dev' and i==0:
                timing={}
                for arm in ARMS:
                    pred,st=replay(net,raw[:START+31],arm,w[arm],[8,2][k-5]); a=mass(x[:32,:5],x[:32,k],arm,w[arm])[:,None]; ref=(1-a)*dense[:,0]+a*dense[:,1]; delta=np.abs(pred-ref); timing[arm]=dict(**st,max=delta.max(),mae=delta.mean()); put(OUT/'timing.json',timing); assert np.max(np.abs(pred.sum(1)-1))<1e-5 and delta.max()<=.01 and delta.mean()<=.0001
        x=np.stack(docs)
        if stage=='train':
            w=fit(x.reshape(-1,x.shape[-1]),k); put(OUT/'checkpoint.json',dict(weights=w,depth=[8,2][k-5],anchor=ANCHOR_SHA,panels=j.digest(OUT/'panels.json')))
        scores={a:loss(x,k,a,w[a]).mean(1) for a in ARMS}; gains={a:interval(s-scores['writable']) for a,s in scores.items()}; passed=all(gains[a][0]>=.001 and gains[a][1]>0 for a in ARMS if a!='writable'); put(OUT/(stage+'.json'),dict(nll={a:interval(s) for a,s in scores.items()},gain=gains,passed=passed,seconds=time.monotonic()-begin));
        if stage=='dev' and not passed: break
    j.emit(complete=True)

if __name__=='__main__':
    if len(sys.argv)==1: main()
    else:
        # Raw binary prefix: 3840..4095 bytes from one document.
        stream,_=setup(); ck=json.loads((OUT/'checkpoint.json').read_text()); raw=Path(sys.argv[1]).read_bytes(); assert START<=len(raw)<4096
        pred,_=replay(load_net().eval(),raw,'writable',ck['weights']['writable'],ck['depth']); np.save(sys.argv[2],pred)
