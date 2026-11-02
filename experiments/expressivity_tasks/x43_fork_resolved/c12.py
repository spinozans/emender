"""C12: disjoint-index dense CE and per-position masked self-index control."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from .c6 import ROOT, count, digest
from .c11 import Boosted
OUT=ROOT/'receipts/c12'
def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def interval(rev,sa,context):
    """Exact reversed-context interval; finite bytes, no backoff or sentinel alias."""
    key=context[::-1]; length=len(key)
    lo=0;hi=len(sa)
    while lo<hi:
        mid=(lo+hi)//2
        if rev[int(sa[mid]):int(sa[mid])+length]<key:lo=mid+1
        else:hi=mid
    left=lo;hi=len(sa)
    while lo<hi:
        mid=(lo+hi)//2
        if rev[int(sa[mid]):int(sa[mid])+length]<=key:lo=mid+1
        else:hi=mid
    return left,lo

def prepare():
    from .direct_sa import suffix_array
    manifest=json.loads((OUT/'corpora.json').read_text())
    for key in ('X','Y'):
        assert digest(ROOT/manifest[key]['path'])==manifest[key]['sha256']
    x=(OUT/'corpus_x.txt').read_bytes(); y=(ROOT/'input.txt').read_bytes()
    rev=x[::-1]; sa=suffix_array(np.append(np.frombuffer(rev,dtype=np.uint8).astype(np.int64),-1))
    # p=0 has no following target in X; p=len(X) has target X[0].
    sa=sa[sa!=0]; data=np.frombuffer(x,dtype=np.uint8)
    z=np.load(ROOT/'receipts/c6/index.npz'); lengths=z['context_length']
    ids=np.empty(len(y),np.int32); rows=[]; cache={}; target_supported=np.zeros(len(y),bool)
    started=time.monotonic()
    for i,l in enumerate(lengths):
        context=y[i-int(l):i]
        if context not in cache:
            lo,hi=interval(rev,sa,context)
            counts=np.bincount(data[len(x)-sa[lo:hi]],minlength=256)
            total=counts.sum(); support=np.count_nonzero(counts)
            row=np.zeros(260,np.float32)
            if total:
                p=counts/total; row[:256]=p
                row[256]=np.log1p(total); row[257]=-(p[p>0]*np.log(p[p>0])).sum()
            row[258]=support>1; row[259]=np.log1p(l)
            cache[context]=len(rows); rows.append(row)
        k=cache[context]; ids[i]=k; target_supported[i]=rows[k][y[i]]>0
        if i%200000==0:print('index',i,time.monotonic()-started,flush=True)
    features=np.array(rows); support=(features[ids,:256]>0).sum(-1)
    np.savez_compressed(OUT/'disjoint.npz',features=features,ids=ids,support=support,target_supported=target_supported)
    write('device.json',dict(corpora_sha256=digest(OUT/'corpora.json'),index_sha256=digest(OUT/'disjoint.npz'),
        source_sha256=digest(Path(__file__)),unique_contexts=len(rows),seconds=time.monotonic()-started,
        train_empty_fraction=float((support[:int(.9*len(y))]==0).mean()),
        train_zero_target_fraction=float((~target_supported[:int(.9*len(y))]).mean()),
        train_nonempty_zero_target=int(((support[:int(.9*len(y))]>0)&~target_supported[:int(.9*len(y))]).sum())))

def evaluate(model,data,features,lengths,split,mode):
    values=[]
    with torch.no_grad():
        for lo in range(split,len(data)-1,16*128):
            hi=min(lo+16*128,len(data)-1); full=(hi-lo)//128*128
            ranges=[]
            if full:ranges.append(torch.arange(lo,lo+full,device='cuda').reshape(-1,128))
            if lo+full<hi:ranges.append(torch.arange(lo+full,hi,device='cuda')[None])
            for at in ranges:
                f=features[at+1]
                if mode=='without':f=torch.zeros_like(f)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    lp,_=model(data[at],f,lengths[at+1])
                values.append((-lp.gather(-1,data[at+1,...,None])[...,0]).flatten().cpu().numpy())
    result=np.concatenate(values).astype(np.float64)
    assert len(result)==len(data)-split-1 and np.isfinite(result).all()
    return result

def train():
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    protocol=digest(ROOT/'PREREGISTRATION_C12.md');assert protocol in (ROOT/'PROGRAM.md').read_text()
    torch.set_num_threads(1)
    raw=(ROOT/'input.txt').read_bytes();split=int(.9*len(raw))
    data=torch.tensor(list(raw),device='cuda',dtype=torch.long)
    z=np.load(ROOT/'receipts/c6/index.npz'); lengths=torch.tensor(z['context_length'],device='cuda',dtype=torch.float32)
    meta=dict(protocol_sha256=protocol,source_sha256=digest(Path(__file__)),seed=4301,batch=16,context=128,
        budget_seconds=180,physical_gpu=os.environ['GPU_LEASE_HELD'],models=[],failures=[],
        inherited_code={p:digest(ROOT/p) for p in ('c11.py','c6.py','run.py')})
    write('training.json',meta)
    target=2_000_000
    for recipe in ('disjoint','masked'):
        if recipe=='disjoint':
            d=np.load(OUT/'disjoint.npz'); per=d['features'][d['ids']]
        else:per=np.load(ROOT/'receipts/c11/device.npz')['features']
        features=torch.tensor(per,device='cuda');del per
        for arm in ('emender','transformer'):
            torch.manual_seed(4301);model=Boosted(arm,target).cuda().train()
            if arm=='emender':target=count(model.head.base)
            opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=.01)
            rng=torch.Generator(device='cuda').manual_seed(4301)
            dropout_rng=torch.Generator(device='cuda').manual_seed(4312)
            steps=0;curve=[];kept=0;seen=0
            torch.cuda.synchronize();start=time.monotonic()
            while time.monotonic()-start<180:
                starts=torch.randint(0,split-128,(16,),device='cuda',generator=rng)
                at=starts[:,None]+torch.arange(128,device='cuda');f=features[at+1]
                if recipe=='masked':
                    keep=torch.rand(at.shape,device='cuda',generator=dropout_rng)>=.5
                    f=f*keep[...,None];kept+=int(keep.sum());seen+=keep.numel()
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    lp,_=model(data[at],f,lengths[at+1]);loss=-lp.gather(-1,data[at+1,...,None])[...,0].mean()
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step();steps+=1
                if steps==1 or steps%100==0:
                    point=dict(step=steps,seconds=time.monotonic()-start,nll=float(loss.detach()));curve.append(point)
                    print(recipe,arm,point,flush=True)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            model.eval();pred={mode:evaluate(model,data,features,lengths,split,mode) for mode in ('with','without')}
            name=arm+'_'+recipe;np.savez_compressed(OUT/(name+'_prediction.npz'),**pred)
            meta['models'].append(dict(name=name,steps=steps,tokens=steps*2048,training_seconds=elapsed,
                trainable_parameters=count(model),keep_fraction=kept/seen if seen else None,curve=curve,
                prediction_sha256=digest(OUT/(name+'_prediction.npz'))))
            write('training.json',meta);print(name,'complete',flush=True)
            del opt,model;torch.cuda.empty_cache()
        del features;torch.cuda.empty_cache()

if __name__=='__main__':
    import sys
    if sys.argv[-1]=='prepare':prepare()
    else:train()
