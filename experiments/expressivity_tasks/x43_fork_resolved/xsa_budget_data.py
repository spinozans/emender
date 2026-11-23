"""Counterfactual read data, or a single actual move, over the native XSA session."""
from collections import Counter
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch
from .xsa_natural_position import HERE, TEST, Controller, Router, Stream, Session, digest, evidence, summarize, occurrences, flagship_identity

EXE=Path('/home/erikg/sxgc/xsa/target/debug/xsa'); RECEIPTS=HERE/'receipts';V1=RECEIPTS/'natural_position_v1'; A=10

def models():
    ck=RECEIPTS/'xsa_controller_v1/controller.pt';vr=json.loads((V1/'result.json').read_text());bc=vr['base_sha'];vc=vr['checkpoint_sha']
    if digest(EXE)!=vr['engine'] or digest(ck)!=bc or digest(V1/'router.pt')!=vc:raise ValueError('binary/checkpoint identity')
    base=Controller().eval();base.load_state_dict(torch.load(ck,weights_only=True)); router=Router(5).eval();router.load_state_dict(torch.load(V1/'router.pt',weights_only=True));return base,router


def atomic(path,obj):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)


def manifest(horizon):
    ident=flagship_identity();s=Stream(ident['source'],ident['manifest']);used=json.loads((TEST/'targets.json').read_text())['final_test']['documents']+json.loads((V1/'documents.json').read_text())['test'];ids={d['doc'] for d in used}; hashes={d['sha256'] for d in used};ans={}
    for split,n in [('train',512),('calibration',64),('final_test',128)]:
        ans[split]=[]
        for d in s.documents(split):
            if d['length']<=horizon or d['doc'] in ids or d['sha256'] in hashes:continue
            ids.add(d['doc']);hashes.add(d['sha256']);ans[split].append(d)
            if len(ans[split])==n:break
        if len(ans[split])!=n:raise ValueError('insufficient eligible documents')
    return dict(identity=ident,docs=ans,horizon=horizon)


def raw_doc(source,d,horizon):
    with Path(source).open('rb') as f:f.seek(d['offset']);raw=f.read(d['length'])
    if hashlib.sha256(raw).hexdigest()!=d['sha256']:raise ValueError('document hash')
    return raw[:horizon+1]


def menu(ctx):
    keys=[None,None];f=[[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0]];seen=set()
    for m in reversed(list(re.finditer(rb'[a-zA-Z]+',ctx[:-3]))):
        k=m.group()[-8:];gap=len(ctx)-3-m.end()
        if len(k)<2 or k in seen:continue
        keys.append((k,gap));f.append([0,0,1,len(seen)/7,len(k)/8,np.log1p(gap)/6,k[0]/128,k[-1]/128]);seen.add(k)
        if len(keys)==A:break
    return keys,np.array(f,dtype='f')


def branch(rpc,k):return Counter({i:v for i,v in enumerate(rpc('branch '+k.hex())['counts']) if v})


def action(rpc,idx,keys,ctx,t):
    if idx==0:return Counter()
    if idx==1:return branch(rpc,ctx[-8:])
    key,gap=keys[idx];hits=rpc(f'locate {ctx[-3:].hex()} 0 {t-1} 4096')['positions'];ends=[j+len(key) for j in rpc(f'locate {key.hex()} 0 {t} 4096')['positions']];h=Counter()
    for j in hits:
        n=bisect_right(ends,j)-1
        if n>=0 and abs(j-ends[n]-gap)<=4:h[rpc(f'byte {j+3}')['byte']]+=1
    return h


def features(h,q):
    n=h.total();return [np.log1p(n),max(h.values(),default=0)/max(n,1),sum(q[b]*v for b,v in h.items())/max(n,1)]


@torch.no_grad()
def extract(base,router,b,H):
    times=list(range(H-1792,H,7)); xs=torch.tensor([list(b[t-16:t]) for t in times]);z=base.encoder(base.embedding(xs).flatten(1));qs=base.head(z).double().softmax(1).numpy();s=Session(EXE);last=0
    feat=np.zeros((256,A,14),'f');py=np.zeros((256,A,3),'f');mask=np.zeros((256,A),bool);cost=np.zeros((256,A),'i');v1=[];v1cost=[];work=Counter()
    try:
        for i,t in enumerate(times):
            ctx=b[max(0,t-192):t];prefix=b[:t];q=qs[i];y=b[t];cache={};count=0
            def rpc(cmd):
                nonlocal count
                count+=1
                if cmd not in cache:
                    a=s.rpc(cmd);op,*args=cmd.split();work[op]+=1
                    if op=='byte':want=prefix[int(args[0])];ok=a['byte']==want
                    elif op=='locate':ok=a['positions']==occurrences(prefix[:int(args[2])],bytes.fromhex(args[0]))
                    else:
                        k=bytes.fromhex(args[0]);hist=Counter(prefix[j+len(k)] for j in occurrences(prefix[:-1],k));ok=a['end']==0 and [(v,n) for v,n in enumerate(a['counts']) if n]==sorted(hist.items())
                    if not ok:raise ValueError('native/oracle mismatch: '+cmd)
                    cache[cmd]=a
                return cache[cmd]
            if s.rpc('append '+b[last:t].hex())!={'visible':t}:raise ValueError('cursor')
            if t//512>last//512:s.rpc('seal')
            last=t;h3=branch(rpc,ctx[-3:]);keys,m=menu(ctx);f3=features(h3,q)
            for j in range(len(keys)):
                count=0;h=action(rpc,j,keys,ctx,t);cost[i,j]=1+count;mask[i,j]=True;feat[i,j]=[*m[j],*f3,*features(h,q)];py[i,j]=[q[y],(q[y]+h3[y])/(1+h3.total()),(q[y]+h[y])/(1+h.total())]
            count=0;hs,_,_=evidence(rpc,ctx[-64:],t);v1cost.append(count);v1.append(summarize(hs,q,y))
    finally:s.close()
    f,p=zip(*v1);wp=router(z,torch.tensor(f,dtype=torch.float32));baseline=(wp*torch.tensor(p,dtype=torch.float32)).sum(1)
    return dict(z=z.numpy(),f=feat,p=py,m=mask,c=cost,v=baseline.numpy(),vc=np.array(v1cost),target=np.array([b[t] for t in times],dtype='u1')),work
