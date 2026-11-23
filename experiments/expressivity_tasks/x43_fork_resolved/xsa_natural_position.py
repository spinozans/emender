"""Causal natural-text positional reads and trained action mixture. CPU reference."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch import nn
from .xsa_binding_live import Session
from .xsa_controller import Controller
from .xsa_controller_data import TEST, digest, load_pieces
from .xsa_neural_evidence import HERE, Stream, flagship_identity

OFFSETS=list(range(256,2048,7)); N=len(OFFSETS); GRID=['local','depth8','depth3','recent_anchor','older_anchor']


def occurrences(text,key):
    hits=[];i=text.find(key)
    while i>=0:
        hits.append(i);i=text.find(key,i+1)
    return hits


def evidence(rpc,context,t):
    """Does not receive target or hidden history. Coordinates absolute in this document."""
    suffix=context[-3:]; candidates=rpc(f'locate {suffix.hex()} 0 {t-1} 4096')['positions']
    payload=[rpc(f'byte {p+3}')['byte'] for p in candidates]
    dense=rpc('branch '+context[-8:].hex()); assert dense['end']==0
    h=[Counter(),Counter({i:n for i,n in enumerate(dense['counts']) if n}),Counter(payload)]
    swapped=h.copy();seen=set(); trace=[]
    for m in reversed(list(re.finditer(rb'[a-zA-Z]+',context[:-3]))):
        key=m.group()[-8:]
        if len(key)<2 or key in seen:continue
        seen.add(key);gap=len(context)-3-m.end();ends=[j+len(key) for j in rpc(f'locate {key.hex()} 0 {t} 4096')['positions']];idx=[]
        for j,p in enumerate(candidates):
            k=bisect_right(ends,p)-1
            if k>=0 and abs(p-ends[k]-gap)<=4:idx.append(j)
        h.append(Counter(payload[j] for j in idx));swapped.append(Counter(payload[(j+1)%len(payload)] for j in idx));trace.append((key.hex(),gap,len(idx)))
        if len(seen)==2:break
    while len(h)<5:h.append(Counter());swapped.append(Counter())
    return h,swapped,trace


def summarize(h,q,y):
    out=[];scores=[]
    for counts in h:
        n=counts.total();out.extend((np.log1p(n),max(counts.values(),default=0)/max(n,1),sum(q[b]*c for b,c in counts.items())/max(n,1)))
        scores.append((q[y]+counts[y])/(1+n))
    return out,scores


@torch.no_grad()
def extract(exe,text,base):
    xs=torch.tensor([list(text[t-16:t]) for t in OFFSETS]);z=base.encoder(base.embedding(xs).flatten(1));q=base.head(z).double().softmax(1).numpy()
    s=Session(exe);before=0;rows=[];work=Counter();times=[];dif=0;nonempty=0
    try:
        for i,t in enumerate(OFFSETS):
            prefix=text[:t];tick=time.perf_counter_ns()
            def rpc(cmd):
                # Real call first. Timings below include independent auditing.
                a=s.rpc(cmd);op,*args=cmd.split();work[op]+=1
                if op=='locate':
                    key=bytes.fromhex(args[0]);end=int(args[2]);want=occurrences(prefix[:end],key)
                    if a['positions']!=want:raise ValueError('position parity')
                    work['positions']+=len(want);work['sealed_rows']+=a['sealed_rows'];work['tail_starts']+=a['tail_starts']
                elif op=='byte' and a['byte']!=prefix[int(args[0])]:raise ValueError('payload parity')
                elif op=='branch':
                    key=bytes.fromhex(args[0]);want=Counter(prefix[j+len(key)] for j in occurrences(prefix,key) if j+len(key)<t)
                    if a!={'counts':[want[k] for k in range(256)],'end':0}:raise ValueError('branch parity')
                return a
            if s.rpc('append '+text[before:t].hex())!={'visible':t}:raise ValueError('cursor')
            if t//512>before//512:s.rpc('seal')
            h,sh,_=evidence(rpc,prefix[-64:],t);times.append((time.perf_counter_ns()-tick)/1e6);before=t
            dif+=any(c and c!=h[2] for c in h[3:]);nonempty+=any(h[3:]);f,v=summarize(h,q[i],text[t]);sf,sv=summarize(sh,q[i],text[t]);rows.append((f,v,sf,sv))
    finally:s.close()
    arrays=[torch.tensor(np.array(col),dtype=torch.float32) for col in zip(*rows)]
    return [z,*arrays],dict(work=work,nonempty=nonempty,different=dif,audited_ms=times)


class Router(nn.Module):
    def __init__(self,k):
        super().__init__();self.k=k;self.net=nn.Sequential(nn.Linear(96+3*k,48),nn.GELU(),nn.Linear(48,k))
    def forward(self,z,f):return self.net(torch.cat((z,f[:,:3*self.k]),1)).softmax(1)


def train(data,k):
    z,f,p,_,_=data;torch.manual_seed(3401);model=Router(k);opt=torch.optim.AdamW(model.parameters(),lr=.002);curve=[]
    rng=torch.Generator().manual_seed(3401)
    for ep in range(10):
        total=0.
        for idx in torch.randperm(len(z),generator=rng).split(256):
            loss=-(model(z[idx],f[idx])*p[idx,:k]).sum(1).log().mean();opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(idx)
        curve.append(total/len(z))
    return model.eval(),curve


def run(exe,out):
    torch.set_num_threads(4);out.mkdir(exist_ok=False,parents=True);ident=flagship_identity();stream=Stream(ident['source'],ident['manifest'])
    pinned=HERE/'receipts';base_dir=pinned/'xsa_controller_v1';w=base_dir/'controller.pt';engine=digest(exe);base=Controller().eval()
    if engine!=json.loads((pinned/'xsa_binding_live.json').read_text())['engine_sha256'] or digest(w)!=json.loads((base_dir/'result.json').read_text())['checkpoint_sha']:raise ValueError('engine or base mismatch')
    base.load_state_dict(torch.load(w,weights_only=True));old=json.loads((TEST/'targets.json').read_text());tr=json.loads((base_dir/'manifest.json').read_text())
    if tr['identity']!=ident or old['identity']!=ident:raise ValueError('stream mismatch')
    docs={'train':tr['train']['documents'],'calibration':old['calibration']['documents']};previous=old['final_test']['documents'];eligible=(d for d in stream.documents('final_test') if d['length']>=2049)
    if [next(eligible) for _ in range(64)]!=previous:raise ValueError('old test order')
    docs['test']=[next(eligible) for _ in range(64)]
    for key in ('doc','sha256'):
        seen=set()
        for ds in [previous,*docs.values()]:
            ids={d[key] for d in ds};
            if seen&ids:raise ValueError('overlap '+key)
            seen|=ids
    (out/'documents.json').write_text(json.dumps(docs,indent=2)+'\n');result=dict(engine=engine,base_sha=digest(w),code_sha=digest(Path(__file__)),document_manifest_sha=digest(out/'documents.json'),actions=GRID);parts={}
    with Path(ident['source']).open('rb') as raw:
        for split,ds in docs.items():
            pieces=[];work=Counter();nonempty=different=0;ms=[];hd=hashlib.sha256()
            for d in ds:
                raw.seek(d['offset']);b=raw.read(d['length']);
                if hashlib.sha256(b).hexdigest()!=d['sha256']:raise ValueError(d)
                a,metadata=extract(exe,b[:2049],base);pieces.append(a);hd.update(bytes(b[t] for t in OFFSETS));work.update(metadata['work']);nonempty+=metadata['nonempty'];different+=metadata['different'];ms.extend(metadata['audited_ms'])
                if split=='test' and len(pieces)==1:first=b[:2049]
            parts[split]=[torch.cat(v) for v in zip(*pieces)];result[split]=dict(docs=len(ds),targets=len(ms),target_sha=hd.hexdigest(),work=work,nonempty=nonempty,different=different,audited_ipc_ms_p50_p99=np.quantile(ms,[.5,.99]).tolist());print(split,len(ms),'targets',flush=True)
            if split=='train':
                full,curve=train(parts[split],5);restricted,rcurve=train(parts[split],3);ck=out/'router.pt';torch.save(full.state_dict(),ck);result.update(curve=curve,restricted_curve=rcurve,checkpoint_sha=digest(ck)); full=Router(5);full.load_state_dict(torch.load(ck,weights_only=True));full.eval()
            if split=='calibration':fixed=int((-parts[split][2].log()).mean(0).argmin());result['fixed_selected']=GRID[fixed]
    with torch.no_grad():
        z,f,p,sf,sp=parts['test'];weights=full(z,f);pred=dict(positional=(weights*p).sum(1),restricted=(restricted(z,f)*p[:,:3]).sum(1),reassigned=(full(z,sf)*sp).sum(1),fixed=p[:,fixed]);pred.update({name:p[:,i] for i,name in enumerate(GRID)});loss={k:-v.double().log().reshape(-1,N).mean(1).numpy() for k,v in pred.items()};ids=np.random.default_rng(3401).integers(0,64,(10000,64));result['nll']={k:float(v.mean()) for k,v in loss.items()};result['doc_nll']={k:v.tolist() for k,v in loss.items()};result['gain_ci']={k:np.quantile((loss[k]-loss['positional'])[ids].mean(1),[.025,.975]).tolist() for k in ('restricted','reassigned','fixed')};result['mean_weights']=weights.mean(0).tolist();a,_=extract(exe,first,base);replayed=(full(a[0],a[1])*a[2]).sum(1);err=float((replayed-pred['positional'][:N]).abs().max());result['replay_max_target_p_error']=err
        if err>1e-6:raise ValueError('reload/replay mismatch')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result['nll'],result['gain_ci'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('exe',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.exe.resolve(),a.out)
