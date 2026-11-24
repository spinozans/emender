"""P2 qualification, paired LM-only learning and observed-prefix inference."""
import argparse
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from .k15_context_run import j, setup, payload, evidence, mass, put, REACH, ANCHOR_SHA, load_net, interval
from .k15_coupled_training import save
from .k15_bookmarks import BookmarkSession
from .k15_horizon import Resident
from .k15_phrase_qualification import PREFIX
from .k15_prefix_machine import PrefixK15, PrefixMachine

OUT=Path('output/k15P2'); O=Path('output/k15O'); R=Path(__file__).parent/'receipts'


def panels(stream):
    root=j.B.parents[1]; old={json.loads(l)['doc'] for p in [REACH,*[p for a in 'DEFGH' for p in (root/f'k15_bookmarks_{a}').glob('*.jsonl')]] for l in p.open()}
    old.add(json.loads((R/'k15_phrase/document.json').read_text())['document']['document'])
    for line in (R/'k15_phrase/hashes.txt').read_text().splitlines():
        h,p=line.split(); p=Path(p)
        if p.name!='panels.json': continue
        assert j.digest(p)==h
        for v in json.loads(p.read_text()).values():
            if isinstance(v,list): old.update(d['doc'] for d in v)
    assigned=[d for d in stream.documents_iter() if d['doc'] in old]; assert len(assigned)==len(old)
    seen={d['sha256'] for d in assigned}
    def fresh(split):
        limit=max(d['offset'] for d in assigned if d['split']==split)
        for d in stream.documents(split):
            if d['length']>=136 and d['offset']>limit and d['sha256'] not in seen and d['doc'] not in old:
                seen.add(d['sha256']); old.add(d['doc']); yield d
    train=fresh('train'); cal=fresh('calibration'); result={}
    for k,n,it in [('train',32,train),('early',8,cal),('dev',16,cal),('locked',16,cal)]:
        result[k]=list(itertools.islice(it,n)); assert len(result[k])==n
    return result


def adapter(m): return {k:v for k,v in m.state_dict().items() if not k.startswith('backbone.')}


def restore(m,path):
    s=torch.load(path,weights_only=True); assert s['anchor']==ANCHOR_SHA and s['panel']==j.digest(OUT/'panels.json') and s['params'].keys()==adapter(m).keys()
    st=m.state_dict(); st.update(s['params']); m.load_state_dict(st,strict=True)


def run(m,raw,opt=None,mode='full'):
    tic=time.monotonic(); session=BookmarkSession(j.EXE); losses=[]; rows=[]; grad={}
    try:
        with torch.set_grad_enabled(opt is not None):
            state=PrefixMachine(m,session,raw[:128],mode)
            for y in raw[128:]:
                p=state.predict(); assert p.min()>0 and abs(float(p.detach().sum())-1)<1e-5
                _,_,edges,_,_=state.pending; e=max(edges,key=lambda e:float(e.weight.detach())); pred=(e.op,*e.cursor)
                part=('start' if e.op==1 else 'continue') if e.op and e.byte==y else ('mismatch' if e.op else 'no-copy')
                w=torch.stack([e.weight.detach() for e in edges]); loss=-p[y].log(); losses.append(loss)
                rows.append(dict(decision=pred,part=part,nll=float(loss.detach()),entropy=float(-torch.xlogy(w,w).sum()),ops=state.observe(y),rows=len(state.rows)))
            loss=torch.stack(losses).mean()
            if opt:
                opt.zero_grad(); loss.backward()
                for k,v in m.named_parameters():
                    if v.requires_grad and v.grad is not None: grad[k]=float(v.grad.norm())
                assert grad and all(np.isfinite(list(grad.values())))
                torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True); opt.step()
    finally: session.close()
    torch.cuda.synchronize(); return dict(nll=float(loss.detach()),rows=rows,grad=grad,seconds=time.monotonic()-tic)


@torch.no_grad()
def baseline(net,raw,weights):
    tic=time.monotonic(); r=Resident(j.EXE,'backoff',audit=True); off=[]; indexed=[]
    try:
        r.append(raw[:128]);
        for y in raw[128:]:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                ids=torch.tensor(list(r.raw[-128:]),device='cuda')[None]; base=net(ids,doors=False)[0][0,-1].float().softmax(0).cpu().numpy()
            hist,f=evidence(base,r.read()); a=mass(np.array(f),0.,'support',weights); off.append(-np.log(base[y])); indexed.append(-np.log((1-a)*base[y]+a*hist[y])); r.append(bytes([y]))
    finally: r.session.close()
    return dict(off=float(np.mean(off)),indexed=float(np.mean(indexed)),baseline_seconds=time.monotonic()-tic)


@torch.no_grad()
def replay(m,raw):
    assert len(raw)>=128; s=BookmarkSession(j.EXE)
    try:
        state=PrefixMachine(m,s,raw[:128]); pred=state.predict()
        for y in raw[128:]: state.observe(y); pred=state.predict()
        return pred
    finally: s.close()


def qualify(m,net,stream):
    cache=O/'dev-000.npy'; assert j.digest(cache)=='a4217db3dd69324d4c691c06e74cf962d1b190d936113baf4e45fa89a46f8556'
    d=json.loads((O/'panels.json').read_text())['dev'][0]; raw=payload(stream,d); err=[]
    with torch.no_grad():
        for i,v in enumerate(np.load(cache)[:8,-2],3840):
            _,p=m.neural(raw[i-128:i],m.write.bias*0); err.append(abs(float(p[raw[i]])-v))
    j.emit(anchor_errors=err); assert max(err)<=.01 and np.mean(err)<=.0001
    save(OUT/'birth.pt',dict(params=adapter(m),anchor=ANCHOR_SHA,panel=j.digest(OUT/'panels.json')))
    clone=PrefixK15(net.backbone).cuda(); restore(clone,OUT/'birth.pt'); s=BookmarkSession(j.EXE)
    try:
        state=PrefixMachine(m,s,PREFIX)
        for i,y in enumerate(b'ABA'):
            p=state.predict(); assert torch.equal(p.detach(),replay(clone,PREFIX+b'ABA'[:i])); assert i>=2 or int(p.argmax())==y
            if not i: first=next(e.weight for e in state.pending[2] if e.byte==65); write=state.pending[3]
            if i==2:
                g=torch.autograd.grad(-p[y].log(),(first,write)); j.emit(delayed_norms=[float(v.norm()) for v in g]); assert all(v.isfinite().all() and v.norm()>0 for v in g)
            state.observe(y)
    finally: s.close()


def main(args):
    assert __debug__ and os.environ['OMP_NUM_THREADS']=='4'; tic=time.monotonic(); stream,ident=setup(); net=load_net().eval(); net.requires_grad_(False)
    def model(compiled):
        torch.manual_seed(3411); return PrefixK15(net.backbone,compiled=compiled).cuda().eval()
    models={'compiled':model(True),'random':model(False),'birth':model(True)}
    if args.replay:
        ck,prefix,out=args.replay; restore(models['compiled'],Path(ck)); np.save(out,replay(models['compiled'],Path(prefix).read_bytes()).cpu().numpy()); return
    p=panels(stream); put(OUT/'panels.json',dict(identity=ident,**p)); qualify(models['compiled'],net,stream); j.emit(qualified=True,seconds=time.monotonic()-tic)
    opts={a:torch.optim.Adam([v for v in models[a].parameters() if v.requires_grad],lr=.001) for a in ('compiled','random')}
    assert j.digest(O/'checkpoint.json')==j.digest(R/'k15_singleton_credit/checkpoint.json'); weights=json.loads((O/'checkpoint.json').read_text())['weights']['support']
    def assess(stage):
        rows=[]
        for d in p[stage]:
            raw=payload(stream,d)[:136]; r=dict(doc=d['doc'],**baseline(net,raw,weights)); details={}
            for a in (*models,'skip','wrong','scramble','freeze'):
                details[a]=run(models.get(a,models['compiled']),raw,mode='full' if a in models else a); r[a]=details[a]['nll']
            r['changes']=sum(x['decision']!=y['decision'] for x,y in zip(details['compiled']['rows'],details['birth']['rows'])); rows.append(r); j.emit(stage=stage,detail=details,**r)
        gains={a:interval([r[a]-r['compiled'] for r in rows]) for a in ('off','indexed','random','birth')}; changed=sum(r['changes'] for r in rows); passed=changed>0 and (gains['birth'][0]>0 if stage=='early' else all(v[0]>=.001 and v[1]>0 for v in gains.values())); j.emit(stage=stage,gains=gains,changes=changed,passed=passed); return passed
    for i,d in enumerate(p['train'],1):
        raw=payload(stream,d)[:136]; t=time.monotonic()
        for a,opt in opts.items():
            row=run(models[a],raw,opt); path=OUT/f'{a}-{i}.pt'; save(path,dict(step=i,params=adapter(models[a]),anchor=ANCHOR_SHA,panel=j.digest(OUT/'panels.json'),optimizer=opt.state_dict())); j.emit(step=i,arm=a,doc=d['doc'],checkpoint_sha=j.digest(path),**row)
        if i==1:
            estimate=time.monotonic()-tic+(time.monotonic()-t)/2*400+120; j.emit(estimated_seconds=estimate); assert estimate<1800
        if i==8 and not assess('early'): return
    if assess('dev'): assess('locked')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--replay',nargs=3,metavar=('CHECKPOINT','PREFIX','OUTPUT_NPY')); main(ap.parse_args())
