"""Stage O singleton extraction, qualification and locked assessment."""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from .k15_context_run import (START, ANCHOR_SHA, OutcomeRows, design, evidence,
                              interval, j, load_net, local, mass, payload, prefill, put, setup)
from .k15_historical_training import REACH

OUT = Path('output/k15O')
ARMS = ('global', 'support', 'online', 'stateless', 'unkeyed', 'writable', 'reset')


def panels(stream):
    root = j.B.parents[1]
    ids = {json.loads(line)['doc'] for path in [REACH, *(p for a in 'DEFGH' for p in (root/f'k15_bookmarks_{a}').glob('*.jsonl'))] for line in path.open()}
    npath = Path('output/k15N/panels.json')
    assert j.digest(npath) == 'dd326f11499b3c0f09578390820ae277fc1b4f6a31b71527aeca0bf981b1d4f4'
    for path in [*(root/a/'panels.json' for a in ('k15_coupled_I','k15_relation_J','k15_counterfactual_K')), Path('output/k15L-pilot/panels.json'), Path('output/k15M/panels.json'), npath]:
        for docs in json.loads(path.read_text()).values():
            if isinstance(docs, list): ids.update(d['doc'] for d in docs)
    old = [d for d in stream.documents_iter() if d['doc'] in ids]
    assert {d['doc'] for d in old} == ids
    seen = {d['sha256'] for d in old}
    limit = max(d['offset'] for d in old if d['split']=='calibration')
    def fresh():
        for d in stream.documents('calibration'):
            if d['offset'] > limit and d['doc'] not in ids and d['length'] >= 4096 and d['sha256'] not in seen:
                ids.add(d['doc']); seen.add(d['sha256']); yield d
    it = fresh(); result = dict(train=json.loads(npath.read_text())['train'], dev=list(itertools.islice(it,48)), locked=list(itertools.islice(it,48)))
    assert [len(v) for v in result.values()] == [96,48,48]
    return result


def singleton(net, prefix):
    assert len(prefix)==128 and not net.training
    return local(net,[prefix])[0]


def extract(net, raw):
    state = prefill(raw); rows = OutcomeRows(); records=[]; dense=[]
    try:
        for t in range(START,4096):
            p=singleton(net,raw[t-128:t]); h,f=evidence(p,state.read()); key=raw[t-2:t]
            c,hit=rows.read(key); u,_=rows.read(b''); y=raw[t]
            records.append([*f,c,u,hit,p[y],h[y]])
            if t<START+32: dense.append([p,h])
            for k in (key,b''): rows.observe(k,p,h,y)
            state.append(bytes([y]))
    finally: state.session.close()
    return np.array(records), np.array(dense)


def credit(x,arm): return x[...,6 if arm=='unkeyed' else 5]


def loss(x,arm,w):
    a=mass(x[...,:5],credit(x,arm),arm,w)
    return -np.log((1-a)*x[...,-2]+a*x[...,-1])


def fit(x):
    weights={}; f=x[:,:5]; p,q=torch.from_numpy(x[:,-2:].T)
    for arm in ARMS[:-1]:
        if arm in ARMS[:3]:
            grid=([v] if arm=='global' else [v,1.] for v in np.linspace(-6,3,128))
            weights[arm]=min(grid,key=lambda w:loss(x,arm,w).mean()); continue
        a=torch.from_numpy(design(f,credit(x,arm),arm)); w=torch.tensor([-1.5]+[0.]*6,dtype=a.dtype,requires_grad=True); opt=torch.optim.Adam([w],lr=.05)
        for _ in range(128):
            z=(a@w).sigmoid()*torch.from_numpy(f[:,0]>0); nll=-((1-z)*p+z*q).log().mean(); opt.zero_grad(); nll.backward(); opt.step()
        weights[arm]=w.detach().tolist()
    weights['reset']=weights['writable']; return weights


def replay(net, observed, arm, weights):
    """Final prediction has no observation; each invocation starts empty."""
    state=prefill(observed); rows=OutcomeRows(); predictions=[]
    try:
        for i in range(START,len(observed)+1):
            if i>START:
                y=observed[i-1]; rows.observe(key,p,h,y); state.append(bytes([y]))
            key=b'' if arm=='unkeyed' else state.raw[-2:]; p=singleton(net,state.raw[-128:]); h,f=evidence(p,state.read())
            a=mass(f,rows.read(key)[0],arm,weights); predictions.append((1-a)*p+a*h)
    finally: state.session.close()
    return np.array(predictions)


def qualify(net,raw,x,dense,weights,stage):
    audit={}
    for arm in ARMS:
        start=time.monotonic(); pred=replay(net,raw[:START+31],arm,weights[arm]); sec=time.monotonic()-start
        a=mass(x[:32,:5],credit(x[:32],arm),arm,weights[arm])[:,None]; ref=(1-a)*dense[:,0]+a*dense[:,1]; err=np.abs(pred-ref)
        audit[arm]=dict(seconds=sec,max=err.max(),mean=err.mean(),normalization=np.abs(pred.sum(1)-1).max(),worst_position=int(np.unravel_index(err.argmax(),err.shape)[0])+START)
        put(OUT/f'{stage}-parity.json',audit); assert err.max()<=.01 and err.mean()<=.0001 and audit[arm]['normalization']<1e-5


def main():
    start=time.monotonic(); stream,ident=setup(); panel=panels(stream); put(OUT/'panels.json',dict(identity=ident,**panel)); net=load_net().eval()
    for stage,docs in panel.items():
        records=[]; begin=time.monotonic()
        for i,d in enumerate(docs):
            raw=payload(stream,d); x,dense=extract(net,raw); records.append(x)
            path=OUT/f'{stage}-{i:03}.npy'; tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f: np.save(f,x)
            tmp.replace(path); j.emit(stage=stage,step=i+1,doc=d['doc'],cache_sha=j.digest(path))
            if i==0:
                first=(raw,x,dense)
                if stage!='train': qualify(net,*first,weights,stage)
            if stage=='train' and i==7:
                elapsed=time.monotonic()-begin; total=time.monotonic()-start+elapsed/8*184+180; hit=np.stack(records)[...,7].mean(); put(OUT/'price.json',dict(seconds=elapsed,estimated_total=total,revisit=hit)); assert total<5200 and hit>=.2
        x=np.stack(records)
        if stage=='train':
            weights=fit(x.reshape(-1,x.shape[-1])); put(OUT/'checkpoint.json',dict(weights=weights,anchor=ANCHOR_SHA,panels=j.digest(OUT/'panels.json'))); qualify(net,*first,weights,stage)
        scores={a:loss(x,a,weights[a]).mean(1) for a in ARMS}; gains={a:interval(s-scores['writable']) for a,s in scores.items() if a!='writable'}; passed=all(v[0]>=.001 and v[1]>0 for v in gains.values()); put(OUT/f'{stage}.json',dict(nll={a:interval(s) for a,s in scores.items()},gain=gains,passed=passed))
        if stage=='dev' and not passed: break
    j.emit(complete=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(); cli.add_argument('--replay',nargs=2,metavar=('PREFIX','NPY')); args=cli.parse_args()
    if args.replay:
        setup(); raw=Path(args.replay[0]).read_bytes(); assert START<=len(raw)<4096
        weights=json.loads((OUT/'checkpoint.json').read_text())['weights']['writable']; np.save(args.replay[1],replay(load_net().eval(),raw,'writable',weights))
    else: main()
