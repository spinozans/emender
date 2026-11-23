"""Bounded, restartable CPU sweep; shards/weights are atomic and bound to inputs."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn
from . import xsa_budget_data as D

PRICES=[0,.002,.01,.04];SPLITS=['train','calibration','final_test']

class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.mix=nn.Sequential(nn.Linear(110,48),nn.GELU(),nn.Linear(48,3));self.value=nn.Sequential(nn.Linear(107,48),nn.GELU(),nn.Linear(48,2))
    def inputs(self,z,f):return torch.cat([z[:,None].expand(-1,f.shape[1],-1),f],-1)
    def weights(self,z,f):
        logits=self.mix(self.inputs(z,f));logits[:,:,2]=logits[:,:,2].masked_fill(f[:,:,0]>0,-1e9);return logits.softmax(-1)
    def estimate(self,z,f):return self.value(self.inputs(z,f[:,:,:11]))
    @staticmethod
    def pick(e,m,price):
        score=e[:,:,0]-price*e[:,:,1].clamp_min(0).expm1().clamp_max(4098);score[:,0]=0;return score.masked_fill(~m,-1e9).argmax(1)


def train(data,seed):
    z,f,p,m,c=[data[k] for k in ('z','f','p','m','c')];torch.manual_seed(seed);net=Policy();rng=torch.Generator().manual_seed(seed);curves={}
    for phase in ('mix','value'):
        opt=torch.optim.AdamW(getattr(net,phase).parameters(),lr=.002);curve=[]
        for _ in range(8):
            total=0.
            for i in torch.randperm(len(z),generator=rng).split(256):
                P=(net.weights(z[i],f[i])*p[i]).sum(-1).clamp_min(1e-30);nll=-P.log();mask=m[i]
                if phase=='mix':v=nll
                else:
                    est=net.estimate(z[i],f[i]);gain=(nll[:,:1]-nll).detach();v=(est[:,:,0]-gain).square()+(est[:,:,1]-c[i].float().clamp_min(1).log()).square()
                loss=(v.masked_fill(~mask,0).sum(1)/mask.sum(1)).mean();opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(i)
            curve.append(total/len(z))
        curves[phase]=curve
    return net.eval(),curves


@torch.no_grad()
def score(net,data):
    z,f,p,m,c=[data[k] for k in ('z','f','p','m','c')];ps=[];es=[]
    for i in torch.arange(len(z)).split(256):
        ps.append((net.weights(z[i],f[i])*p[i]).sum(-1));es.append(net.estimate(z[i],f[i]))
    P=torch.cat(ps);E=torch.cat(es);chosen=[net.pick(E,m,a) for a in PRICES];idx=torch.arange(len(z));fixed=[torch.where(m[:,k],k,0) for k in range(D.A)]
    def stats(j):
        loss=-P[idx,j].double().log();return dict(nll=float(loss.mean()),reads=float(c[idx,j].float().mean()),doc=loss.reshape(-1,256).mean(1).tolist(),moves=j.bincount(minlength=D.A).tolist())
    return [stats(j) for j in chosen],[stats(j) for j in fixed],float(-P.masked_fill(~m,0).max(1).values.double().log().mean()),P,E


@torch.no_grad()
def live(net,base,b,H,price,refp,refcost,expected):
    s=D.Session(D.EXE);last=0;lat=[];moves=Counter();errors=[];reads=0
    try:
        for i,t in enumerate(range(H-1792,H,7)):
            start=time.perf_counter_ns();ctx=b[max(0,t-192):t];count=0
            def rpc(cmd):
                nonlocal count
                count+=1;return s.rpc(cmd)
            s.rpc('append '+b[last:t].hex());
            if t//512>last//512:s.rpc('seal')
            last=t;z=base.encoder(base.embedding(torch.tensor([list(ctx[-16:])])).flatten(1));q=base.head(z).double().softmax(1)[0].numpy();h3=D.branch(rpc,ctx[-3:]);keys,menu=D.menu(ctx);K=len(keys);f=torch.zeros(1,K,14);f[0,:,:8]=torch.from_numpy(menu);f[0,:,8:11]=torch.tensor(D.features(h3,q));j=int(net.pick(net.estimate(z,f),torch.ones(1,K,dtype=torch.bool),price));h=D.action(rpc,j,keys,ctx,t);f[0,j,11:]=torch.tensor(D.features(h,q));w=net.weights(z,f)[0,j].numpy();pr=w[0]*q.copy()
            for wi,hist in zip(w[1:],(h3,h)):
                vv=q.copy()
                for byte,n in hist.items():vv[byte]+=n
                pr+=wi*vv/(1+hist.total())
            lat.append((time.perf_counter_ns()-start)/1e6);moves[j]+=1;reads+=count
            if j!=int(expected[i]) or count!=int(refcost[i,j]):raise ValueError(f'live choose/cost mismatch at {t}')
            errors.append(abs(float(pr[b[t]])-float(refp[i,j])))
    finally:s.close()
    if max(errors)>1e-5:raise ValueError('live probability mismatch')
    return dict(targets=len(lat),reads=reads,moves=moves,max_probability_error=max(errors),observe_predict_ms_p50_p99=np.quantile(lat,[.5,.99]).tolist())


def checkpoint(root,**kw):D.atomic(root/'status.json',dict(time_unix=time.time(),pid=os.getpid(),**kw))


def run(root,deadline):
    torch.set_num_threads(4);base,v1=D.models();bound=dict(engine=D.digest(D.EXE),data=D.digest(Path(D.__file__)),runner=D.digest(Path(__file__)),reg=D.digest(D.HERE/'PREREGISTRATION_BUDGETED_READS.md'),prior=D.digest(D.V1/'result.json'));bind=root/'binding.json'
    if bind.exists() and json.loads(bind.read_text())!=bound:raise ValueError('resume identity changed')
    D.atomic(bind,bound);overall=[];manifests=[]
    for H in (2048,8192):
        folder=root/str(H);folder.mkdir(exist_ok=True);manifest=D.manifest(H);src=manifest['identity']['source'];manifests.append(manifest);D.atomic(folder/'manifest.json',manifest);parts={}
        for split in SPLITS:
            chunks=[];work=Counter();targets=hashlib.sha256();docs=manifest['docs'][split];receipt=folder/(split+'_progress.json');hashes=json.loads(receipt.read_text()) if receipt.exists() else {}
            for i,d in enumerate(docs):
                if time.time()>deadline:raise TimeoutError('allocation expired')
                key=f'{i:04d}-{d["sha256"]}';path=folder/f'{split}-{key}.npz'
                if key not in hashes:
                    b=D.raw_doc(src,d,H);arrays,counts=D.extract(base,v1,b,H);tmp=path.with_suffix('.partial')
                    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
                    os.replace(tmp,path);hashes[key]=dict(sha=D.digest(path),work=counts);D.atomic(receipt,hashes)
                if D.digest(path)!=hashes[key]['sha']:raise ValueError('damaged shard')
                with np.load(path) as a:chunks.append({k:torch.from_numpy(a[k]) for k in a.files})
                targets.update(chunks[-1]['target'].numpy().tobytes());work.update(hashes[key]['work'])
                if i%8==0 or i+1==len(docs):checkpoint(root,phase='extract',horizon=H,split=split,documents_complete=i+1,documents_total=len(docs))
            parts[split]={k:torch.cat([v[k] for v in chunks]) for k in chunks[0]};D.atomic(folder/(split+'_identity.json'),dict(target_sha=targets.hexdigest(),targets=len(parts[split]['target']),counterfactual_native_work=work));del chunks
            if split=='train':
                for seed in (3401,3402):
                    ck=folder/f'{seed}.pt';rec=folder/f'{seed}_training.json';checkpoint(root,phase='train',horizon=H,seed=seed)
                    if not rec.exists():
                        net,curves=train(parts['train'],seed);tmp=ck.with_suffix('.partial');torch.save(net.state_dict(),tmp);os.replace(tmp,ck);D.atomic(rec,dict(curves=curves,sha=D.digest(ck)))
                    if D.digest(ck)!=json.loads(rec.read_text())['sha']:raise ValueError('checkpoint drift')
                del parts['train']
        for seed in (3401,3402):
            checkpoint(root,phase='score',horizon=H,seed=seed);net=Policy();net.load_state_dict(torch.load(folder/f'{seed}.pt',weights_only=True));net.eval();cal=parts['calibration'];test=parts['final_test'];calprice,calfixed,_,_,_=score(net,cal);allowed=[i for i,s in enumerate(calprice) if s['reads']<=cal['vc'].double().mean().item()]
            if not allowed:raise ValueError('no admissible price')
            pick=min(allowed,key=lambda j:calprice[j]['nll']);fix=min(range(D.A),key=lambda j:calfixed[j]['nll']);prices,fixed,oracle,P,E=score(net,test);sel=prices[pick];vdoc=-test['v'].double().log().reshape(-1,256).mean(1).numpy();delta=np.array(sel['doc'])-vdoc;ids=np.random.default_rng(3401).integers(0,len(delta),(10000,len(delta)));ci=np.quantile(delta[ids].mean(1),[.025,.975]);c=float(test['vc'].double().mean());b=D.raw_doc(src,manifest['docs']['final_test'][0],H);proof=live(net,base,b,H,PRICES[pick],P,test['c'],net.pick(E,test['m'],PRICES[pick]));result=dict(seed=seed,horizon=H,selected_price=PRICES[pick],calibration_prices=calprice,test_prices=prices,selected=sel,stop=fixed[0],fixed=fix,fixed_result=fixed[fix],oracle_nll=oracle,v1_nll=float(vdoc.mean()),v1_reads=c,v1_doc=vdoc.tolist(),delta_ci=ci.tolist(),passes=bool(ci[1]<=.01 and sel['reads']<=.75*c),strict_gain=bool(ci[1]<0),live=proof);D.atomic(folder/f'{seed}_result.json',result);overall.append({k:result[k] for k in ('seed','horizon','selected_price','v1_nll','v1_reads','delta_ci','passes','strict_gain')}|dict(nll=sel['nll'],reads=sel['reads']));D.atomic(root/'results_so_far.json',overall)
    overlap={s:len(set(d['sha256'] for d in manifests[0]['docs'][s])&set(d['sha256'] for d in manifests[1]['docs'][s])) for s in SPLITS};D.atomic(root/'complete.json',dict(cells=overall,horizon_document_overlap=overlap));checkpoint(root,phase='complete',cells=len(overall))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    try:run(a.root,a.deadline)
    except Exception:
        checkpoint(a.root,phase='failed',error=traceback.format_exc());raise
