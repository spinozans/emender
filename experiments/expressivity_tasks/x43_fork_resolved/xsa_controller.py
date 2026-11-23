"""A trainable byte predictor with a conditional gate on native XSA branch reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn
from .xsa_controller_data import TEST, BINARY, digest, load_pieces

SEED=3401

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(256,16)
        self.encoder=nn.Sequential(nn.Linear(256,96),nn.GELU())
        self.head=nn.Linear(96,256)
        self.gate=nn.Linear(99,1)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def forward(self,x,h,mode='learned'):
        z=self.encoder(self.embedding(x.long()).flatten(1)); q=self.head(z).softmax(-1)
        support=h.sum(1,keepdim=True); dist=h/support.clamp(min=1)
        if mode=='learned':
            features=torch.cat((z,support.log1p(),dist.amax(1,keepdim=True),(q*dist).sum(1,keepdim=True)),1)
            g=self.gate(features).sigmoid()
        else:
            g=torch.full_like(support,float(mode))
        g=g*(support>0)
        return q*(1-g)+dist*g,g[:,0],q,dist

class Data:
    def __init__(self,pieces,path):
        # Context for target 256 starts at 240, never uses the target itself.
        self.x=torch.from_numpy(np.concatenate([np.lib.stride_tricks.sliding_window_view(np.frombuffer(p,dtype='u1'),16)[240:2032] for p in pieces]))
        self.y=torch.from_numpy(np.concatenate([np.frombuffer(p[256:2048],dtype='u1') for p in pieces])).long()
        self.h=torch.zeros(len(self.y),256,dtype=torch.int16)
        with np.load(path) as archive:
            rows,bs,cs=archive['full'].T
            self.h[rows,bs]=torch.from_numpy(cs.copy()).short()
        assert len(self.x)==len(self.y)==1792*len(pieces)


def nll(p,y): return -p.gather(1,y[:,None])[:,0].clamp_min(1e-30).log()

@torch.no_grad()
def evaluate(model,data,mode,alter=None):
    losses=[]; correct=0; gates=[]
    for x,h,y in zip(data.x.split(512),data.h.split(512),data.y.split(512)):
        h=h.float()
        if alter=='rotate':h=h.roll(1,1)
        if alter=='mask':h=h*0
        p,g,_,_=model(x,h,mode); losses.append(nll(p,y)); correct+=int((p.argmax(1)==y).sum());gates.extend(g[h.sum(1)>0].tolist())
    loss=torch.cat(losses).double().view(-1,1792).mean(1).numpy(); return loss,dict(nll=float(loss.mean()),accuracy=correct/len(data.y),supported_gate_quantiles=np.quantile(gates,[0,.1,.5,.9,1]).tolist() if gates else None)

@torch.no_grad()
def choose_constant(model,data):
    grid=torch.arange(20,dtype=torch.float64)/20; totals=torch.zeros_like(grid)
    for x,h,y in zip(data.x.split(512),data.h.split(512),data.y.split(512)):
        h=h.float(); _,_,q,d=model(x,h); mask=(h.sum(1)>0)[:,None]; g=grid[None,:]*mask
        qy=q.gather(1,y[:,None]);dy=d.gather(1,y[:,None]);totals-=((1-g)*qy+g*dy).clamp_min(1e-30).log().sum(0)
    return float(grid[totals.argmin()])

@torch.no_grad()
def replay(model,exe,text,data):
    # Predict before ingesting each observed target. Compare the ENTIRE distribution.
    maximum=0.; correct=0; total=0.
    with subprocess.Popen([exe,'session','8'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as proc:
        def rpc(cmd):
            proc.stdin.write(cmd.encode()+b'\n');proc.stdin.flush();return json.loads(proc.stdout.readline())
        rpc('append '+text[:256].hex()); expected=model(data.x[:1792],data.h[:1792].float())[0]
        for i in range(256,2048):
            if i%512==0:rpc('seal')
            reply=rpc('branch '+text[i-8:i].hex());h=torch.tensor(reply['counts'])[None,:].float();x=torch.tensor(list(text[i-16:i]))[None,:]
            p=model(x,h)[0][0]; maximum=max(maximum,float((p-expected[i-256]).abs().max())); y=text[i];total-=float(p[y].log());correct+=int(p.argmax()==y)
            rpc('append '+text[i:i+1].hex())
        proc.stdin.close()
        if proc.wait(timeout=10) or maximum>2e-5:raise RuntimeError(f'live replay failed: {maximum}')
    return dict(targets=1792,max_probability_difference=maximum,nll=total/1792,correct=correct)


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    m=json.loads((args.data/'manifest.json').read_text()); evidence=json.loads((TEST/'evidence.json').read_text())
    if not (digest(TEST/'targets.json')==m['test_manifest_sha']==evidence['targets_manifest_sha256'] and digest(args.exe)==m['engine']==BINARY):raise ValueError('engine/manifest drift')
    train_path=args.data/'train.npz';test_path=TEST/'final_test.npz'
    if digest(train_path)!=m['evidence']['sha256'] or digest(test_path)!=evidence['final_test']['sha256']:raise ValueError('evidence digest')
    train=Data(load_pieces(m['train'],m['identity']['source']),train_path); models={}; result={}; epochs={}
    for name,mode in [('controller','learned'),('fixed',.5),('local',0.)]:
        torch.manual_seed(SEED);model=Controller();optim=torch.optim.AdamW(model.parameters(),lr=.002);rng=torch.Generator().manual_seed(SEED);epochs[name]=[]
        for epoch in range(3):
            total=0.
            for idx in torch.randperm(len(train.y),generator=rng).split(512):
                p,*_=model(train.x[idx],train.h[idx].float(),mode);loss=nll(p,train.y[idx]).mean();optim.zero_grad();loss.backward();optim.step();total+=float(loss.detach())*len(idx)
            epochs[name].append(total/len(train.y));print(name,epoch+1,epochs[name][-1],flush=True)
        models[name]=model.eval()
    scalar=choose_constant(models['controller'],train); ck=out/'controller.pt';tmp=ck.with_suffix('.partial');torch.save(models['controller'].state_dict(),tmp);os.replace(tmp,ck)
    restored=Controller();restored.load_state_dict(torch.load(ck,weights_only=True));restored.eval()
    pieces=load_pieces(json.loads((TEST/'targets.json').read_text())['final_test'],m['identity']['source']);test=Data(pieces,test_path); per_doc={}
    for name,model,mode,alter in [('controller',restored,'learned',None),('masked',restored,'learned','mask'),('rotated',restored,'learned','rotate'),('scalar',restored,scalar,None),('fixed',models['fixed'],.5,None),('local',models['local'],0.,None)]:
        per_doc[name],result[name]=evaluate(model,test,mode,alter)
    ids=np.random.default_rng(SEED).integers(0,len(pieces),(10000,len(pieces)));paired={}
    for k,loss in per_doc.items():
        delta=loss-per_doc['controller'];paired[k]=dict(mean=float(delta.mean()),interval=np.quantile(delta[ids].mean(1),[.025,.975]).tolist())
    receipt=dict(scores=result,paired_minus_controller=paired,epoch_train_nll=epochs,scalar=scalar,targets=len(test.y),train_targets=len(train.y),parameters=sum(p.numel() for p in restored.parameters()),per_doc_nll={k:v.tolist() for k,v in per_doc.items()},live=replay(restored,args.exe,pieces[0],test),checkpoint_sha=digest(ck),engine=BINARY,data_manifest_sha=digest(args.data/'manifest.json'),code_sha=digest(Path(__file__)))
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('exe',type=Path);a.add_argument('data',type=Path);a.add_argument('out',type=Path);run(a.parse_args())
