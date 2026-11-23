"""Answer-supervised Emender operand selection over native XSA range registers."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from ..x45_matrix_emender.model import MatrixShape, MatrixBlock, RoutingPolicy
from .xsa_binding_fixture import fixture, K
from .xsa_binding_live import Session


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(6,24)
        self.block=MatrixBlock(MatrixShape(width=24,hidden=48,heads=1,value_dim=8,rank=4,rank_max=4),RoutingPolicy(),1)
        self.head=nn.Linear(24,3)

    def forward(self,counts):
        available=counts>=0
        x=torch.cat((counts.clamp(min=0).log1p(),available.float()),-1)
        return self.head(self.block(self.embed(x))).masked_fill(~available,-1e9)


class Machine:
    def __init__(self,s,prefix,seed):
        self.s=s;self.hi=len(prefix)-17;q=prefix[-17:];self.keys=[q[1:6],q[6:11],q[11:16]]
        random.Random(6571+seed).shuffle(self.keys);self.calls=0;self.positions=0
        s.rpc('append '+prefix.hex());s.rpc('seal')

    def locate(self,key,lo,hi):
        self.calls+=1;out=self.s.rpc(f'locate {key.hex()} {lo} {hi} 4096')['positions'];self.positions+=len(out);return out

    def byte(self,p):
        self.calls+=1;return self.s.rpc(f'byte {p}')['byte']

    def step(self,span,used,history,path,choose,rotate=False):
        # Enumerate only in training. Greedy test code takes one path, no retry.
        lo,hi=span
        hits=[None if i in used else self.locate(k,lo,hi) for i,k in enumerate(self.keys)]
        obs=history+[[-1 if h is None else len(h) for h in hits]]
        picks=range(3) if choose is None else [choose(obs)]
        for i in picks:
            if hits[i] is None or len(hits[i])!=1: continue  # abstention outcome
            pos,=hits[i]; key=self.keys[i]; tail=pos+len(key)
            if len(used)==2:
                if tail+K>=hi: continue
                if rotate:
                    peers=self.locate(key,0,self.hi);pos=peers[(peers.index(pos)+1)%len(peers)];tail=pos+len(key)
                yield obs,path+[i],self.byte(tail+K)
            else:
                end=hi
                for at in self.locate(key[:1],tail,hi):
                    if at+1<hi and self.byte(at+1) in b'0123456789abcdef':end=at;break
                yield from self.step((tail,end),used|{i},obs,path+[i],choose,rotate)

    def run(self,policy=None,rotate=False):
        return list(self.step((0,self.hi),set(),[],[],policy,rotate))


def examples(exe,seeds,shape):
    for seed in seeds:
        for raw,answer in fixture(seed,*shape):
            s=Session(exe)
            try: yield Machine(s,raw,seed),answer
            finally:s.close()


def test(exe,seeds,shape,net):
    stats={}
    for name in ['learned','unique_first','rotate']:
        def choose(hist):
            if name=='unique_first':return next((i for i,n in enumerate(hist[-1]) if n==1),0)
            return int(net(torch.tensor([hist]).float())[0,-1].argmax())
        hits=[];calls=pos=0;sample=None
        with torch.no_grad():
            for machine,y in examples(exe,seeds,shape):
                result=machine.run(choose,name=='rotate');hits.append(bool(result) and result[0][2]==y)
                calls+=machine.calls;pos+=machine.positions
                if sample is None and result:sample=dict(counts=result[0][0],actions=result[0][1],got=result[0][2],expected=y)
        stats[name]=dict(correct=sum(hits),targets=len(hits),pairs_correct=sum(a and b for a,b in zip(hits[::2],hits[1::2])),index_calls=calls,positions_returned=pos,first=sample)
    return stats


def main(exe,out):
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    here=Path(__file__).resolve().parent
    if digest(exe)!=json.loads((here/'receipts/xsa_binding_live.json').read_text())['engine_sha256']:raise ValueError('unvalidated engine')
    torch.set_num_threads(4);torch.manual_seed(3401);out.mkdir(parents=True,exist_ok=False)
    obs=[];paths=[];is_correct=[]
    for machine,y in examples(exe,range(1000,1064),(4,2,3)):
        ends=machine.run()
        if len(ends)!=1:raise RuntimeError('marginal implementation requires one nonabstaining execution')
        o,a,v=ends[0];obs.append(o);paths.append(a);is_correct.append(v==y)
    x=torch.tensor(obs).float();a=torch.tensor(paths);right=torch.tensor(is_correct).float();net=Policy();optim=torch.optim.AdamW(net.parameters(),lr=.003);losses=[]
    for t in range(200):
        idx=torch.randint(len(x),(16,)); logits=net(x[idx]); prob=logits.softmax(-1).gather(-1,a[idx,:,None]).squeeze(-1).prod(-1)
        loss=-((1-prob)/256+right[idx]*prob).log().mean();optim.zero_grad();loss.backward();optim.step();losses.append(float(loss.detach()))
    ck=out/'selector.pt';torch.save(net.state_dict(),ck);net=Policy();net.load_state_dict(torch.load(ck,weights_only=True));net.eval()
    stats=dict(iid=test(exe,range(2000,2032),(4,2,3),net),large=test(exe,range(3000,3032),(8,4,4),net))
    receipt=dict(engine_sha256=digest(exe),code_sha256=digest(Path(__file__)),checkpoint_sha256=digest(ck),train_targets=len(x),train_nll_first=losses[0],train_nll_last=losses[-1],parameters=sum(p.numel() for p in net.parameters()),test=stats)
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('exe',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.exe.resolve(),a.out)
