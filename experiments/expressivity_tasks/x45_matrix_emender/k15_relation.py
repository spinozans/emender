"""Frozen K15 with an occurrence-set handle and a distilled middle-mixer decoder."""
if not __debug__:
    raise RuntimeError("K15 I/J requires Python assertions; unset PYTHONOPTIMIZE and omit -O")

import torch
from torch import nn


def reads(raw, session, mode='full', audit=False):
    sites=[]; stats=dict(locate=0,tail_starts=0,sealed_rows=0,alternative=0,hits=0)
    for s in range(32,len(raw)+1,32):
        if session.rpc('append '+raw[s-32:s].hex())['visible'] != s:
            raise RuntimeError('native append did not expose the expected prefix')
        if s==1024: break
        for depth in (8,4,2):
            pat=raw[s-depth:s]; ans=session.rpc(f'locate {pat.hex()} 0 {s-1} {s}')
            for k in ('tail_starts','sealed_rows'): stats[k]+=ans[k]
            stats['locate']+=1; hits=ans['positions']
            if audit: assert hits==[i for i in range(s-depth) if raw[i:i+depth]==pat]
            if hits: break
        if not hits: continue
        handle=[i+depth for i in hits]; stats['hits']+=len(handle)
        stats['alternative']+=int(len({raw[i] for i in handle})>1)
        for lag in range(min(32,len(raw)-s+1)):
            if lag and mode=='immediate': break
            if lag==1 and mode=='scramble': handle=sorted((a+s//2)%s for a in handle)
            # Refine using only consumed bytes; candidate payloads predate handle creation.
            if lag: handle=[a for a in handle if a+lag<s and raw[a+lag-1]==raw[s+lag-1]]
            if not handle: break
            bs=[raw[a+lag] for a in handle]; t=s+lag-1
            sites.append((t,bs if mode!='echo' else [raw[t]]*len(bs),handle.copy(),s))
    return sites,stats


class RelationPort(nn.Module):
    mode='full'
    def __init__(self,dim):
        super().__init__(); self.value=nn.Embedding(256,dim); nn.init.zeros_(self.value.weight)
        self.recency=nn.Parameter(torch.tensor(1.0))

    def forward(self,mixer,hidden,ids,session,forced=None,audit=False):
        assert ids.shape[0]==1 and 1<=ids.shape[1]<=1024
        sites,stats=reads(bytes(ids[0].tolist()),session,self.mode,audit)
        out=hidden.new_zeros(ids.shape[1],self.value.embedding_dim).float()
        if sites:
            t,b,a=zip(*[(t,b,(addr-s)/s) for t,bs,addrs,s in sites for b,addr in zip(bs,addrs)])
            t=torch.tensor(t,device=out.device); b=torch.tensor(b,device=out.device)
            w=(self.recency.clamp(-8,8)*torch.tensor(a,device=out.device)).exp()
            den=out.new_zeros(len(out)).index_add(0,t,w); out=out.index_add(0,t,(w/den[t])[:,None]*self.value(b))
        return out[None],sites,stats,[]
