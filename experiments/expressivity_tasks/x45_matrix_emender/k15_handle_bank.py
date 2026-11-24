"""Prefix-bound occurrence sets. A recall selects an entire saved seed for 32 targets."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Seed:
    stop: int
    pattern: bytes
    addresses: tuple


def continuations(raw, block, seed, restart, short):
    out = np.full(32, -1, np.int16)
    stops = np.full(32, -1, np.int16)
    if seed is None:
        return out, stops
    live = () if restart else seed.addresses
    origin = block
    pattern = seed.pattern[-1:] if short else seed.pattern
    for lag in range(min(32,len(raw)-block+1)):
        p = block + lag
        if lag and restart and raw[p-len(pattern):p] == pattern:
            live, origin = seed.addresses, p
        elif lag:
            live = tuple(a for a in live if a+p-origin < seed.stop and raw[a+p-origin-1] == raw[p-1])
        if live:
            out[lag] = raw[live[-1]+p-origin]; stops[lag] = origin
    return out, stops


def enumerate_bank(raw, session, short=False, audit=False, only_rank=None):
    assert 32<=len(raw)<=1024
    bank=[]; blocks=[]; traces=[]; features=[]; work=dict(locate=0, tail_starts=0, sealed_rows=0, misses=0, hits=0)
    for s in range(32,len(raw)+1,32):
        assert session.rpc('append '+raw[s-32:s].hex())['visible']==s
        if s==1024: break
        current=None
        for depth in (8,4,2):
            pat=raw[s-depth:s]; res=session.rpc(f'locate {pat.hex()} 0 {s-1} {s}'); hits=res['positions']
            if audit: assert hits==[i for i in range(s-depth) if raw[i:i+depth]==pat]
            for k in ('tail_starts','sealed_rows'): work[k]+=res[k]
            work['locate']+=1; work['misses']+=not hits
            if hits:
                current=Seed(s,pat,tuple(i+depth for i in hits)); work['hits']+=len(hits); break
        base=continuations(raw,s,current,False,False)[0]; options=[base]; desc=[[0.,0.,0.,0.]]; log=[]
        if callable(only_rank):
            metadata=[[0.,0.,0.,0.]]+ [[(s-seed.stop)/1024,len(seed.pattern)/8,len(seed.addresses)/s,1.] for seed in bank]+[[0.]*4]*(4-len(bank))
            selected=only_rank(s,metadata)
        else: selected=only_rank
        for rank in range(4):
            seed=bank[rank] if rank<len(bank) else None
            alt, origins=continuations(raw,s,seed if selected in (None,rank+1) else None,True,short)
            options.append(np.where(base>=0,base,alt)); desc.append([(s-seed.stop)/1024,len(seed.pattern)/8,len(seed.addresses)/s,1.] if seed else [0.]*4)
            if seed:
                for lag in np.flatnonzero((base<0)&(alt>=0)):
                    log.append(dict(t=int(s+lag-1),block=s,rank=rank+1,seed_stop=seed.stop,pattern=seed.pattern.hex(),addresses=seed.addresses,restart=int(origins[lag]),byte=int(alt[lag]),current_stop=current.stop if current else None))
        blocks.append(options); features.append(desc); traces.append(log)
        if current: bank=([current]+bank)[:4]
    return np.array(blocks), np.array(features), traces, work


class BankPort(nn.Module):
    short=False
    only_rank=None
    def forward(self,mixer,hidden,ids,session,forced=None,audit=False):
        opts,meta,trace,work=enumerate_bank(bytes(ids[0].tolist()),session,self.short,audit,self.only_rank)
        return hidden.new_zeros(1,1024,mixer.value_dim).float(), (opts,meta), (trace,work), hidden[0,31:992:32].float()


class Recall(nn.Module):
    def __init__(self,width):
        super().__init__(); self.score=nn.Sequential(nn.Linear(width+4,32),nn.Tanh(),nn.Linear(32,1))

    def forward(self,hidden,meta):
        score=self.score(torch.cat((hidden[:,None,:].expand(-1,5,-1),meta),-1)).squeeze(-1)
        mask=meta[:,:,3]==0; mask[:,0]=False
        return score.masked_fill(mask,-1e9)

class DecodePort(nn.Module):
    def __init__(self,controller=None):
        super().__init__(); self.controller=controller

    def forward(self,mixer,hidden,ids,session,forced=None,audit=False):
        picks=[]
        def select(stop,metadata):
            if self.controller is None: rank=0
            else:
                with torch.autocast('cuda',enabled=False):
                    rank=int(self.controller(hidden[:,stop-1].float(),torch.tensor(metadata,device=ids.device)[None]).argmax())
            picks.append(rank); return rank
        opts,_,_,work=enumerate_bank(bytes(ids[0].tolist()),session,only_rank=select)
        return hidden.new_zeros(1,ids.shape[1],mixer.value_dim).float(),opts[np.arange(len(picks)),picks],work,picks


def decode_last(logits,selected):
    pos=logits.shape[1]-1; index=pos-31; dist=logits[0,-1].float().softmax(-1)
    if 0<=index<selected.size:
        byte=selected.flatten()[index]
        if byte>=0:
            dist=dist*.95; dist[int(byte)]+=.05
    return dist
