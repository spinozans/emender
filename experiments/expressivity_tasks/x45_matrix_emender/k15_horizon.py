"""Resident label-free memory companion for a fixed-prefix K15 distribution."""
from collections import Counter
from math import sqrt
from pathlib import Path
import time
from experiments.expressivity_tasks.x43_fork_resolved.xsa_binding_live import Session
from .k15_handle_bank import Seed, continuations

POLICIES=('depth8','backoff','current','newest','aggregate')


def occurrences(raw,key):
    i=raw.find(key)
    while i>=0:
        yield i
        i=raw.find(key,i+1)


class Resident:
    def __init__(self,exe,mode,audit=False):
        assert mode in POLICIES
        self.session=Session(exe); self.mode=mode; self.audit=audit
        self.raw=b''; self.bank=[]; self.old=[]; self.seed=None; self.work=Counter()

    def rpc(self,cmd):
        self.work[cmd.split()[0]]+=1
        return self.session.rpc(cmd)

    def append(self,chunk):
        self.raw+=chunk; raw=self.raw; s=len(raw)
        assert self.rpc('append '+chunk.hex())['visible']==s
        if s%1024==0: assert self.rpc('seal')['sealed']==s
        if s%32 or self.mode in ('depth8','backoff'): return
        self.old=self.bank; self.seed=None
        for depth in (8,4,2):
            pat=raw[-depth:]; r=self.rpc(f'locate {pat.hex()} 0 {s-1} {s}'); hits=r['positions']
            if self.audit: assert hits==list(occurrences(raw[:-1],pat))
            for k in ('sealed_rows','tail_starts'): self.work[k]+=r[k]
            self.work['positions']+=len(hits)
            if hits:
                self.seed=Seed(s,pat,tuple(i+depth for i in hits)); self.bank=([self.seed]+self.bank)[:4]; break

    def read(self):
        raw=self.raw; s=len(raw)
        if self.mode in ('depth8','backoff'):
            for depth in ((8,) if self.mode=='depth8' else (8,4,2)):
                pat=raw[-depth:]; r=self.rpc('branch '+pat.hex()); assert r['end']==0
                h=Counter({i:v for i,v in enumerate(r['counts']) if v})
                if self.audit: assert h==Counter(raw[i+depth] for i in occurrences(raw[:-1],pat))
                if h: return h
            return Counter()
        block=s//32*32; lag=s-block
        seeds=[self.seed]+(self.old[:1] if self.mode=='newest' else self.old if self.mode=='aggregate' else [])
        h=Counter()
        for i,seed in enumerate(seeds):
            if not seed: continue
            v=int(continuations(raw,block,seed,i!=0,False)[0][lag]); self.work['handle_checks']+=1
            if v>=0:
                h[v]+=len(seed.pattern)*sqrt(len(seed.addresses)) if self.mode=='aggregate' else 1
                if self.mode!='aggregate': break
        return h


def serve(exe,mode,raw,start,audit=False):
    """Only consumed bytes are passed into the live object. No label tensor."""
    begin=time.perf_counter(); state=Resident(exe,mode,audit); rows=[]; latency=[]
    try:
        for s in range(0,start,32): state.append(raw[s:s+32])
        prefill=time.perf_counter()-begin
        for t in range(start,len(raw)+1):
            tic=time.perf_counter()
            if t>start: state.append(raw[t-1:t])
            rows.append(state.read()); latency.append(time.perf_counter()-tic)
        rss=int(next(l for l in Path(f'/proc/{state.session.p.pid}/status').read_text().splitlines() if l.startswith('VmHWM:')).split()[1])
    finally: state.session.close()
    return rows,dict(total=time.perf_counter()-begin,prefill=prefill,latency=latency,native_peak_kib=rss,work=dict(state.work))
