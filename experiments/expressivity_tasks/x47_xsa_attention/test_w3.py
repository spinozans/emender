"""Independent document/causal/count and memory-credit qualification (CPU)."""
import hashlib
import io
import numpy as np
import pytest
import torch
from .w3_data import ROW,compile_shard,getwindow,T
from .w3_model import Head,likelihood,ARMS


class Tiny:
    def __init__(self,docs):
        self.counts={}
        for doc in docs:
            for i in range(len(doc)+1):
                for n in range(min(i,8)+1):
                    k=doc[i-n:i]; v=doc[i] if i<len(doc) else 256
                    h=self.counts.setdefault(k,{}); h[v]=h.get(v,0)+1
        self.keys=[None,*sorted(self.counts)]; self.id={k:i for i,k in enumerate(self.keys)}; self.root=self.id[b'']
    def suffix(self,b):
        return np.array([[self.id.get(b[i+1-j:i+1],0) if j<=i+1 else 0 for j in range(1,9)] for i in range(len(b))],np.uint32)
    def rows(self,ids):
        r=np.zeros(len(ids),ROW)
        for i,id in enumerate(ids):
            if id:
                k=self.keys[id]
                for c,n in self.counts[k].items():
                    r['h'][i,c]=n
                    if c!=256: r['child'][i,c]=self.id.get(k+bytes([c]),0)
        return r


def case():
    svc=Tiny([b'whatM1',b'whatM1',b'whatS2',b'\0\x1e\xff'])
    raw=b'..whatM1'+b' something more than sixteen bytes pass;'+b'whatM1whatS2'; return svc,raw


def test_every_prefix_matches_singleton_and_live_scan():
    svc,raw=case(); all_=compile_shard(svc,[raw]); active=0
    for i in range(len(raw)-1):
        prefix=raw[:i+1]; y=raw[i+1]; piece=compile_shard(svc,[prefix+bytes([255-y])]); a=(0,i)
        for k in ('feat','ptr','prior','codes','on'):
            np.testing.assert_array_equal(all_[k][a],piece[k][a])
        np.testing.assert_array_equal(all_['table'][all_['ci'][a]],piece['table'][piece['ci'][a]])
        # independent one-document exhaustive eligible continuation counts and address
        for d in (8,4,2):
            if len(prefix)<d: continue
            events=[prefix[j+d] for j in range(len(prefix)-d) if prefix[j:j+d]==prefix[-d:]]
            if events: break
        else: events=[]
        np.testing.assert_allclose(all_['target_mass'][a][1],events.count(y)/len(events) if events else 0)
        j=-1
        for d in (4,2):
            k=prefix[-d:]; last=[p for p in range(d-1,i-17) if prefix[p+1-d:p+1]==k and k in svc.counts]
            if last: j=last[-1]+2; break
        assert all_['ptr'][a]==j
        active+=int(all_['on'][a]); assert not all_['on'][a] or i-j>=16
    assert active>0
    duo=compile_shard(svc,[raw,raw]); np.testing.assert_array_equal(duo['ptr'][1],all_['ptr'][0])


def test_full_distribution_birth_and_delayed_gradient():
    svc,raw=case(); z=compile_shard(svc,[raw]); torch.manual_seed(4703); m=Head(); ti=lambda k:torch.from_numpy(z[k])
    x=ti('raw')[:,:-1].long(); h=torch.randn(1,len(raw)-1,256,requires_grad=True)
    args=(h,x,ti('feat'),ti('ptr').long(),ti('codes').long(),ti('prior'),ti('table')[ti('ci').long()],ti('on'))
    i=int(np.flatnonzero(z['on'][0])[0]); j=int(z['ptr'][0,i]); p=args[5][0,i]; c=args[6][0,i]; tag=x[0,j]
    w,e,_=m(*args,'full'); expected=torch.stack((p,1-p))*(1+32*c[:,tag]); torch.testing.assert_close(e[0,i],expected/expected.sum())
    assert torch.allclose(w.sum(-1),torch.ones_like(w[...,0])); assert (i-j)>=16
    wb,eb,_=m(*args,'blind'); torch.testing.assert_close(eb[0,i],torch.stack((p,1-p)))
    direct,_,_=m(*args,'direct'); assert (direct[...,-1]==0).all()
    g,=torch.autograd.grad(-e[0,i].log()[0],m.writer.weight); assert g.abs().max()>0
    # The target gathers agree with building all 256 entries of this distribution.
    a=svc.counts[raw[i+1-int(round(float(z['feat'][0,i,0])*8)):i+1]]; ap=np.zeros(256)
    for b,n in a.items():
        if b!=256: ap[b]=n
    ap/=ap.sum(); y=torch.tensor(z['raw'][:,1:]).long(); q=torch.full_like(h,1/256)
    py=likelihood(m(*args,'full'),q[...,0],ti('target_mass'),args[4],y)
    # Recover live distribution independently, including structural data bytes.
    lp=np.zeros(256); d=round(float(z['feat'][0,i,3])*8)
    if d:
        for l in range(i+1-d):
            if raw[l:l+d]==raw[i+1-d:i+1]: lp[raw[l+d]]+=1
        lp/=lp.sum()
    copy=np.eye(256)[raw[j-1]]; edge=np.zeros(256); edge[z['codes'][0,i]]=e[0,i].detach().numpy()
    dense=w[0,i].detach().numpy()@np.stack((np.ones(256)/256,ap,lp,copy,edge)); assert abs(dense.sum()-1)<1e-5
    assert abs(float(py[0,i].detach())-dense[y[0,i]])<1e-5


def test_hash_and_window_seam_refused():
    raw=b'x'*(T+1); d=dict(length=len(raw),offset=0,start=0,sha256=hashlib.sha256(raw).hexdigest())
    assert getwindow(io.BytesIO(raw),d)==raw
    with pytest.raises(AssertionError): getwindow(io.BytesIO(raw),dict(d,start=1))
    with pytest.raises(AssertionError): getwindow(io.BytesIO(raw[:-1]+b'y'),d)
