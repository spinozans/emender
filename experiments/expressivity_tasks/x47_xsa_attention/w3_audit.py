"""Real first-training-window spot replay before GPU use; stock CLI, no working-set IDs."""
import json
import re
import subprocess
from functools import lru_cache

import numpy as np
from .w3_data import OUT,STOCK,getwindow,sha,dump
from .b1_mount import HERE,ROOT as NVME
from .workbench_witness import require


def verify():
    panel=json.loads((OUT/'panels.json').read_text()); r=panel['panels']['train'][0]
    with open(panel['source']['path'],'rb') as f: raw=getwindow(f,r)
    with np.load(OUT/'train/000.npz') as z: data={k:z[k] for k in z.files}
    require(data['raw'][0].tobytes()==raw,'stored bytes')
    index=NVME/json.loads((HERE/'b1_mounts.json').read_text())['P100']['files'][0]['path']
    @lru_cache(None)
    def stock(key):
        txt=subprocess.run([STOCK,'branch-of',index,'hex:'+key.hex()],capture_output=True,text=True,check=True,timeout=30).stdout
        if txt.startswith('absent '): return None
        h=np.zeros(257,np.uint64)
        for label,c in re.findall(r'^child (END|0x[0-9a-f]{2}) count=(\d+) node=',txt,re.M): h[256 if label=='END' else int(label,16)]=int(c)
        require(h.sum()==int(re.search(r'present count=(\d+)',txt)[1]),'stock closure'); return h
    rows=[]
    for start in (257,769,1537):
        choices=np.flatnonzero(data['on'][0,start:]); i=int(start+choices[0]) if len(choices) else start
        obs=raw[:i+1]; y=raw[i+1]; f=data['feat'][0,i]; ptr=int(data['ptr'][0,i]);
        for d in (8,6,4,2,1,0):
            h=stock(obs[-d:] if d else b'')
            if h is not None: break
        require(round(f[0]*8)==d and abs(data['target_mass'][0,i,0]-h[y]/max(1,h[:256].sum()))<1e-6,'archive base')
        for ld in (8,4,2):
            key=obs[-ld:]; hits=[obs[j+ld] for j in range(i+1-ld) if obs[j:j+ld]==key]
            if hits: break
        else: ld=0; hits=[]
        require(round(f[3]*8)==ld and abs(data['target_mass'][0,i,1]-hits.count(y)/max(1,len(hits)))<1e-6,'live base')
        for jd in (4,2):
            key=obs[-jd:]; j=obs.rfind(key,0,i-18+1)
            if j!=-1 and stock(key) is not None: break
        else: j=-1; jd=0
        require(round(f[7]*4)==jd and ptr==(j+jd+1 if j>=0 else -1),'delayed write')
        active=bool(data['on'][0,i]); err=0.
        if active:
            hp=stock(key)[:256]; pair=np.argsort(-hp.astype(np.int64),kind='stable')[:2]
            require(np.array_equal(pair,data['codes'][0,i]),'bound branch codes')
            require(abs(hp[pair[0]]/hp[pair].sum()-data['prior'][0,i])<1e-6,'parent prior')
            for b,idx in zip(pair,data['ci'][0,i],strict=True):
                hc=stock(key+bytes([b]))[:256]; err=max(err,float(np.abs(hc/hc.sum()-data['table'][idx]).max()))
            require(err<1e-6,'child row values')
        rows.append(dict(position=i,base_depth=d,live_depth=ld,join_depth=jd,addr=ptr,active=active,max_child_error=err))
    result=dict(document=r['sha256'],panel_sha=sha(OUT/'panels.json'),stock_sha=sha(STOCK),calls=stock.cache_info().misses,rows=rows)
    dump(OUT/'natural-parity.json',result); print(json.dumps(result))

if __name__=='__main__': verify()
