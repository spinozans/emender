"""W3: verified document windows and batched fixed-policy P100 branch working sets."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

import numpy as np
from .b1_mount import ArchiveDescriptor, HERE, ROOT as NVME
from .workbench_run import sha, dump
from .workbench_witness import require

OUT=Path('output/w3'); EXE=Path.home()/'.cache/xsa-w3-target/release/xsa'
STOCK=Path.home()/'.cache/xsa-b2-target/release/xsa'
FLAG=HERE.parent/'x43_fork_resolved/receipts/flagship_stream'
MANIFEST=FLAG/'documents.jsonl.gz'; MAN_SHA='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'
T=2048; SHARD=32; N={'train':2048,'dev':128,'confirm':128}
ROW=np.dtype([('h','<u8',(257,)),('child','<u4',(256,))])


def prior_hashes(obj):
    if isinstance(obj,dict):
        if 'sha256' in obj and isinstance(obj['sha256'],str): yield obj['sha256']
        for v in obj.values(): yield from prior_hashes(v)
    elif isinstance(obj,list):
        for v in obj: yield from prior_hashes(v)


def select():
    reg=json.loads((FLAG/'registration.json').read_text()); src=reg['source']; st=Path(src['path']).stat()
    require((st.st_ino,st.st_size,st.st_mtime_ns)==tuple(src[k] for k in ('inode','size','mtime_ns')),'source identity')
    require(sha(MANIFEST)==reg['manifest_sha256']==MAN_SHA,'document index')
    old=[NVME/s/'panels.json' for s in ('k15_coupled_I','k15_relation_J','k15_counterfactual_K')]
    old += [Path('output')/s/'panels.json' for s in ('k15L-pilot','k15M','k15N','k15O','k15P2')]
    old += [NVME/'xsa_budget_cap8192_v2'/s/'manifest.json' for s in ('2048','8192')]
    seen=set(); old_ids={}
    for p in old:
        excluded=set(prior_hashes(json.loads(p.read_text()))); require(excluded,'unrecognized exclusion manifest')
        old_ids[str(p)]=sha(p); seen.update(excluded)
    split={'train':'train','calibration':'dev','final_test':'confirm'}; eligible={s:[] for s in N}
    with gzip.open(MANIFEST,'rt') as f:
        for line in f:
            d=json.loads(line); off,n,h=d['byte_offset'],d['byte_length'],d['sha256']
            if off<100_600_000_000 or n<T+1 or h in seen: continue
            seen.add(h); label=split[d['split']]; start=int(h[16:32],16)%(n-T)
            rank=hashlib.sha256(('W3-natural-v1:'+h).encode()).hexdigest()
            eligible[label].append((rank,dict(doc=d.get('document',d.get('source_index')),offset=off,length=n,sha256=h,start=start)))
    panels={s:[d for _,d in sorted(l)[:N[s]]] for s,l in eligible.items()}
    require(all(len(v)==N[s] for s,v in panels.items()),'not enough eligible documents')
    require(len({d['doc'] for l in panels.values() for d in l})==sum(N.values()),'duplicate document ID')
    return dict(source=src,manifest_sha=MAN_SHA,registration_sha=sha(FLAG/'registration.json'),exclusions=old_ids,
                eligible={s:len(l) for s,l in eligible.items()},panels=panels)


def getwindow(file,d):
    file.seek(d['offset']); raw=file.read(d['length']); j=d['start']
    require(len(raw)==d['length'] and hashlib.sha256(raw).hexdigest()==d['sha256'],'full raw hash mismatch')
    require(0<=j and j+T+1<=len(raw),'cross-document window')
    return raw[j:j+T+1]


class Pages:
    def __init__(self, *, pinned_fds=None):
        start = time.perf_counter_ns()
        self.exe_sha=sha(EXE); require((OUT/'binary.sha').read_text().split()[0]==self.exe_sha,'native build mismatch')
        self.desc=ArchiveDescriptor('P100',json.loads((HERE/'b1_mounts.json').read_text())['P100'],NVME)
        self.attest_ns = time.perf_counter_ns() - start
        self._start_workspace(pinned_fds)

    def _start_workspace(self, pinned_fds):
        # Both openers reach the same real native launch and protocol check. A retained
        # descriptor never substitutes an old PID, header, root, or response pipe.
        start = time.perf_counter_ns()
        exe, index = EXE, self.desc.index
        inherited = ()
        if pinned_fds is not None:
            exe_fd, index_fd = inherited = tuple(pinned_fds)
            require(all(os.path.samestat(os.fstat(fd), path.stat()) for fd, path in zip(inherited, (exe, index), strict=True)), 'changed pinned workspace file')
            exe, index = f'/proc/self/fd/{exe_fd}', f'/proc/self/fd/{index_fd}'
        self.proc=subprocess.Popen([exe,'branch-of',index,'--workspace'],pass_fds=inherited,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        try:
            # Reject malformed external input as an admission error. Do not expose the
            # historical benchmark assertion type or publish a partially opened worker.
            head = self.proc.stdout.read(16)
            if len(head) != 16 or head[:8] != b'W3BP0001':
                raise ValueError('invalid m20 workspace header')
            root, nodes = struct.unpack_from('<II', head, 8)
            if not 0 < root <= nodes == 21707038:
                raise ValueError('wrong m20 workspace tree')
            self.root, self.nodes = root, nodes
            self.workspace_ns = time.perf_counter_ns() - start
        except BaseException:
            self.close(); raise

    def recv(self,n):
        data=self.proc.stdout.read(n); require(len(data)==n,'incomplete page frame'); return data

    def close(self):
        try: self.proc.stdin.close(); self.proc.wait(timeout=3)
        except (BrokenPipeError,subprocess.TimeoutExpired): self.proc.kill(); self.proc.wait()
        finally: self.proc.stdout.close()

    def rpc(self,op,n,payload,size):
        require(1<=n<=4096,'frame limit'); self.proc.stdin.write(op+struct.pack('<I',n)+payload); self.proc.stdin.flush()
        return self.recv(size)

    def suffix(self,raw):
        data=self.rpc(b'S',len(raw),raw,len(raw)*32); return np.frombuffer(data,'<u4').reshape(-1,8)

    def rows(self,ids):
        ids=np.asarray(ids,dtype='<u4'); result=np.zeros(len(ids),ROW); ix=np.flatnonzero(ids)
        for batch in np.array_split(ix,max(1,(len(ix)+4095)//4096)):
            if len(batch): result[batch]=np.frombuffer(self.rpc(b'R',len(batch),ids[batch].tobytes(),len(batch)*ROW.itemsize),ROW)
        return result

    def qualify(self):
        require(sha(STOCK)=='fbc436aac93190368bbed637a5f69cca0fa5848fc81a311e60c4ebe1b4d3bf50','stock binary')
        checks=[]
        for k in (b'the ',b'\x00\x1e',b'\xff'*8,b' the a'):
            out=subprocess.run([STOCK,'branch-of',self.desc.index,'hex:'+k.hex()],capture_output=True,text=True,timeout=30,check=True).stdout
            node=self.suffix(k)[-1,len(k)-1]; present=out.startswith('present '); require(bool(node)==present,'stock coverage')
            h=np.zeros(257,np.uint64)
            for label,c in re.findall(r'^child (END|0x[0-9a-f]{2}) count=(\d+) node=',out,re.M):
                h[256 if label=='END' else int(label,16)]=int(c)
            require(np.array_equal(self.rows([node])['h'][0],h),'stock histogram')
            if present: require(int(re.search(r'present count=(\d+)',out)[1])==int(h.sum()),'incomplete stock oracle')
            checks.append(dict(key_hex=k.hex(),present=present,support=int(h.sum()),hist_sha=hashlib.sha256(h.tobytes()).hexdigest()))
        return checks


def live_statistics(raw):
    """At i first register the transition ending AT i, then query suffix ending at i.
    Only the likelihood column sees y, never support/peak/depth or model features."""
    t=len(raw)-1; out=np.zeros((t,4),np.float32)
    for d in (2,4,8):
        hist={}
        for i in range(d-1,t):
            if i>=d:
                prev=raw[i-d:i]; b=raw[i]; entry=hist.get(prev)
                if entry is None: hist[prev]=[1,1,{b:1}]
                else:
                    entry[0]+=1; cnt=entry[2]; cnt[b]=cnt.get(b,0)+1; entry[1]=max(entry[1],cnt[b])
            row=hist.get(raw[i+1-d:i+1])
            if row: out[i]=(d,row[0],row[1]/row[0],row[2].get(raw[i+1],0)/row[0])
    return out


def addresses(ids):
    n,t,_=ids.shape; past=np.full((n,t),-1,np.int16); locus=np.zeros((n,t),np.uint32); depth=np.zeros((n,t),np.uint8)
    for b in range(n):
        # A materialized node identifies one literal string on this bound mount.
        for k in (2,4):
            seq=ids[b,:,k-1].tolist(); recent={}
            for i in range(18,t):
                j=i-18
                if seq[j]: recent[seq[j]]=j
                p=recent.get(seq[i]);
                if p is not None: past[b,i]=p+2; locus[b,i]=seq[i]; depth[b,i]=k
    return past,locus,depth


def compile_shard(svc,windows):
    raw=np.stack([np.frombuffer(w,np.uint8) for w in windows]); x,y=raw[:,:-1],raw[:,1:]
    ids=np.stack([svc.suffix(w[:-1]) for w in windows]); ptr,locus,jdepth=addresses(ids)
    base=np.full_like(locus,svc.root); depth=np.zeros_like(jdepth)
    for d in (1,2,4,6,8):
        col=ids[:,:,d-1]; pos=col!=0; base[pos]=col[pos]; depth[pos]=d
    uniq=np.union1d(base,locus); pages=svc.rows(uniq); counts=pages['h'][:,:256]; sums=counts.sum(-1)
    ix=np.searchsorted(uniq,base); denom=np.maximum(sums[ix],1)
    archive_y=counts[ix,y]/denom; archive_peak=counts.max(-1)[ix]/denom
    # Rank the pair only once per distinct historical locus, not per occurrence.
    keys,inv=np.unique(locus,return_inverse=True); rows=np.searchsorted(uniq,keys); c=counts[rows].copy(); ar=np.arange(len(c))
    a=c.argmax(1); ca=c[ar,a].copy(); c[ar,a]=0; b=c.argmax(1); cb=c[ar,b]; pair=np.stack((a,b),-1)
    nodes=pages['child'][rows[:,None],pair]; nodes[(cb==0)|(keys==0)]=0
    cu=np.union1d(nodes,[0]).astype(np.uint32); child=svc.rows(cu)['h'][:,:256]; total=child.sum(-1)
    dist=(child/np.maximum(total,1)[:,None]).astype(np.float32); ci=np.searchsorted(cu,nodes)
    on=(nodes!=0).all(1)&(total[ci]>0).all(1); inv=inv.reshape(x.shape)
    prior=ca/np.maximum(ca+cb,1); coverage=(ca+cb)/np.maximum(sums[rows],1)
    live=np.stack([live_statistics(w) for w in windows]); lag=np.where(ptr>=0,np.arange(x.shape[1])[None,:]-ptr,0)
    feat=np.stack((depth/8,np.log1p(sums[ix])/24,archive_peak,live[:,:,0]/8,np.log1p(live[:,:,1])/8,live[:,:,2],
                   np.log1p(lag)/8,jdepth/4,coverage[inv]),-1).astype(np.float32)
    return dict(raw=raw,ptr=ptr,codes=pair[inv].astype(np.uint8),prior=prior[inv].astype(np.float32),
                ci=ci[inv].astype(np.int32),table=dist,on=on[inv],feat=feat,
                # Labels / target projections separated from the decision inputs.
                target_mass=np.stack((archive_y,live[:,:,3]),-1).astype(np.float32))


def prepare(stages):
    tic=time.monotonic(); OUT.mkdir(exist_ok=True); panel=select(); path=OUT/'panels.json'
    if path.exists(): require(json.loads(path.read_text())==panel,'changed selection')
    else: dump(path,panel)
    svc=Pages()
    try:
        stock=svc.qualify()
        for stage in stages:
            if stage=='confirm': require(json.loads((OUT/'dev.json').read_text())['pass_gate'],'confirmation closed')
            work=OUT/stage; work.mkdir(exist_ok=False); docs=panel['panels'][stage]; rows=[]; t0=time.monotonic()
            with Path(panel['source']['path']).open('rb') as f:
                for j in range(0,len(docs),SHARD):
                    ws=[getwindow(f,d) for d in docs[j:j+SHARD]]; part=compile_shard(svc,ws)
                    p=work/f'{j//SHARD:03}.npz'; np.savez(p,**part)
                    rows.append(dict(file=p.name,sha256=sha(p),size=p.stat().st_size,targets=part['on'].size,
                                     addressed=int((part['ptr']>=0).sum()),joins=int(part['on'].sum())))
                    if (j//SHARD)%8==0: print(f'{stage} {j+len(ws)}/{len(docs)} docs',flush=True)
            receipt=dict(stage=stage,panel_sha=sha(path),native_ref=(OUT/'native-ref').read_text().strip(),exe_sha=svc.exe_sha,
                         mount=svc.desc.version,stock_parity=stock,shards=rows,seconds=time.monotonic()-t0)
            dump(work/'receipt.json',receipt); print(stage,'complete',time.monotonic()-tic,flush=True)
    finally: svc.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['prepare','confirm']); args=ap.parse_args()
    prepare(['train','dev'] if args.mode=='prepare' else ['confirm'])
