"""Hash-disjoint by-offset heldout windows, gated on independent index checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from . import c14_branch_serve as serving
from .p100_verify import OUT as VERIFY, RAW, INDEX, LIMIT, identity
from .c14_feature_extract import digest
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/p100_heldout'
DATA=Path('/home/erikg/c14_p100_heldout')

def write(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def gate():
    v=json.loads((VERIFY/'verdict.json').read_text());assert v['status']=='passed' and v['region_bytes_scanned']==LIMIT
    r=json.loads((VERIFY/'registration.json').read_text());assert identity(RAW)==r['raw'] and identity(INDEX)==r['index']
    assert digest(serving.XSA)==r['binary_sha256']
    return digest(VERIFY/'verdict.json')

def select():
    bound=gate();assert not (OUT/'selection.json').exists()
    audit=ROOT/'receipts/c14_registered_scale/stream_audit.json';a=json.loads(audit.read_text())
    offsets=Path('/home/erikg/c14_dense_bytes/pile_offsets.jsonl');assert digest(offsets)==a['offsets_sha256']
    candidates=[];position=LIMIT
    with RAW.open('rb') as f:
        f.seek(position);buf=f.read(1<<20);cut=buf.find(b'\x1e');assert cut>=0
        position+=cut+1;f.seek(position)
        # Candidate documents are selected by bytes first. Hash exclusions follow.
        while len(candidates)<1024:
            f.seek(position);parts=[];length=0
            while True:
                bb=f.read(1<<20);assert bb
                at=bb.find(b'\x1e')
                if at>=0:parts.append(bb[:at]);length+=at;break
                parts.append(bb);length+=len(bb)
                assert length<64<<20,'unexpected giant document: fail closed'
            raw=b''.join(parts)
            if len(raw)>1:candidates.append(dict(byte_offset=position,byte_length=length,sha256=hashlib.sha256(raw).hexdigest()))
            position+=length+1
    wanted={d['sha256'] for d in candidates};excluded=set();seen=0
    with offsets.open() as f:
        for line in f:
            d=json.loads(line);seen+=1
            if d['sha256'] in wanted:excluded.add(d['sha256'])
    docs=[];windows=[];unique=set()
    for d in candidates:
        h=d['sha256']
        if h in excluded or h in unique:continue
        unique.add(h);doc=len(docs);docs.append(dict(d,document=doc,split='final_test'))
        for start in range(0,min(d['byte_length']-1,64*128),128):
            n=min(128,d['byte_length']-1-start)
            windows.append(dict(document=doc,start=start,targets=n))
            if len(windows)==4096:break
        if len(windows)==4096:break
    assert len(windows)==4096 and len(docs)>=64
    assert all(d['byte_offset']>=LIMIT for d in docs)
    x=np.zeros((len(windows),128),np.uint8);y=np.zeros_like(x);lengths=np.zeros(len(windows),np.int32)
    with RAW.open('rb') as f:
        for d in docs:
            f.seek(d['byte_offset']);bb=f.read(d['byte_length']);assert hashlib.sha256(bb).hexdigest()==d['sha256']
            for i,w in enumerate(windows):
                if w['document']!=d['document']:continue
                n=w['targets'];s=w['start'];assert s+n<d['byte_length'];x[i,:n]=np.frombuffer(bb[s:s+n],np.uint8);y[i,:n]=np.frombuffer(bb[s+1:s+n+1],np.uint8);lengths[i]=n
    np.savez(DATA/'windows.npz',x=x,y=y,lengths=lengths)
    write('selection.json',dict(status='passed',verification_sha256=bound,stream_audit_sha256=digest(audit),offsets_sha256=digest(offsets),frozen_documents_checked=seen,candidate_documents=len(candidates),hash_overlap_exclusions=len(excluded),documents=docs,windows=windows,targets=int(lengths.sum()),windows_file=str(DATA/'windows.npz'),windows_sha256=digest(DATA/'windows.npz'),corpus_materialized=False,scope='full-document hashes; at most64 windows per selected document; all final_test; no calibration/refitting'))

def extract():
    gate();s=json.loads((OUT/'selection.json').read_text());assert digest(DATA/'windows.npz')==s['windows_sha256']
    data=np.load(DATA/'windows.npz');x=data['x'];lengths=data['lengths'];n=len(x)
    assert not (OUT/'features.json').exists()
    ff=np.lib.format.open_memmap(DATA/'features.npy',mode='w+',dtype=np.float32,shape=(n,128,260));ff[:]=0
    kk=np.zeros((n,128),np.int32)
    serving.N=LIMIT+1
    server=serving.BranchServer(OUT/'extract_server.stderr',jobs=8,index=INDEX)
    support_stats={};calls=0
    try:
        for lo in range(0,n,16):
            lefts=[bytes(x[j,:i+1]) for j in range(lo,min(lo+16,n)) for i in range(int(lengths[j]))]
            rows,rates,queries=serving.select(lefts,server.query);at=0
            for j in range(lo,min(lo+16,n)):
                for i in range(int(lengths[j])):
                    row=rows[at]
                    terms=[(c/row['support_size'])*math.log(c/row['support_size']) for b,c in row['branches'] if b!=30] if row['support_size'] else []
                    entropy=correction=0.
                    for term in terms:
                        combined=entropy+term
                        correction+=(entropy-combined)+term if abs(entropy)>=abs(term) else (term-combined)+entropy
                        entropy=combined
                    corrected=entropy+correction if correction and math.isfinite(correction) else entropy
                    assert np.float32(-entropy).tobytes()==np.float32(-corrected).tobytes(),'entropy identity gate'
                    row['entropy']=-corrected
                    ff[j,i]=serving.features(row);kk[j,i]=row['context_bytes'];at+=1
            for key,v in rates.items():
                q=support_stats.setdefault(key,dict(eligible=0,supported=0))
                for field in q:q[field]+=v[field]
            calls+=queries;ff.flush();np.save(DATA/'contexts.npy',kk)
            write('extraction_progress.json',dict(status='running',next_window=min(lo+16,n),queries=calls,support=support_stats))
            print('features',min(lo+16,n),'/',n,flush=True)
    finally:server.close()
    write('features.json',dict(status='complete',windows=n,targets=int(lengths.sum()),features=str(DATA/'features.npy'),features_sha256=digest(DATA/'features.npy'),contexts=str(DATA/'contexts.npy'),contexts_sha256=digest(DATA/'contexts.npy'),selection_sha256=digest(OUT/'selection.json'),support=support_stats,queries=calls,geometry='unchanged registered260 floats; byte30 excluded as terminal; document-window resets; context includes current input byte only'))

def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('select','extract'));a=p.parse_args();OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    {'select':select,'extract':extract}[a.phase]()

if __name__=='__main__':main()
