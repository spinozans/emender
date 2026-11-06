"""Sampled native branch answers versus independent full raw-region byte scans."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from . import c14_branch_serve as serving

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/p100_query_verification'
RAW=Path('/mnt/nvme2n1/erikg/pile.txt')
INDEX=Path('/mnt/nvme2n1/erikg/P100c/pile-head100.sxi2')
LIMIT=100_000_000_000
CHUNK=32<<20

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(p):
    st=p.stat();return dict(path=str(p),size=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns)
def write(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('register','verify'));a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.phase=='register':
        assert not (OUT/'registration.json').exists()
        patterns=[]
        with RAW.open('rb') as f:
            for offset in (1_000_003,17_000_000_019,51_000_000_023,99_999_000_029):
                f.seek(offset);bb=f.read(64);assert len(bb)==64
                for n in (8,16,32,64):patterns.append(dict(offset=offset,bytes=n,hex=bb[:n].hex()))
        patterns.append(dict(offset=None,bytes=64,hex=(b'\0'*64).hex()))
        write('registration.json',dict(status='registered_before_query_measurement',raw=identity(RAW),index=identity(INDEX),limit=LIMIT,patterns=patterns,binary=identity(serving.XSA),binary_sha256=sha(serving.XSA),runner_sha256=sha(Path(__file__))))
        return
    reg=json.loads((OUT/'registration.json').read_text())
    assert identity(RAW)==reg['raw'] and identity(INDEX)==reg['index']
    assert sha(serving.XSA)==reg['binary_sha256'] and sha(Path(__file__))==reg['runner_sha256']
    assert not (OUT/'verdict.json').exists(),'refuse overwrite terminal verification'
    patterns=[bytes.fromhex(x['hex']) for x in reg['patterns']]
    if not (OUT/'native.json').exists():
        serving.N=LIMIT+1
        server=serving.BranchServer(OUT/'server.stderr',jobs=4,index=INDEX)
        try:answers=server.query(patterns)
        finally:server.close()
        write('native.json',answers)
    answers=json.loads((OUT/'native.json').read_text())
    progress=OUT/'scan.json';cursor=0;counts=[[0]*256 for _ in patterns];eofs=[0]*len(patterns)
    if progress.exists():
        saved=json.loads(progress.read_text());cursor=saved['next_offset'];counts=saved['counts'];eofs=saved['eofs']
    # A resumed scan rehashes the prefix: no opaque serialized hash state.
    h=hashlib.sha256();started=time.monotonic()
    with RAW.open('rb') as f:
        while f.tell()<cursor:h.update(f.read(min(CHUNK,cursor-f.tell())))
        while cursor<LIMIT:
            n=min(CHUNK,LIMIT-cursor)
            f.seek(cursor);bb=f.read(min(n+max(map(len,patterns)),LIMIT-cursor));assert len(bb)>=n
            h.update(bb[:n])
            for i,pat in enumerate(patterns):
                pos=bb.find(pat)
                while 0<=pos<n:
                    end=pos+len(pat)
                    if cursor+end==LIMIT:eofs[i]+=1
                    else:counts[i][bb[end]]+=1
                    pos=bb.find(pat,pos+1)
            cursor+=n
            if cursor%(CHUNK*16)==0 or cursor==LIMIT:
                write('scan.json',dict(status='scanning' if cursor<LIMIT else 'scan_complete',next_offset=cursor,region_bytes_scanned=cursor,counts=counts,eofs=eofs,raw_prefix_sha256=h.hexdigest(),seconds_this_invocation=time.monotonic()-started))
                print(json.dumps(dict(offset=cursor,total=LIMIT,seconds=time.monotonic()-started)),flush=True)
    checks=[]
    for i,(answer,c,eof) in enumerate(zip(answers,counts,eofs)):
        got=dict(answer['branches']);expected={b:n for b,n in enumerate(c) if n}
        checks.append(dict(pattern=i,match=got==expected and answer['width']==sum(c)+eof and answer['eof']==eof,raw_branches=sorted(expected.items()),native=answer,raw_width=sum(c)+eof,raw_eof=eof))
    assert identity(RAW)==reg['raw'] and identity(INDEX)==reg['index']
    passed=all(c['match'] for c in checks)
    write('verdict.json',dict(status='passed' if passed else 'failed',region_bytes_scanned=cursor,raw_region_sha256=h.hexdigest(),checks=checks,registration_sha256=sha(OUT/'registration.json'),scope='17 sampled complete branch histograms; not exhaustive index certification'))
    assert passed,'native branch mismatch; heldout pipeline forbidden'

if __name__=='__main__':main()
