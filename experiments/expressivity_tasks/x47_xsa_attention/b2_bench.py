"""Registered local B2 comparison. No queries supplied by argv."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from time import perf_counter_ns as clock

from .b1_mount import Archive, EXE_SHA, XSA, digest, HERE
from .b2_worker import PositionalWorker

EXE = Path.home()/'.cache/xsa-b2-target/release/xsa'


def original(desc, pats):
    t=clock()
    with tempfile.NamedTemporaryFile() as f:
        data=b''.join(p.hex().encode()+b'\n' for p in pats); f.write(data); f.flush()
        proc=subprocess.run(['/usr/bin/time','-f','B2_RSS %M',XSA,'search',desc.index,'--patterns',f.name,'--hex','--jsonl','--max-occ','8','--corpus',desc.text],capture_output=True,check=True,timeout=30)
    wall=clock()-t
    rows=[[] for p in pats]
    for line in proc.stdout.splitlines():
        h=json.loads(line); assert h['cross']==0 and h['len']==len(pats[h['id']]); rows[h['id']].append(h['pos'])
    assert b'refused=0' in proc.stderr and b'SEARCH_BOUNDARY mode=no-cross-default structure=member-m15' in proc.stderr
    return rows,dict(elapsed_ns=wall,rss_kib=int(re.search(rb'B2_RSS (\d+)',proc.stderr)[1]),tx=len(data),rx=len(proc.stdout),stderr_bytes=len(proc.stderr))


def run():
    assert digest(XSA.read_bytes())==EXE_SHA
    for name in ('F6','P100'):
        t=clock(); desc=Archive(name); validation_ns=clock()-t; keys=desc.patterns[:2]
        if name=='P100': assert len(set(keys))==2 and all(len(k)==32 for k in keys)
        expected={p:sorted(int(desc.starts[a.doc])+a.start for a,v in hits) for p,hits in desc.lookup(keys).items()}
        if name=='P100': assert [len(expected[p]) for p in keys]==[2,1]
        persistent=PositionalWorker(desc,EXE); calls=0
        try:
            for rep in range(5):
                order=[rep%2,1-rep%2]
                for ids in [[order[0]],[order[1]],order]:
                    pats=[keys[i] for i in ids]; modes=['original','oneshot','persistent']
                    for mode in (modes if rep%2==0 else reversed(modes)):
                        extra={}
                        if mode=='original': pos,record=original(desc,pats)
                        else:
                            t=clock(); w=persistent if mode=='persistent' else PositionalWorker(desc,EXE)
                            try:
                                spans,record=w.request(pats); response_ns=clock()-t
                                pos=[[int(desc.starts[a.doc])+a.start for a in row] for row in spans]
                                rss=int(re.search(r'VmHWM:\s+(\d+)',Path(f'/proc/{w.proc.pid}/status').read_text())[1]); extra=dict(ready_bytes=w.ready_bytes,start_ns=w.start_ns,ready=w.ready,response_ns=response_ns,rss_kib=rss)
                            finally:
                                if mode=='oneshot': w.close()
                            extra["inclusive_ns"]=clock()-t
                            if mode=='persistent': calls+=1; extra['resident_call']=calls
                        assert [sorted(r) for r in pos]==[expected[p] for p in pats]
                        print(json.dumps(dict(bank=name,version=desc.version,validation_ns=validation_ns,rep=rep,mode=mode,ids=[(2,6)[i] if name=='P100' else i for i in ids],positions=pos,pattern_hashes=[hashlib.sha256(p).hexdigest() for p in pats],**record,**extra)),flush=True)
        finally: persistent.close()

if __name__=='__main__': run()
