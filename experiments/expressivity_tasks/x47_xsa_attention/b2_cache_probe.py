import json
import struct
import subprocess
from time import perf_counter_ns as clock

from .b1_mount import Archive, XSA, EXE_SHA, digest


def run():
    assert digest(XSA.read_bytes())==EXE_SHA
    d=Archive('P100'); keys=d.patterns
    expected=[[int(d.starts[a.doc])+a.start for a,_ in row] for row in d.lookup(keys).values()]
    assert [len(r) for r in expected]==[2,1] and all(len(k)==32 for k in keys)
    procs={}
    try:
        for mib in (64,256):
            p=subprocess.Popen([XSA,'seeds',d.index,'--serve','-j','1','--min-len','32','--max-len','32','--stride','32','--max-seeds','1','--min-mult','1','--max-mult','8','--anchor-max-per-seed','8','--rss-gb','1','--cache-gb',str(mib/1024)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            procs[mib]=p
            while True:
                line=p.stderr.readline()
                if line.startswith(b'SEEDS_SERVE_READY'): break
                if not line: raise RuntimeError('seeds not ready')
            print(json.dumps(dict(mib=mib,ready=line.decode().strip())),flush=True)
        for rep in range(6):
            for i in ([0,1] if rep%2==0 else [1,0]):
                for mib in ([64,256] if rep%2==0 else [256,64]):
                    p=procs[mib]; t=clock(); p.stdin.write(struct.pack('<I',32)+keys[i]); p.stdin.flush()
                    n,=struct.unpack('<I',p.stdout.read(4)); assert 16<=n<=272
                    data=p.stdout.read(n); ns=clock()-t
                    _,num,seeds=struct.unpack_from('<QII',data); rows=list(struct.iter_unpack('<IIQQQ',data[16:]))
                    assert seeds==1 and num==len(rows) and all(off==0 and m==32 and mult==num for off,m,mult,_,_ in rows)
                    positions=sorted(r[3] for r in rows); assert positions==sorted(expected[i]); print(json.dumps(dict(rep=rep,mib=mib,key=(2,6)[i],ns=ns,positions=positions)),flush=True)
    finally:
        for mib,p in procs.items():
            p.stdin.close(); stats=p.stderr.read().decode().splitlines(); assert p.wait()==0
            print(json.dumps(dict(mib=mib,stats=next(s for s in stats if s.startswith('SEEDS_SERVE_STATS')))),flush=True)
            p.stdout.close(); p.stderr.close()

if __name__=='__main__': run()
