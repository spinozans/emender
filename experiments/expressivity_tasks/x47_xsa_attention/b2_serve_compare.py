"""Small positive-only comparison with the already-framed native anchor server."""
import json
import struct
import subprocess
from time import perf_counter_ns as clock

from .b1_mount import Archive, XSA, EXE_SHA, digest
from .b2_bench import EXE
from .b2_worker import PositionalWorker


def run():
    assert digest(XSA.read_bytes())==EXE_SHA
    for name in ('F6','P100'):
        desc=Archive(name); keys=desc.patterns[:2]; depth=len(keys[0]); assert len(keys[1])==depth
        expected=[[int(desc.starts[a.doc])+a.start for a,_ in hits] for hits in desc.lookup(keys).values()]
        for mode in ('seeds','positional'):
            if mode=='seeds':
                t=clock(); proc=subprocess.Popen([XSA,'seeds',desc.index,'--serve','-j','1','--min-len',str(depth),'--max-len',str(depth),'--stride',str(depth),'--min-mult','1','--max-mult','8','--max-seeds','1','--anchor-max-per-seed','8'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                while True:
                    line=proc.stderr.readline()
                    if b'SEEDS_SERVE_READY' in line: break
                    if not line: raise RuntimeError('seeds exited before ready')
                start_ns=clock()-t
            else:
                w=PositionalWorker(desc,EXE); start_ns=w.start_ns
            try:
                plan=[('first',[i]) for i in (0,1)]+[('same',[i]) for i in (0,1) for _ in range(5)]+[('alternate',[i]) for _ in range(5) for i in (0,1)]
                if mode=='positional': plan += [('dedup',[0,0,1,1])]*5
                for seq,(phase,ids) in enumerate(plan):
                    t=clock(); unique=list(dict.fromkeys(ids)); pats=[keys[i] for i in unique]
                    if mode=='seeds':
                        p,=pats; frame=struct.pack('<I',depth)+p; proc.stdin.write(frame); proc.stdin.flush()
                        n,=struct.unpack('<I',proc.stdout.read(4)); assert n<=16+8*32
                        data=proc.stdout.read(n); wid,num,seeds=struct.unpack_from('<QII',data); assert wid==seq and len(data)==16+32*num and seeds==1
                        rows=list(struct.iter_unpack('<IIQQQ',data[16:])); assert all(off==0 and m==depth and mult==num<=8 and doc==int(desc.starts.searchsorted(pos,'right'))-1 for off,m,mult,pos,doc in rows)
                        positions=[[row[3] for row in rows]]; tx,rx=len(frame),4+n
                    else:
                        spans,r=w.request(pats); positions=[[int(desc.starts[a.doc])+a.start for a in row] for row in spans]; tx,rx=r['tx'],r['rx']
                    by_id=dict(zip(unique,positions,strict=True)); expanded=[by_id[i] for i in ids]; wall=clock()-t
                    assert [sorted(row) for row in expanded]==[sorted(expected[i]) for i in ids]
                    print(json.dumps(dict(bank=name,mode=mode,phase=phase,seq=seq,start_ns=start_ns,ids=[(2,6)[i] if name=='P100' else i for i in ids],distinct=len(unique),wall_ns=wall,tx=tx,rx=rx,positions=expanded)),flush=True)
            finally:
                if mode=='positional': w.close()
                else:
                    proc.stdin.close(); proc.stderr.read(); assert proc.wait()==0; proc.stderr.close(); proc.stdout.close()

if __name__=='__main__': run()
