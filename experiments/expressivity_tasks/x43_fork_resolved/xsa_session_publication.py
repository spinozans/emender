"""CPU append/query latency while replacing a sealed generation (depth 8)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from xsa_session_latency import stats, timed


def run(exe, data, n):
    with subprocess.Popen([exe, 'session', '8'], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        def rpc(s):
            proc.stdin.write(s.encode()+b'\n'); proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f'server stopped on {s[:16]}')
            return json.loads(line)
        for i in range(0,n,65536):
            rpc('append '+data[i:min(n,i+65536)].hex())
        rpc('seal'); rpc('branch '+data[n-8:n].hex())
        # Old base exists. Rebuild at a later cutoff with a nonempty tail.
        rpc('append '+data[n:n+512].hex()); cut=n+512
        started=time.perf_counter()
        submitted, submit_ms=timed(lambda: rpc('seal_async'))
        if submitted != {'submitted':cut}:
            raise ValueError(submitted)
        samples=[]; status={'pending':True}; i=cut
        while status['pending']:
            if time.perf_counter()-started>60 or i>=len(data):
                raise RuntimeError('publication did not complete within benchmark budget')
            def step():
                rpc('append '+data[i:i+1].hex()); return rpc('branch '+data[i-7:i+1].hex())
            row, ms=timed(step); samples.append(ms); i+=1
            if len(samples)%16==0:
                status=rpc('status')
        wall_ms=(time.perf_counter()-started)*1000
        if status!={'pending':False,'cut':cut,'visible':i}:
            raise ValueError(status)
        p=data[i-8:i]; counts=[0]*256; j=data.find(p,0,i)
        while j>=0 and j+8<i:
            counts[data[j+8]]+=1; j=data.find(p,j+1,i)
        if row!={'counts':counts,'end':0}:
            raise ValueError('post-publication answer mismatch')
        info=Path(f'/proc/{proc.pid}/status').read_text()
        peak=int(next(x for x in info.splitlines() if x.startswith('VmHWM:')).split()[1]);proc.stdin.close()
        if proc.wait(timeout=10): raise RuntimeError('unclean exit')
    return dict(prefix=n,cut=cut,visible=i,observations=len(samples),submit_ms=submit_ms,
                publish_observed_ms=wall_ms,append_query_ms=stats(samples),peak_rss_kib=peak)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('exe','document','output'): ap.add_argument(name,type=Path)
    a=ap.parse_args();data=a.document.read_bytes()
    if len(data)<1_100_000: ap.error('single document too short')
    rows=[run(a.exe.resolve(),data,n) for n in (100000,500000,1000000)]
    result=dict(document_sha256=hashlib.sha256(data).hexdigest(),binary_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),rows=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows),flush=True)
