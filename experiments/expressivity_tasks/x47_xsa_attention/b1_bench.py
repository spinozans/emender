"""Mounted request-to-residual benchmark. Only via b1_leased.sh."""
import json
import time
import resource
import torch
from .b1_lease import approved
from .b1_mount import Registry
from .b1_head import pack,transfer,XSAResidual
from .bench_read import sdpa_fp32,timing
from .bank_read import reference_read
from .bank_read_triton import triton_read

clock=time.perf_counter

def emit(**kw): print(json.dumps(kw),flush=True)


@torch.no_grad()
def main():
    if not approved(): raise RuntimeError('approved ancestor lease required')
    torch.manual_seed(3401); r=Registry({'F6','P100','context'}); head=XSAResidual().cuda(); pats={}
    try:
        emit(device=torch.cuda.get_device_name(),torch=torch.__version__)
        for n in ('context','F6','P100'):
            t=clock(); m=r.open(n)
            if n=='context':
                m.observe(b'ABA',[(1,)*8,(2,)*8,(3,)*8]); pats[n]=(b'A',b'B')
            else: pats[n]=m.patterns[:2]
            emit(mount=n,host_open_ms=(clock()-t)*1000)
        for count in (1,8):
            state=torch.ones(count,8,device='cuda'); weights=[.5,1.,2.]; ev,ev2=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for rep in range(3):
                cpu=[]; phases={}; gold=[]; torch.cuda.synchronize(); start=clock()
                for n in pats:
                    m=r.mounts[n]; t=clock(); requests=[[pats[n][i%2],pats[n][(i+int(i%3==0))%2]] for i in range(count)]
                    rows=r.resolve(n,requests); u=clock(); cpu.append(pack(rows,m,n)); v=clock()
                    phases[n]=dict(resolve_ms=(u-t)*1000,cli_inclusive_ms=getattr(m,'cli_ms',None),native_inner_ms=getattr(m,'search_ms',None),pack_ms=(v-u)*1000); gold.append(rows)
                t=clock(); banks=[transfer(b,'cuda') for b in cpu]; torch.cuda.synchronize(); u=clock(); ev.record(); y=head(state,list(zip(weights,banks))); ev2.record(); v=clock(); ev2.synchronize(); end=clock()
                expected=[]
                for rows in gold: expected.append(torch.stack([torch.tensor([x for _,x in row],dtype=torch.float32).mean(0) for row in rows]))
                torch.testing.assert_close(y.cpu(),1+sum(s*g for s,g in zip(weights,expected)),rtol=1e-4,atol=3e-5)
                emit(rows=count,rep=rep,phases=phases,h2d_sync_ms=(u-t)*1000,head_submit_ms=(v-u)*1000,head_stream_ms=ev.elapsed_time(ev2),sync_ms=(end-v)*1000,total_ms=(end-start)*1000,transfer_bytes=sum(t.numel()*t.element_size() for k,v,p,b in (z.checked() for z in cpu) for t in (k,v,p.ids,b)),gpu_peak_alloc=torch.cuda.max_memory_allocated(),host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            for op in [reference_read,sdpa_fp32,triton_read]:
                fn=lambda:head(state,list(zip(weights,banks)),read=op)
                torch.testing.assert_close(fn(),y,rtol=1e-4,atol=3e-5); emit(rows=count,resident_head=op.__name__,**timing(fn,10))
            for ids in [(0,),(1,),(2,),(0,2),(2,1,0)]:
                torch.testing.assert_close(head(state,[(weights[i],banks[i]) for i in ids]).cpu(),1+sum(weights[i]*expected[i] for i in ids),rtol=1e-4,atol=3e-5)
            # Read controls a later *native request*. Decode only on observed live keys.
            if count==1:
                for scale in (0.,1.):
                    t=clock(); state2=head(state,[(scale,banks[0])]); key_id=int((state2[0,0]>2).item()); key=pats['context'][key_id]
                    rows=r.resolve('context',[[key]]); bank=transfer(pack(rows,r.mounts['context'],'context'),'cuda'); final=head(state2,[(1.,bank)])
                    torch.cuda.synchronize(); elapsed=1000*(clock()-t); assert key_id==int(scale)
                    torch.testing.assert_close(final.cpu(),state2.cpu()+2,rtol=1e-4,atol=3e-5); emit(feedback_scale=scale,selected_key=key_id,two_read_wall_ms=elapsed)
    finally: r.close()

if __name__=='__main__': main()
