"""Leased-GPU profile of resident pages only, not SXGC or end-to-end decode."""
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F
import triton

from .bank_read import BoundPage, reference_read
from .bank_read_triton import triton_read


def sdpa(q,k,v,page,b):
    live = page.ids >= 0
    rows = page.ids.clamp_min(0).long()
    scores = b.masked_fill(~live, -torch.inf)[:,None,None,:]
    return F.scaled_dot_product_attention(q[:,None,None,:],k[rows][:,None],v[rows][:,None],
                                         attn_mask=scores,dropout_p=0.)[:,0,0].float()


def emit(**data):
    print(json.dumps(data),flush=True)


def sdpa_fp32(q,k,v,page,b):
    rows=page.ids.clamp_min(0).long()
    scores=b.masked_fill(page.ids < 0,-torch.inf)[:,None,None,:]
    return F.scaled_dot_product_attention(q[:,None,None,:].float(),k[rows][:,None].float(),v[rows][:,None].float(),attn_mask=scores,dropout_p=0.)[:,0,0]


def timing(fn,iters):
    torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
    first_wall_ms = (time.perf_counter()-t)*1e3
    for _ in range(5): fn()
    torch.cuda.synchronize(); a,b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    a.record(); b.record(); b.synchronize()
    wall,stream = [],[]
    for _ in range(iters):
        t = time.perf_counter(); a.record(); fn(); b.record(); b.synchronize()
        wall.append((time.perf_counter()-t)*1e6); stream.append(1e3*a.elapsed_time(b))
    def percentiles(xs):
        return [round(statistics.median(xs),2),round(sorted(xs)[(95*len(xs)+99)//100-1],2)]
    return dict(first_call_wall_ms=first_wall_ms,samples=iters,wall_us_p50_p95=percentiles(wall),stream_us_p50_p95=percentiles(stream))


def main():
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1') or os.environ.get('CUDA_VISIBLE_DEVICES') != os.environ['GPU_LEASE_HELD']:
        raise RuntimeError('one leased physical card 0/1 required')
    torch.manual_seed(3401); torch.backends.cuda.matmul.allow_tf32=False
    emit(device=torch.cuda.get_device_name(),cc=torch.cuda.get_device_capability(),torch=torch.__version__,triton=triton.__version__,lease=os.environ['GPU_LEASE_HELD'])
    # Workloads fixed before launch. Each row may represent a different bank.
    cases=[(1,8,64,128,'random'),(8,32,64,512,'random'),(128,32,64,4096,'random'),
           (2048,64,128,65536,'random'),(2048,64,128,65536,'shared'),(128,32,64,4096,'linear')]
    for r,f,d,n,layout in cases:
        ids=torch.randint(n,(r,f)); ids[::11,-1:]=-1
        if layout=='shared': ids=torch.randint(8,(r,f))
        if layout=='linear': ids=(torch.arange(r*f).reshape(r,f)%n)
        page=BoundPage.from_cpu(ids,n,device='cuda')
        for dtype in (torch.float32,torch.bfloat16):
            q,k,v=[torch.randn(z,device='cuda',dtype=dtype).requires_grad_() for z in [(r,d),(n,d),(n,d)]]
            bias=torch.randn(r,f,device='cuda',requires_grad=True); up=torch.randn(r,d,device='cuda')
            args=(q,k,v,page,bias)
            with torch.no_grad(): gold=reference_read(*args)
            comparators=[('triton',triton_read),('eager',reference_read),('gather_sdpa_fp32',sdpa_fp32)]
            if dtype==torch.bfloat16:
                comparators.append(('gather_sdpa_native',lambda q,k,v,p,b: sdpa(q,k,v,p,b.to(q.dtype))))
            for name,op in comparators:
                torch.cuda.synchronize(); started=time.perf_counter()
                with torch.no_grad(): actual=op(*args)
                torch.cuda.synchronize(); first_ms=(time.perf_counter()-started)*1e3
                err=(actual-gold).abs().max().item()
                tol=dict(rtol=1e-4,atol=3e-5) if dtype==torch.float32 else dict(rtol=.03,atol=.02)
                torch.testing.assert_close(actual,gold,**tol)
                emit(rows=r,fan=f,width=d,dtype=str(dtype),layout=layout,op=name,first_observed_fwd_ms=first_ms,max_abs=err)
                def fwd():
                    with torch.no_grad(): return op(*args)
                def pair(): return torch.autograd.grad(op(*args),(q,k,v,bias),grad_outputs=up)
                for mode,fn,iters in [('fwd',fwd,31),('fwd+bwd',pair,15)]:
                    emit(rows=r,fan=f,width=d,records=n,layout=layout,dtype=str(dtype),op=name,mode=mode,**timing(fn,iters))


if __name__=='__main__': main()
