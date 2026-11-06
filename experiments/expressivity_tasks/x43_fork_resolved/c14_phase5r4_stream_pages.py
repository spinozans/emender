"""Round4 opt-in synthesis: bounded streaming pairs within tunable slot pages.

JIT arithmetic is shared with R3; only READ launch boundaries change. Persistent
FP32 parity carries preserve the full reduction chain across page boundaries.
"""
import triton
from .c14_phase5r3_split_kernel import initialize, control, score_pages, normalize, read_page


class StreamingPages:
    """Diagnostic only. Dedicated scratch per call; no production installer."""
    compiled=None
    read_compiled=None
    page_size=1024
    def __init__(self, page_size=1024):
        assert page_size in (256,512,1024)
        self.page_size=page_size
    def __getitem__(self,grid):
        def launch(*a,**kw):
            import torch
            b=grid[0];s=kw['S'];bs=kw['BS'];tmax=kw['T']
            score=torch.empty((b,bs),device=a[0].device)
            product=torch.empty_like(score)
            scalars=torch.empty((b,8),device=a[0].device,dtype=torch.int32)
            carry=torch.empty((b,2,128),device=a[0].device)
            opts=dict(num_warps=8,enable_fp_fusion=False)
            initialize[(b,triton.cdiv(s,128))](*a[17:21],S=s,**opts)
            for t in range(tmax):
                self.compiled=control[grid](*a,score,product,scalars,t,Phase=0,**kw)
                score_pages[(b,triton.cdiv(bs,128))](a[1],a[17],score,a[15],t,T=tmax,S=s,BS=bs,**opts)
                normalize[grid](score,a[19],a[20],product,scalars,a[15],t,S=s,BS=bs,**opts)
                for page in range(triton.cdiv(bs,self.page_size)):
                    self.read_compiled=read_page[grid](a[18],product,carry,a[16],a[15],t,page,T=tmax,S=s,BS=bs,PAGE=self.page_size,**opts)
                self.compiled=control[grid](*a,score,product,scalars,t,Phase=1,**kw)
            return self.compiled
        return launch

kernel=StreamingPages()
