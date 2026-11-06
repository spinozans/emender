"""Round8 three-kernel probe; control separate from score/normalize/READ."""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from .c14_phase5r3_split_kernel import control, normalize, initialize, as_rows, as_cols, fixed_sum
from .c14_phase5r3_split_kernel import read_page

@gluon.jit
def pre_read(K,Q,V,A,Pop,Append,Logit,Coef,Match,Error,Own,Pred,Expiry,Keep,Ids,Valid,
           Out,FinalK,FinalV,FinalS,FinalMeta,Trace,
           Score,Product,Scalars,Carry,t,
           T:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,
           Decay:tl.constexpr,Free:tl.constexpr,PopThreshold:tl.constexpr,
           Enable:tl.constexpr,Eps:tl.constexpr,MatchWire:tl.constexpr,
           OccSlope:tl.constexpr,OccOnset:tl.constexpr,AppendOnly:tl.constexpr):
    b=tl.program_id(0)
    tl.thread_barrier()
    if t < tl.load(Valid+b):
        kd=tl.arange(0,64,layout=tl.BlockedLayout([1],[32],[8],[0]))
        q=tl.load(Q+(b*T+t)*64+kd)
        for page in range(tl.cdiv(BS,128)):
            rs=page*128+tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
            keys=tl.load(FinalK+(b*S+as_rows(rs))*64+as_cols(kd),as_rows(rs)<S,0.)
            score=fixed_sum(keys*as_cols(q),1)*(2./tl.sqrt(64.))
            tl.store(Score+b*BS+rs,score,rs<BS)
    tl.thread_barrier()
    normalize(Score,FinalS,FinalMeta,Product,Scalars,Valid,t,S=S,BS=BS)
    tl.thread_barrier()
    for page in range(tl.cdiv(BS,128)):
        read_page(FinalV,Product,Carry,Out,Valid,t,page,T=T,S=S,BS=BS,PAGE=128)
        tl.thread_barrier()

class FusedCycle:
    compiled=None
    read_compiled=None
    page_size=128
    def __getitem__(self,grid):
        def launch(*a,**kw):
            import torch
            b=grid[0];s=kw['S'];bs=kw['BS']
            score=torch.empty((b,bs),device=a[0].device)
            product=torch.empty_like(score)
            scalars=torch.empty((b,8),device=a[0].device,dtype=torch.int32)
            carry=torch.empty((b,2,128),device=a[0].device)
            initialize[(b,triton.cdiv(s,128))](*a[17:21],S=s,num_warps=8,enable_fp_fusion=False)
            for t in range(kw["T"]):
                self.compiled=control[grid](*a,score,product,scalars,t,Phase=0,**kw)
                self.read_compiled=pre_read[grid](*a,score,product,scalars,carry,t,**kw)
                self.compiled=control[grid](*a,score,product,scalars,t,Phase=1,**kw)
            return self.compiled
        return launch
kernel=FusedCycle()
