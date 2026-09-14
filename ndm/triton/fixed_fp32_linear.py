"""Non-dispatched candidate: fixed FP32 row-tile arithmetic, no autotuning.

For qualification only. No numerical policy selects this implementation yet.
"""
import math
import torch
import triton
import triton.language as tl

KERNEL_ID='e97-row-fixed-fp32-m16-n64-k32-w4-s2-v1'


@triton.jit(do_not_specialize=['M'],do_not_specialize_on_alignment=['M'])
def _fixed_linear(X,W,B,Y,M,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr):
    rows=tl.program_id(0)*16+tl.arange(0,16)
    cols=tl.program_id(1)*64+tl.arange(0,64)
    ks=tl.arange(0,32)
    acc=tl.full((16,64),0,tl.float32)
    for block in range(tl.cdiv(K,32)):
        kk=block*32+ks
        a=tl.load(X+rows[:,None]*K+kk[None,:],(rows[:,None]<M)&(kk[None,:]<K),other=0)
        b=tl.load(W+cols[None,:]*K+kk[:,None],(cols[None,:]<N)&(kk[:,None]<K),other=0)
        acc=tl.dot(a,b,acc,input_precision='ieee')
    if HAS_BIAS:
        bias=tl.load(B+cols,cols<N,other=0)
        acc=acc+bias[None,:]
    tl.store(Y+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


def fixed_fp32_linear(x,w,b=None):
    if x.device.type!='cuda' or w.device!=x.device or (b is not None and b.device!=x.device):raise ValueError('fixed Linear requires one CUDA device')
    if x.dtype!=torch.float32 or w.dtype!=torch.float32 or (b is not None and b.dtype!=torch.float32):raise ValueError('fixed Linear requires transient FP32 operands')
    if x.ndim<2 or w.ndim!=2 or x.shape[-1]!=w.shape[1] or min(w.shape)<=0:raise ValueError('fixed Linear shape')
    if b is not None and b.shape!=(w.shape[0],):raise ValueError('fixed Linear bias shape')
    if not x.is_contiguous() or not w.is_contiguous() or (b is not None and not b.is_contiguous()):raise ValueError('fixed Linear contiguous operands required')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x,w,b) if t is not None):raise ValueError('standalone fixed Linear autograd is not implemented')
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':raise ValueError('highest FP32/no TF32 required')
    m=math.prod(x.shape[:-1]);n,k=w.shape
    if m<=0:raise ValueError('empty Linear rows')
    out=torch.empty((*x.shape[:-1],n),device=x.device,dtype=torch.float32)
    _fixed_linear[(triton.cdiv(m,16),triton.cdiv(n,64))](x,w,w if b is None else b,out,m,n,k,b is not None,
        num_warps=4,num_stages=2,enable_fp_fusion=True)
    return out
