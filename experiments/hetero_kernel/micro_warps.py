import sys, time, math
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
import torch, triton
import ndm.triton.gdn2_nonlin_fused as M
from ndm.triton.gdn2_nonlin_fused import _gdn2_nonlin_fwd_kernel, _gdn2_nonlin_bwd_kernel, _next_pow2, PHI_TANH
dev='cuda'; B,T,K,V=2,2048,32,32
def mk(H):
    g=torch.Generator(device=dev).manual_seed(0)
    return (torch.randn(B,T,H,K,device=dev,dtype=torch.bfloat16,generator=g),
            torch.randn(B,T,H,K,device=dev,dtype=torch.bfloat16,generator=g),
            0.3*torch.randn(B,T,H,V,device=dev,dtype=torch.bfloat16,generator=g),
            -0.05-0.05*torch.rand(B,T,H,device=dev,dtype=torch.float32,generator=g),
            2.0*torch.rand(B,T,H,device=dev,dtype=torch.float32,generator=g))
def tm(fn,n=20,w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3
H=4; q,k,v,gg,beta=mk(H); bk=_next_pow2(K); bv=_next_pow2(V); C=64; SC=1.0/math.sqrt(K)
def fwd(nw):
    o=torch.empty((B,T,H,V),device=dev,dtype=q.dtype); st=torch.empty((T+1,B,H,K,V),device=dev,dtype=q.dtype)
    def f():
        _gdn2_nonlin_fwd_kernel[(B,H)](q,k,v,gg,beta,o,st,T=T,B=B,H=H,K_DIM=K,V_DIM=V,BLOCK_K=bk,BLOCK_V=bv,STATE_CHUNK=C,PHI_MODE=PHI_TANH,SCALE=SC,num_warps=nw)
    return f,o,st
_,o,st=fwd(4)[0](),*fwd(4)[1:]
o,st=fwd(4)[1],fwd(4)[2]
# need real states for bwd
oo=torch.empty((B,T,H,V),device=dev,dtype=q.dtype); st=torch.empty((T+1,B,H,K,V),device=dev,dtype=q.dtype)
_gdn2_nonlin_fwd_kernel[(B,H)](q,k,v,gg,beta,oo,st,T=T,B=B,H=H,K_DIM=K,V_DIM=V,BLOCK_K=bk,BLOCK_V=bv,STATE_CHUNK=C,PHI_MODE=PHI_TANH,SCALE=SC,num_warps=4)
do=torch.randn_like(oo)
def bwd(nw):
    dq=torch.empty((B,T,H,K),device=dev,dtype=torch.float32);dk=torch.empty_like(dq)
    dv=torch.empty((B,T,H,V),device=dev,dtype=torch.float32);dgg=torch.empty((B,T,H),device=dev,dtype=torch.float32);dbeta=torch.empty_like(dgg)
    def f():
        _gdn2_nonlin_bwd_kernel[(B,H)](q,k,v,gg,beta,st,do,dq,dk,dv,dgg,dbeta,T=T,B=B,H=H,K_DIM=K,V_DIM=V,BLOCK_K=bk,BLOCK_V=bv,STATE_CHUNK=C,PHI_MODE=PHI_TANH,SCALE=SC,num_warps=nw)
    return f
print("H=4 num_warps sweep:")
for nw in (1,2,4,8):
    ff,_,_=fwd(nw); tf=tm(ff)
    tb=tm(bwd(nw))
    print(f"  nw={nw}: fwd={tf:.3f} bwd={tb:.3f} tot={tf+tb:.3f}")
