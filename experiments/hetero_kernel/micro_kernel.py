import sys, time, math
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
import torch
from ndm.triton.gdn2_nonlin_fused import (gdn2_nonlinear_scan_forward, gdn2_nonlinear_scan_backward, PHI_TANH)
dev='cuda'; B,T,K,V=2,2048,32,32
def mk(H):
    g=torch.Generator(device=dev).manual_seed(0)
    q=torch.randn(B,T,H,K,device=dev,dtype=torch.bfloat16,generator=g)
    k=torch.randn(B,T,H,K,device=dev,dtype=torch.bfloat16,generator=g)
    v=0.3*torch.randn(B,T,H,V,device=dev,dtype=torch.bfloat16,generator=g)
    gg=-0.05-0.05*torch.rand(B,T,H,device=dev,dtype=torch.float32,generator=g)
    beta=2.0*torch.rand(B,T,H,device=dev,dtype=torch.float32,generator=g)
    return q,k,v,gg,beta
def tm(fn,n=20,w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3
print(f"{'H':>3} {'C':>4} {'fwd_ms':>8} {'bwd_ms':>8} {'tot':>8} {'stateMB':>8}")
for H in (4,8):
  for C in (64,):
    q,k,v,gg,beta=mk(H)
    def fwd(): return gdn2_nonlinear_scan_forward(q,k,v,gg,beta,C,PHI_TANH)
    o,states=fwd()
    do=torch.randn_like(o)
    def bwd(): return gdn2_nonlinear_scan_backward(q,k,v,gg,beta,states,do,C,PHI_TANH)
    tf=tm(fwd); tb=tm(bwd)
    print(f"{H:>3} {C:>4} {tf:>8.3f} {tb:>8.3f} {tf+tb:>8.3f} {states.numel()*2/1e6:>8.1f}")
