import os, sys, time, math
sys.path.insert(0, '/home/erikg/ndm/.wg-worktrees/agent-1255')
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from ndm.triton.gdn2_nonlin_fused import fused_nonlinear_gated_delta_scan

dev='cuda'; B,T,K,V=2,2048,32,32
def mk(H, rg=False):
    g_=torch.Generator(device=dev).manual_seed(0)
    q=torch.randn(B,T,H,K,device=dev,dtype=torch.bfloat16,generator=g_,requires_grad=rg)
    k=torch.randn(B,T,H,K,device=dev,dtype=torch.bfloat16,generator=g_,requires_grad=rg)
    v=(0.3*torch.randn(B,T,H,V,device=dev,dtype=torch.bfloat16,generator=g_)).requires_grad_(rg)
    g=(-0.05-0.05*torch.rand(B,T,H,device=dev,dtype=torch.float32,generator=g_)).requires_grad_(rg)
    beta=(2.0*torch.rand(B,T,H,device=dev,dtype=torch.float32,generator=g_)).requires_grad_(rg)
    return q,k,v,g,beta

def timeit(fn, n=12, w=4):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n

def fla_fwdbwd(H):
    q,k,v,g,beta=mk(H,rg=True)
    def f():
        o,_=chunk_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True)
        o.sum().backward()
        for t in (q,k,v,g,beta): t.grad=None
    return f
def fused_fwdbwd(H, C=64):
    q,k,v,g,beta=mk(H,rg=True)
    def f():
        o=fused_nonlinear_gated_delta_scan(q,k,v,g,beta,state_chunk=C,state_nonlin='tanh')
        o.sum().backward()
        for t in (q,k,v,g,beta): t.grad=None
    return f

print(f"{'H':>4} {'fla_ms':>9} {'fused_ms':>9} {'ratio':>7} {'fused_peakMB':>13}")
for H in (64,32,16,8,4,2):
    tf=timeit(fla_fwdbwd(H))*1e3
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tu=timeit(fused_fwdbwd(H))*1e3
    peak=torch.cuda.max_memory_allocated()/1e6
    print(f"{H:>4} {tf:>9.3f} {tu:>9.3f} {tf/tu:>7.3f} {peak:>13.1f}")
