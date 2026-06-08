import sys, time, math
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
import torch
from ndm.triton.gdn2_nonlin_fused import fused_nonlinear_gated_delta_scan, nonlinear_gated_delta_torch_reference
dev='cuda'
def mk(B,T,H,K,V,seed,rg):
    g=torch.Generator(device=dev).manual_seed(seed)
    q=torch.randn(B,T,H,K,device=dev,generator=g,requires_grad=rg)
    k=torch.randn(B,T,H,K,device=dev,generator=g,requires_grad=rg)
    v=(0.3*torch.randn(B,T,H,V,device=dev,generator=g)).requires_grad_(rg)
    gg=(-0.05-0.05*torch.rand(B,T,H,device=dev,generator=g)).requires_grad_(rg)
    beta=(2.0*torch.rand(B,T,H,device=dev,generator=g)).requires_grad_(rg)
    return q,k,v,gg,beta
def rel(a,b): return (a-b).abs().max().item()/b.abs().max().clamp_min(1e-8).item()
# parity vs torch reference (which normalizes internally)
for nl in ('tanh','relu','softplus_c'):
    t=mk(2,40,3,16,16,7,True); r=tuple(x.detach().clone().requires_grad_(True) for x in t)
    o=fused_nonlinear_gated_delta_scan(*t,state_chunk=8,state_nonlin=nl,prenorm=True)
    oref=nonlinear_gated_delta_torch_reference(*r,state_chunk=8,state_nonlin=nl)
    g=torch.randn_like(o); o.backward(g); oref.backward(g)
    gr=max(rel(t[i].grad.float(),r[i].grad.float()) for i in range(5))
    print(f"  prenorm {nl}: fwd={rel(o.float(),oref.float()):.2e} maxgrad={gr:.2e}")
# timing prenorm vs in-kernel, fwd+bwd, H=4 T=2048
B,T,K,V=2,2048,32,32
def tm(fn,n=20,w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3
for H in (4,8):
    q,k,v,gg,beta=mk(B,T,H,K,V,0,True)
    for pn in (False,True):
        def f():
            o=fused_nonlinear_gated_delta_scan(q,k,v,gg,beta,state_chunk=64,state_nonlin='tanh',prenorm=pn)
            o.sum().backward()
            for t_ in (q,k,v,gg,beta): t_.grad=None
        print(f"  H={H} prenorm={pn}: {tm(f):.3f} ms (fwd+bwd)")
