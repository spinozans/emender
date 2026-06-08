import os, sys
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
import torch
from ndm.models.typed_head_mixture import TypedHeadMixtureLayer

dev='cuda'; torch.manual_seed(0)
dim, n_heads = 512, 16
# blend: gdn-neg bulk + a few shell heads
import torch.nn.functional as F
# logits over 8 types; want ~ gdn + shell
def mk(overlap):
    torch.manual_seed(0)
    m=TypedHeadMixtureLayer(dim=dim, n_state=32, n_heads=n_heads,
        head_type_logits=[2.0,-9,-9,-9,-9,0.0,-9,-9],  # gdn2_recall + shell
        shell_state_nonlin='tanh', shell_state_chunk=16, shell_fused=True,
        overlap_streams=overlap).cuda().bfloat16()
    return m
m0=mk(False); m1=mk(True)
m1.load_state_dict(m0.state_dict())
print("alloc:", m0.alloc['counts'])
x=torch.randn(2,128,dim,device=dev,dtype=torch.bfloat16)
x0=x.clone().requires_grad_(True); x1=x.clone().requires_grad_(True)
o0=m0(x0); o1=m1(x1)
go=torch.randn_like(o0)
o0.backward(go); o1.backward(go)
torch.cuda.synchronize()
def rel(a,b): return (a-b).abs().max().item()/b.abs().max().clamp_min(1e-8).item()
print("fwd relerr:", rel(o1.float(),o0.float()))
print("dx  relerr:", rel(x1.grad.float(),x0.grad.float()))
# param grads
mx=0
for (n,p0),(_,p1) in zip(m0.named_parameters(), m1.named_parameters()):
    if p0.grad is not None and p1.grad is not None:
        r=rel(p1.grad.float(),p0.grad.float()); mx=max(mx,r)
print("max param-grad relerr:", mx)
print("OK" if rel(o1.float(),o0.float())<1e-2 and mx<5e-2 else "FAIL")
