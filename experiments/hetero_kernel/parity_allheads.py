"""fwd+bwd bf16 parity per head type, and overlap-vs-sequential parity for the
blended cell. Correctness only (contention-insensitive)."""
import sys; sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
import torch
from ndm.models.typed_head_mixture import TypedHeadMixtureLayer, TYPE_NAMES
def relerr(a,b): return (a-b).abs().max().item()/b.abs().max().clamp_min(1e-8).item()
dev='cuda'; dim=512; nh=16
def logits_for(types):
    lg=[-9.0]*8
    for t in types: lg[TYPE_NAMES.index(t)]=2.0
    return lg
def mk(types, overlap, seed=0):
    torch.manual_seed(seed)
    return TypedHeadMixtureLayer(dim=dim,n_state=32,n_heads=nh,head_type_logits=logits_for(types),
        shell_state_nonlin='tanh',shell_state_chunk=16,shell_fused=True,
        e97_state_nonlin='tanh',use_chunked_e97_delta=True,e97_chunk_size=16,
        overlap_streams=overlap).cuda().bfloat16()
print("=== overlap vs sequential parity (fwd+bwd) ===")
for types in (['gdn2_recall','gdn2_nonlin_shell'],
              ['gdn2_recall','e97_delta','gdn2_nonlin_shell'],
              ['gdn2_recall','e97_raw','e97_delta','gdn2_nonlin_shell']):
    m0=mk(types,False); m1=mk(types,True); m1.load_state_dict(m0.state_dict())
    x=torch.randn(2,96,dim,device=dev,dtype=torch.bfloat16)
    x0=x.clone().requires_grad_(True); x1=x.clone().requires_grad_(True)
    o0=m0(x0); o1=m1(x1); g=torch.randn_like(o0); o0.backward(g); o1.backward(g)
    torch.cuda.synchronize()
    pg=max((relerr(p1.grad.float(),p0.grad.float()) for (_,p0),(_,p1) in zip(m0.named_parameters(),m1.named_parameters()) if p0.grad is not None and p1.grad is not None), default=0)
    cnt={k:v for k,v in m0.alloc['counts'].items() if v>0}
    ok = relerr(o1.float(),o0.float())<1e-2 and relerr(x1.grad.float(),x0.grad.float())<5e-2 and pg<5e-2
    print(f"  {str(cnt):60s} fwd={relerr(o1.float(),o0.float()):.1e} dx={relerr(x1.grad.float(),x0.grad.float()):.1e} pg={pg:.1e} {'OK' if ok else 'FAIL'}")
