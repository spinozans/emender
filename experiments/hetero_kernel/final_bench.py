"""hetero-kernel FINAL benchmark — one clean-GPU run captures everything:
  A. isolated nonlinear scan kernel: prenorm on/off, fwd+bwd, H=4/8 (kernel evidence)
  B. full 1.3B LadderLM cell (dim=2240, depth=18, 64 heads), fwd+bwd, bf16:
     - gdn_pure (GDN-2 ceiling)
     - gdn+shell{4,8}              overlap on/off
     - gdn+e97_delta+shell{4,8}    overlap on/off  (wider chunkable bulk)
REAL models, REAL token batches, no mocks. Writes final_bench.json.
"""
import os, sys, json, time
_T=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255/experiments/e97_delta_1p3b_cma')
import torch
from shapes import BASE, VOCAB_SIZE, fracs_to_logits8
from ndm.models.ladder_lm import LadderLM
from ndm.triton.gdn2_nonlin_fused import fused_nonlinear_gated_delta_scan
DIM=2240

# ---- A. isolated scan kernel prenorm speedup ----
def mk_scan(H, B=2, T=2048, K=32, V=32):
    g=torch.Generator(device='cuda').manual_seed(0)
    q=torch.randn(B,T,H,K,device='cuda',dtype=torch.bfloat16,generator=g,requires_grad=True)
    k=torch.randn(B,T,H,K,device='cuda',dtype=torch.bfloat16,generator=g,requires_grad=True)
    v=(0.3*torch.randn(B,T,H,V,device='cuda',dtype=torch.bfloat16,generator=g)).requires_grad_(True)
    gg=(-0.05-0.05*torch.rand(B,T,H,device='cuda',dtype=torch.float32,generator=g)).requires_grad_(True)
    beta=(2.0*torch.rand(B,T,H,device='cuda',dtype=torch.float32,generator=g)).requires_grad_(True)
    return q,k,v,gg,beta
def tm(fn,n=30,w=8):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3
def scan_bench():
    rows=[]
    for H in (4,8):
        q,k,v,gg,beta=mk_scan(H)
        for pn in (False,True):
            def f():
                o=fused_nonlinear_gated_delta_scan(q,k,v,gg,beta,state_chunk=64,state_nonlin='tanh',prenorm=pn)
                o.sum().backward()
                for t_ in (q,k,v,gg,beta): t_.grad=None
            ms=tm(f); rows.append(dict(H=H,prenorm=pn,ms=round(ms,3)))
            print(f"  scan H={H} prenorm={pn}: {ms:.3f} ms (fwd+bwd)",flush=True)
    return rows

# ---- B. full cell ----
def build(logits, overlap):
    lk=dict(head_type_logits=[float(x) for x in logits], gdn_allow_neg_eigval=True,
            lam_max=1.585, beta_max=2.747, shell_state_nonlin='tanh', shell_state_chunk=64,
            shell_fused=True, overlap_streams=overlap, use_triton_e97=True,
            cast_recurrent_bf16=True, e97_state_nonlin='tanh', use_chunked_e97_delta=True, e97_chunk_size=32)
    return LadderLM(vocab_size=VOCAB_SIZE, dim=DIM, depth=BASE['depth'], level='typed-gdn2-lm',
                    n_heads=BASE['n_heads'], n_state=BASE['n_state'], expansion=BASE['expansion'],
                    layer_kwargs=lk, mlp_ratio=BASE['mlp_ratio'])
def tok_s(m, B=2, T=2048, n=15, w=6):
    x=torch.randint(0,VOCAB_SIZE,(B,T),device='cuda')
    for _ in range(w):
        l=m(x,return_loss=True); l=l[0] if isinstance(l,tuple) else l; l.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n):
        l=m(x,return_loss=True); l=l[0] if isinstance(l,tuple) else l; l.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); return B*T*n/(time.time()-t0)
def counts(m):
    for _,mod in m.named_modules():
        if hasattr(mod,'alloc'): return mod.alloc['counts']
def run(label, logits, overlap):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m=build(logits, overlap).cuda().bfloat16()
    c=counts(m); t=tok_s(m); peak=torch.cuda.max_memory_allocated()/1e6
    del m; torch.cuda.empty_cache()
    cc={k:v for k,v in c.items() if v>0}
    print(f"  {label:34s} {t:>9.1f} tok/s  peak={peak:>6.0f}MB  {cc}",flush=True)
    return dict(label=label, tok_s=round(t,1), peak_mb=round(peak,1), counts=c, overlap=overlap)

if __name__=='__main__':
    print("=== A. isolated nonlinear scan: prenorm speedup (fwd+bwd) ===")
    scan=scan_bench()
    print("=== B. full 1.3B cell: blended tok/s vs GDN-2 ===")
    ceil=run('gdn_pure (GDN-2 baseline)', fracs_to_logits8({'gdn2_recall':1.0}), False)
    rows=[ceil]
    for n in (4,8):
        f=n/64.0
        rows.append(run(f'gdn+shell{n}_seq', fracs_to_logits8({'gdn2_recall':1-f,'gdn2_nonlin_shell':f}), False))
        rows.append(run(f'gdn+shell{n}_overlap', fracs_to_logits8({'gdn2_recall':1-f,'gdn2_nonlin_shell':f}), True))
    for n in (4,8):
        f=n/64.0; rest=(1-f)/2
        lg=fracs_to_logits8({'gdn2_recall':rest,'e97_delta':rest,'gdn2_nonlin_shell':f})
        rows.append(run(f'gdn+e97d+shell{n}_seq', lg, False))
        rows.append(run(f'gdn+e97d+shell{n}_overlap', lg, True))
    C=ceil['tok_s']
    for r in rows: r['ratio']=round(r['tok_s']/C,4)
    out=dict(ceiling=C, scan=scan, rows=rows)
    json.dump(out, open(os.path.join(_T,'final_bench.json'),'w'), indent=2)
    print("\n=== RATIOS vs GDN-2 ===")
    for r in rows: print(f"  {r['label']:34s} ratio={r['ratio']}")
    print("WROTE final_bench.json")
