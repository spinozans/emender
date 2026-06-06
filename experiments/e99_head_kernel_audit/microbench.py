"""Tiny single-batch fwd+bwd throughput micro-benchmark for the four E99 head
execution paths. Scope-allowed: proves perf characteristics, no LM/CMA run.
One GPU, B/T fixed, matched head geometry. Reports ms/iter and relative slowdown.
"""
import os, sys, json, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
import torch
from ndm.models.gdn2_nonlin_shell import GDN2NonlinShellLayer
from ndm.models.unified_cell import UnifiedCellLayer
from fla.layers import GatedDeltaNet

dev='cuda'; dt=torch.bfloat16
B,T,DIM,H,N = 4, 2048, 1024, 16, 32
def mk_native():
    return GatedDeltaNet(hidden_size=DIM, expand_v=1.0, head_dim=N, num_heads=H,
        use_gate=True, use_short_conv=True, conv_size=4, allow_neg_eigval=True,
        mode='chunk', layer_idx=0).to(dev).to(dt)
def mk_shell(chunk):
    return GDN2NonlinShellLayer(dim=DIM, n_state=N, n_heads=H, state_nonlin='tanh',
        state_chunk=chunk).to(dev).to(dt)
def mk_unified():
    return UnifiedCellLayer(dim=DIM, n_state=N, n_heads=H, knob_mode='fixed_pop',
        n_spread_corners=4, corner_mixture=[0,0,0,1.0], split_gate=True,
        lam_max=1.585, beta_max=2.747, igain_max=2.0, head_norm=True,
        use_gate=True, gate_activation='silu').to(dev).to(dt)

def bench(name, m, native_call=False, iters=20):
    x = torch.randn(B,T,DIM, device=dev, dtype=dt, requires_grad=True)
    def step():
        out = m(x, use_cache=False)[0] if native_call else m(x)
        out.sum().backward()
    for _ in range(3): step(); 
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(iters): step()
    torch.cuda.synchronize()
    ms=(time.time()-t0)/iters*1e3
    return ms

if __name__=='__main__':
    torch.manual_seed(0)
    res={}
    res['native_gdn2_recall']      = bench('native', mk_native(), native_call=True)
    res['shell_nonlin_chunk64']    = bench('shell64', mk_shell(64))
    res['shell_nonlin_chunk1']     = bench('shell1',  mk_shell(1))
    res['unified_nonlin_fused']    = bench('unified', mk_unified())
    base=res['native_gdn2_recall']
    out={k:{'ms_per_iter':round(v,2),'slowdown_vs_native':round(v/base,2)} for k,v in res.items()}
    out['_config']=dict(B=B,T=T,dim=DIM,n_heads=H,n_state=N,dtype='bf16')
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(os.path.dirname(__file__),'microbench.json'),'w'), indent=2)
    print("WROTE microbench.json")
