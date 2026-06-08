import os,sys,time
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255/experiments/e97_delta_1p3b_cma')
import torch
from shapes import BASE, VOCAB_SIZE, fracs_to_logits8
from ndm.models.ladder_lm import LadderLM
DIM=2240
def build(ov):
    f=4/64.
    lk=dict(head_type_logits=[float(x) for x in fracs_to_logits8({'gdn2_recall':1-f,'gdn2_nonlin_shell':f})],
            gdn_allow_neg_eigval=True, lam_max=1.585, beta_max=2.747, shell_state_nonlin='tanh',
            shell_state_chunk=64, shell_fused=True, overlap_streams=ov)
    return LadderLM(vocab_size=VOCAB_SIZE,dim=DIM,depth=BASE['depth'],level='typed-gdn2-lm',
        n_heads=BASE['n_heads'],n_state=BASE['n_state'],expansion=BASE['expansion'],layer_kwargs=lk,mlp_ratio=BASE['mlp_ratio'])
def timeit(m, bwd, n=15,w=5):
    x=torch.randint(0,VOCAB_SIZE,(2,2048),device='cuda')
    for _ in range(w):
        l=m(x,return_loss=True); l=l[0] if isinstance(l,tuple) else l
        if bwd: l.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n):
        l=m(x,return_loss=True); l=l[0] if isinstance(l,tuple) else l
        if bwd: l.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); return 2*2048*n/(time.time()-t0)
for ov in (False,True):
    m=build(ov).cuda().bfloat16()
    with torch.no_grad(): fwd=timeit(m,False)
    fb=timeit(m,True)
    print(f"overlap={ov}: fwd_only={fwd:.0f}  fwd+bwd={fb:.0f}")
    del m; torch.cuda.empty_cache()
