"""Confirm the >=0.95x result is stable: 4 trials each of gdn_pure and
gdn+shell4_overlap (the winning composition), report per-trial + mean ratio.
Standard fwd+bwd throughput (matches repo timed_tok_s). REAL 1.3B cell."""
import os,sys,json,time
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255/experiments/e97_delta_1p3b_cma')
import torch
from shapes import BASE, VOCAB_SIZE, fracs_to_logits8
from ndm.models.ladder_lm import LadderLM
DIM=2240
def build(logits, overlap):
    lk=dict(head_type_logits=[float(x) for x in logits],gdn_allow_neg_eigval=True,lam_max=1.585,
        beta_max=2.747,shell_state_nonlin='tanh',shell_state_chunk=64,shell_fused=True,
        overlap_streams=overlap)
    return LadderLM(vocab_size=VOCAB_SIZE,dim=DIM,depth=BASE['depth'],level='typed-gdn2-lm',
        n_heads=BASE['n_heads'],n_state=BASE['n_state'],expansion=BASE['expansion'],layer_kwargs=lk,
        mlp_ratio=BASE['mlp_ratio']).cuda().bfloat16()
def toks(m,n=20,w=8):
    x=torch.randint(0,VOCAB_SIZE,(2,2048),device='cuda')
    for _ in range(w):
        l=m(x,return_loss=True);l=l[0] if isinstance(l,tuple) else l;l.backward();m.zero_grad(set_to_none=True)
    torch.cuda.synchronize();t0=time.time()
    for _ in range(n):
        l=m(x,return_loss=True);l=l[0] if isinstance(l,tuple) else l;l.backward();m.zero_grad(set_to_none=True)
    torch.cuda.synchronize();return 2*2048*n/(time.time()-t0)
f=4/64.
base=build(fracs_to_logits8({'gdn2_recall':1.0}),False)
blend=build(fracs_to_logits8({'gdn2_recall':1-f,'gdn2_nonlin_shell':f}),True)
ratios=[]
for i in range(4):
    b=toks(base); s=toks(blend); ratios.append(s/b)
    print(f"  trial {i}: gdn_pure={b:.0f}  blend={s:.0f}  ratio={s/b:.4f}",flush=True)
import statistics
print(f"MEAN ratio = {statistics.mean(ratios):.4f}  (min {min(ratios):.4f}, max {max(ratios):.4f})")
json.dump(dict(ratios=ratios,mean=statistics.mean(ratios)),open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'confirm.json'),'w'),indent=2)
