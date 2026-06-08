"""Contention-robust A/B: two models resident, ALTERNATE one fwd+bwd each per
iteration so both experience the SAME time-varying GPU contention. Ratio of summed
times is valid even on a shared GPU. Reports baseline vs blended (overlap) and
overlap vs sequential. REAL 1.3B LadderLM, B=2 T=2048 bf16. No mocks."""
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
        overlap_streams=overlap,use_triton_e97=True,cast_recurrent_bf16=True,e97_state_nonlin='tanh',
        use_chunked_e97_delta=True,e97_chunk_size=32)
    return LadderLM(vocab_size=VOCAB_SIZE,dim=DIM,depth=BASE['depth'],level='typed-gdn2-lm',
        n_heads=BASE['n_heads'],n_state=BASE['n_state'],expansion=BASE['expansion'],layer_kwargs=lk,
        mlp_ratio=BASE['mlp_ratio']).cuda().bfloat16()
def step(m,x):
    l=m(x,return_loss=True); l=l[0] if isinstance(l,tuple) else l; l.backward(); m.zero_grad(set_to_none=True)
def interleave(mA,mB,x,n=40,w=8):
    for _ in range(w): step(mA,x); step(mB,x)
    torch.cuda.synchronize()
    tA=tB=0.0
    for _ in range(n):
        torch.cuda.synchronize(); t0=time.time(); step(mA,x); torch.cuda.synchronize(); tA+=time.time()-t0
        torch.cuda.synchronize(); t0=time.time(); step(mB,x); torch.cuda.synchronize(); tB+=time.time()-t0
    return tA/n, tB/n
def main():
    x=torch.randint(0,VOCAB_SIZE,(2,2048),device='cuda')
    f4=4/64.; f8=8/64.
    res={}
    pairs=[
      ('gdn_pure','gdn+shell4_overlap', fracs_to_logits8({'gdn2_recall':1.0}), False,
        fracs_to_logits8({'gdn2_recall':1-f4,'gdn2_nonlin_shell':f4}), True),
      ('gdn+shell4_seq','gdn+shell4_overlap', fracs_to_logits8({'gdn2_recall':1-f4,'gdn2_nonlin_shell':f4}), False,
        fracs_to_logits8({'gdn2_recall':1-f4,'gdn2_nonlin_shell':f4}), True),
      ('gdn_pure','gdn+e97d+shell4_overlap', fracs_to_logits8({'gdn2_recall':1.0}), False,
        fracs_to_logits8({'gdn2_recall':(1-f4)/2,'e97_delta':(1-f4)/2,'gdn2_nonlin_shell':f4}), True),
    ]
    for la,lb,lga,oa,lgb,ob in pairs:
        mA=build(lga,oa); mB=build(lgb,ob)
        tA,tB=interleave(mA,mB,x)
        ratio=tA/tB  # tok/s_B / tok/s_A = tA/tB
        print(f"  {la:20s} {1/tA*4096:8.0f}tok/s  vs  {lb:26s} {1/tB*4096:8.0f}tok/s  ratio(B/A)={ratio:.4f}",flush=True)
        res[f"{la}__vs__{lb}"]=dict(msA=round(tA*1e3,3),msB=round(tB*1e3,3),ratio_B_over_A=round(ratio,4))
        del mA,mB; torch.cuda.empty_cache()
    json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'interleaved.json'),'w'),indent=2)
    print("WROTE interleaved.json")
if __name__=='__main__': main()
