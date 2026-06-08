import os, sys, json
_T=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255/experiments/e97_wallclock_cma')
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255/experiments/e97_delta_1p3b_cma')
import torch
from wc_common import timed_tok_s
from shapes import BASE, VOCAB_SIZE, fracs_to_logits8
from ndm.models.ladder_lm import LadderLM
DIM=2240
def build(logits, overlap, **kw):
    lk=dict(head_type_logits=[float(x) for x in logits], gdn_allow_neg_eigval=True,
            lam_max=1.585, beta_max=2.747, shell_state_nonlin='tanh',
            shell_state_chunk=64, shell_fused=True, overlap_streams=overlap, **kw)
    return LadderLM(vocab_size=VOCAB_SIZE, dim=DIM, depth=BASE['depth'],
                    level='typed-gdn2-lm', n_heads=BASE['n_heads'], n_state=BASE['n_state'],
                    expansion=BASE['expansion'], layer_kwargs=lk, mlp_ratio=BASE['mlp_ratio'])
def run(label, logits, overlap):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m=build(logits, overlap).cuda().bfloat16()
    tok=timed_tok_s(m,'cuda'); del m; torch.cuda.empty_cache()
    print(f"  {label:28s} {tok:>9.1f} tok/s",flush=True)
    return tok
ceil=run('gdn_pure', fracs_to_logits8({'gdn2_recall':1.0}), False)
res={'ceiling':ceil,'rows':[]}
for n in (4,8,16):
    f=n/64.0; lg=fracs_to_logits8({'gdn2_recall':1-f,'gdn2_nonlin_shell':f})
    seq=run(f'shell_n{n}_seq', lg, False)
    ov =run(f'shell_n{n}_overlap', lg, True)
    res['rows'].append(dict(n=n, seq=seq, overlap=ov, ratio_seq=seq/ceil, ratio_ov=ov/ceil))
print("\n=== OVERLAP vs SEQUENTIAL (full 1.3B cell, ratio vs gdn_pure) ===")
print(f"  gdn_pure: {ceil:.1f} tok/s")
for r in res['rows']:
    print(f"  shell={r['n']:>2}  seq={r['ratio_seq']:.4f}  overlap={r['ratio_ov']:.4f}  gain={r['ratio_ov']-r['ratio_seq']:+.4f}")
json.dump(res, open(os.path.join(_T,'overlap_sweep.json'),'w'), indent=2)
