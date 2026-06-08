import os, sys, time, json
_T=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_T)
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255')
sys.path.insert(0,'/home/erikg/ndm/.wg-worktrees/agent-1255/experiments/e97_wallclock_cma')
import torch
from wc_common import build_shell_ladder, timed_tok_s
from shapes import fracs_to_logits8

DIM=2240
def run(label, logits, **kw):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m=build_shell_ladder(DIM, logits, **kw).cuda().bfloat16()
    cnt=None
    for nm,mod in m.named_modules():
        if hasattr(mod,'alloc'): cnt=mod.alloc['counts']; break
    tok=timed_tok_s(m,'cuda')
    peak=torch.cuda.max_memory_allocated()/1e6
    del m; torch.cuda.empty_cache()
    r=dict(label=label, tok_s=round(tok,1), peak_mb=round(peak,1),
           shell=cnt['gdn2_nonlin_shell'] if cnt else None)
    print(f"  {label:24s} {r['tok_s']:>9} tok/s  shell={r['shell']}  peak={r['peak_mb']}MB",flush=True)
    return r

out=[]
base=run('gdn_pure', fracs_to_logits8({'gdn2_recall':1.0}))
ceil=base['tok_s']
for n in (4,8,16,32):
    f=n/64.0
    out.append(run(f'shell_tanh_n{n}', fracs_to_logits8({'gdn2_recall':1-f,'gdn2_nonlin_shell':f}),
                   shell_state_nonlin='tanh', shell_state_chunk=64, shell_fused=True))
for r in [base]+out:
    r['ratio']=round(r['tok_s']/ceil,4)
print("\n=== FULL-CELL THROUGHPUT vs GDN (sequential kernel, current) ===")
print(f"  gdn_pure baseline: {ceil} tok/s")
for r in out:
    print(f"  shell={r['shell']:>2}  {r['tok_s']:>9} tok/s  ratio={r['ratio']}")
json.dump(dict(ceiling=ceil, base=base, results=out), open(os.path.join(_T,'baseline_seq.json'),'w'), indent=2)
