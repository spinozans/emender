import sys, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig
from experiments.expressivity_tasks.x31_tape_cell_fused import cell_fused as x31
import experiments.expressivity_tasks.x44_chunk_scan.velocity_adapter as A
import experiments.expressivity_tasks.x44_chunk_scan.chunked_cell as cc
import experiments.expressivity_tasks.x44_chunk_scan.sequential_replay as sr
cfg = TapeHeadConfig(n_heads=1, budget=73, decay_rate=0.9861, pop_threshold=0.225, free_threshold=1e-3)
rows, tok = 1, 2048
torch.manual_seed(0)
k=torch.randn(rows,tok,1,cfg.key_dim,device='cuda'); q=torch.randn_like(k)
v=torch.randn(rows,tok,1,cfg.value_dim,device='cuda')
beta=torch.ones(rows,tok,1,device='cuda'); pop=torch.rand(rows,tok,1,device='cuda')
logit=torch.randn(rows,tok,1,device='cuda')
out,state,trace,pressure = x31.tape_scan(k,q,v,beta,pop,logit,None,cfg)
torch.cuda.synchronize()
sched = A.schedule_from_call(k.reshape(rows,tok,-1), logit, pop, torch.zeros(1,device='cuda'), trace)
kf,qf,vf = (x.reshape(rows,tok,-1).contiguous() for x in (k,q,v))
r_seq, f_seq = sr.sequential_tape_scan(kf,qf,vf,sched,cfg)
r_chk, f_chk = cc.chunked_tape_scan(kf,qf,vf,sched,cfg,chunk_size=256,row_block=rows)
cert = out.reshape(rows,1,tok,cfg.value_dim)[:,0]
print('cert vs seq  read dev', (cert-r_seq).abs().max().item())
print('cert vs chk  read dev', (cert-r_chk).abs().max().item())
print('seq  vs chk  read dev', (r_seq-r_chk).abs().max().item())
d=(cert-r_seq).abs().amax(-1)
i=torch.nonzero(d>1e-3)
print('first differing token (row,tok):', i[:5].tolist())
if len(i):
    r,t=i[0].tolist()
    print('cert', cert[r,t,:6].tolist())
    print('seq ', r_seq[r,t,:6].tolist())
    print('live at t', int(trace.pop_fired[r,t,0]), trace.alloc_target[r,t,0].item(), trace.matched[r,t,0].item(), trace.case[r,t,0].item())
