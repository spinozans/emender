import sys, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig
from experiments.expressivity_tasks.x31_tape_cell_fused import cell_fused as x31
from experiments.expressivity_tasks.x42_pile_scale import velocity_forward as vf
import experiments.expressivity_tasks.x44_chunk_scan.velocity_adapter as A
import experiments.expressivity_tasks.x44_chunk_scan.chunked_cell as cc
from experiments.expressivity_tasks.x44_chunk_scan import bench_chunk_scan as B
cfg = TapeHeadConfig(n_heads=1, budget=73, decay_rate=0.9861, pop_threshold=0.225, free_threshold=1e-3)
rows, tok = 8, 2048
torch.manual_seed(0)
k=torch.randn(rows,tok,1,cfg.key_dim,device='cuda'); q=torch.randn_like(k)
v=torch.randn(rows,tok,1,cfg.value_dim,device='cuda')
beta=torch.ones(rows,tok,1,device='cuda'); pop=torch.rand(rows,tok,1,device='cuda')
logit=torch.randn(rows,tok,1,device='cuda')
out,state,trace,pressure = x31.tape_scan(k,q,v,beta,pop,logit,None,cfg)
torch.cuda.synchronize()
nat=state.native
sched = A.schedule_from_call(k.reshape(rows,tok,-1), logit, pop, torch.zeros(1,device='cuda'), trace)
read, final = cc.chunked_tape_scan(k.reshape(rows,tok,-1), q.reshape(rows,tok,-1), v.reshape(rows,tok,-1),
                                   sched, cfg, chunk_size=256, row_block=rows)
print('read dev', (read.reshape(rows,1,tok,cfg.value_dim)-out).abs().max().item())
print('values dev', (final.values-nat.values.reshape(rows,cfg.slots,-1)).abs().max().item())
print('strength dev', (final.strength-nat.strength.reshape(rows,cfg.slots)).abs().max().item())
print('live mismatch', int((final.live!=nat.live.reshape(rows,cfg.slots)).sum()))
print('age dev', int((final.age-nat.age.reshape(rows,cfg.slots)).abs().max()))
print('read scale', out.abs().max().item(), 'read rms', out.pow(2).mean().sqrt().item())
