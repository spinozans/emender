import sys, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig
from experiments.expressivity_tasks.x31_tape_cell_fused import cell_fused as x31
import experiments.expressivity_tasks.x44_chunk_scan.velocity_adapter as A
import experiments.expressivity_tasks.x44_chunk_scan.chunked_cell as cc
cfg = TapeHeadConfig(n_heads=1, budget=73, decay_rate=0.9861, pop_threshold=0.225, free_threshold=1e-3)
rows, tok = 8, 2048
torch.manual_seed(0)
# tied key/query geometry, as the pure arm initialises it (query.load_state_dict(key))
feat = torch.randn(rows, tok, 64, device='cuda')
Wk = torch.randn(64,64, device='cuda')/8
k = (feat @ Wk).reshape(rows,tok,1,64)*4
q = k.clone()
v=torch.randn(rows,tok,1,cfg.value_dim,device='cuda')
beta=torch.ones(rows,tok,1,device='cuda'); pop=(torch.rand(rows,tok,1,device='cuda')*0.3)
logit=torch.randn(rows,tok,1,device='cuda')
out,state,trace,pressure = x31.tape_scan(k,q,v,beta,pop,logit,None,cfg)
torch.cuda.synchronize()
sched = A.schedule_from_call(k.reshape(rows,tok,-1), logit, pop, torch.zeros(1,device='cuda'), trace)
kf,qf,vf = (x.reshape(rows,tok,-1).contiguous() for x in (k,q,v))
r_chk, f_chk = cc.chunked_tape_scan(kf,qf,vf,sched,cfg,chunk_size=256,row_block=rows)
cert = out.reshape(rows,1,tok,cfg.value_dim)[:,0]
d = (cert-r_chk).abs()
print('read max dev', d.max().item(), 'median', d.median().item())
print('frac of (token,row) with dev>1e-3:', float((d.amax(-1)>1e-3).float().mean()))
nat=state.native
print('values dev', (f_chk.values-nat.values.reshape(rows,cfg.slots,-1)).abs().max().item(),
      'strength', (f_chk.strength-nat.strength.reshape(rows,cfg.slots)).abs().max().item(),
      'live mismatch', int((f_chk.live!=nat.live.reshape(rows,cfg.slots)).sum()))
# how peaked is the softmax? report the max probability per token for a sample row
sc = (q[0].reshape(tok,64) @ nat.keys.reshape(rows,cfg.slots,64)[0].T)*(2/64**0.5)
p = torch.softmax(sc.masked_fill(~nat.live.reshape(rows,cfg.slots)[0].expand(tok,-1), -1e30), -1)
print('softmax max prob: median', p.max(-1).values.median().item(), 'p99', p.max(-1).values.quantile(0.99).item())
top2 = p.topk(2,-1).values
print('top1-top2 margin median', (top2[:,0]-top2[:,1]).median().item())
