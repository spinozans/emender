import sys, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig
from experiments.expressivity_tasks.x31_tape_cell_fused import cell_fused as x31
cfg = TapeHeadConfig(n_heads=1, budget=73, decay_rate=0.9861, pop_threshold=0.225, free_threshold=1e-3)
rows, tok = 1, 2048
torch.manual_seed(0)
k=torch.randn(rows,tok,1,cfg.key_dim,device='cuda'); q=torch.randn_like(k)
v=torch.randn(rows,tok,1,cfg.value_dim,device='cuda')
beta=torch.ones(rows,tok,1,device='cuda'); pop=torch.rand(rows,tok,1,device='cuda')*0.3
logit=torch.randn(rows,tok,1,device='cuda')
outs=[]
for i in range(3):
    out,state,trace,_ = x31.tape_scan(k,q,v,beta,pop,logit,None,cfg)
    torch.cuda.synchronize(); outs.append(out.clone())
print('kernel run-to-run read deviation:', (outs[0]-outs[1]).abs().max().item(), (outs[0]-outs[2]).abs().max().item())
