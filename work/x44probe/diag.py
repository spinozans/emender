import sys, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig
import experiments.expressivity_tasks.x44_chunk_scan.velocity_adapter as A
import experiments.expressivity_tasks.x44_chunk_scan.chunked_cell as cc
import experiments.expressivity_tasks.x44_chunk_scan.sequential_replay as sr
from experiments.expressivity_tasks.x44_chunk_scan.schedule import DecisionSchedule
d=torch.load('work/x44probe/bad.pt', weights_only=False)
cfg=TapeHeadConfig(n_heads=1, budget=d['cfg']['slots'], decay_rate=d['cfg']['decay_rate'],
                   free_threshold=d['cfg']['free_threshold'])
rows,tok=d['k'].shape[0],d['k'].shape[1]
t=d['trace']
class T: pass
tr=T(); tr.alloc_target=t['alloc']; tr.matched=t['matched']; tr.pop_fired=t['pop_fired']; tr.pop_target=t['pop_target']; tr.w_on=t['w_on']
sched=A.schedule_from_call(d['k'].reshape(rows,tok,-1), d['append_logit'], d['pop'], d['error_coef'], tr)
kf,qf,vf=(x.reshape(rows,tok,-1).contiguous() for x in (d['k'],d['q'],d['v']))
r_chk,f_chk=cc.chunked_tape_scan(kf,qf,vf,sched,cfg,chunk_size=256,row_block=rows)
r_seq,f_seq=sr.sequential_tape_scan(kf,qf,vf,sched,cfg)
cert=d['out'].reshape(rows,1,tok,-1)[:,0]
for name,r in (('chunked',r_chk),('sequential',r_seq)):
    dv=(cert-r).abs()
    print(name,'max',dv.max().item(),'median',dv.median().item(),
          'frac>1e-3',float((dv.amax(-1)>1e-3).float().mean()))
print('state: values',(f_chk.values-d['values'].reshape(rows,cfg.slots,-1)).abs().max().item(),
      'strength',(f_chk.strength-d['strength'].reshape(rows,cfg.slots)).abs().max().item(),
      'live',int((f_chk.live!=d['live'].reshape(rows,cfg.slots)).sum()))
print('rows',rows,'tokens',tok,'slots',cfg.slots)
dv=(cert-r_chk).abs().amax(-1)
i=torch.nonzero(dv>1e-2)
print('n bad tokens',len(i),'first',i[:6].tolist())


print('--- recorded in-arm ---')
if d.get('chunked_read') is not None:
    rec = d['chunked_read'].reshape(rows,tok,-1)
    print('recorded chunked vs recorded cert', (rec-cert).abs().max().item())
    print('fresh chunked   vs recorded cert', (r_chk-cert).abs().max().item())
    print('recorded chunked vs fresh chunked', (rec-r_chk).abs().max().item())
    dv=(rec-cert).abs()
    j=torch.nonzero(dv.amax(-1)>1e-2)[:3]
    print('bad (row,token)', j.tolist())
    if len(j):
        rr,tt=j[0].tolist()
        print('rec  ',rec[rr,tt,:4].tolist())
        print('fresh',r_chk[rr,tt,:4].tolist())
        print('cert ',cert[rr,tt,:4].tolist())
        print('call index', d.get('call'))
        print('alloc at t', sched.alloc[rr,tt].item(), 'correct', sched.correct[rr,tt].item(),
              'pop', sched.pop_slot[rr,tt].item())
        print('round tokens around t alloc', sched.alloc[rr,max(0,tt-3):tt+3].tolist())
