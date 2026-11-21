import sys, time, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig
import experiments.expressivity_tasks.x44_chunk_scan.chunked_cell as cc
import experiments.expressivity_tasks.x44_chunk_scan.sequential_replay as sr
import experiments.expressivity_tasks.x44_chunk_scan.graph_scan as gs
from experiments.expressivity_tasks.x44_chunk_scan.schedule import DecisionSchedule
import experiments.expressivity_tasks.x44_chunk_scan.bench_chunk_scan as B
cfg=TapeHeadConfig(n_heads=1,budget=73,decay_rate=B.__dict__.get('DECAY',0.9861),pop_threshold=0.225,free_threshold=1e-3)
rows,tok=8,2048
cfg,k,q,v,beta,pop,logit=B.inputs(rows,tok,cfg.slots)
cfg=TapeHeadConfig(n_heads=1,budget=73,decay_rate=0.9861,pop_threshold=0.225,free_threshold=1e-3)
sched,_=B.routed_schedule(cfg,k,q,v,beta,pop,logit)
seq=B.statistics.median(B.timed(lambda: sr.sequential_tape_scan(k,q,v,sched,cfg),3)[0])
print('sequential_replay per call ms', round(seq*1e3,2))
for c in (64,128,256,512):
    g=gs.ChunkedScanGraph(cfg,rows,tok,chunk_size=c,row_block=rows)
    ts=[]
    for _ in range(5):
        t0=time.perf_counter(); g(k,q,v,sched); torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1e3)
    print(f'armshape chunked graph C={c}: {B.statistics.median(ts):.2f} ms  ({seq*1e3/B.statistics.median(ts):.0f}x vs sequential_replay)')
    eager=[]
    for _ in range(3):
        t0=time.perf_counter(); cc.chunked_tape_scan(k,q,v,sched,cfg,chunk_size=c,row_block=rows); torch.cuda.synchronize(); eager.append((time.perf_counter()-t0)*1e3)
    print('                 eager %.2f ms' % B.statistics.median(eager))
