import sys, time, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x44_chunk_scan import bench_chunk_scan as B, chunked_cell, graph_scan, sequential_replay
shapes = [('s117 T2048 b64',2048,64,117), ('s311 T2048 b64',2048,64,311),
          ('s826 T2048 b64',2048,64,826), ('s2196 T2048 b64',2048,64,2196),
          ('s2196 T256 b64',256,64,2196), ('s117 T256 b64',256,64,117),
          ('s2196 T2048 b8',2048,8,2196)]
chunks = [16,32,48,64,96,128,192,256]
for name,tok,rows,slots in shapes:
    cfg,k,q,v,beta,pop,logit = B.inputs(rows, tok, slots)
    sched,_ = B.routed_schedule(cfg,k,q,v,beta,pop,logit)
    seq = B.timed(lambda: sequential_replay.sequential_tape_scan(k,q,v,sched,cfg), 3)[0]
    out=[]
    for c in chunks:
        if c > tok: continue
        rb = max(1, min(rows, 2**24//(c*slots)))
        t0=time.perf_counter()
        g = graph_scan.ChunkedScanGraph(cfg,rows,tok,chunk_size=c,row_block=rb)
        torch.cuda.synchronize(); cap=time.perf_counter()-t0
        ts=[]
        for _ in range(3):
            t0=time.perf_counter(); g(k,q,v,sched); torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1e3)
        med=sorted(ts)[1]
        out.append((c, med, cap))
    best=min(out,key=lambda x:x[1])
    print(f'{name:18s} seq {B.statistics.median(seq)*1e3:7.1f} ms  best C={best[0]:4d} {best[1]:7.1f} ms  ({B.statistics.median(seq)*1e3/best[1]:.1f}x)  ' +
          ' '.join(f'c{c}:{t:.0f}' for c,t,_ in out))
