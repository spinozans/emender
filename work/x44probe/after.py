import sys, time, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x44_chunk_scan import bench_chunk_scan as B, chunked_cell, graph_scan, sequential_replay
from torch.profiler import profile, ProfilerActivity
for slots, rows, tok in ((117,64,2048),(2196,64,2048),(2196,64,256),(2196,8,2048)):
    cfg,k,q,v,beta,pop,logit = B.inputs(rows, tok, slots)
    sched,_ = B.routed_schedule(cfg,k,q,v,beta,pop,logit)
    seq = B.statistics.median(B.timed(lambda: sequential_replay.sequential_tape_scan(k,q,v,sched,cfg),3)[0])
    best=None
    for c in (16,32,48,64,96,128,256):
        if c>tok: continue
        rb=max(1,min(rows,2**24//(c*slots)))
        g=graph_scan.ChunkedScanGraph(cfg,rows,tok,chunk_size=c,row_block=rb)
        ts=[]
        for _ in range(3):
            t0=time.perf_counter(); g(k,q,v,sched); torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1e3)
        med=sorted(ts)[1]
        if best is None or med<best[1]: best=(c,med)
    # node count at the best C
    c=best[0]
    chunked_cell.chunked_tape_scan(k,q,v,sched,cfg,chunk_size=c,row_block=max(1,min(rows,2**24//(c*slots))))
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        chunked_cell.chunked_tape_scan(k,q,v,sched,cfg,chunk_size=c,row_block=max(1,min(rows,2**24//(c*slots))))
        torch.cuda.synchronize()
    nodes=sum(e.count for e in prof.key_averages())
    print(f'slots {slots:5d} rows {rows:3d} T {tok:5d}: best C={c:3d} graph {best[1]:7.1f} ms  seq {seq*1e3:7.1f} ms -> {seq*1e3/best[1]:6.1f}x   nodes {nodes} ({nodes/((tok+c-1)//c):.0f}/chunk)')
