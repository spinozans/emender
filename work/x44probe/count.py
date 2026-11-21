import sys, time, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x44_chunk_scan import bench_chunk_scan as B, chunked_cell, graph_scan
from torch.profiler import profile, ProfilerActivity
for slots in (117, 2196):
    cfg,k,q,v,beta,pop,logit = B.inputs(64, 2048, slots)
    sched,_ = B.routed_schedule(cfg,k,q,v,beta,pop,logit)
    for vec in (False, True):
        chunked_cell.chunked_tape_scan(k,q,v,sched,cfg,chunk_size=64,row_block=64,vectorized=vec)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            chunked_cell.chunked_tape_scan(k,q,v,sched,cfg,chunk_size=64,row_block=64,vectorized=vec)
            torch.cuda.synchronize()
        total=sum(e.count for e in prof.key_averages())
        print(f'slots {slots} vectorized {vec}: kernels {total}  (per chunk {total/32:.0f})')
    for vec in (False, True):
        t0=time.perf_counter()
        g = graph_scan.ChunkedScanGraph(cfg,64,2048,chunk_size=64,row_block=64)
        torch.cuda.synchronize(); cap=time.perf_counter()-t0
        ts=[]
        for _ in range(3):
            t0=time.perf_counter(); g(k,q,v,sched); torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1e3)
        print(f'slots {slots} vec {vec} graph replay ms {[round(t,1) for t in ts]} capture {cap:.1f}s')
        break
