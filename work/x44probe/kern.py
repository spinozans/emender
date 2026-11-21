import sys, torch
sys.path.insert(0,'/home/erikg/emender')
from experiments.expressivity_tasks.x44_chunk_scan import bench_chunk_scan as B, chunked_cell, triton_scan
for slots in (117, 2196):
    cfg,k,q,v,beta,pop,logit = B.inputs(64, 2048, slots)
    sched,_ = B.routed_schedule(cfg,k,q,v,beta,pop,logit)
    entry = chunked_cell.empty_state(64, slots, cfg.key_dim, cfg.value_dim, 'cuda')
    # real entry state: run one chunk-length of the scan first
    _, entry = chunked_cell.chunked_tape_scan(k[:, :64], q[:, :64], v[:, :64], sched.slice, cfg) if False else (None, entry)
    entry = chunked_cell.empty_state(64, slots, cfg.key_dim, cfg.value_dim, 'cuda')
    entry.live[:, :64] = True; entry.strength[:, :64] = 0.9; entry.age[:, :64] = 3
    s = type(sched)(*(getattr(sched, n)[:, :64].contiguous() for n in ('alloc','gate','correct','beta','pop_slot','pop_factor')))
    for derive in (True, False):
        ok, dev = triton_scan.strength_liveness_matches_torch(s, entry.strength, entry.live,
                                                              decay=cfg.decay_rate, threshold=cfg.free_threshold, derive=derive)
        print(f'slots {slots} derive {derive}: exact={ok} dev={dev:.3e}')
