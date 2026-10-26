"""Compare compact journals under equal append-event streams, after parity."""
import argparse
import gc
import os
import torch
from .all_field_journal import AllFieldJournal
from .cell_fused import tape_scan, TapeHeadConfig
from .train_gates import require_parity, save


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity-receipt',required=True)
    args=parser.parse_args()
    require_parity(args.parity_receipt)
    held=os.environ.get('GPU_LEASE_HELD')
    if held not in tuple(str(i) for i in range(7)) or os.environ.get('CUDA_VISIBLE_DEVICES')!=held:
        raise RuntimeError('single broker lease on cards 0..6 required')
    b,t,h,dk,dv=64,16,4,64,128
    rows=[]
    for s in (736,11776):
        cfg=TapeHeadConfig(key_dim=dk,value_dim=dv,n_heads=h,budget=s)
        k=torch.zeros(b,t,h,dk,device='cuda')
        v=torch.zeros(b,t,h,dv,device='cuda')
        on=torch.ones(b,t,h,device='cuda')
        off=torch.zeros_like(on)
        _,state,_,_=tape_scan(k,k,v,on,off,off,None,cfg,record_all=True)
        journal=state.all_field_journal
        row=dict(slots=s,stores=b*h,tokens=t,initial_live=0,
                 append_events=b*h*t,recorded_entries=int(journal.counts.sum()),
                 retained_capacity_per_store=journal.capacity,
                 retained_reserved_bytes=journal.nbytes,
                 retained_storage_bytes=sum(x.untyped_storage().nbytes() for x in
                     (journal.addresses,journal.before,journal.widths,journal.counts,
                      journal.boundaries,journal.gates,journal.errors)),
                 capture_capacity_per_store=AllFieldJournal.capacity_for(b*h,t,s,dk,dv,0),
                 capture_reserved_bytes=b*h*(20*AllFieldJournal.capacity_for(b*h,t,s,dk,dv,0)+16*t+8),
                 transient_capture_scratch_bytes=b*h*max(s,dv)*8)
        rows.append(row)
        print(row,flush=True)
        del state,journal,k,v,on,off
        gc.collect();torch.cuda.empty_cache()
    ratio=rows[1]['retained_reserved_bytes']/rows[0]['retained_reserved_bytes']
    capture_ratio=rows[1]['capture_reserved_bytes']/rows[0]['capture_reserved_bytes']
    equal=rows[0]['recorded_entries']==rows[1]['recorded_entries']
    save('storage_part6.json',dict(physical_gpu=held,gpu=torch.cuda.get_device_name(),
        measurements=rows,equal_recorded_events=equal,retained_reservation_ratio=ratio,
        capture_reservation_ratio=capture_ratio,
        storage_independence_gate=equal and ratio<=1.5 and capture_ratio<=1.5,
        note='Component allocations, not measured peak or CUDA caching-allocator reservation. Capture scratch and base/replay state remain O(S); retained logs use at most twice the largest store event count per store plus per-token routing records.'))


if __name__=='__main__':
    main()
