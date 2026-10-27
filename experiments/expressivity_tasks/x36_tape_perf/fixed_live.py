"""Controlled T ablation: constant live prefix, no writes/pops, four warps."""
import json
import os
from pathlib import Path
import statistics
import time
import torch
from .cell_fused import tape_scan,FusedState,TapeHeadConfig
from ..x30_tape_cell.naive_reference import init_state


def main():
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):
        raise RuntimeError('broker lease required')
    torch.set_num_threads(1)
    rows=[]
    for t in (512,4096):
        torch.manual_seed(3602)
        cfg=TapeHeadConfig(n_heads=4,budget=2944)
        native=init_state(2,cfg,'cuda')
        native.keys.normal_();native.values.normal_()
        native.live[...,:64]=True;native.strength[...,:64]=1.
        history=torch.zeros(8,128,device='cuda',dtype=torch.int32)
        torch.manual_seed(3603)
        inputs=[torch.randn(2,1,4,d,device='cuda').expand(2,t,4,d).contiguous().requires_grad_()
                for d in (64,64,128)]
        inputs += [torch.zeros(2,t,4,device='cuda',requires_grad=True) for _ in range(3)]
        inputs[0].requires_grad_(False)
        inputs[3].requires_grad_(False)
        coef=torch.zeros(4,device='cuda',requires_grad=True)
        def run():
            for x in (*inputs,coef):x.grad=None
            state=FusedState(native,history,live_bound=64)
            begin,middle,end=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
            torch.cuda.synchronize();tic=time.perf_counter();begin.record()
            out,final,_,_=tape_scan(*inputs,state,cfg,append_err_coef=coef)
            middle.record();out.sum().backward();end.record()
            torch.cuda.synchronize()
            return dict(wall_ms=(time.perf_counter()-tic)*1000,
                        scan_ms=begin.elapsed_time(middle),adjoint_ms=middle.elapsed_time(end)),final.native.live
        for _ in range(2):run()
        samples=[]
        for _ in range(5):
            sample,live=run();samples.append(sample)
        occupancy=live.sum(-1)
        if not bool((occupancy==64).all()):
            raise AssertionError('fixed occupancy changed')
        from .backward import reverse_kernel
        compiled=[c for cache in reverse_kernel.device_caches.values() for c in cache[0].values()]
        current=compiled[-1]
        if current.metadata.num_warps!=4:
            raise AssertionError('controlled T comparison changed warp count')
        med={key:statistics.median(s[key] for s in samples) for key in samples[0]}
        row=dict(batch=2,heads=4,slots=2944,tokens=t,live=64,
                 writes=False,pops=False,decay=cfg.decay_rate,raw_ms=samples,median_ms=med,
                 key_adjoint_tracked=False,
                 per_token_ms={key:value/t for key,value in med.items()},
                 warps=4,registers=current.n_regs,spills=current.n_spills,
                 scope='Controlled cell-only ablation; not a B64 model cost gate')
        rows.append(row)
        print(json.dumps(row),flush=True)
    root=Path(__file__).resolve().parent
    (root/'receipts/part3_fixed_live.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
