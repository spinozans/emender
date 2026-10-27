"""Matched registered shape gates, raw samples, occupancy and memory evidence."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time
import torch
from .model import LM
from .measure import step, instrumentation
from .loss import bounded_loss
from .steady import FiniteGradients
from ..x34_lm_pilot.data import batch
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tokens',type=int)
    parser.add_argument('--slots',type=int)
    parser.add_argument('--output',default='part2_shape_gates.json')
    args=parser.parse_args()
    if Path(args.output).name!=args.output:
        raise ValueError('output must be a receipt filename')
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):
        raise RuntimeError('broker lease required')
    torch.set_num_threads(1)
    results=[]
    shapes=[(512,s) for s in (736,2944,11776)]+[(t,2944) for t in (128,1024,4096)]
    shapes=[(t,s) for t,s in shapes if (args.tokens is None or t==args.tokens) and (args.slots is None or s==args.slots)]
    for t,s in shapes:
        arms=('base_lm','tape_lm') if s==2944 else ('tape_lm',)
        for arm in arms:
            torch.manual_seed(3401)
            record=dict(arm=arm,batch=64,tokens=t,heads=4,slots=s,
                        loss_projection='chunked-1024' if t>=1024 else 'dense',
                        warmups=2,samples=5)
            model=opt=x=y=None
            handle=None
            live=[]
            try:
                model=LM(arm).cuda().train()
                if arm=='tape_lm':
                    model.blocks[2].cell.cfg=TapeHeadConfig(n_heads=4,budget=s)
                    handle=model.blocks[2].cell.register_forward_hook(lambda m,a,o: live.__setitem__(slice(None),[o[1].native.live]))
                opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
                opt._x36_finite=FiniteGradients('cuda')
                x,y=(v.cuda() for v in batch('train',0,batch_size=64,context=t))
                def run():
                    if t<1024:
                        return step(model,opt,x,y)
                    opt.zero_grad(set_to_none=True)
                    loss=bounded_loss(model,x,y)
                    loss.backward()
                    opt._x36_finite.clip(model.parameters())
                    opt.step()
                    return loss
                torch.cuda.reset_peak_memory_stats()
                for _ in range(2):run()
                times=[]
                for _ in range(5):
                    torch.cuda.synchronize();tic=time.perf_counter()
                    run();torch.cuda.synchronize()
                    times.append((time.perf_counter()-tic)*1000)
                opt._x36_finite.check()
                record.update(status='completed',raw_ms=times,median_ms=statistics.median(times),
                              peak_allocated_bytes=torch.cuda.max_memory_allocated())
                if arm=='tape_lm':
                    component_raw=[]
                    with instrumentation(model,opt) as events:
                        for _ in range(5):
                            events.clear();run();torch.cuda.synchronize()
                            component_raw.append({name:sum(a.elapsed_time(b) for a,b in pairs)
                                                  for name,pairs in events.items()
                                                  if name in ('cell_capture','replay_backward')})
                    record['component_raw_ms']=component_raw
                    record['component_median_ms']={name:statistics.median(r[name] for r in component_raw)
                                                   for name in component_raw[0]}
                opt._x36_finite.check()
                if arm=='tape_lm':
                    from .backward import reverse_kernel
                    compiled=[c for cache in reverse_kernel.device_caches.values() for c in cache[0].values()]
                    current=compiled[-1]
                    record['reverse_compiler_stats']=dict(warps=current.metadata.num_warps,
                        registers_per_thread=current.n_regs,spills=current.n_spills,
                        shared_bytes=current.metadata.shared,
                        layout='transposed' if s<4096 else 'oracle_aos')
                if live:
                    occupancy=live[0].sum(-1).float()
                    record['final_live_min_mean_max']=[occupancy.min().item(),occupancy.mean().item(),occupancy.max().item()]
            except (torch.cuda.OutOfMemoryError,ValueError,RuntimeError) as exc:
                record.update(status='failed',error=repr(exc))
            finally:
                if handle is not None:handle.remove()
                results.append(record)
                (ROOT/'receipts'/args.output).write_text(json.dumps(results,indent=2)+'\n')
                print(json.dumps(record),flush=True)
                del model,opt,x,y
                live.clear()
                gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
