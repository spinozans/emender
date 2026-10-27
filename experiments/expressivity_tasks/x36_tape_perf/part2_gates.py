"""Matched registered shape gates, raw samples, occupancy and memory evidence."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch
import triton
import triton.language as tl
from .model import LM
from .measure import step, instrumentation
from .loss import bounded_loss
from .steady import FiniteGradients
from ..x34_lm_pilot.data import batch
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig

ROOT=Path(__file__).resolve().parent


@triton.jit
def count_frees(Freed,Out,S:tl.constexpr,Block:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    slot=tl.arange(0,Block)
    flag=tl.load(Freed+row*S+slot,slot<S,0)
    tl.store(Out+row,tl.sum((flag!=0).to(tl.int32),0))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tokens',type=int)
    parser.add_argument('--slots',type=int)
    parser.add_argument('--profile',action='store_true',help='Separate diagnostic profile, excluded from wall samples')
    parser.add_argument('--traffic',action='store_true',help='Separate unmeasured routing/live-slot trace')
    parser.add_argument('--output',default='part2_shape_gates.json')
    args=parser.parse_args()
    if Path(args.output).name!=args.output:
        raise ValueError('output must be a receipt filename')
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):
        raise RuntimeError('broker lease required')
    torch.set_num_threads(1)
    results=[]
    shapes=[(512,s) for s in (736,2944,5888,11776)]+[(t,2944) for t in (128,1024,4096)]
    if args.tokens is not None and args.slots is not None:
        shapes=[(args.tokens,args.slots)]
    shapes=[(t,s) for t,s in shapes if (args.tokens is None or t==args.tokens) and (args.slots is None or s==args.slots)]
    for t,s in shapes:
        arms=('base_lm','tape_lm') if s==2944 else ('tape_lm',)
        for arm in arms:
            torch.manual_seed(3401)
            record=dict(arm=arm,batch=64,tokens=t,heads=4,slots=s,
                        physical_gpu=os.environ['GPU_LEASE_HELD'],gpu=torch.cuda.get_device_name(),
                        python=sys.executable,torch_version=torch.__version__,
                        loss_projection='chunked-1024' if t>=1024 else 'dense',
                        warmups=2,samples=5)
            model=opt=x=y=None
            handle=None
            input_handle=None
            live=[]
            saved_inputs=[]
            try:
                model=LM(arm).cuda().train()
                if arm=='tape_lm':
                    model.blocks[2].cell.cfg=TapeHeadConfig(n_heads=4,budget=s)
                    handle=model.blocks[2].cell.register_forward_hook(lambda m,a,o: live.__setitem__(slice(None),[o[1].native.live]))
                    input_handle=model.blocks[2].cell.register_forward_pre_hook(lambda m,a: saved_inputs.__setitem__(slice(None),[(a[0].detach(),a[1])]))
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
                for warmup in range(2):
                    run()
                    if arm=='tape_lm':
                        from .backward import reverse_kernel
                        compiled=[c for cache in reverse_kernel.device_caches.values() for c in cache[0].values()]
                        current=compiled[-1]
                        print(json.dumps(dict(progress='warmup_compiler',tokens=t,slots=s,warmup=warmup+1,
                                              warps=current.metadata.num_warps,registers=current.n_regs,
                                              spills=current.n_spills)),flush=True)
                times=[]
                for _ in range(5):
                    torch.cuda.synchronize();tic=time.perf_counter()
                    run();torch.cuda.synchronize()
                    times.append((time.perf_counter()-tic)*1000)
                    print(json.dumps(dict(progress='wall_sample',arm=arm,tokens=t,slots=s,
                                          sample=len(times),ms=times[-1])),flush=True)
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
                    if args.profile:
                        cpu_x,cpu_y=(v.pin_memory() for v in batch('train',0,batch_size=64,context=t))
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True) as prof:
                            x,y=cpu_x.cuda(non_blocking=True),cpu_y.cuda(non_blocking=True)
                            with instrumentation(model,opt):
                                run()
                            torch.cuda.synchronize()
                        trace=ROOT/'receipts'/f'{Path(args.output).stem}_{arm}_t{t}_s{s}_trace.json'
                        prof.export_chrome_trace(str(trace))
                        from .profile_summary import summarize
                        record['profile']=summarize(trace)
                        record['profile_trace']=trace.name
                        record['profile_includes_nonblocking_pinned_ingestion']=True
                        record['host_allocation_cpu_ms']=sum(e.self_cpu_time_total for e in prof.events() if e.name in ('aten::empty','aten::empty_strided','aten::empty_like'))/1000
                        lane_allocation=0.
                        for event in prof.events():
                            if event.name not in ('aten::empty','aten::empty_strided','aten::empty_like'):
                                continue
                            parent=event.cpu_parent
                            while parent is not None:
                                if parent.name in ('x36::scan_forward','x36::replay_backward'):
                                    lane_allocation+=event.self_cpu_time_total/1000
                                    break
                                parent=parent.cpu_parent
                        record['scan_replay_host_allocation_cpu_ms']=lane_allocation
                opt._x36_finite.check()
                if arm=='tape_lm':
                    from .backward import reverse_kernel,replay_uses_soa
                    compiled=[c for cache in reverse_kernel.device_caches.values() for c in cache[0].values()]
                    current=compiled[-1]
                    record['reverse_compiler_stats']=dict(warps=current.metadata.num_warps,
                        registers_per_thread=current.n_regs,spills=current.n_spills,
                        shared_bytes=current.metadata.shared,
                        sanitize_overflow=getattr(current.metadata,'sanitize_overflow',None),
                        automatic_overflow_assert_sites=current.asm['ttir'].count('overflow detected'),
                        ptx_assert_calls=current.asm['ptx'].count('__assertfail')-1,
                        ptx_local_load_instructions=current.asm['ptx'].count('ld.local'),
                        ptx_local_store_instructions=current.asm['ptx'].count('st.local'),
                        layout='transposed' if replay_uses_soa(s) else 'oracle_aos')
                    if args.profile and t==512 and s==11776:
                        codegen=[]
                        for kind in ('ttir','ptx'):
                            artifact=f'{Path(args.output).stem}_reverse_s{s}.{kind}'
                            (ROOT/'receipts'/artifact).write_text(current.asm[kind])
                            codegen.append(artifact)
                        record['reverse_codegen_files']=codegen
                    from .kernels import scan_kernel
                    scans=[c for cache in scan_kernel.device_caches.values() for c in cache[0].values()]
                    scan=scans[-1]
                    record['forward_compiler_stats']=dict(warps=scan.metadata.num_warps,
                        registers_per_thread=scan.n_regs,spills=scan.n_spills,
                        shared_bytes=scan.metadata.shared)
                if live:
                    occupancy=live[0].sum(-1).float()
                    record['final_live_min_mean_max']=[occupancy.min().item(),occupancy.mean().item(),occupancy.max().item()]
                if saved_inputs:
                    pure=[]
                    with torch.no_grad():
                        for _ in range(5):
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record()
                            result=model.blocks[2].cell(*saved_inputs[0])
                            end.record();torch.cuda.synchronize()
                            pure.append(start.elapsed_time(end))
                            del result
                    record['pure_cell_forward_raw_ms']=pure
                    record['journal_increment_estimate_ms']=record['component_median_ms']['cell_capture']-statistics.median(pure)
                    record['journal_increment_method']='Matched no-grad cell ablation; not a direct in-kernel timer'
                if args.traffic and arm=='tape_lm':
                    counters={}
                    def count_routes(module,inputs,output):
                        trace=output[2]
                        needed=output[0].grad_fn.needs_input_grad
                        counters['key_grad']=bool(needed[4] or needed[11])
                        counters['allocations']=(trace.alloc_target>=0).to(torch.int64)
                        counters['frees']=torch.empty_like(trace.alloc_target,dtype=torch.int32)
                        count_frees[(trace.alloc_target.numel(),)](trace.freed,counters['frees'],s,
                                                                   triton.next_power_of_2(s))
                        counters['final']=output[1].native.live.sum(-1)
                    traffic_handle=model.blocks[2].cell.register_forward_hook(count_routes)
                    try:
                        run();torch.cuda.synchronize()
                        delta=counters['allocations'].cpu()-counters['frees'].cpu()
                        after=delta.cumsum(1)
                        before=after-delta
                        if not torch.equal(after[:,-1],counters['final'].cpu()) or before.min()<0 or after.max()>s:
                            raise AssertionError('routing occupancy reconstruction failed')
                        pairs=int(before.sum())
                        record['live_traffic']=dict(scope='Separate unmeasured post-profile step; logical accesses, not measured DRAM bytes',
                            initial_live=0,total_live_slot_tokens=pairs,
                            mean_live_per_token=before.double().mean((0,2)).tolist(),
                            final_live_min_max=[int(after[:,-1].min()),int(after[:,-1].max())],
                            allocations=int(counters['allocations'].sum()),frees=int(counters['frees'].sum()),
                            key_adjoint_tracked=counters['key_grad'],
                            forward_payload_logical_bytes=pairs*(64+128)*4,
                            replay_payload_logical_bytes=pairs*(3*128+2*64+(128 if counters['key_grad'] else 0))*4,
                            replay_payload_formula='Native values read; value adjoints read/write; native keys read twice; key adjoints read/write only when tracked')
                    finally:
                        traffic_handle.remove()
            except (torch.cuda.OutOfMemoryError,ValueError,RuntimeError) as exc:
                record.update(status='failed',error=repr(exc))
            finally:
                if handle is not None:handle.remove()
                if input_handle is not None:input_handle.remove()
                results.append(record)
                (ROOT/'receipts'/args.output).write_text(json.dumps(results,indent=2)+'\n')
                print(json.dumps(record),flush=True)
                del model,opt,x,y
                live.clear()
                saved_inputs.clear()
                gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
