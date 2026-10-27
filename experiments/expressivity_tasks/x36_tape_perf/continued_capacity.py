"""Fixed-occupancy arbitrary-slot continuation diagnostic, not an LM ratio gate.

Run through run_leased.sh continued-capacity. State construction and snapshot
are excluded from samples; import/continuation copies inside scan are included.
Production occupancy-sizing transfers remain inside the measured ranges.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

from .cell_fused import tape_scan, TapeHeadConfig
from ..x30_tape_cell.naive_reference import init_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--live', type=int, default=256)
    parser.add_argument('--near-budget', action='store_true', help='Use S-32 live rows at each capacity; occupancy stress, not fixed-live doubling')
    parser.add_argument('--profile', action='store_true', help='Separate untimed continued S11776 profile')
    parser.add_argument('--slots', type=int, nargs='+', default=[2944, 5888, 11776, 8192, 8224])
    parser.add_argument('--output', default='part6_continued_capacity.json')
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        raise ValueError('output must be a receipt filename')
    if os.environ.get('GPU_LEASE_HELD') not in ('0', '1', '5', '6'):
        raise RuntimeError('broker lease required')
    if min(args.batch, args.tokens, args.live, *args.slots) < 1 or args.live > min(args.slots):
        raise ValueError('positive dimensions and live <= slots required')
    torch.set_num_threads(1)
    receipt = dict(scope='Cell-only diagnostic; not the registered whole-model ratio gate',
                   purpose='Fixed live count with holes and live rows at the highest logical slot',
                   sample_cleanup='Drop public outputs and collect garbage outside timing; no private replay-node modification',
                   retention_cause_established=True,
                   retention_scope='Discarded raw-output owner cycle reproduced by the original-production negative control; not a claim that all allocator paths are closed',
                   retention_evidence=['part6_lifetime_original.txt','part6_matrix_parity.txt'],
                   allocation_policy='Production measured entry cardinality reserves journal lifetimes; production prepare allocates exact maximum entry/allocation-union rows, independently of padded kernel lanes',
                   peak_scope='Per-sample process peak during forward/backward, including persistent seed/template and snapshot baseline; validation/cleanup/profile outside samples',
                   imported_host_metadata_transfers_per_chunk=2,
                   host_metadata='Forward packs entry-live maximum and exceptional-cold flag into one transfer; replay reads the union maximum',
                   python=sys.executable, physical_gpu=os.environ['GPU_LEASE_HELD'],
                   warmups=2, samples=5, results=[], gates=[])
    path = Path(__file__).resolve().parent / 'receipts' / args.output
    for slots in args.slots:
        live_count=slots-32 if args.near_budget else args.live
        torch.manual_seed(3605)
        cfg = TapeHeadConfig(n_heads=4, budget=slots, decay_rate=1.)
        native = init_state(args.batch, cfg, 'cuda')
        native.keys.normal_()
        native.values.normal_()
        indices = torch.linspace(0, slots - 1, live_count, device='cuda').long()
        native.live[..., indices] = True
        native.strength[..., indices] = 1.
        native.age[..., indices] = 1
        def inputs(tokens):
            tensors = [torch.randn(args.batch, tokens, 4, d, device='cuda') for d in (64, 64, 128)]
            tensors += [torch.zeros(args.batch, tokens, 4, device='cuda') for _ in range(3)]
            return tensors
        # One actual preceding chunk produces a continued FusedState with
        # persistent metadata. Both modes therefore start from the same bytes.
        with torch.no_grad():
            _, template, _, _ = tape_scan(*inputs(1), native, cfg)
        xs = inputs(args.tokens)
        for x in xs[1:3]:
            x.requires_grad_()
        for mode in ('imported', 'continued'):
            samples = []
            for sample in range(7):
                snapshot = template.snapshot()
                state = snapshot.native if mode == 'imported' else snapshot
                for x in xs:
                    x.grad = None
                start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tic = time.perf_counter()
                start.record()
                out, final, _, _ = tape_scan(*xs, state, cfg)
                node = out.grad_fn
                middle.record()
                out.sum().backward()
                end.record()
                torch.cuda.synchronize()
                measured = dict(step_ms=(time.perf_counter() - tic) * 1000,
                                forward_ms=start.elapsed_time(middle),
                                replay_ms=middle.elapsed_time(end),
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved())
                if not torch.equal(final.native.live, template.native.live):
                    raise AssertionError('fixed live set changed')
                if sample >= 2:
                    samples.append(measured)
                native_gradient_requests=list(node.needs_input_grad[11:14])
                if any(native_gradient_requests):
                    raise AssertionError('capacity control unexpectedly requests native-state gradients')
                pool_stats=getattr(node,'pool_stats',None)
                if pool_stats is None:
                    raise AssertionError('capacity control bypassed the production sparse pool')
                if pool_stats is not None:
                    pool_stats={key:value.cpu().tolist() if isinstance(value,torch.Tensor) else value
                                for key,value in pool_stats.items()}
                    pool_stats['intra_chunk_event_bound']=pool_stats.pop('intra_chunk_events')
                    events=(node.trace[2]>=0).sum(-1).flatten().cpu().tolist()
                    pool_stats['allocation_events_per_store']=events
                    correction_events=((node.trace[0]==4)|(node.trace[0]==5)).sum(-1).flatten().cpu().tolist()
                    pool_stats['correction_events_per_store']=correction_events
                    if any(correction_events):
                        raise AssertionError('fixed-live control corrected a row')
                    pool_stats['dead_value_adjoint_read_write_logical_bytes']=args.batch*4*args.tokens*live_count*cfg.value_dim*8
                    pool_stats['measured_union_bound_per_store']=[min(slots,n+e)
                                                                for n,e in zip(pool_stats['entry_counts'],events)]
                    pool_stats['worst_case_rows_per_store']=[min(slots,n+args.tokens)
                                                           for n in pool_stats['entry_counts']]
                    if pool_stats['allocated_rows'] != max(1,max(pool_stats['union_counts'])):
                        raise AssertionError('production pool allocation is not measured union maximum')
                    if any(n>bound for n,bound in zip(pool_stats['union_counts'],pool_stats['worst_case_rows_per_store'])):
                        raise AssertionError('adjoint union exceeds lifetime bound')
                    if any(events):
                        raise AssertionError('fixed-live control allocated a row')
                del out, final, state, snapshot, node
                gc.collect()
            from .backward import reverse_kernel
            compiled = [c for cache in reverse_kernel.device_caches.values() for c in cache[0].values()]
            kernel = compiled[-1]
            record = dict(mode=mode, slots=slots, batch=args.batch, heads=4,
                          tokens=args.tokens, live=live_count, maximum_live_slot=slots - 1,
                          raw_ms=samples,
                          median_ms={k: statistics.median(s[k] for s in samples) for k in samples[0]},
                          compiler=dict(warps=kernel.metadata.num_warps, registers=kernel.n_regs,
                                        spills=kernel.n_spills, shared_bytes=kernel.metadata.shared),
                          representation='exact-union adjoints and hot replay payloads; native cold backing and logical-S reductions retained',
                          pool=pool_stats,near_budget_stress=args.near_budget,
                          native_gradient_requests=native_gradient_requests,
                          dense_native_unpack_calls=0,
                          dense_unpack_evidence='native_grad returns before unpack when needs_input_grad[11:14] are all false; byte fixtures forbid unpack with a raising mock',
                          allocated_after_samples_bytes=torch.cuda.memory_allocated())
            if args.profile and slots==11776 and mode=='continued':
                for x in xs:
                    x.grad=None
                state=template.snapshot()
                from unittest.mock import patch
                from .backward import Replay
                original_backward=Replay.backward
                def profiled_backward(*a,**kw):
                    with torch.profiler.record_function('x36::replay_backward'):
                        return original_backward(*a,**kw)
                with patch.object(Replay,'backward',staticmethod(profiled_backward)):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        with torch.profiler.record_function('x36::scan_forward'):
                            out,final,_,_=tape_scan(*xs,state,cfg)
                        loss=out.sum()
                        loss.backward()
                        torch.cuda.synchronize()
                trace=path.with_name(path.stem+'_continued_s11776_trace.json')
                prof.export_chrome_trace(str(trace))
                from .profile_summary import summarize
                record['profile']=summarize(trace)
                record['profile_trace']=trace.name
                record['profile_scope']='Separate cell-only diagnostic; backward range on actual autograd thread; loss sum and leaf accumulation outside ranges; not the registered whole-LM ratio gate'
                del out,final,state,loss
                gc.collect()
            receipt['results'].append(record)
            path.write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps(record), flush=True)
        del native, template, xs
        gc.collect()
        torch.cuda.empty_cache()
    for mode in ('imported', 'continued'):
        by_slots = {r['slots']: r for r in receipt['results'] if r['mode'] == mode}
        for a, b in ((2944, 5888), (5888, 11776), (8192, 8224)):
            if a not in by_slots or b not in by_slots:
                continue
            ratios = {key: by_slots[b]['median_ms'][key] / by_slots[a]['median_ms'][key]
                      for key in ('step_ms', 'replay_ms')}
            receipt['gates'].append(dict(mode=mode, slots=[a, b], limit=1.5, ratios=ratios,
                                        occupancy_matched=not args.near_budget,
                                        live_counts=[by_slots[a]['live'],by_slots[b]['live']],
                                        occupied_work_ratio=by_slots[b]['live']/by_slots[a]['live'],
                                        status='passed' if max(ratios.values()) <= 1.5 else 'failed'))
    path.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
