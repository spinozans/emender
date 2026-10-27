"""Fixed-occupancy arbitrary-slot continuation diagnostic, not an LM ratio gate.

Run through run_leased.sh continued-capacity. State construction and snapshot
are excluded from samples; import/continuation copies inside scan are included.
No occupancy query occurs inside a measured range.
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
    parser.add_argument('--slots', type=int, nargs='+', default=[2944, 5888, 11776, 8192, 8224])
    parser.add_argument('--output', default='part5_continued_capacity_entry.json')
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
                   sample_cleanup='After timing, explicitly release discarded replay node owner references; no retained/repeated backward is measured',
                   retention_cause_established=False,
                   python=sys.executable, physical_gpu=os.environ['GPU_LEASE_HELD'],
                   warmups=2, samples=5, results=[], gates=[])
    path = Path(__file__).resolve().parent / 'receipts' / args.output
    for slots in args.slots:
        torch.manual_seed(3605)
        cfg = TapeHeadConfig(n_heads=4, budget=slots, decay_rate=1.)
        native = init_state(args.batch, cfg, 'cuda')
        native.keys.normal_()
        native.values.normal_()
        indices = torch.linspace(0, slots - 1, args.live, device='cuda').long()
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
                                replay_ms=middle.elapsed_time(end))
                if not torch.equal(final.native.live, template.native.live):
                    raise AssertionError('fixed live set changed')
                if sample >= 2:
                    samples.append(measured)
                # These discarded nodes cannot be reused for backward. Release
                # private owner references explicitly outside timing. GC alone
                # did not prevent diagnostic OOM; its cause is not established.
                node.storage = None
                node.address_owners = ()
                node.replay = ()
                del out, final, state, snapshot, node
                gc.collect()
            from .backward import reverse_kernel
            compiled = [c for cache in reverse_kernel.device_caches.values() for c in cache[0].values()]
            kernel = compiled[-1]
            record = dict(mode=mode, slots=slots, batch=args.batch, heads=4,
                          tokens=args.tokens, live=args.live, maximum_live_slot=slots - 1,
                          raw_ms=samples,
                          median_ms={k: statistics.median(s[k] for s in samples) for k in samples[0]},
                          compiler=dict(warps=kernel.metadata.num_warps, registers=kernel.n_regs,
                                        spills=kernel.n_spills, shared_bytes=kernel.metadata.shared),
                          representation='entry baseline: dense unknown-import replay, logical-S reductions')
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
                                        status='passed' if max(ratios.values()) <= 1.5 else 'failed'))
    path.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
