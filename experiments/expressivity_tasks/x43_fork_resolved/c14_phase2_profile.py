"""Phase-2 baseline: native timing and compiler resources; no kernel mutation."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch
from . import c14_b_control_native as native
from .c14_launch_shape import install
from .c14_b_step_composition import bits
from ..x42_pile_scale.velocity_candidate import Policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rowkeys', action='store_true', help='explicit layouts and row-addressed key writes')
    p.add_argument('--fixed-t257', action='store_true', help='hold T257 fixed for all four live peaks')
    p.add_argument('--gluon', action='store_true', help='use explicit reference reduction layouts; implies staging')
    p.add_argument('--staged', action='store_true', help='uncertified global-staging candidate')
    a = p.parse_args()
    a.staged = a.staged or a.gluon or a.rowkeys
    assert os.environ.get('GPU_LEASE_HELD') in ('3', '5', '6')
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    provenance = {}
    for name in ('c14_phase2_profile.py', 'c14_phase2_staged_kernel.py', 'c14_phase2_gluon_kernel.py', 'c14_phase2_gluon_rowkeys_kernel.py', 'c14_launch_shape.py', 'c14_b_control_native.py'):
        path = Path(__file__).with_name(name)
        provenance[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (a.output/name).write_bytes(path.read_bytes())
    (a.output/'provenance.json').write_text(json.dumps(dict(sources=provenance, arguments=vars(a) | {'output':str(a.output)}),indent=2)+'\n')
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    original = native.kernel
    original_forward = native.native_forward
    rows = []
    policy = Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.)
    class Variant:
        def __init__(self, warps):
            self.warps = warps
            self.compiled = None
        def __getitem__(self, grid):
            def launch(*args, **kw):
                kw['num_warps'] = self.warps
                target = original
                if a.staged and mode == 'bounded':
                    from .c14_phase2_staged_kernel import kernel as target
                    if a.gluon:
                        from .c14_phase2_gluon_kernel import kernel as target
                    if a.rowkeys:
                        from .c14_phase2_gluon_rowkeys_kernel import kernel as target
                self.compiled = target[grid](*args, **kw)
                return self.compiled
            return launch
    try:
        # Same input generation seed per shape, capacity fixed at 257 for
        # the controlled live-slot curve; include budgeted capacity separately.
        for cap, peak in ((257,48), (257,110), (257,128), (257,257)):
            b, t = 16, 257 if a.fixed_t257 else (128 if peak <= 128 else 257)
            fields = dict(vars(policy.configs()[0]))
            fields.update(budget=cap, slots=cap, array_append_only=False, decay_rate=1.)
            cfg = SimpleNamespace(**fields)
            xs = [torch.randn(b,t,d,device='cuda') for d in (64,64)]
            xs += [torch.zeros(b,t,d,device='cuda') for d in (128,128)]
            xs += [torch.zeros(b,t,device='cuda'), torch.zeros(b,t,device='cuda'), torch.full((b,t),20.,device='cuda')]
            xs[6][:,peak:] = -20.
            xs += [torch.zeros((),device='cuda') for _ in range(3)]
            metadata = ([[None]*t for _ in range(b)], [b'a'*t]*b, [(j,0) for j in range(b)], [t]*b, [[False]*t for _ in range(b)], [[97]*t for _ in range(b)])
            reference = None
            for mode in ('reference', 'bounded'):
                warps = 8
                variant = Variant(warps)
                native.kernel = variant
                native.native_forward = original_forward
                if mode == 'bounded':
                    install()
                def run():
                    return native.native_forward(xs,cfg,policy.gate_policy(0),metadata)
                for _ in range(5):
                    run()
                first = run()
                fingerprint = bits(first)
                repeat = fingerprint == bits(run())
                if reference is None:
                    reference = fingerprint
                exact = fingerprint == reference
                reports = native.native_reports(first[1][1], cfg, [t]*b)
                assert all(r['peak_occupancy'] == peak and r['capacity_merges'] == 0 for r in reports)
                durations = []
                for _ in range(20):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    run()
                    torch.cuda.synchronize()
                    durations.append((time.perf_counter()-start)*1000)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        run()
                    torch.cuda.synchronize()
                (a.output/f'{mode}_{peak}_phases.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
                compiled = variant.compiled
                props = torch.cuda.get_device_properties(0)
                # Resource ceiling, not achieved occupancy: each program is one CTA.
                threads = warps * 32
                register_ctas = props.regs_per_multiprocessor // (compiled.n_regs * threads)
                shared_ctas = props.shared_memory_per_multiprocessor // compiled.metadata.shared
                thread_ctas = props.max_threads_per_multi_processor // threads
                resource_ctas = min(register_ctas, shared_ctas, thread_ctas)
                (a.output/f'{mode}_{peak}.ptx').write_text(compiled.asm['ptx'])
                row = dict(staged=a.staged and mode == 'bounded',mode=mode,cap=cap,peak_live=peak,batch=b,tokens=t,warps=warps,repeat_exact=repeat,reference_exact=exact,
                    native_ms=durations,median_ms=statistics.median(durations),
                    n_regs=compiled.n_regs,n_spills=compiled.n_spills,metadata=str(compiled.metadata),
                    resource_ctas_per_sm=resource_ctas,sm_count=props.multi_processor_count,grid_ctas=b,
                    grid_warp_occupancy_upper_bound=b*threads/(props.multi_processor_count*props.max_threads_per_multi_processor),
                    resource_warp_occupancy_ceiling=resource_ctas*threads/props.max_threads_per_multi_processor,
                    fingerprint=fingerprint,scope='native-only controlled zero-value append probe, fixed T within each pair, decay=1; not training steps/s')
                rows.append(row)
                (a.output/'result.json').write_text(json.dumps(dict(gpu=os.environ['GPU_LEASE_HELD'],rows=rows,source_sha256=hashlib.sha256(inspect.getsource(native).encode()).hexdigest()),indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('fingerprint','native_ms','metadata')}),flush=True)
    finally:
        native.kernel = original
        native.native_forward = original_forward
    assert rows and all(r['reference_exact'] and r['repeat_exact'] for r in rows)

if __name__ == '__main__':
    main()
