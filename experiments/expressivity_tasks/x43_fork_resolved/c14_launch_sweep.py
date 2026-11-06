"""Read-only launch experiment; production launcher and registrations untouched."""
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
from .c14_b_step_composition import bits
from ..x42_pile_scale.velocity_candidate import Policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('3', '5', '6')
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    original = native.kernel
    rows = []
    policy = Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.)
    class Variant:
        def __init__(self, warps):
            self.warps = warps
            self.compiled = None
        def __getitem__(self, grid):
            def launch(*args, **kw):
                kw['num_warps'] = self.warps
                self.compiled = original[grid](*args, **kw)
                return self.compiled
            return launch
    try:
        # Same input generation seed per shape, capacity fixed at 257 for
        # the controlled live-slot curve; include budgeted capacity separately.
        for cap, peak in ((48,48), (257,48), (257,110), (257,257)):
            b, t = 16, peak
            fields = dict(vars(policy.configs()[0]))
            fields.update(budget=cap, slots=cap, array_append_only=False)
            cfg = SimpleNamespace(**fields)
            xs = [torch.randn(b,t,d,device='cuda') for d in (64,64)]
            xs += [torch.zeros(b,t,d,device='cuda') for d in (128,128)]
            xs += [torch.zeros(b,t,device='cuda'), torch.zeros(b,t,device='cuda'), torch.full((b,t),20.,device='cuda')]
            xs += [torch.zeros((),device='cuda') for _ in range(3)]
            metadata = ([[None]*t for _ in range(b)], [b'a'*t]*b, [(j,0) for j in range(b)], [t]*b, [[False]*t for _ in range(b)], [[97]*t for _ in range(b)])
            reference = None
            for warps in (8,16,32):
                variant = Variant(warps)
                native.kernel = variant
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
                compiled = variant.compiled
                row = dict(cap=cap,peak_live=peak,batch=b,tokens=t,warps=warps,repeat_exact=repeat,reference_exact=exact,
                    native_ms=durations,median_ms=statistics.median(durations),
                    n_regs=compiled.n_regs,n_spills=compiled.n_spills,metadata=str(compiled.metadata),
                    fingerprint=fingerprint,scope='native-only synthetic zero-value append probe; variable sequence length, not a training steps/s curve')
                rows.append(row)
                (a.output/'result.json').write_text(json.dumps(dict(gpu=os.environ['GPU_LEASE_HELD'],rows=rows,source_sha256=hashlib.sha256(inspect.getsource(native).encode()).hexdigest()),indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('fingerprint','native_ms','metadata')}),flush=True)
    finally:
        native.kernel = original

if __name__ == '__main__':
    main()
