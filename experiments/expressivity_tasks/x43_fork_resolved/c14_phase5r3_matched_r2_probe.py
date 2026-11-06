"""Disposable phase-5 parity/resource probe; no certification or training promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from . import c14_b_control_native as native
from .c14_b_step_composition import bits
from .c14_phase2_staged import install_rowkeys
from .c14_phase5r3_split_kernel import kernel
from ..x42_pile_scale.velocity_candidate import Policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('3', '4', '5', '6')
    assert not args.output.exists()
    torch.set_num_threads(1)
    torch.manual_seed(8135)
    install_rowkeys()
    from .c14_phase5r2_selected_kernel import kernel as reference
    policy = Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.)
    rows = []
    class Observer:
        compiled = None
        def __getitem__(self, grid):
            def launch(*a, **kw):
                selected = kernel
                self.compiled = selected[grid](*a, **kw)
                return self.compiled
            return launch
    observer = Observer()
    shapes = ((1,1024),)
    if args.quick:
        shapes = ((1,32),)
    try:
        for b,t in shapes:
            cap=2*t+1
            cfg=SimpleNamespace(**(vars(policy.configs()[0]) | dict(budget=cap,slots=cap,array_append_only=False,decay_rate=1.)))
            xs=[torch.randn(b,t,d,device='cuda') for d in (64,64)]
            xs += [torch.randn(b,1,128,device='cuda').expand(b,t,128).contiguous(),torch.zeros(b,t,128,device='cuda')]
            xs += [torch.zeros(b,t,device='cuda'),torch.full((b,t),20.,device='cuda'),torch.full((b,t),20.,device='cuda')]
            xs[6][:,257:]=-20.
            xs += [torch.zeros((),device='cuda') for _ in range(3)]
            metadata=([[None]*t for _ in range(b)],[b'a'*t]*b,[(j,0) for j in range(b)],[t]*b,[[False]*t for _ in range(b)],[[97]*t for _ in range(b)])
            def run():
                return native.native_forward(xs,cfg,policy.gate_policy(0),metadata)
            native.kernel=reference
            expected=bits(run())
            reference_samples=[]
            for _ in range(5):
                torch.cuda.synchronize(); start=time.monotonic(); run(); torch.cuda.synchronize()
                reference_samples.append((time.monotonic()-start)*1000)
            native.kernel=observer
            actual=bits(run())
            repeated=bits(run())
            row=dict(reference_samples_ms=reference_samples,reference_median_ms=statistics.median(reference_samples),batch=b,tokens=t,cap=cap,exact=expected==actual,repeat_exact=actual==repeated,
                     reference=expected,candidate=actual,registers=observer.compiled.n_regs,spills=observer.compiled.n_spills)
            samples=[]
            for _ in range(5):
                torch.cuda.synchronize(); start=time.monotonic(); run(); torch.cuda.synchronize()
                samples.append((time.monotonic()-start)*1000)
            row.update(samples_ms=samples,median_ms=statistics.median(samples))
            rows.append(row)
            args.output.write_text(json.dumps(dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('c14_phase5r3_split_kernel.py'))},rows=rows),indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('reference','candidate')}),flush=True)
            assert row['exact'] and row['repeat_exact'], 'byte parity failed; retained as failure'
    finally:
        native.kernel=reference

if __name__=='__main__':
    main()
