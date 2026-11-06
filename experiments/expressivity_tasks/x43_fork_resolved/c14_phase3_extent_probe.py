"""Non-certifying single-stream physical-extent timing for scale projection."""
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch
from . import c14_b_control_native as native
from .c14_phase2_staged import install_rowkeys
from .c14_b_step_composition import bits
from ..x42_pile_scale.velocity_candidate import Policy


def main():
    assert os.environ.get('GPU_LEASE_HELD') in ('3','5','6')
    out = Path(__file__).resolve().parent/'receipts/scalability_phase3/extent_probe.json'
    assert not out.exists()
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    policy = Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.)
    install_rowkeys()
    original = native.kernel
    class Observer:
        compiled = None
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.compiled = original[grid](*args, **kwargs)
                return self.compiled
            return launch
    observer = Observer()
    native.kernel = observer
    rows = []
    try:
        for t, cap in ((257,257),(512,1025),(1024,2049)):
            row = dict(batch=1,tokens=t,cap=cap,peak_live=257,
                       scope='native forward only, zero-value controlled append schedule; repeat test only, no reference/gradient qualification')
            try:
                cfg = SimpleNamespace(**(vars(policy.configs()[0]) | dict(budget=cap,slots=cap,array_append_only=False,decay_rate=1.)))
                xs = [torch.randn(1,t,d,device='cuda') for d in (64,64)]
                xs += [torch.zeros(1,t,d,device='cuda') for d in (128,128)]
                xs += [torch.zeros(1,t,device='cuda'),torch.zeros(1,t,device='cuda'),torch.full((1,t),20.,device='cuda')]
                xs[6][:,257:] = -20.
                xs += [torch.zeros((),device='cuda') for _ in range(3)]
                metadata = ([[None]*t],[b'a'*t],[(0,0)],[t],[[False]*t],[[97]*t])
                def run():
                    return native.native_forward(xs,cfg,policy.gate_policy(0),metadata)
                for _ in range(3):
                    run()
                first = run()
                assert bits(first) == bits(run())
                reports = native.native_reports(first[1][1],cfg,[t])
                assert reports[0]['peak_occupancy'] == 257 and reports[0]['capacity_merges'] == 0
                samples = []
                for _ in range(10):
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    run()
                    torch.cuda.synchronize()
                    samples.append((time.monotonic()-start)*1000)
                row.update(status='timed_repeat_exact_not_certified',samples_ms=samples,median_ms=statistics.median(samples),
                           registers=observer.compiled.n_regs,spills=observer.compiled.n_spills,metadata=str(observer.compiled.metadata))
            except Exception as exc:
                row.update(status='failed',error=repr(exc))
            rows.append(row)
            out.write_text(json.dumps(dict(gpu=os.environ['GPU_LEASE_HELD'],rows=rows),indent=2)+'\n')
            print(json.dumps(row),flush=True)
    finally:
        native.kernel = original
    assert all(r['status'] != 'failed' for r in rows), 'extent probe failures retained'


if __name__ == '__main__':
    main()
