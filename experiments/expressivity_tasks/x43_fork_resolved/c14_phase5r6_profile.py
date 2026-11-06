"""Unchanged R5 component attribution; timing is not physical traffic."""
import json
import sys
from pathlib import Path
import torch
from . import c14_phase5r5_probe as probe

original = probe.kernel


class Profile:
    calls = 0

    def __getattr__(self, name):
        return getattr(original, name)

    def __getitem__(self, grid):
        def launch(*a, **kw):
            self.calls += 1
            if self.calls != 2:
                return original[grid](*a, **kw)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                result = original[grid](*a, **kw)
                torch.cuda.synchronize()
            by = {}
            controls = 0
            gpu = sorted((e for e in prof.events() if e.device_type.name == 'CUDA'),
                         key=lambda e: e.time_range.start)
            for e in gpu:
                name = e.name
                if name == 'control':
                    name = 'pre_control' if controls % 2 == 0 else 'post_control'
                    controls += 1
                row = by.setdefault(name, dict(calls=0, device_us=0.))
                row['calls'] += 1
                row['device_us'] += e.device_time_total
            assert controls == 2 * kw['T']
            launch_names = {e.name for e in prof.events() if 'LaunchKernel' in e.name}
            launches = [e for e in prof.events() if e.name in launch_names]
            trace = a[21].cpu().reshape(grid[0], kw['T'], 18)
            assert (trace[:, :, [0, 1, 2, 7, 12, 14]] == 0).all()
            live = torch.cat((torch.zeros(grid[0], 1, dtype=trace.dtype),
                              trace[:, :-1, 15]), dim=1)
            assert live.tolist()[0] == [min(t, 257) for t in range(kw['T'])]
            output.write_text(json.dumps(dict(
                shape=dict(batch=grid[0], T=kw['T'], S=kw['S'], BS=kw['BS']),
                kernels=by, observed_live_slot_visits=int(live.sum()),
                observed_peak=int(live.max()),
                kernel_sum_us=sum(e.device_time_total for e in gpu),
                kernel_span_us=max(e.time_range.end for e in gpu) - min(e.time_range.start for e in gpu),
                cpu_launch_names=sorted(launch_names),
                cpu_launch_calls=len(launches) if launches else None,
                cpu_launch_total_us=sum(e.cpu_time_total for e in launches) if launches else None,
                scope='One warmed, CUPTI-instrumented forward store call. Inputs are preprojected. '
                      'No projection or backward occurs. Pre/post include value updates and metadata; '
                      'READ includes gather/carries, not separately resolved. Span-minus-sum is device idle '
                      'time inside this instrumented span, not uninstrumented launch overhead. '
                      'No physical traffic measurement.'), indent=2) + '\n')
            return result
        return launch


if __name__ == '__main__':
    output = Path(sys.argv[1])
    assert not output.exists()
    probe.kernel = Profile()
    sys.argv = [sys.argv[0], '--only1024', '--matched-r4', '--output',
                str(output.with_name(output.stem + '_sweep.json'))]
    probe.main()
