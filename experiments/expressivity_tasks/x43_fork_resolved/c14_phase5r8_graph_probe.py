"""Measure identical warmed store launches eagerly and through torch graph replay."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch
from . import c14_phase5r5_probe as probe

original = probe.kernel
output = Path(sys.argv[1])
assert not output.exists()
assert os.environ.get('GPU_LEASE_HELD') in ('3', '4', '5', '6')

class Finished(Exception):
    pass

class Measure:
    def __getattr__(self, name):
        return getattr(original, name)

    def __getitem__(self, grid):
        def launch(*a, **kw):
            def eager():
                return original[grid](*a, **kw)
            for _ in range(3):
                eager()
            torch.cuda.synchronize()
            # Initialization is inside capture: each replay starts from empty state.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                eager()
            graph.replay()
            torch.cuda.synchronize()
            expected = [x.detach().clone() for x in a[16:22]]
            eager()
            torch.cuda.synchronize()
            exact = all(torch.equal(x.view(torch.uint8), y.view(torch.uint8))
                        for x, y in zip(a[16:22], expected))
            samples = {'eager': [], 'graph': []}
            for _ in range(7):
                for name, run in (('eager', eager), ('graph', graph.replay)):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    run()
                    torch.cuda.synchronize()
                    samples[name].append((time.perf_counter()-start)*1000)
            counts = {}
            for name, run in (('eager', eager), ('graph', graph.replay)):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    run()
                    torch.cuda.synchronize()
                events = prof.events()
                counts[name] = dict(device_kernels=sum(e.device_type.name == 'CUDA' for e in events),
                                    cpu_kernel_launches=sum('LaunchKernel' in e.name for e in events),
                                    cpu_graph_launches=sum('GraphLaunch' in e.name for e in events))
            result = dict(shape=dict(B=grid[0],T=kw['T'],S=kw['S'],BS=kw['BS']),
                          samples_ms=samples, median_ms={k:statistics.median(v) for k,v in samples.items()},
                          counts=counts, graph_eager_byte_exact=exact,
                          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          scope='Preprojected forward store only; graph includes initialization and scratch allocations. No backward or model-step timing. NVIDIA tested only; ROCm torch.cuda graph API not validated; no other vendor claim.')
            output.write_text(json.dumps(result,indent=2)+'\n')
            assert exact
            raise Finished()
        return launch

if __name__ == '__main__':
    probe.kernel = Measure()
    sys.argv = [sys.argv[0], '--only1024', '--matched-r4', '--output', str(output.with_suffix('.unused.json')), '--batch', os.environ.get('ROUND8_BATCH', '1')]
    try:
        probe.main()
    except Finished:
        pass
