"""Receipt-first X34 training instrumentation; no historical source edits."""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import statistics
import time
import torch
from torch.nn import functional as F
from ..x34_lm_pilot import model as x34
from ..x34_lm_pilot.data import batch
from ..x31_tape_cell_fused.backward import Replay

ROOT = Path(__file__).resolve().parent


def write(name, data):
    (ROOT/'receipts'/name).write_text(json.dumps(data, indent=2)+'\n')


@contextmanager
def instrumentation(model):
    replay = Replay
    if type(model.blocks[2].cell).__module__.endswith("x36_tape_perf.cell_fused"):
        from .backward import Replay as replay
    """Stream events avoid introducing boundary synchronizations."""
    records = {}
    handles = []
    def wrap(name, fn):
        def call(*a, **kw):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.profiler.record_function('x36::'+name):
                result = fn(*a, **kw)
            end.record()
            records.setdefault(name, []).append((start, end))
            return result
        return call
    old_trace, old_reverse = x34.trace_metrics, replay.backward
    x34.trace_metrics = wrap('telemetry', old_trace)
    replay.backward = staticmethod(wrap('replay_backward', old_reverse))
    for name, module in [('cell_capture', model.blocks[2].cell), ('read_out', model.blocks[2].read_out)]:
        old = module.forward
        module.forward = wrap(name, old)
        handles.append((module, old))
    try:
        yield records
    finally:
        x34.trace_metrics = old_trace
        replay.backward = staticmethod(old_reverse)
        for module, old in handles:
            module.forward = old


def step(model, opt, x, y):
    opt.zero_grad(set_to_none=True)
    logits, _ = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return loss


def measure(b, samples, variant):
    torch.manual_seed(3401)
    from .model import LM
    model = (x34.LM('tape_lm') if variant == 'original' else LM('tape_lm', optimized_replay=variant == 'optimized')).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95), weight_decay=.1)
    x, y = (v.cuda() for v in batch('train', 0, batch_size=b))
    for _ in range(2):
        step(model, opt, x, y)
    torch.cuda.synchronize()
    raw = []
    saved = []
    hook = model.blocks[2].cell.register_forward_pre_hook(lambda m, a: saved.append((a[0].detach(), a[1])))
    with instrumentation(model) as records:
        for _ in range(samples):
            records.clear()
            torch.cuda.synchronize()
            tic = time.perf_counter()
            step(model, opt, x, y)
            torch.cuda.synchronize()
            row = {'step': (time.perf_counter()-tic)*1000}
            row.update({k: sum(s.elapsed_time(e) for s,e in pairs) for k,pairs in records.items()})
            raw.append(row)
    hook.remove()
    cell_inputs = saved[-1]
    saved.clear()
    opt.zero_grad(set_to_none=True)
    gc.collect()
    pure = []
    with torch.no_grad():
        for _ in range(samples):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = model.blocks[2].cell(*cell_inputs)
            end.record()
            torch.cuda.synchronize()
            pure.append(start.elapsed_time(end))
            del result
    med = {k: statistics.median(r[k] for r in raw) for k in raw[0]}
    rows = {
        'pure_cell_forward_ablation': statistics.median(pure),
        'journal_capture_increment_estimate': med['cell_capture']-statistics.median(pure),
        'replay_backward': med['replay_backward'],
        'trace_metrics': med.get('telemetry', 0.),
        'read_out_forward': med['read_out'],
    }
    rows['everything_else_including_projection_backward_optimizer_and_host_launch_gaps'] = med['step']-sum(rows.values())
    result = dict(variant=variant, batch=b, tokens=512, heads=4, slots=2944, raw_ms=raw,
                  pure_forward_raw_ms=pure, table_ms=rows, step_ms=med['step'],
                  attribution='Journal increment is matched forward ablation; other rows stream events, remainder closes wall step.')
    write(f'{variant}_b{b}.json', result)
    # Profile a separate step, never use profiler timings as wall performance.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        step(model, opt, x, y)
        torch.cuda.synchronize()
    events = prof.events()
    kernels = {}
    syncs = {}
    for event in events:
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name] = kernels.get(event.name, 0)+1
        if any(s in event.name.lower() for s in ('synchronize', 'memcpy dtoh', '_local_scalar_dense')):
            syncs[event.name] = syncs.get(event.name, 0)+1
    result['launch_api_cpu_ms'] = sum(e.cpu_time_total for e in events if e.name in ('cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC'))/1000
    result['kernel_counts'] = kernels
    result['host_sync_events_including_measurement_boundary'] = syncs
    result['cuda_activity_count'] = sum(kernels.values())
    result['lane_token_kernel_count'] = sum(n for k,n in kernels.items() if any(s in k for s in ('read_tiles','reduce_read','scan_kernel')))
    result['launch_note'] = 'CUDA activities counted by profiler; memcpy/memset may also appear. Per-token three read/reduce/mutate launches; reverse_kernel already contains a device token loop.'
    write(f'{variant}_b{b}.json', result)
    print(json.dumps(result), flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant', choices=['original','telemetry','optimized'], default='original')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--samples', type=int, default=5)
    args=p.parse_args()
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):
        raise RuntimeError('broker lease required')
    torch.set_num_threads(1)
    write('environment.json', dict(gpu=torch.cuda.get_device_name(), physical_gpu=os.environ['GPU_LEASE_HELD'], torch=torch.__version__))
    measure(args.batch, args.samples, args.variant)

if __name__ == '__main__':
    main()
