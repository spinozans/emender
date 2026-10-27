"""CPU-only counterexamples and analysis of existing, immutable X34 evidence.

Run with the project's approved interpreter as a module. This is a mechanism
preflight, not GPU parity certification or a training runner.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

import torch

from ..x30_tape_cell.naive_reference import (
    TapeHeadConfig, fade_horizon, init_state, tape_scan,
)

ROOT = Path(__file__).resolve().parent
TASKS = ROOT.parent
CONTROL = TASKS / 'x34_lm_pilot/receipts/probe_pilot_tape_lm.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decay_receipt():
    cfg = TapeHeadConfig()
    requested = math.exp(math.log(cfg.free_threshold) / 128)
    fp32 = struct.unpack('<f', struct.pack('<f', requested))[0]
    return dict(free_threshold=cfg.free_threshold, control_decay=cfg.decay_rate,
                control_horizon=fade_horizon(cfg), requested_decay=requested,
                effective_fp32_decay=fp32,
                effective_horizon=math.log(cfg.free_threshold) / math.log(fp32),
                caveat='Untouched strength-1 slot lifetime; refresh and POP can change it.')


def slope(lengths, peaks):
    if len(lengths) != len(peaks) or len(lengths) < 2:
        raise ValueError('need paired lengths and peaks')
    if any(x <= 0 for x in lengths) or any(p <= 0 for p in peaks):
        raise ValueError('log slope undefined for nonpositive lengths/occupancy')
    x, y = list(map(math.log, lengths)), list(map(math.log, peaks))
    xm, ym = statistics.mean(x), statistics.mean(y)
    denominator = sum((v-xm)**2 for v in x)
    if denominator == 0:
        raise ValueError('need distinct context lengths')
    return sum((a-xm)*(b-ym) for a,b in zip(x,y)) / denominator


def control_receipt():
    source = json.loads(CONTROL.read_text())
    rows = []
    for row in source['context']:
        traces = [window[0] for window in row['telemetry']]
        peak = [max(t['peak_live_per_head'][h] for t in traces) for h in range(4)]
        routing = {name: sum(t['routing'][name] for t in traces)
                   for name in traces[0]['routing']}
        total = sum(routing.values())
        allocations = sum(routing[n] for n in ('R-EMPTY', 'R-NOMATCH', 'R-ZERO-RESIDUAL'))
        rows.append(dict(context=row['context'], windows=row['windows'],
                         peak_live_per_head=peak, mean_live_per_head=None,
                         occupancy_profile=None, gate_strength=None,
                         write_enabled_fraction=1-routing['NO-WRITE']/total,
                         successful_allocations_per_token_head=allocations/total,
                         append_requests_per_token_head=(allocations+routing['R-CAP'])/total,
                         pressure_fraction_per_window=[t['pressure_fraction'] for t in traces],
                         routing=routing, perplexity=row['perplexity']))
    lengths = [r['context'] for r in rows]
    if lengths != [512, 1024, 2048, 4096]:
        raise ValueError('control context grid changed')
    story_peaks = [r['telemetry'][0]['peak_live_per_head'][0] for r in source['stories']]
    return dict(source=str(CONTROL.relative_to(TASKS)), sha256=digest(CONTROL),
                tokens=source['tokens'], rows=rows,
                slope_per_head=[slope(lengths, [r['peak_live_per_head'][h] for r in rows])
                                for h in range(4)],
                story_peak_median=statistics.median(story_peaks),
                story_peak_max=max(story_peaks),
                missing=['mean live occupancy', 'per-token occupancy profile', 'gate strength'],
                interpretation='Transcript growth, already measured through T=4096; no retraining.')


def scan_example(logit=0., repeated_keys=False, zero_values=False, decay=None, tokens=32):
    """Small adversarial oracle example; fixed controls isolate one mechanism."""
    cfg = TapeHeadConfig(key_dim=4, value_dim=4, n_heads=4, budget=64,
                         **({} if decay is None else dict(decay_rate=decay)))
    gen = torch.Generator().manual_seed(3901)
    k = torch.randn(1, tokens, 4, 4, generator=gen)
    if repeated_keys:
        k = k[:, :1].expand_as(k).clone()
    q = torch.randn(1, tokens, 4, 4, generator=gen)
    v = torch.randn(1, tokens, 4, 4, generator=gen)
    if zero_values:
        v.zero_()
    beta = torch.ones(1, tokens, 4)
    pop = torch.zeros_like(beta)
    logits = torch.full_like(beta, logit, requires_grad=True)
    read, final, trace = tape_scan(k,q,v,beta,pop,logits,init_state(1,cfg,'cpu'),cfg)
    return read, final, trace, logits


def mechanism_receipt():
    low = scan_example(logit=-20.)
    high = scan_example(logit=20.)
    keys = scan_example(repeated_keys=True)
    normal = scan_example()
    zeros = scan_example(zero_values=True)
    same_gate_allocations = torch.equal(low[2].alloc_target, high[2].alloc_target)
    same_key_routing = torch.equal(normal[2].case, keys[2].case)
    physical_rate = (low[2].alloc_target >= 0).float().mean()
    # The hard allocation trace is integral and disconnected from gate logits.
    assert not physical_rate.requires_grad
    grad = torch.autograd.grad(low[0].square().sum(), low[3])[0]
    assert grad.abs().sum() > 0
    assert same_gate_allocations and same_key_routing
    zero_rate = (zeros[2].alloc_target >= 0).float().mean().item()
    assert zero_rate == 1.
    return dict(status='blocked', oracle='x30 naive_reference, CPU; not X36 GPU parity',
                gate_extremes=[-20.,20.], gate_extremes_same_allocation_trace=same_gate_allocations,
                allocation_rate_requires_grad=physical_rate.requires_grad,
                allocation_rate=float(physical_rate),
                gate_logit_read_gradient_nonzero=bool(grad.abs().sum() > 0),
                repeated_vs_random_keys_same_routing=same_key_routing,
                zero_residual_allocation_rate=zero_rate,
                zero_residual_routing=zeros[2].counts(),
                blockers=[
                    'Append gate is differentiable value amplitude, not allocation probability.',
                    'Write matching is age-1 adjacency; keys do not implement novelty routing.',
                    'R-ZERO-RESIDUAL appends instead of skipping allocation.',
                    'Requested batch 64 differs from receipted X34 effective batch 8.',
                ])


def main():
    torch.set_num_threads(1)
    sources = ['x30_tape_cell/naive_reference.py', 'x31_tape_cell_fused/cell_fused.py',
               'x34_lm_pilot/model.py', 'x34_lm_pilot/cli.py',
               'x36_tape_perf/cell_fused.py', 'x36_tape_perf/kernels.py',
               'x37_head_differentiation/model.py']
    result = dict(status='blocked-before-training', training_runs=0, gpu_runs=0,
                  preregistration_sha256=digest(ROOT/'RECEIPTS.md'),
                  source_sha256={p:digest(TASKS/p) for p in sources},
                  decay=decay_receipt(), control=control_receipt(),
                  mechanism=mechanism_receipt())
    output = ROOT/'receipts/preflight.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],
                         control_slope=result['control']['slope_per_head'],
                         decay=result['decay'], evidence=str(output))))


if __name__ == '__main__':
    main()
