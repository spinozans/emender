"""Precision budget of the chunk-boundary composition (Lever 4 measurement).

The runtime path carries the chunk-exit state into the next chunk exactly, so no
boundary arithmetic happens at all.  This probe prices the *alternative* the
formulation would need if the chunks were fully parallel: composing the
per-chunk ``(P, I)`` summaries over the chunk axis and materialising each chunk
entry by renormallsing against the composed prefix.  ``affine.prefix_compose``
prices it in fp32 and fp64.

Reported per configuration: max/median deviation from an fp64 exact fold, and
wall time.  Nothing here changes the implementation; it is the cost/benefit of
an option, as required.

Usage::

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x44_chunk_scan.precision_probe
"""
from __future__ import annotations

import json
import statistics
import time

import torch

from . import affine


def exact_fold(multiplier, injection, entry=None):
    """fp64 reference: the plain sequential recurrence, one step at a time."""
    steps, lanes = multiplier.shape
    state = torch.zeros(lanes, dtype=torch.float64, device=multiplier.device)
    if entry is not None:
        state = entry.double().clone()
    states = torch.empty(steps, lanes, dtype=torch.float64, device=multiplier.device)
    for i in range(steps):
        state = multiplier[i].double() * state + injection[i].double()
        states[i] = state
    return states


def carried(multiplier, injection, entry=None, chunk=64):
    """The runtime path: exact chunk-to-chunk carry of the state."""
    steps, lanes = multiplier.shape
    states = torch.empty(steps, lanes, dtype=multiplier.dtype, device=multiplier.device)
    state = torch.zeros(lanes, dtype=multiplier.dtype, device=multiplier.device)
    if entry is not None:
        state = entry.clone()
    for begin in range(0, steps, chunk):
        stop = min(steps, begin + chunk)
        local, _ = affine.scan_with_entry(multiplier[begin:stop], injection[begin:stop],
                                          state)
        states[begin:stop] = local
        state = local[-1]
    return states


def composed(multiplier, injection, entry=None, chunk=64, boundary_dtype=torch.float64):
    """Fully chunk-parallel form: composed summaries, renormalised entries."""
    steps, lanes = multiplier.shape
    chunks = (steps + chunk - 1) // chunk
    pad = chunks * chunk - steps
    if pad:
        multiplier = torch.cat([multiplier, torch.ones(pad, lanes, dtype=multiplier.dtype,
                                                       device=multiplier.device)], 0)
        injection = torch.cat([injection, torch.zeros(pad, lanes, dtype=injection.dtype,
                                                      device=injection.device)], 0)
    shape = (chunks, chunk, lanes)
    states, _ = affine.chunked_prefix_scan(multiplier.view(shape), injection.view(shape),
                                           boundary_dtype=boundary_dtype, entry=entry)
    return states.view(chunks * chunk, lanes)[:steps]


def measure(function, *args, **kwargs):
    function(*args, **kwargs)
    torch.cuda.synchronize()
    times = []
    result = None
    for _ in range(3):
        start = time.perf_counter()
        result = function(*args)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    report = {'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
              'rows': []}
    for slots, decay, tokens, chunk in ((2196, 0.999, 4096, 64), (2196, 0.999, 4096, 16),
                                        (117, 0.999, 4096, 256), (117, 0.5, 4096, 256)):
        lanes = 64 * slots
        # POP/overwrite zeros are part of the data, as in the live scan.
        multiplier = torch.full((tokens, lanes), decay, device=device)
        multiplier[::7] = 0.0
        injection = torch.zeros(tokens, lanes, device=device)
        injection[::7] = 1.0
        truth = exact_fold(multiplier, injection)
        entry = torch.zeros(lanes, device=device)
        row = {'slots': slots, 'decay': decay, 'tokens': tokens, 'chunk': chunk,
               'lanes': lanes}
        for name, function, kwargs in (
                ('carried_exact', carried, {}),
                ('composed_fp64_boundary', composed, {'boundary_dtype': torch.float64}),
                ('composed_fp32_boundary', composed, {'boundary_dtype': torch.float32})):
            seconds, states = measure(lambda f=function, k=kwargs: f(
                multiplier, injection, entry, chunk=chunk, **k))
            deviation = (states.double() - truth).abs()
            scale = truth.abs().max().item()
            row[name] = {
                'seconds': seconds,
                'max_abs_deviation': deviation.max().item(),
                'median_abs_deviation': deviation.median().item(),
                'relative_max': deviation.max().item() / scale,
            }
        report['rows'].append(row)
        print(json.dumps(row))


if __name__ == '__main__':
    main()
