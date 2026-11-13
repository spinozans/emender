"""Speed and state-traffic measurement for the chunked parallel scan.

Compares three paths on identical inputs and one identical decision schedule
(the schedule is routed by the certified X31 fused kernel, so it is a real
model-produced decision sequence rather than a synthetic pattern):

* ``certified`` : ``x31_tape_cell_fused.cell_fused.tape_scan`` -- the Triton
                  kernel the X40/X42 arms run, sequential in T, one program per
                  (batch, head);
* ``sequential``: ``sequential_replay.sequential_tape_scan`` -- the same
                  semantics as the chunked scan, per-token PyTorch, one launch
                  per token;
* ``chunked``   : ``chunked_cell.chunked_tape_scan`` at several chunk lengths.

``state_bytes`` is the analytic slot-store traffic ``rows * tokens * slots *
(key_dim + value_dim) * 4`` for the sequential path and ``rows * ceil(tokens /
chunk) * slots * (key_dim + value_dim) * 4`` for the chunked one: the chunk
streams the chunk-entry store once instead of once per token.  Reporting bytes
next to wall time separates the serialisation win from the bandwidth win.

Usage::

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x44_chunk_scan.bench_chunk_scan
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig
from experiments.expressivity_tasks.x31_tape_cell_fused import cell_fused as x31
from experiments.expressivity_tasks.x44_chunk_scan import chunked_cell, sequential_replay
from experiments.expressivity_tasks.x44_chunk_scan.schedule import DecisionSchedule

# Unit of the X42 production arms: batch 64 x context 2048 through four heads
# with budgets 117/311/826/2196 (velocity_production_v1/g10 policy).
PRODUCTION_SHAPES = (
    ('head0_t2048_b64_s117', 2048, 64, 117),
    ('head1_t2048_b64_s311', 2048, 64, 311),
    ('head2_t2048_b64_s826', 2048, 64, 826),
    ('head3_t2048_b64_s2196', 2048, 64, 2196),
    ('short_t256_b64_s2196', 256, 64, 2196),
    ('short_t256_b64_s117', 256, 64, 117),
    ('batch8_t2048_s2196', 2048, 8, 2196),
)


def inputs(rows, tokens, slots, device='cuda', seed=0):
    """fp32 inputs in the flattened ``[rows, tokens, dim]`` layout."""
    cfg = TapeHeadConfig(n_heads=1, budget=slots, decay_rate=0.999,
                         pop_threshold=0.225, free_threshold=1e-3)
    torch.manual_seed(seed)
    k = torch.randn(rows, tokens, cfg.key_dim, device=device)
    q = torch.randn(rows, tokens, cfg.key_dim, device=device)
    v = torch.randn(rows, tokens, cfg.value_dim, device=device)
    beta = (torch.rand(rows, tokens, device=device) > .3).float()
    pop = torch.rand(rows, tokens, device=device)
    logit = torch.randn(rows, tokens, device=device)
    return cfg, k, q, v, beta, pop, logit


def routed_schedule(cfg, k, q, v, beta, pop, logit):
    """Run the certified kernel once and read its routing decisions back.

    The returned schedule is the real decision channel of that kernel: append
    target, correction target, pop target and magnitude, write magnitude.  The
    append gate is ``sigmoid(append_logit)`` (the X31 kernel's gate, with a zero
    write-error coefficient).
    """
    rows, tokens = k.shape[:2]
    out, state, trace, _ = x31.tape_scan(
        k.unsqueeze(2), q.unsqueeze(2), v.unsqueeze(2), beta.unsqueeze(2),
        pop.unsqueeze(2), logit.unsqueeze(2), None, cfg)
    schedule = DecisionSchedule(
        alloc=trace.alloc_target.squeeze(2).contiguous(),
        gate=torch.sigmoid(logit),
        correct=trace.matched.squeeze(2).contiguous(),
        beta=beta,
        pop_slot=torch.where(trace.pop_fired.squeeze(2), trace.pop_target.squeeze(2),
                             torch.full_like(trace.pop_target.squeeze(2), -1)),
        pop_factor=torch.where(trace.pop_fired.squeeze(2), 1.0 - pop,
                               torch.ones_like(pop)))
    schedule.check(rows, tokens, cfg.slots).fail_on_collision()
    statistics_of = dict(
        append=float((schedule.alloc >= 0).float().mean()),
        correct=float((schedule.correct >= 0).float().mean()),
        pop=float((schedule.pop_slot >= 0).float().mean()),
        traced_deaths=float((trace.freed != 0).float().mean()))
    return schedule, statistics_of


def timed(fn, repeats):
    torch.cuda.synchronize()
    fn()                                    # warm-up (Triton JIT, allocator)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times, result


def state_bytes(rows, tokens, slots, dims, chunk=None):
    passes = tokens if chunk is None else math.ceil(tokens / chunk)
    return rows * passes * slots * dims * 4


def measure_case(name, tokens, rows, slot_count, repeats, chunks, device='cuda'):
    cfg, k, q, v, beta, pop, logit = inputs(rows, tokens, slot_count, device)
    schedule, stats = routed_schedule(cfg, k, q, v, beta, pop, logit)
    dims = cfg.key_dim + cfg.value_dim
    record = {'name': name, 'tokens': tokens, 'rows': rows, 'slots': slot_count,
              'event_rates': stats, 'repeats': repeats, 'paths': {}}

    certified_times, _ = timed(
        lambda: x31.tape_scan(k.unsqueeze(2), q.unsqueeze(2), v.unsqueeze(2),
                              beta.unsqueeze(2), pop.unsqueeze(2), logit.unsqueeze(2),
                              None, cfg), repeats)
    sequential_times, sequential_read, = (lambda t: (t, None))(timed(
        lambda: sequential_replay.sequential_tape_scan(k, q, v, schedule, cfg),
        repeats)[0])
    _, (reference_read, reference_final) = timed(
        lambda: sequential_replay.sequential_tape_scan(k, q, v, schedule, cfg), 1)
    for label, times, traffic in (
            ('certified_x31_kernel', certified_times,
             state_bytes(rows, tokens, slot_count, dims)),
            ('sequential_replay', sequential_times,
             state_bytes(rows, tokens, slot_count, dims))):
        record['paths'][label] = {
            'median_seconds': statistics.median(times),
            'mean_seconds': statistics.mean(times),
            'min_seconds': min(times),
            'tokens_per_second': rows * tokens / statistics.median(times),
            'state_bytes': traffic,
            'state_gigabytes_per_second': traffic / statistics.median(times) / 1e9,
        }
    for chunk in chunks:
        death_log = []
        times, (read, final) = timed(
            lambda chunk=chunk: chunked_cell.chunked_tape_scan(
                k, q, v, schedule, cfg, chunk_size=chunk, row_block=rows,
                deaths=death_log),
            repeats)
        read_error = (read - reference_read).abs()
        worst = max(read_error.max().item(),
                    (final.strength - reference_final.strength).abs().max().item(),
                    (final.values - reference_final.values).abs().max().item())
        traffic = state_bytes(rows, tokens, slot_count, dims, chunk)
        record['paths'][f'chunked_c{chunk}'] = {
            'median_seconds': statistics.median(times),
            'mean_seconds': statistics.mean(times),
            'min_seconds': min(times),
            'tokens_per_second': rows * tokens / statistics.median(times),
            'state_bytes': traffic,
            'state_gigabytes_per_second': traffic / statistics.median(times) / 1e9,
            'max_abs_deviation_vs_sequential': worst,
            'median_abs_deviation_vs_sequential': read_error.median().item(),
        }
    return record


def bandwidth_probe(device='cuda', megabytes=512, repeats=5):
    """Achievable device copy bandwidth, the ceiling for a state-traffic bound."""
    source = torch.empty(megabytes * 1024 * 1024 // 4, device=device)
    best = float('inf')
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        source.clone()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    # clone touches the source once and the destination once
    return 2 * megabytes * 1024 * 1024 / best / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--chunks', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--shapes', nargs='*', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = [shape for shape in PRODUCTION_SHAPES
              if args.shapes is None or shape[0] in args.shapes]
    result = {'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
              'copy_bandwidth_gigabytes_per_second': bandwidth_probe(device),
              'cases': []}
    for name, tokens, rows, slots in shapes:
        record = measure_case(name, tokens, rows, slots, args.repeats, args.chunks, device)
        result['cases'].append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'paths'}))
        for path, values in record['paths'].items():
            print(f"  {path:22s} {values['median_seconds']*1e3:9.2f} ms "
                  f"{values['tokens_per_second']:12.1f} tok/s "
                  f"{values['state_bytes']/1e9:8.2f} GB "
                  f"{values['state_gigabytes_per_second']:7.1f} GB/s"
                  + (f"  dev {values['max_abs_deviation_vs_sequential']:.3e}"
                     if 'max_abs_deviation_vs_sequential' in values else ''))
    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps({'copy_bandwidth_gigabytes_per_second':
                      result['copy_bandwidth_gigabytes_per_second']}))


if __name__ == '__main__':
    main()
