"""Reproducible maintenance and complete-cell timers, with separate scopes.

Maintenance times exclude reads, writes of content, tensor/host transfer, and
chunk snapshots. They must not be interpreted as complete-cell speedups.
"""
import argparse
import json
import os
import platform
import statistics
import time
import torch
from .cell import Management, TapeHeadConfig, tape_scan, from_native
from . import naive_reference
from .train_stage1 import ROOT, save_json, device_for


def timer(fn, iterations, device):
    for _ in range(10):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        begin = time.perf_counter()
        for _ in range(iterations):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        samples.append((time.perf_counter()-begin)/iterations*1e6)
    return statistics.median(samples)


def maintenance(slots, dev):
    cfg = TapeHeadConfig(n_heads=1, budget=slots)
    # A full, inert store with no deaths in the measured interval. Every
    # append request degrades to cap. Native must find newest/first-free
    # and scan decay deaths every token; efficient store uses pointers.
    m = Management(1, cfg)
    for j in range(slots):
        m.add(0, j)
    strength = torch.ones(slots, device=dev)
    age = torch.zeros(slots, dtype=torch.long, device=dev)  # same birth ties as m
    live = torch.ones(slots, dtype=torch.bool, device=dev)
    def naive():
        nonlocal strength, age, live
        free_exists = (~live).any()
        newest = torch.where(live, age, 2**31).argmin()
        strength = torch.where(live, strength*cfg.decay_rate, strength)
        age = age + live.long()
        live = live & ~(live & (strength <= cfg.free_threshold))
        return free_exists, newest
    def efficient():
        free_exists = not m.free[0].full()
        newest = m.newest[0]
        due = m.events.get(m.t+1, ())
        m.t += 1
        return free_exists, newest, due
    a, b = timer(naive, 1000, dev), timer(efficient, 1000, torch.device('cpu'))
    return dict(slots=slots, native_us=a, efficient_us=b, reduction=a/b,
                native_minimum_element_ops_per_token=slots*7,
                efficient_scalar_ops_per_token_upper_estimate=20,
                scope='inert full-store cap maintenance only; no content/read/snapshot/synchronization')


@torch.no_grad()
def complete(heads, slots, dev):
    cfg = TapeHeadConfig(key_dim=16, value_dim=16, n_heads=heads, budget=slots)
    shape = (1, 32, heads)
    k = torch.randn(*shape, 16, device=dev)
    q, v = torch.randn_like(k), torch.randn_like(k)
    beta = torch.ones(*shape, device=dev)
    pop = torch.zeros_like(beta)
    logit = torch.zeros_like(beta)
    def run():
        return tape_scan(k, q, v, beta, pop, logit, None, cfg)
    return dict(heads=heads, slots=slots, batch=1, tokens=32, key=16, value=16,
                us_per_token=timer(run, 3, dev)/32,
                scope='complete forward incl reads/host routing/chunk snapshots; empty initial store')


@torch.no_grad()
def full_store(slots, dev):
    cfg = TapeHeadConfig(key_dim=16, value_dim=16, n_heads=1, budget=slots)
    state = naive_reference.init_state(1, cfg, dev)
    state.keys.normal_()
    state.values.normal_()
    state.strength.fill_(1.)
    state.live.fill_(True)
    state.age[:] = torch.arange(1, slots+1, device=dev)
    efficient_state = from_native(state, cfg)
    shape = (1, 16, 1)
    k = torch.randn(*shape, 16, device=dev)
    q, v = torch.randn_like(k), torch.randn_like(k)
    beta = torch.ones(*shape, device=dev)
    pop = torch.zeros_like(beta)
    logit = torch.zeros_like(beta)
    args = k, q, v, beta, pop, logit
    def native():
        return naive_reference.tape_scan(*args, state, cfg, collect_trace=False)
    def efficient():
        return tape_scan(*args, efficient_state, cfg)
    a, b = timer(native, 3, dev)/16, timer(efficient, 3, dev)/16
    return dict(slots=slots, heads=1, batch=1, tokens=16, key=16, value=16,
                native_us_per_token=a, efficient_us_per_token=b, ratio=a/b,
                scope='complete forward from a full store, includes reads and snapshots; no backward')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output', default=str(ROOT/'runs/cost_cpu.json'))
    args = p.parse_args()
    torch.set_num_threads(1)
    dev = device_for(args.device)
    torch.manual_seed(0)
    result = dict(device=str(dev), torch=torch.__version__, host=platform.node(),
                  physical_gpu=os.environ.get('GPU_LEASE_HELD'),
                  maintenance=[maintenance(s, dev) for s in (736, 11776)],
                  full_cell=[complete(h, 736, dev) for h in (1, 4, 16)],
                  full_store=[full_store(s, dev) for s in (736, 11776)])
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
