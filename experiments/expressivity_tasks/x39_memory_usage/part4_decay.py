"""Broker-only CPU/X36 decay death-schedule comparison, not training parity."""
import json
import os
from pathlib import Path

import torch

from ..x30_tape_cell.naive_reference import TapeHeadConfig, init_state, tape_scan as oracle
from ..x36_tape_perf.cell_fused import tape_scan
from .audit import TASKS, digest


def main():
    if os.environ.get('GPU_LEASE_HELD') not in ('0', '1', '5', '6'):
        raise RuntimeError('X39 GPU lease required')
    torch.set_num_threads(1)
    cfg = TapeHeadConfig(n_heads=4, budget=2944, decay_rate=0.9474635124206543)
    # Orthogonal, isolated births avoid correction, POP and cap effects.
    k = torch.zeros(1, 512, 4, 64)
    v = torch.zeros(1, 512, 4, 128)
    beta = torch.zeros(1, 512, 4)
    for i, t in enumerate(range(0, 512, 32)):
        k[:, t, :, i] = 8
        v[:, t, :, i] = 1
        beta[:, t] = 1
    inputs = (k, k.clone(), v, beta, torch.zeros_like(beta), torch.full_like(beta, 20.))
    with torch.no_grad():
        _, cpu, ct = oracle(*inputs, init_state(1, cfg, 'cpu'), cfg)
        _, gpu, gt, _ = tape_scan(*(x.cuda() for x in inputs), None, cfg)
    for name in ('freed', 'alloc_target', 'case'):
        assert torch.equal(getattr(ct, name), getattr(gt, name).cpu()), name
    assert torch.equal(cpu.live, gpu.native.live.cpu())
    deaths = (ct.freed == 2).nonzero().tolist()
    assert deaths, 'vacuous parity: no decay deaths'
    result = dict(status='passed', decay_rate_fp32=cfg.decay_rate,
        physical_gpu=os.environ['GPU_LEASE_HELD'], deaths=deaths,
        comparison='bit-exact freed, alloc_target, case and final live against CPU oracle',
        scope='isolated births at B1 H4 K64 V128 S2944 T512; not complete training/configuration parity',
        source_sha256={p:digest(TASKS/p) for p in (
            'x39_memory_usage/part4_decay.py', 'x36_tape_perf/forward.py',
            'x36_tape_perf/kernels.py', 'x30_tape_cell/naive_reference.py')})
    (Path(__file__).parent/'receipts/part4_decay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('deaths','source_sha256')}))


if __name__ == '__main__':
    main()
