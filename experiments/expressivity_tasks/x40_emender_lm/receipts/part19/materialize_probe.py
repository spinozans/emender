"""Isolated X31 materialization operand proof; process-local candidate only."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
import triton
import triton.language as tl
from experiments.expressivity_tasks.x31_tape_cell_fused import kernels

assert sys.executable == '/home/erikg/emender/.venv/bin/python'
assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3', '4')
p = argparse.ArgumentParser()
p.add_argument('--masked', action='store_true')
p.add_argument('--control', action='store_true')
p.add_argument('--record', action='store_true')
a = p.parse_args()

@triton.jit
def initialize_birth(Birth, Live, N: tl.constexpr, Block: tl.constexpr):
    x = tl.program_id(0)*Block + tl.arange(0, Block)
    live = tl.load(Live+x, x<N, 0)
    tl.store(Birth+x, 3, (x<N) & live)

if a.masked:
    source = kernels.materialize_state.src
    old = 'born=tl.load(Birth+n*S+slots,valid,Clock)'
    assert source.count(old) == 1
    kernels.materialize_state._unsafe_update_src(source.replace(old,
        'born=tl.load(Birth+n*S+slots,valid & live,Clock)'))
    kernels.materialize_state.hash = None

rows = []
for batch, slots in ((8, 2944), (64, 64), (64, 256), (64, 736), (128, 1472)):
    for tile_mode in ('mixed', 'empty', 'full'):
        if a.control and (batch, slots, tile_mode) != (64, 64, 'mixed'):
            continue
        shape = (batch, slots)
        live = torch.zeros(shape, device='cuda', dtype=torch.bool)
        if tile_mode == 'mixed':
            live[:, ::31] = True
        elif tile_mode == 'full':
            live.fill_(True)
        # Unwritten dead Birth lanes model the actual allocation lifecycle.
        birth = torch.empty(shape, device='cuda', dtype=torch.int64)
        initialize_birth[(triton.cdiv(batch*slots, 256),)](birth, live, batch*slots, 256)
        ref = torch.full(shape, .75, device='cuda')
        touch = torch.full(shape, 7, device='cuda', dtype=torch.int64)
        strength = torch.full(shape, -99., device='cuda')
        age = torch.full(shape, -99, device='cuda', dtype=torch.int64)
        dummy = torch.empty(1, device='cuda', dtype=torch.int64)
        if a.record:
            capacity = 2*slots
            addresses = torch.empty(batch, capacity, device='cuda', dtype=torch.uint64)
            before = torch.empty_like(addresses)
            widths = torch.empty(batch, capacity, device='cuda', dtype=torch.int32)
            counts = torch.zeros(batch, device='cuda', dtype=torch.int64)
        else:
            capacity = 0
            addresses = before = widths = counts = dummy
        kernels.materialize_state[(batch,)](ref, touch, birth, live, strength, age,
            addresses, before, widths, counts, capacity, a.record, slots, (slots+31)//32,
            11, 1., enable_fp_fusion=False, debug=True, num_warps=4)
        torch.cuda.synchronize()
        assert torch.equal(strength[live], ref[live])
        assert torch.equal(age[live], torch.full_like(age[live], 8))
        assert (strength[~live] == -99.).all()
        assert (age[~live] == -99).all()
        if a.record:
            assert torch.equal(counts, 2*live.sum(1))
        rows.append(dict(batch=batch, slots=slots, mode=tile_mode, status='passed'))
result = dict(status='passed', masked=a.masked, record=a.record, fixtures=rows,
              scope='isolated materialization; not full certification')
tag = ('masked' if a.masked else 'original') + ('_record' if a.record else '')
out = Path(__file__).with_name(f'materialize_{tag}.json')
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
