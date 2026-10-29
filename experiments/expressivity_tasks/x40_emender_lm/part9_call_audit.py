"""Diagnose the proposed duplicate-forward removal without changing kernels.

This is a B1 single-block call audit, not a sustained scale throughput receipt.
Run only through run_leased.sh. Existing production certification runs first.
"""
import gc
import hashlib
import json
import os
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from . import backward, model
from .parity import certify
from .part6_checkpoint import checkpoint_block
from .scale_surface import MemoryBlock, SHAPES, SurfaceLM
from .probe_rebase import TransposedLaunch
from ..x31_tape_cell_fused import backward as oracle_backward

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part9'


def main():
    held = os.environ.get('GPU_LEASE_HELD')
    assert held in ('1', '5', '6')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == held
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    result = dict(status='failed', physical_gpu=held, python=os.sys.executable,
                  scope='B1 single-block structural audit; no sustained throughput claim',
                  sources={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                           for name in ('model.py', 'backward.py', 'forward.py',
                                        'part6_checkpoint.py', 'part9_call_audit.py')},
                  flags={k: os.environ.get(k, '0') for k in
                         ('X40_BATCHED_READS', 'X40_FRESH_POOL',
                          'X40_FUSED_INIT', 'X40_STABLE_MAPPING')}, rows=[])
    try:
        for context in (512, 2048):
            # Same long-context oracle launch-axis adapter as part5_gate.
            # Raw X31 exceeds CUDA grid.y at T2048; floating arithmetic is unchanged.
            with patch.object(oracle_backward, 'rebase', TransposedLaunch()) if context > 512 else nullcontext():
                gate = certify(batch=1, context=context)
            gate['oracle_launch_adapter'] = 'TransposedLaunch' if context > 512 else None
            (OUT/f'production_T{context}.json').write_text(json.dumps(gate, indent=2)+'\n')
            assert gate['status'] == 'passed' and gate['pressure_gate']
            for scale in ('22M', '1.3B'):
                torch.manual_seed(3401)
                with torch.device('meta'):
                    shape = SurfaceLM(scale, 'pure')
                    hidden = shape.hidden
                del shape
                width = SHAPES[scale][0]
                block = MemoryBlock(width)
                block.ff = torch.nn.Sequential(torch.nn.Linear(width, hidden),
                                               torch.nn.GELU(), torch.nn.Linear(hidden, width))
                block = block.cuda().train()
                x = torch.randn(1, context, width, device='cuda', requires_grad=True)
                ids = torch.randint(1, 32768, (1, context), device='cuda')
                counts = Counter()
                phase = 'forward'
                native_forward = model.wired_forward
                native_reverse = backward.Replay.backward

                def scan(*args, **kwargs):
                    counts[f'{phase}:native_forward'] += 1
                    return native_forward(*args, **kwargs)

                def reverse(*args):
                    nonlocal phase
                    old = phase
                    phase = 'inside_reverse'
                    counts['backward:reverse'] += 1
                    try:
                        return native_reverse(*args)
                    finally:
                        phase = old

                with patch.object(model, 'wired_forward', scan), patch.object(
                        backward.Replay, 'backward', staticmethod(reverse)):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out, gate = checkpoint_block(block, x, ids)
                        loss = out.square().mean() + .01 * gate
                    phase = 'checkpoint_recompute'
                    loss.backward()
                expected = {'forward:native_forward': 4,
                            'checkpoint_recompute:native_forward': 4,
                            'backward:reverse': 4}
                assert dict(counts) == expected, dict(counts)
                assert bool(torch.isfinite(x.grad).all())
                row = dict(scale=scale, width=width, ff_hidden=hidden, batch=1,
                           context=context, counts=dict(counts),
                           native_forwards_inside_reverse=counts['inside_reverse:native_forward'])
                result['rows'].append(row)
                print(json.dumps(row), flush=True)
                del block, x, ids, out, gate, loss
                gc.collect()
                torch.cuda.empty_cache()
        result['status'] = 'passed'
        result['conclusion'] = ('Checkpoint recomputation already records the journal used by '
                                'the reverse. Reverse launches no second native forward.')
    except Exception as exc:
        result['error'] = repr(exc)
        raise
    finally:
        (OUT/'call_audit.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
