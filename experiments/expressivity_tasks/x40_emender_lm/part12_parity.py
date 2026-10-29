"""Byte parity of fresh pooled replay against unchanged dense/prefix replay."""
from contextlib import nullcontext
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import torch
import numpy as np
from .model import POLICIES, scan
from ..x36_tape_perf import adjoint_pool

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part12'


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1', '5', '6')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['GPU_LEASE_HELD']
    assert json.loads((OUT / 'support_gate.json').read_text())['status'] == 'passed'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', choices=('random', 'empty', 'popped', 'decay', 'correction'), default='random')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--fused-forward', action='store_true')
    options = parser.parse_args()
    fixture = options.fixture
    from . import backward
    from .part12_decompose import Split
    split = Split() if options.split else None
    torch.set_num_threads(1)
    result = dict(status='failed', fixture=fixture, rows=[], physical_gpu=os.environ['GPU_LEASE_HELD'],
                  scope='B32 fresh native 1.3B head geometry, FP32 replay; not a full-model curve',
                  sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in ('backward.py', 'forward.py', 'part12_parity.py')})
    if split is not None:
        result['sources'].update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ('part12_decompose.py', 'part12_split_generated.py')})
    try:
        for context in (512, 2048):
            for head, cfg in enumerate(POLICIES):
                torch.manual_seed(4012)
                shape = (32, context, 1)
                k = torch.randn((*shape, 64), device='cuda') * .3
                q = torch.randn_like(k) * .3
                v = torch.randn((*shape, 128), device='cuda')
                pop = torch.rand(shape, device='cuda')
                append = torch.randn_like(pop)
                logits = torch.randn_like(pop)
                mask = torch.ones_like(pop)
                mask[:, ::17] = 0
                if fixture != 'random':
                    mask.zero_()
                    pop.zero_()
                    logits.fill_(2.)
                    if fixture == 'popped':
                        mask[:, -2] = 1.
                        pop[:, -1] = 1.
                    elif fixture == 'decay':
                        mask[:, 0] = 1.
                    elif fixture == 'correction':
                        mask[:, :2] = 1.
                baseline = None
                counts = []
                original = adjoint_pool.build
                def build(*args):
                    rows, inverse, sizes = original(*args)
                    counts.extend(sizes.cpu().tolist())
                    return rows, inverse, sizes
                for pooled in (False, True):
                    xs = [x.detach().clone().requires_grad_(True) for x in (q, v, pop, append)]
                    ac = torch.tensor([.03], device='cuda', requires_grad=True)
                    with patch.dict(os.environ, X40_FRESH_POOL=str(int(pooled)), X40_LONG_POOL=str(int(pooled)), X40_BATCHED_READS='0' if pooled and options.fused_forward else os.environ.get('X40_BATCHED_READS', '0')), patch.object(adjoint_pool, 'build', build), patch.object(backward, 'reverse_kernel', split) if pooled and split is not None else nullcontext():
                        (out, st, tr, pressure), e, err, gate = scan(k, xs[0], xs[1], mask, xs[2], xs[3], cfg, ac,
                            logits, torch.tensor([-1.], device='cuda'), torch.tensor([.07], device='cuda'))
                        values = (out, st.native.keys, st.native.values, st.native.strength)
                        loss = sum(x.square().sum() for x in values)
                        grads = torch.autograd.grad(loss, [*xs, ac])
                    observed = {**{f'value_{i}': x.detach().cpu() for i, x in enumerate(values)},
                                **{f'gradient_{i}': x.detach().cpu() for i, x in enumerate(grads)},
                                **{name: getattr(tr, name).detach().cpu() for name in ('case', 'matched', 'alloc_target', 'pop_target', 'pop_fired', 'freed', 'w_on')},
                                'pressure': pressure.detach().cpu(), 'live': st.native.live.cpu(), 'age': st.native.age.cpu()}
                    if not pooled:
                        baseline = observed
                    else:
                        failures = {}
                        for name, x in observed.items():
                            a = np.frombuffer(baseline[name].numpy().tobytes(), dtype=np.uint8)
                            z = np.frombuffer(x.numpy().tobytes(), dtype=np.uint8)
                            if not np.array_equal(a, z):
                                failures[name] = dict(differing_bytes=int((a != z).sum()),
                                    max_abs=float((baseline[name].to(torch.float64) - x.to(torch.float64)).abs().max()))
                        row = dict(context=context, head=head, slots=cfg.slots, batch=32,
                                   union_counts=counts, union_fraction=sum(counts) / (32 * cfg.slots), failures=failures,
                                   status='failed' if failures else 'passed')
                        result['rows'].append(row)
                        print(json.dumps(row), flush=True)
                        assert not failures, row
                    del out, st, tr, pressure, values, grads, loss, xs, ac
                    torch.cuda.empty_cache()
        result['status'] = 'passed'
        if split is not None:
            result['split_wall_seconds'] = split.seconds
            result['split_calls'] = split.calls
            result['split_launches'] = split.launches
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        (OUT / (f'parity_fused_forward_{fixture}.json' if options.fused_forward else f'parity_split_{fixture}.json' if options.split else 'parity.json' if fixture == 'random' else f'parity_{fixture}.json')).write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
