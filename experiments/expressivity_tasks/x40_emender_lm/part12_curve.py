"""Paired 1.3B ignition-v1 real-data curve and full gradient byte hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import torch
from ..x34_lm_pilot import data
from ..x36_tape_perf.loss import ChunkedLoss
from .scale_surface import SurfaceLM, counts
from .part6_surface import features
from .part6_journal import packed_journals

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part12'


def digest_parameters(model, gradients=False):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        x = p.grad if gradients else p
        digest.update(name.encode())
        if x is not None:
            digest.update(x.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('base', 'candidate'), required=True)
    parser.add_argument('--context', choices=(512, 2048), type=int, required=True)
    a = parser.parse_args()
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1', '5', '6')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['GPU_LEASE_HELD']
    for fixture in ('', '_empty', '_popped', '_decay', '_correction'):
        assert json.loads((OUT / f'parity{fixture}.json').read_text())['status'] == 'passed'
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    result = dict(status='failed', arm=a.arm, context=a.context, rows=[], physical_gpu=os.environ['GPU_LEASE_HELD'],
                  scope='1.3B SurfaceLM ignition-v1, TinyStories, B64/B32 checkpoint, production bf16 dense and FP32 replay',
                  sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('backward.py', 'part12_curve.py', 'model.py', 'scale_surface.py')})
    start = time.perf_counter()
    try:
        model = SurfaceLM('1.3B', 'pure').cuda().train()
        result['parameters'] = counts(model)
        result['initial_parameters_sha256'] = digest_parameters(model)
        baseline = None
        if a.arm == 'candidate':
            baseline = json.loads((OUT / f'curve_base_T{a.context}.json').read_text())
            assert baseline['status'] == 'passed'
            assert result['initial_parameters_sha256'] == baseline['initial_parameters_sha256']
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1, fused=True)
        data_digest = hashlib.sha256()
        with patch.dict(os.environ, X40_FRESH_POOL='1', X40_LONG_POOL=str(int(a.arm == 'candidate'))):
            for step in range((2_000_000 + 64*a.context - 1) // (64*a.context)):
                opt.zero_grad(set_to_none=True)
                x, y = data.batch('train', step, batch_size=64, context=a.context)
                data_digest.update(x.numpy().tobytes())
                data_digest.update(y.numpy().tobytes())
                x, y = x.cuda(), y.cuda()
                total = 0.
                for begin in (0, 32):
                    with packed_journals([]):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            h, gates = features(model, x[begin:begin+32], 'pure', True)
                        loss = (ChunkedLoss.apply(h.float(), model.head.weight, y[begin:begin+32], 1024) + .01*torch.stack(gates).mean()) * .5
                        loss.backward()
                        total += float(loss.detach())
                        del h, gates, loss
                gradient_sha = digest_parameters(model, gradients=True)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                row = dict(step=step+1, tokens=(step+1)*64*a.context, loss=total, norm=float(norm), gradient_sha256=gradient_sha)
                result['rows'].append(row)
                print(json.dumps(row), flush=True)
                if baseline is not None:
                    assert row == baseline['rows'][step], dict(candidate=row, baseline=baseline['rows'][step])
        result.update(status='passed', tokens=result['rows'][-1]['tokens'], data_sha256=data_digest.hexdigest(), final_parameters_sha256=digest_parameters(model))
        if baseline is not None:
            assert result['data_sha256'] == baseline['data_sha256']
            assert result['final_parameters_sha256'] == baseline['final_parameters_sha256']
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        result['seconds'] = time.perf_counter() - start
        (OUT / f'curve_{a.arm}_T{a.context}.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
