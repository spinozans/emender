"""Fail-closed prerequisite probe; never enables a sparse optimization.

B32, native 1.3B head geometry, fresh state. Counterexamples are native state
witnesses, not evidence that entry/allocation-union pooling itself is incorrect.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import torch

from .forward import tape_scan
from .model import POLICIES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part11'


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1', '5', '6')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['GPU_LEASE_HELD']
    # This probes native state semantics only, without journals or backward.
    assert os.environ.get('X40_BATCHED_READS', '0') == '0'
    torch.set_num_threads(1)
    torch.manual_seed(4011)
    OUT.mkdir(exist_ok=True)
    result = dict(status='error', physical_gpu=os.environ['GPU_LEASE_HELD'],
                  python=sys.executable, batch=32, rows=[],
                  scope=__doc__, production_changed=False,
                  sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in ('part11_support_gate.py', 'forward.py',
                                        'backward.py', 'kernels.py', 'model.py')})
    try:
        with torch.no_grad():
            for context in (512, 2048):
                for head, cfg in enumerate(POLICIES):
                    for fixture in ('never_written', 'popped', 'decay'):
                        shape = (32, context, 1)
                        k = torch.ones((*shape, cfg.key_dim), device='cuda') * .25
                        q = torch.zeros_like(k)
                        v = torch.ones((*shape, cfg.value_dim), device='cuda') * .5
                        beta = torch.zeros(shape, device='cuda')
                        pop = torch.zeros_like(beta)
                        logit = torch.zeros_like(beta)
                        if fixture == 'popped':
                            beta[:, -2] = 1.
                            pop[:, -1] = 1.
                        elif fixture == 'decay':
                            beta[:, 0] = 1.
                        out, state, trace, pressure = tape_scan(
                            k, q, v, beta, pop, logit, None, cfg,
                            wire_logits=torch.ones_like(beta),
                            match_coef=torch.zeros(1, device='cuda'),
                            error_coef=torch.zeros(1, device='cuda'),
                            evidence=torch.empty_like(beta),
                            error_evidence=torch.empty_like(beta))
                        torch.cuda.synchronize()
                        n = state.native
                        written = torch.zeros_like(n.live)
                        a = trace.alloc_target.transpose(1, 2).reshape(32, -1)
                        extended = torch.zeros((32, cfg.slots + 1),
                                               device='cuda', dtype=torch.bool)
                        extended.scatter_(1, torch.where(a >= 0, a, cfg.slots), True)
                        written.copy_(extended[:, :cfg.slots].reshape_as(written))
                        dead = ~n.live
                        payload_nonzero = (n.keys != 0).any(-1) | (n.values != 0).any(-1)
                        dead_written = dead & written
                        bad_payload = dead_written & payload_nonzero
                        bad_strength = dead_written & (n.strength != 0)
                        never_bad = (~written) & (payload_nonzero | (n.strength != 0))
                        row = dict(context=context, head=head, slots=cfg.slots,
                                   key_dim=cfg.key_dim, value_dim=cfg.value_dim,
                                   fixture=fixture, live=int(n.live.sum()),
                                   allocations=int((a >= 0).sum()),
                                   pop_events=int(trace.pop_fired.sum()),
                                   decay_events=int((trace.freed == 2).sum()),
                                   dead_written=int(dead_written.sum()),
                                   never_written_nonzero=int(never_bad.sum()),
                                   dead_written_nonzero_payload=int(bad_payload.sum()),
                                   dead_written_nonzero_strength=int(bad_strength.sum()))
                        witnesses = (bad_payload | bad_strength | never_bad).nonzero()
                        if len(witnesses):
                            idx = tuple(witnesses[0].tolist())
                            row['witness'] = dict(index=list(idx), live=bool(n.live[idx]),
                                written=bool(written[idx]), strength=float(n.strength[idx]),
                                strength_bits=int(n.strength.view(torch.int32)[idx]),
                                first_key=float(n.keys[idx][0]),
                                first_value=float(n.values[idx][0]))
                        result['rows'].append(row)
                        print(json.dumps(row), flush=True)
                        del out, state, trace, pressure, n, k, q, v, beta, pop, logit
                        torch.cuda.empty_cache()
        counterexamples = [r for r in result['rows'] if 'witness' in r]
        result.update(status='failed' if counterexamples else 'passed',
                      counterexample_count=len(counterexamples),
                      decision='STOP: no sparse skip or production change authorized by this probe'
                      if counterexamples else 'Prerequisites sampled only; parity still required')
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        (OUT / 'support_gate.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
