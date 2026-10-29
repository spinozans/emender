"""Fresh-origin union prerequisite; native B32 head shapes, no optimization."""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import torch

from .forward import tape_scan
from .fused_init import fresh_state
from .model import POLICIES
from ..x30_tape_cell.naive_reference import init_state
from ..x31_tape_cell_fused.cell_fused import FusedState

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part12'


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1', '5', '6')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1)
    torch.manual_seed(4012)
    OUT.mkdir(exist_ok=True)
    result = dict(status='error', rows=[], physical_gpu=os.environ['GPU_LEASE_HELD'],
                  python=sys.executable, production_changed=False,
                  scope='Fresh B32 native head fixtures; not trained allocation telemetry or full-model parity',
                  sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in ('part12_support_gate.py', 'forward.py', 'kernels.py', 'fused_init.py', 'model.py')})
    try:
        with torch.no_grad():
            for context in (512, 2048):
                for head, cfg in enumerate(POLICIES):
                    for init in ('torch', 'fused'):
                        for fixture in ('empty', 'popped', 'decay', 'correction', 'mixed'):
                            if init == 'fused':
                                native, history = fresh_state(32, cfg, 'cuda')
                            else:
                                native = init_state(32, cfg, 'cuda')
                                history = torch.zeros(32, 128, device='cuda', dtype=torch.int32)
                            assert all(not x.view(torch.int32).any() for x in (native.keys, native.values, native.strength))
                            state = FusedState(native, history, live_bound=0)
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
                            elif fixture == 'correction':
                                beta[:, :2] = 1.
                                v[:, 1] = .75
                            elif fixture == 'mixed':
                                beta[:, ::2] = 1.
                                pop[:, 3::7] = 1.
                                v[:, 2::4] = .75
                            out, final, trace, pressure = tape_scan(
                                k, q, v, beta, pop, logit, state, cfg, record_all=True,
                                wire_logits=torch.ones_like(beta),
                                match_coef=torch.zeros(1, device='cuda'),
                                error_coef=torch.zeros(1, device='cuda'),
                                evidence=torch.empty_like(beta), error_evidence=torch.empty_like(beta))
                            journal = final.all_field_journal
                            n = final.native
                            touched = torch.zeros((32, cfg.slots), device='cuda', dtype=torch.bool)
                            coverage = {}
                            # Derive touched slots from actual journal addresses, including corrections.
                            for name, tensor in (('keys', n.keys), ('values', n.values), ('strength', n.strength)):
                                width = tensor.shape[-1] if tensor.ndim == 4 else 1
                                seen = torch.zeros(tensor.numel(), device='cuda', dtype=torch.bool)
                                for store in range(32):
                                    addresses = journal.addresses[store, :int(journal.counts[store])].to(torch.int64)
                                    offset = (addresses - tensor.data_ptr()) // 4
                                    valid = (addresses >= tensor.data_ptr()) & (addresses < tensor.data_ptr() + tensor.numel() * 4)
                                    seen[offset[valid]] = True
                                rows = seen.reshape(32, cfg.slots, width).any(-1)
                                touched |= rows
                                bad = (~seen) & (tensor.reshape(-1).view(torch.int32) != 0)
                                coverage[name] = dict(journaled_elements=int(seen.sum()), unjournaled_nonzero_elements=int(bad.sum()))
                            untouched = ~touched
                            payload = (n.keys.view(torch.int32) != 0).any(-1) | (n.values.view(torch.int32) != 0).any(-1)
                            bad = untouched & (payload.reshape_as(touched) | (n.strength.view(torch.int32) != 0).reshape_as(touched))
                            counts = touched.sum(-1)
                            row = dict(context=context, head=head, slots=cfg.slots, batch=32, init=init, fixture=fixture,
                                       union_counts=counts.cpu().tolist(), union_fraction=float(counts.float().mean() / cfg.slots),
                                       never_touched_nonzero=int(bad.sum()), coverage=coverage,
                                       allocations=int((trace.alloc_target >= 0).sum()), corrections=int(((trace.case == 4) | (trace.case == 5)).sum()))
                            # A complete rollback is an independent check for unjournaled mutations.
                            journal.undo_(final)
                            row['rollback_payload_strength_zero_bits'] = all(not x.view(torch.int32).any() for x in (n.keys, n.values, n.strength))
                            result['rows'].append(row)
                            print(json.dumps(row), flush=True)
                            assert not row['never_touched_nonzero'] and row['rollback_payload_strength_zero_bits']
                            assert all(not item['unjournaled_nonzero_elements'] for item in coverage.values())
                            del out, final, trace, pressure, journal, n, state, native, k, q, v, beta, pop, logit
                            torch.cuda.empty_cache()
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        (OUT / 'support_gate.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
