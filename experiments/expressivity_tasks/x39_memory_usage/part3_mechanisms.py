"""Fail-closed CPU receipts. These do not authorize GPU training."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ..x30_tape_cell.naive_reference import TapeHeadConfig, init_state, tape_scan
from ..x30_tape_cell.cell import tape_scan as pressure_scan
from .audit import digest, TASKS


def poverty_receipt():
    n, h = 24, 4
    # One write every other token avoids age-1 corrections. Recall follows all
    # writes, so cap merges can destroy bindings that the rich store retains.
    eye = torch.eye(n)
    keys = torch.zeros(1, 3*n, h, n)
    values = torch.zeros_like(keys)
    for i in range(n):
        keys[:, 2*i] = 8*eye[i]
        values[:, 2*i] = eye[i]
        keys[:, 2*n+i] = 8*eye[i]
    beta = torch.zeros(1, 3*n, h)
    beta[:, :2*n:2] = 1
    rows = []
    for budget in (8, 16, 32):
        cfg = TapeHeadConfig(key_dim=n, value_dim=n, n_heads=h, budget=budget)
        v = values.clone().requires_grad_()
        args = (keys, keys, v, beta, torch.zeros_like(beta), torch.full_like(beta, 20.))
        read, final, trace = tape_scan(*args, init_state(1, cfg, 'cpu'), cfg)
        logits = 20*read[:, 2*n:]
        targets = torch.arange(n).view(1,n,1).expand(1,n,h)
        loss = F.cross_entropy(logits.reshape(-1,n), targets.reshape(-1))
        loss.backward()
        with torch.no_grad():
            lazy_read, _, _, pressure = pressure_scan(*args, None, cfg)
        torch.testing.assert_close(read, lazy_read, atol=2e-5, rtol=2e-5)
        occupancy = ((trace.alloc_target >= 0).long() - (trace.freed != 0).sum(-1)).cumsum(1)
        assert torch.equal(occupancy[:, -1], final.live.sum(-1))
        rows.append(dict(budget=budget, peak_live_per_head=occupancy.amax((0,1)).tolist(),
                         routing=trace.counts(), pressure_fraction=float((pressure>0).float().mean()),
                         pressure_max=float(pressure.max()), recall_cross_entropy=float(loss.detach()),
                         recall_accuracy=float((logits.argmax(-1)==targets).float().mean()),
                         value_gradient_norm=float(v.grad.norm()), occupancy=occupancy[0].tolist()))
    assert all(r['routing']['R-CAP'] > 0 and r['pressure_max'] > 0 for r in rows[:2])
    assert all(r['peak_live_per_head'] == [r['budget']]*h for r in rows[:2])
    assert rows[0]['recall_cross_entropy'] > rows[1]['recall_cross_entropy'] > rows[2]['recall_cross_entropy']
    assert rows[2]['routing']['R-CAP'] == 0
    return dict(status='passed', scope='synthetic recall CE through capacity-limited reads; not LM perplexity',
                pressure='actual X30 CPU cell 128-token cap/request register; telemetry is not an added loss', rows=rows)


def semantic_receipt():
    n, h, d = 8, 4, 16
    # Alias identities 0..7 and 8..15 denote the same eight concepts.
    # Shared hand-set W_in folds alias pairs onto the first eight coordinates.
    x = torch.eye(d)
    shared = torch.cat((torch.eye(n), torch.eye(n)), dim=1)
    beta = torch.zeros(1, 2*d, h)
    # W_write(x) = +2 for both presentations; threshold occurs before scan.
    beta[:, ::2] = torch.sigmoid(torch.tensor(2.)).gt(.5).float()
    rows, traces = [], []
    for semantic in (False, True):
        z = F.pad(F.linear(x, shared), (0,n)) if semantic else x
        keys = torch.zeros(1, 2*d, h, d)
        keys[:, ::2] = 8*z.view(1,d,1,d)
        values = torch.zeros_like(keys)
        values[:, ::2, :, :n] = torch.eye(n).repeat(2,1).view(1,d,1,n)
        cfg = TapeHeadConfig(key_dim=d, value_dim=d, n_heads=h, budget=32)
        with torch.no_grad():
            read, _, trace = tape_scan(keys, keys, values, beta, torch.zeros_like(beta),
                                      torch.full_like(beta,20.), init_state(1,cfg,'cpu'), cfg)
        repeats = read[:, 2*n::2, :, :n]
        target = torch.arange(n).view(1,n,1,1).expand(1,n,h,1)
        rows.append(dict(keys='content-sensitive shared hand-set W_in' if semantic else 'identity',
                         repeat_target_read=float(repeats.gather(-1,target).mean()),
                         repeat_write_enabled_fraction=float(trace.w_on[:, 2*n::2].float().mean()),
                         repeat_allocation_fraction=float((trace.alloc_target[:, 2*n::2]>=0).float().mean()),
                         routing=trace.counts()))
        traces.append(trace)
    assert rows[1]['repeat_target_read'] > .99
    assert rows[0]['repeat_target_read'] < .2
    assert torch.equal(traces[0].w_on, traces[1].w_on)
    assert torch.equal(traces[0].alloc_target, traces[1].alloc_target)
    return dict(status='failed', gate='STOP before GPU', rows=rows,
                missing='Content-sensitive keys retrieve stored repeats, but W_write(x) has no input from the current tape read. Shared W_in changes keys/queries only. No repeat-selective gate circuit was established.',
                limits='Hand-set constructive retrieval and counterexample, not a proof that contextual transformer features cannot learn repetition indirectly. No new read-to-gate routing or ISA change was introduced.')


def main():
    torch.set_num_threads(1)
    result = dict(status='failed-semantic-mechanism; stopped-before-GPU', poverty=poverty_receipt(),
                  semantic_repeat=semantic_receipt(), gpu_parity='not-run: fail-closed mechanism gate',
                  training='not-run', evaluation='not-run', source_sha256={p:digest(TASKS/p) for p in (
                      'x30_tape_cell/naive_reference.py','x30_tape_cell/cell.py',
                      'x36_tape_perf/cell_fused.py','x37_head_differentiation/model.py',
                      'x39_memory_usage/part3_mechanisms.py')})
    path = Path(__file__).resolve().parent/'receipts/part3_mechanisms.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','poverty','semantic_repeat')}))
    print(json.dumps({'poverty':result['poverty']['rows'], 'semantic':result['semantic_repeat']}, default=str))


if __name__ == '__main__':
    main()
