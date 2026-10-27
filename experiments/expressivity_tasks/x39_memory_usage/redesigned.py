"""CPU mechanism checks for the redesigned X39 levers; never GPU certification."""
import json
from pathlib import Path

import torch

from ..x30_tape_cell.naive_reference import TapeHeadConfig, init_state, tape_scan
from .audit import decay_receipt, digest, TASKS


def oracle_trace(write_strength, *, budget=128, decay=None):
    tokens = write_strength.shape[1]
    cfg = TapeHeadConfig(key_dim=4, value_dim=4, n_heads=4, budget=budget,
                        **({} if decay is None else {'decay_rate': decay}))
    gen = torch.Generator().manual_seed(3902)
    k = torch.randn(1, tokens, 4, 4, generator=gen)
    q = torch.randn(1, tokens, 4, 4, generator=gen)
    v = torch.randn(1, tokens, 4, 4, generator=gen)
    with torch.no_grad():
        enabled = (write_strength.detach() > cfg.write_enable_threshold).float()
        _, final, trace = tape_scan(k, q, v, enabled, torch.zeros_like(write_strength),
                                   torch.zeros_like(write_strength), init_state(1, cfg, 'cpu'), cfg)
    allocated = (trace.alloc_target >= 0).long()
    occupancy = (allocated - (trace.freed != 0).sum(-1)).cumsum(1)
    assert torch.equal(occupancy[:, -1], final.live.sum(-1))
    return dict(allocation_rate=float(allocated.float().mean()),
                write_enabled_fraction=float(trace.w_on.float().mean()),
                occupancy=occupancy[0].tolist(), routing=trace.counts(),
                decay_deaths=int((trace.freed == 2).sum()))


def write_gate_receipt():
    """Optimize only the actual sigmoid input, with unchanged oracle thresholds.

    This isolates the indirect controller; it does not claim LM quality or
    content selectivity. Initial +2 is the certified W_write bias.
    """
    rows = []
    features = torch.linspace(-1, 1, 128).reshape(1, 128, 1).expand(1, 128, 4)
    for cost in (0., .001, .01, .1):
        bias = torch.nn.Parameter(torch.tensor(2.))
        optimizer = torch.optim.SGD([bias], lr=10.)
        initial_grad = None
        for _ in range(100):
            optimizer.zero_grad()
            penalty = cost * (features + bias).sigmoid().mean()
            penalty.backward()
            if initial_grad is None:
                initial_grad = float(bias.grad)
            optimizer.step()
        strength = (features + bias.detach()).sigmoid()
        row = oracle_trace(strength)
        row.update(cost=cost, initial_bias=2., final_bias=float(bias.detach()),
                   initial_cost_gradient=initial_grad, mean_sigmoid=float(strength.mean()))
        rows.append(row)
    rates = [r['allocation_rate'] for r in rows]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    assert rates[-1] < rates[0]
    assert rows[0]['write_enabled_fraction'] == 1.
    return dict(status='passed', scope='CPU controller optimization only; no language-model training',
                optimizer='SGD lr=10, 100 steps; fixed features [-1,1]', rows=rows)


def horizon_receipt():
    row = oracle_trace(torch.ones(1, 512, 4), budget=512,
                       decay=decay_receipt()['effective_fp32_decay'])
    tail = row['occupancy'][256:]
    assert row['decay_deaths'] > 0
    assert max(max(x) for x in tail) - min(min(x) for x in tail) <= 1
    return dict(status='passed', horizon=decay_receipt(), trace=row)


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    result = dict(status='partial-mechanism-preflight; training-not-run', batch=64,
                  pilot_steps=611, pilot_tokens=20021248, full_steps=3052,
                  full_tokens=100007936, write_gate=write_gate_receipt(),
                  decay=horizon_receipt(),
                  source_sha256={p: digest(TASKS/p) for p in
                      ('x30_tape_cell/naive_reference.py', 'x34_lm_pilot/model.py',
                       'x36_tape_perf/cell_fused.py', 'x37_head_differentiation/model.py')},
                  remaining=['poverty: prove saturation affects model loss; pressure telemetry alone is insufficient',
                             'semantic keys plus cost: learned repeat-selective gate CPU receipt',
                             'all parameter receipts, fresh GPU parity, ReplayBackward, training and evaluation'])
    target = root/'receipts/redesigned_mechanisms.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'receipt': str(target), 'status': result['status'],
                      'allocation_rates': [r['allocation_rate'] for r in result['write_gate']['rows']]}))


if __name__ == '__main__':
    main()
