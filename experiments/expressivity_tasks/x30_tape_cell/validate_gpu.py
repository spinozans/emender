"""Small leased-device native/effective parity receipt (no training)."""
import argparse
import json
import os
import torch
from . import naive_reference as naive
from .cell import from_native, tape_scan
from .routing_trace import assert_traces_equal
from .train_stage1 import ROOT, device_for, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output', default=str(ROOT/'runs/parity_gpu.json'))
    args = p.parse_args()
    torch.set_num_threads(1)
    dev = device_for(args.device)
    rows = []
    for budget, heads in ((3, 1), (11, 4), (736, 16), (11776, 1)):
        for seed in (0, 7):
            torch.manual_seed(seed)
            cfg = naive.TapeHeadConfig(key_dim=8, value_dim=8, n_heads=heads,
                                        budget=budget, decay_rate=.8, free_threshold=.15)
            shape = (1, 7, heads)
            k = torch.randn(*shape, 8, device=dev)
            q = torch.randn_like(k, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            beta = (torch.rand(*shape, device=dev) > .3).float()
            pop = torch.rand(*shape, device=dev, requires_grad=True)
            logit = torch.randn(*shape, device=dev, requires_grad=True)
            coef = torch.randn(heads, device=dev, requires_grad=True)*.1
            init = naive.init_state(1, cfg, dev)
            init.keys.normal_()
            init.values.normal_()
            init.strength.uniform_(.3, 1.)
            init.live = torch.rand_like(init.strength) > .3
            init.age.random_(0, 4)
            source = (k, q, v, beta, pop, logit)
            expected, last, trace = naive.tape_scan(*source, init, cfg, append_err_coef=coef)
            loss_n = expected.square().mean() + last.values.square().mean()*.1 + last.strength.mean()*.2
            leaves = q, v, pop, logit, coef
            grad_n = torch.autograd.grad(loss_n, leaves, allow_unused=True)
            actual, final, tr, _ = tape_scan(*source, from_native(init, cfg), cfg,
                                            append_err_coef=coef, chunk_size=3)
            assert_traces_equal(trace, tr.dense(dev))
            mat = final.materialize()
            loss_e = actual.square().mean() + mat['values'].square().mean()*.1 + mat['strength'].mean()*.2
            grad_e = torch.autograd.grad(loss_e, leaves, allow_unused=True)
            pairs = [(expected, actual)] + [(getattr(last, name), value) for name, value in mat.items()]
            pairs += [(a if a is not None else torch.zeros_like(leaf),
                       b if b is not None else torch.zeros_like(leaf))
                      for a, b, leaf in zip(grad_n, grad_e, leaves)]
            max_error = 0.
            for a, b in pairs:
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
                if a.is_floating_point():
                    max_error = max(max_error, float((a-b).detach().abs().max()))
            rows.append(dict(budget=budget, heads=heads, seed=seed,
                             trace_equal=True, max_absolute_error=max_error))
    result = dict(device=str(dev), physical_gpu=os.environ.get('GPU_LEASE_HELD'),
                  torch=torch.__version__, atol=1e-5, rtol=1e-5,
                  usage_ema_excluded=True, cases=rows)
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
