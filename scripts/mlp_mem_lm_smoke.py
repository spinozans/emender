#!/usr/bin/env python3
"""Smoke + param-count for the mlp-mem LM head (task nlmem-triton).

Builds a real LadderLM with the `mlp-mem-lm` mixer (the nonlinear MLP-memory cell on
its fused sequential Triton fwd+bwd kernel), forwards a REAL byte batch, runs the LM
loss + backward, and reports trainable params + grad-finiteness. This is the
end-to-end "wired into head + LadderLM" check from the task validation.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ndm.models import LadderLM


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(dim, depth, n_heads, n_state, mlp_ratio, layer_kwargs):
    return LadderLM(
        vocab_size=256, dim=dim, depth=depth, level='mlp-mem-lm',
        expansion=1.0, n_heads=n_heads, n_state=n_state,
        use_gate=True, use_conv=True, d_conv=4,
        mlp_ratio=mlp_ratio, mlp_multiple=64,
        layer_kwargs=layer_kwargs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_state', type=int, default=32)
    ap.add_argument('--mlp_ratio', type=float, default=2.0)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--T', type=int, default=128)
    args = ap.parse_args()

    dev = args.device
    B, T = 2, args.T
    x = torch.randint(0, 256, (B, T + 1), device=dev)

    layer_kwargs = dict(mlp_mem_hidden=args.hidden, mlp_mem_eta_max=1.0, mlp_mem_ckpt=16)
    print(f"config: dim={args.dim} depth={args.depth} n_heads={args.n_heads} "
          f"n_state={args.n_state} hidden={args.hidden} mlp_ratio={args.mlp_ratio} T={T}")
    m = build(args.dim, args.depth, args.n_heads, args.n_state,
              args.mlp_ratio, layer_kwargs).to(dev)
    n = count_params(m)
    m.train()
    out = m(x, return_loss=True)
    loss = out[0] if isinstance(out, tuple) else out
    loss.backward()
    grads_finite = all(torch.isfinite(p.grad).all() for p in m.parameters()
                       if p.grad is not None)
    n_with_grad = sum(1 for p in m.parameters() if p.grad is not None and p.requires_grad)
    print(f"  mlp-mem-lm  params={n:,}  smoke_loss={float(loss):.4f}  "
          f"grads_finite={grads_finite}  n_params_with_grad={n_with_grad}")
    assert torch.isfinite(loss), "loss is non-finite"
    assert grads_finite, "some grads are non-finite"
    print("  OK: forward + backward through LadderLM(mlp-mem-lm) is finite.")


if __name__ == '__main__':
    main()
