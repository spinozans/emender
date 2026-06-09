#!/usr/bin/env python3
"""Smoke + param-count for the complex-eig-lm comparison (task complex-eig-lm).

Builds the real-eigenvalue baseline and the complex-eigenvalue candidate(s) at a
fixed LM config, forwards a REAL byte batch through each, and reports trainable
parameter counts so the driver can equalize totals via per-side mlp_ratio.
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ndm.models import LadderLM


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(level, dim, depth, n_heads, n_state, mlp_ratio, layer_kwargs=None):
    return LadderLM(
        vocab_size=256, dim=dim, depth=depth, level=level,
        expansion=1.0, n_heads=n_heads, n_state=n_state,
        use_gate=True, use_conv=True, d_conv=4,
        mlp_ratio=mlp_ratio, mlp_multiple=64,
        layer_kwargs=layer_kwargs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_state', type=int, default=64)
    ap.add_argument('--mlp_ratio', type=float, default=2.0)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    dev = args.device
    B, T = 2, 128
    x = torch.randint(0, 256, (B, T + 1), device=dev)

    configs = {
        'real-eig-gdn': dict(level='real-eig-gdn', layer_kwargs=None),
        'complex-everywhere': dict(level='complex-eig-lm',
                                   layer_kwargs=dict(nonlin_subset_frac=0.0)),
        'complex+nonlin': dict(level='complex-eig-lm',
                               layer_kwargs=dict(nonlin_subset_frac=0.125,
                                                 nonlin_subset_phi='hardtanh')),
    }
    print(f"config: dim={args.dim} depth={args.depth} n_heads={args.n_heads} "
          f"n_state={args.n_state} mlp_ratio={args.mlp_ratio}")
    for name, cfg in configs.items():
        m = build(level=cfg['level'], dim=args.dim, depth=args.depth,
                  n_heads=args.n_heads, n_state=args.n_state,
                  mlp_ratio=args.mlp_ratio,
                  layer_kwargs=cfg['layer_kwargs']).to(dev)
        n = count_params(m)
        m.train()
        out = m(x, return_loss=True)
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        finite = all(torch.isfinite(p.grad).all() for p in m.parameters()
                     if p.grad is not None)
        print(f"  {name:22s} params={n:,}  smoke_loss={float(loss):.4f}  grads_finite={finite}")
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # build() takes level as kw; fix call signature
    main()
