"""Param counting + matching for the e97-raw-plus comparison.

Builds depth=1 and depth=2 LadderLM models for a given geometry, derives the
exact per-layer and base (embedding/tied-head/final-norm) param counts, then
solves for the depth or n_heads that hits a target param budget. REAL model
construction (ndm.models.LadderLM) — no formulas, no mocks.
"""
import argparse
import torch
from ndm.models import LadderLM

VOCAB = 50281  # p50k_base


def build_count(level, dim, depth, n_heads, n_state, expansion, mlp_ratio,
                e88_raw_write=False, use_gate=True, gate_activation='silu',
                linear_state=False, mlp_multiple=64):
    m = LadderLM(
        vocab_size=VOCAB, dim=dim, depth=depth, level=level,
        n_heads=n_heads, n_state=n_state, expansion=expansion,
        use_gate=use_gate, gate_activation=gate_activation,
        linear_state=linear_state, e88_raw_write=e88_raw_write,
        mlp_ratio=mlp_ratio, mlp_multiple=mlp_multiple,
        use_triton=False,  # construction only; kernel choice irrelevant to param count
    )
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


def per_layer_and_base(**kw):
    p1 = build_count(depth=1, **kw)
    p2 = build_count(depth=2, **kw)
    per_layer = p2 - p1
    base = p1 - per_layer  # embedding (tied head) + final norm
    return per_layer, base


def count_for_depth(depth, per_layer, base):
    return base + per_layer * depth


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', required=True)
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--n_heads', type=int, required=True)
    ap.add_argument('--n_state', type=int, default=32)
    ap.add_argument('--expansion', type=float, default=1.0)
    ap.add_argument('--mlp_ratio', type=float, default=0.0)
    ap.add_argument('--raw_write', type=int, default=0)
    ap.add_argument('--linear_state', type=int, default=0)
    ap.add_argument('--target', type=float, default=1.30e9)
    ap.add_argument('--depth', type=int, default=0, help='if >0 just count this depth')
    ap.add_argument('--solve', choices=['depth', 'none'], default='depth')
    args = ap.parse_args()

    kw = dict(level=args.level, dim=args.dim, n_heads=args.n_heads,
              n_state=args.n_state, expansion=args.expansion,
              mlp_ratio=args.mlp_ratio, e88_raw_write=bool(args.raw_write),
              linear_state=bool(args.linear_state))

    if args.depth > 0:
        n = build_count(depth=args.depth, **kw)
        print(f"depth={args.depth} -> {n:,} params ({n/1e6:.1f}M)")
    else:
        per_layer, base = per_layer_and_base(**kw)
        print(f"per_layer={per_layer:,}  base={base:,}")
        best = None
        for d in range(1, 60):
            n = count_for_depth(d, per_layer, base)
            if best is None or abs(n - args.target) < abs(best[1] - args.target):
                best = (d, n)
        d, n = best
        print(f"best depth={d} -> {n:,} ({n/1e6:.1f}M), off by {(n-args.target)/1e6:+.1f}M "
              f"({100*(n-args.target)/args.target:+.2f}%)")
        for dd in [d-1, d, d+1]:
            if dd >= 1:
                nn = count_for_depth(dd, per_layer, base)
                print(f"   depth={dd}: {nn/1e6:.1f}M ({100*(nn-args.target)/args.target:+.2f}%)")
