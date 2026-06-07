"""Solve n_heads at fixed dim/depth to hit a param target, for a given mlp_ratio.
Keeps the e97-raw cell identity (dim, n_state, depth) fixed and trades mixer
head-count for the SwiGLU MLP. REAL LadderLM construction."""
import argparse, torch
from ndm.models import LadderLM
VOCAB = 50281

def count(level, dim, depth, n_heads, n_state, expansion, mlp_ratio, raw_write, linear_state):
    m = LadderLM(vocab_size=VOCAB, dim=dim, depth=depth, level=level,
                 n_heads=n_heads, n_state=n_state, expansion=expansion,
                 use_gate=True, gate_activation='silu', linear_state=bool(linear_state),
                 e88_raw_write=bool(raw_write), mlp_ratio=mlp_ratio, use_triton=False)
    n = sum(p.numel() for p in m.parameters()); del m; return n

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', required=True)
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--depth', type=int, required=True)
    ap.add_argument('--n_state', type=int, default=32)
    ap.add_argument('--expansion', type=float, default=1.0)
    ap.add_argument('--mlp_ratio', type=float, required=True)
    ap.add_argument('--raw_write', type=int, default=1)
    ap.add_argument('--linear_state', type=int, default=0)
    ap.add_argument('--target', type=float, default=1.3007e9)
    ap.add_argument('--hi', type=int, default=400)
    args = ap.parse_args()
    common = dict(level=args.level, dim=args.dim, depth=args.depth, n_state=args.n_state,
                  expansion=args.expansion, mlp_ratio=args.mlp_ratio,
                  raw_write=args.raw_write, linear_state=args.linear_state)
    # binary search n_heads in [1, hi]
    lo, hi = 1, args.hi
    while lo < hi:
        mid = (lo + hi) // 2
        n = count(n_heads=mid, **common)
        if n < args.target: lo = mid + 1
        else: hi = mid
    for h in [lo-1, lo]:
        if h >= 1:
            n = count(n_heads=h, **common)
            print(f"mlp_ratio={args.mlp_ratio} depth={args.depth} n_heads={h} -> {n/1e6:.1f}M ({100*(n-args.target)/args.target:+.2f}%)")
