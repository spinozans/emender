"""Direct E91 vs E88 layer benchmark.

Bypasses LadderLM. Tests just the recurrence layers head-to-head.
Includes proper warmup (50 iters) and longer measurement (100 iters).
"""
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ndm.models.e88_fla_hybrid import E88FLAHybrid
from ndm.models.e91_matmat import E91MatMat


def bench_layer(name, layer, B, T, dim, n_warmup=20, n_iter=100):
    layer = layer.cuda().train()
    x = torch.randn(B, T, dim, device='cuda', dtype=torch.bfloat16, requires_grad=False)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)

    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = layer(x.float())
            if isinstance(out, tuple):
                out = out[0]
        loss = out.float().sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(n_iter):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = layer(x.float())
            if isinstance(out, tuple):
                out = out[0]
        loss = out.float().sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**2
    tokens = B * T * n_iter
    print(f"  {name:>20s}: {elapsed:>7.2f}s   {tokens/elapsed:>10.0f} tok/s   peak={peak:>6.0f}MB")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=1)
    ap.add_argument('--T', type=int, default=512)
    ap.add_argument('--dim', type=int, default=1280)
    ap.add_argument('--n_heads', type=int, default=141)
    ap.add_argument('--n_state', type=int, default=16)
    args = ap.parse_args()

    print(f"Direct layer bench: B={args.B} T={args.T} dim={args.dim} H={args.n_heads} N={args.n_state}\n")

    # E88 via E88FLAHybrid directly
    e88 = E88FLAHybrid(
        dim=args.dim, n_heads=args.n_heads, n_state=args.n_state,
        expansion=1.0, use_gate=True, gate_activation='silu',
    )
    bench_layer('E88FLAHybrid (CUDA)', e88, args.B, args.T, args.dim)

    # E91 rank=1
    e91_r1 = E91MatMat(
        dim=args.dim, n_heads=args.n_heads, n_state=args.n_state, rank=1,
        use_gate=True, gate_activation='silu',
    )
    bench_layer('E91 rank=1 (Triton)', e91_r1, args.B, args.T, args.dim)

    # E91 rank=N
    e91_rN = E91MatMat(
        dim=args.dim, n_heads=args.n_heads, n_state=args.n_state, rank=args.n_state,
        use_gate=True, gate_activation='silu',
    )
    bench_layer(f'E91 rank={args.n_state} (Triton)', e91_rN, args.B, args.T, args.dim)


if __name__ == '__main__':
    main()
