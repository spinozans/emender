"""Throughput benchmark: E91 vs E88 at matched model sizes.

Tests:
  - Forward + backward time for fixed B, T, dim, depth
  - Memory peak
  - Per-step throughput

Goal: see how much rank-r helps (or hurts) compared to rank-1 (E88).
"""
import os, sys, time, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def bench_one(name, build_fn, B, T, dim, n_iter=20, warmup=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_fn().cuda()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randint(0, 256, (B, T), device='cuda')

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
        loss = out.sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iter):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
        loss = out.sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**2
    tokens = B * T * n_iter
    print(f"  {name:>20s}: {elapsed:>6.2f}s   {tokens/elapsed:>8.0f} tok/s   peak={peak:>6.0f}MB   params={n_params:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=8)
    ap.add_argument('--T', type=int, default=512)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_state', type=int, default=16)
    args = ap.parse_args()

    from ndm.models import LadderLM

    print(f"Bench: B={args.B} T={args.T} dim={args.dim} depth={args.depth} heads={args.n_heads} N={args.n_state}\n")

    # E88 baseline
    bench_one('E88 (rank-1)',
              lambda: LadderLM(level='E88', vocab_size=256, dim=args.dim, depth=args.depth,
                                n_heads=args.n_heads, n_state=args.n_state, expansion=1.0,
                                use_gate=True, gate_activation='silu'),
              args.B, args.T, args.dim)

    # E91 rank=1 (should match E88 mathematically; throughput compares same op count)
    bench_one('E91 rank=1',
              lambda: LadderLM(level='E91', vocab_size=256, dim=args.dim, depth=args.depth,
                                n_heads=args.n_heads, n_state=args.n_state, rank=1,
                                use_gate=True, gate_activation='silu'),
              args.B, args.T, args.dim)

    # E91 rank=N (full-rank, more state change per step)
    bench_one(f'E91 rank={args.n_state}',
              lambda: LadderLM(level='E91', vocab_size=256, dim=args.dim, depth=args.depth,
                                n_heads=args.n_heads, n_state=args.n_state, rank=args.n_state,
                                use_gate=True, gate_activation='silu'),
              args.B, args.T, args.dim)


if __name__ == '__main__':
    main()
