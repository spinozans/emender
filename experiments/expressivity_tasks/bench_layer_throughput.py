"""Real-throughput layer benchmark: E88 (CUDA) vs E92 (Triton) vs E93 (Triton).

Measures forward + backward time for each layer at production-scale config.
50 warmup iters, 200 timed iters with optimizer step.

Goal: see if Triton kernels are within striking distance of E88's hand-tuned
CUDA kernels.
"""
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def bench(name, build_fn, B, T, dim, n_warmup=50, n_iter=200):
    torch.cuda.empty_cache()
    layer = build_fn().cuda().train()
    n_params = sum(p.numel() for p in layer.parameters())
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)
    x = torch.randn(B, T, dim, device='cuda', dtype=torch.float32)

    for _ in range(n_warmup):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = layer(x)
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
            out = layer(x)
            if isinstance(out, tuple):
                out = out[0]
        loss = out.float().sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    tokens = B * T * n_iter
    tps = tokens / elapsed
    print(f"  {name:>30s}  {elapsed:>7.2f}s  {tps:>9.0f} tok/s  peak={peak_mb:>6.0f}MB  params={n_params:,}")
    return tps


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=1)
    ap.add_argument('--T', type=int, default=512)
    args = ap.parse_args()

    from ndm.models.e88_fla_hybrid import E88FLAHybrid
    from ndm.models.e92_matmat import E92MatMat
    from ndm.models.e93_minimal import E93Minimal

    print(f"\nLayer-only fwd+bwd throughput (B={args.B}, T={args.T}, bf16 autocast)\n")

    # E88 at its winning shape: dim=1280, H=877, N=16
    print("--- E88 winning shape: dim=1280, H=877, N=16 ---")
    bench('E88 (CUDA, H=877)',
          lambda: E88FLAHybrid(dim=1280, n_heads=877, n_state=16, expansion=1.0,
                                use_gate=True, gate_activation='silu'),
          args.B, args.T, 1280)

    # E92 at its (unfinished) winning shape: dim=2944, H=213, N=16
    print("\n--- E92 winning shape: dim=2944, H=213, N=16 ---")
    bench('E92 (Triton, H=213)',
          lambda: E92MatMat(dim=2944, n_heads=213, n_state=16),
          args.B, args.T, 2944)

    # E93 at a 1B-class config: dim=2048, depth alone — single layer here
    print("\n--- E93 single layer: dim=2048, N=16, M=2048 ---")
    bench('E93 (Triton, M=dim)',
          lambda: E93Minimal(dim=2048, n_state=16),
          args.B, args.T, 2048)


if __name__ == '__main__':
    main()
