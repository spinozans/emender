#!/usr/bin/env python3
"""Throughput benchmark: fused mlp-mem sequential scan vs FLA GDN-2 (chunk_gated_delta_rule).

Kernel-level wall-clock comparison at matched (B,T,H,K=N,V) shapes, fp32, CUDA-event
timed with warmup. mlp-mem is a NON-associative sequential per-token scan (no chunking
possible — see NONLIN_MEMORY_SPEC.md sec 4); GDN-2 is the chunked gated-delta-rule
baseline. The spec predicts mlp-mem pays a sequential-scan wall-clock penalty vs the
chunkable GDN-2 — this script measures it honestly (fwd-only and fwd+bwd) so the
validate doc reports a real throughput number, not an assertion.

Both run via gpu-broker leases (caller sets CUDA_VISIBLE_DEVICES).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from ndm.triton.mlp_mem_fused import mlp_mem_triton
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def _time(fn, iters, warmup=5):
    """Return mean ms/iter via CUDA events after warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def mk(B, T, H, N, V, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    k = torch.randn(B, T, H, N, device=dev, dtype=torch.float32, generator=g) * 0.5
    q = torch.randn(B, T, H, N, device=dev, dtype=torch.float32, generator=g) * 0.5
    v = torch.randn(B, T, H, V, device=dev, dtype=torch.float32, generator=g) * 0.5
    eta = torch.sigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32, generator=g)) * 0.5
    gamma = torch.sigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32, generator=g)) * 0.3 + 0.6
    return k, q, v, eta, gamma


def bench_shape(B, T, H, N, V, HID, dev, iters):
    k, q, v, eta, gamma = mk(B, T, H, N, V, dev)
    toks = B * T

    # ---- mlp-mem fused sequential scan ----
    def mm_fwd():
        mlp_mem_triton(k, q, v, eta, gamma, HID, ckpt_interval=16)

    ka = k.clone().requires_grad_(True)
    qa = q.clone().requires_grad_(True)
    va = v.clone().requires_grad_(True)
    ea = eta.clone().requires_grad_(True)
    ga = gamma.clone().requires_grad_(True)

    def mm_fwdbwd():
        for t in (ka, qa, va, ea, ga):
            t.grad = None
        out, _, _ = mlp_mem_triton(ka, qa, va, ea, ga, HID, ckpt_interval=16)
        out.pow(2).sum().backward()

    mm_f = _time(mm_fwd, iters)
    mm_fb = _time(mm_fwdbwd, iters)

    # ---- FLA GDN-2 chunked gated-delta-rule (matched shapes) ----
    g_log = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H, device=dev, dtype=torch.float32)) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32))

    def gd_fwd():
        chunk_gated_delta_rule(q, k, v, g_log, beta, use_qk_l2norm_in_kernel=True)

    qb = q.clone().requires_grad_(True)
    kb = k.clone().requires_grad_(True)
    vb = v.clone().requires_grad_(True)
    gb = g_log.clone().requires_grad_(True)
    bb = beta.clone().requires_grad_(True)

    def gd_fwdbwd():
        for t in (qb, kb, vb, gb, bb):
            t.grad = None
        o = chunk_gated_delta_rule(qb, kb, vb, gb, bb, use_qk_l2norm_in_kernel=True)
        out = o[0] if isinstance(o, tuple) else o
        out.pow(2).sum().backward()

    gd_f = _time(gd_fwd, iters)
    gd_fb = _time(gd_fwdbwd, iters)

    return {
        'T': T, 'B': B, 'H': H, 'N': N, 'V': V, 'HID': HID, 'toks': toks,
        'mm_fwd_ms': mm_f, 'mm_fwdbwd_ms': mm_fb,
        'gd_fwd_ms': gd_f, 'gd_fwdbwd_ms': gd_fb,
        'mm_fwd_tps': toks / (mm_f * 1e-3), 'gd_fwd_tps': toks / (gd_f * 1e-3),
        'mm_fb_tps': toks / (mm_fb * 1e-3), 'gd_fb_tps': toks / (gd_fb * 1e-3),
        'ratio_fwd': gd_f / mm_f, 'ratio_fwdbwd': gd_fb / mm_fb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=4)
    ap.add_argument('--H', type=int, default=8)
    ap.add_argument('--N', type=int, default=32)
    ap.add_argument('--V', type=int, default=32)
    ap.add_argument('--HID', type=int, default=32)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--Ts', type=int, nargs='+', default=[128, 512, 1024])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA (lease a GPU via gpu-broker)"
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False  # honest fp32

    print(f"device={torch.cuda.get_device_name(0)}  fp32  "
          f"B={args.B} H={args.H} N={args.N} V={args.V} HID={args.HID} iters={args.iters}")
    print(f"{'T':>6} | {'mlp-mem fwd':>14} {'GDN-2 fwd':>14} {'ratio':>7} | "
          f"{'mlp-mem f+b':>14} {'GDN-2 f+b':>14} {'ratio':>7}")
    print(f"{'':>6} | {'(ms / tok/s)':>14} {'(ms / tok/s)':>14} {'GDN/mm':>7} | "
          f"{'(ms / tok/s)':>14} {'(ms / tok/s)':>14} {'GDN/mm':>7}")
    print('-' * 104)
    rows = []
    for T in args.Ts:
        r = bench_shape(args.B, T, args.H, args.N, args.V, args.HID, dev, args.iters)
        rows.append(r)
        print(f"{T:>6} | "
              f"{r['mm_fwd_ms']:6.2f}/{r['mm_fwd_tps']/1e6:5.2f}M  "
              f"{r['gd_fwd_ms']:6.2f}/{r['gd_fwd_tps']/1e6:5.2f}M  "
              f"{r['ratio_fwd']:6.3f}x | "
              f"{r['mm_fwdbwd_ms']:6.2f}/{r['mm_fb_tps']/1e6:5.2f}M  "
              f"{r['gd_fwdbwd_ms']:6.2f}/{r['gd_fb_tps']/1e6:5.2f}M  "
              f"{r['ratio_fwdbwd']:6.3f}x")
    print('-' * 104)
    print("ratio = GDN-2 time / mlp-mem time  (>1 => mlp-mem FASTER; <1 => mlp-mem SLOWER).")
    print("Expectation (spec sec 4): mlp-mem is a non-chunkable sequential scan, so it is")
    print("SLOWER than the chunked GDN-2 baseline, esp. as T grows (ratio < 1, shrinking).")


if __name__ == '__main__':
    main()
