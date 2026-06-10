#!/usr/bin/env python3
"""Throughput benchmark: fused chunked `refit` (TTT inner-opt write) vs FLA GDN-2.

Kernel-level wall-clock comparison at matched (B,T,H,N=V) shapes, fp32, CUDA-event
timed with warmup. Unlike the non-chunkable mlp-mem cell, `refit` IS chunk-parallel
(TTT_WRITE_SPEC sec 3: the joint (S,M) recurrence is an upper-triangular 2x2 (g,mu)
companion per channel -> affine in the state -> the intra-chunk twiddle factorizes as
Phi = A1 @ A2, two [C,C] tensor-core dots). So this benchmark tests whether refit pays
the "pure-torch penalty" the validate criterion warns about — it should NOT, because the
hot path is two real @triton.jit kernels (fwd+bwd), no torch fallback. GDN-2 is the
chunked gated-delta baseline (`fla.ops.gated_delta_rule.chunk_gated_delta_rule`); at
mu=0 refit reduces to exactly that delta write, so the ratio measures the chunked refit's
tensor-core efficiency relative to the FLA reference at the same algorithmic class.

Both the momentum path (has_mom=True, the full `refit`) and the delta special case
(has_mom=False) are timed so the doc reports the cost of the extra mu-gate twiddle.

ratio = GDN-2 time / refit time  (>1 => refit FASTER; <1 => refit SLOWER).
Run via gpu-broker leases (caller sets CUDA_VISIBLE_DEVICES).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from ndm.triton.refit_chunked_autograd import refit_chunked_triton
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


def _l2n(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def mk(B, T, H, N, V, dev, seed=0):
    """GDN-convention inputs: L2-normed q/k, sigmoid erase/write, decay 0.6-0.9,
    momentum 0.1-0.5 — the heavy-ball-stable regime TTT_WRITE_SPEC assumes."""
    g = torch.Generator(device=dev).manual_seed(seed)
    k = _l2n(torch.randn(B, T, H, N, device=dev, dtype=torch.float32, generator=g))
    q = _l2n(torch.randn(B, T, H, N, device=dev, dtype=torch.float32, generator=g))
    v = torch.randn(B, T, H, V, device=dev, dtype=torch.float32, generator=g) * 0.5
    decay = torch.sigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32, generator=g)) * 0.3 + 0.6
    mu = torch.sigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32, generator=g)) * 0.4 + 0.1
    e = torch.sigmoid(torch.randn(B, T, H, N, device=dev, dtype=torch.float32, generator=g)) * 0.5
    w = torch.sigmoid(torch.randn(B, T, H, V, device=dev, dtype=torch.float32, generator=g))
    return k, q, v, decay, mu, e, w


def _bench_refit(k, q, v, decay, mu, e, w, C, has_mom, iters):
    def fwd():
        refit_chunked_triton(k, v, q, decay, e, w, mu, chunk_size=C, has_mom=has_mom)

    ka, qa, va = (t.clone().requires_grad_(True) for t in (k, q, v))
    da, ma, ea, wa = (t.clone().requires_grad_(True) for t in (decay, mu, e, w))

    def fwdbwd():
        for t in (ka, qa, va, da, ma, ea, wa):
            t.grad = None
        out, _ = refit_chunked_triton(ka, va, qa, da, ea, wa, ma, chunk_size=C, has_mom=has_mom)
        out.pow(2).sum().backward()

    return _time(fwd, iters), _time(fwdbwd, iters)


def bench_shape(B, T, H, N, V, C, dev, iters):
    k, q, v, decay, mu, e, w = mk(B, T, H, N, V, dev)
    toks = B * T

    rf_mom_f, rf_mom_fb = _bench_refit(k, q, v, decay, mu, e, w, C, True, iters)
    rf_del_f, rf_del_fb = _bench_refit(k, q, v, decay, mu, e, w, C, False, iters)

    # ---- FLA GDN-2 chunked gated-delta-rule (matched shapes) ----
    g_log = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H, device=dev, dtype=torch.float32)) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32))

    def gd_fwd():
        chunk_gated_delta_rule(q, k, v, g_log, beta, use_qk_l2norm_in_kernel=True)

    qb, kb, vb = (t.clone().requires_grad_(True) for t in (q, k, v))
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
        'T': T,
        'rf_mom_f': rf_mom_f, 'rf_mom_fb': rf_mom_fb,
        'rf_del_f': rf_del_f, 'rf_del_fb': rf_del_fb,
        'gd_f': gd_f, 'gd_fb': gd_fb, 'toks': toks,
        'ratio_mom_f': gd_f / rf_mom_f, 'ratio_mom_fb': gd_fb / rf_mom_fb,
        'ratio_del_f': gd_f / rf_del_f, 'ratio_del_fb': gd_fb / rf_del_fb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=4)
    ap.add_argument('--H', type=int, default=8)
    ap.add_argument('--N', type=int, default=32)
    ap.add_argument('--V', type=int, default=32)
    ap.add_argument('--C', type=int, default=32)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--Ts', type=int, nargs='+', default=[128, 512, 1024])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA (lease a GPU via gpu-broker)"
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False  # honest fp32

    print(f"device={torch.cuda.get_device_name(0)}  fp32  "
          f"B={args.B} H={args.H} N={args.N} V={args.V} C={args.C} iters={args.iters}")
    hdr = (f"{'T':>6} | {'refit-mom f+b':>14} {'refit-del f+b':>14} {'GDN-2 f+b':>14} "
           f"{'mom':>6} {'del':>6}")
    print(hdr)
    print(f"{'':>6} | {'(ms / Mtok/s)':>14} {'(ms / Mtok/s)':>14} {'(ms / Mtok/s)':>14} "
          f"{'ratio':>6} {'ratio':>6}")
    print('-' * len(hdr))
    rows = []
    for T in args.Ts:
        r = bench_shape(args.B, T, args.H, args.N, args.V, args.C, dev, args.iters)
        rows.append(r)
        tps = lambda ms: r['toks'] / (ms * 1e-3) / 1e6
        print(f"{T:>6} | "
              f"{r['rf_mom_fb']:6.2f}/{tps(r['rf_mom_fb']):5.2f}M  "
              f"{r['rf_del_fb']:6.2f}/{tps(r['rf_del_fb']):5.2f}M  "
              f"{r['gd_fb']:6.2f}/{tps(r['gd_fb']):5.2f}M  "
              f"{r['ratio_mom_fb']:5.2f}x {r['ratio_del_fb']:5.2f}x")
    print('-' * len(hdr))
    print("ratio = GDN-2 time / refit time  (>1 => refit FASTER; <1 => refit SLOWER).")
    print("Both refit paths are FUSED chunked Triton (no torch fallback); the delta")
    print("special case (has_mom=False) is the FLA-class algorithm, so its ratio is the")
    print("apples-to-apples 'no pure-torch penalty' check. mom adds the mu-gate twiddle.")
    # fwd-only line for completeness
    print()
    print(f"{'T':>6} | {'refit-mom fwd':>14} {'refit-del fwd':>14} {'GDN-2 fwd':>14} "
          f"{'mom':>6} {'del':>6}")
    for r in rows:
        tps = lambda ms: r['toks'] / (ms * 1e-3) / 1e6
        print(f"{r['T']:>6} | "
              f"{r['rf_mom_f']:6.2f}/{tps(r['rf_mom_f']):5.2f}M  "
              f"{r['rf_del_f']:6.2f}/{tps(r['rf_del_f']):5.2f}M  "
              f"{r['gd_f']:6.2f}/{tps(r['gd_f']):5.2f}M  "
              f"{r['ratio_mom_f']:5.2f}x {r['ratio_del_f']:5.2f}x")


if __name__ == '__main__':
    main()
