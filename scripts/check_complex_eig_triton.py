"""Parity + throughput harness for the fused-Triton complex gated-delta kernel.

Compares ComplexEigChunkedFn (fused Triton, real/imag-decomposed, no torch.complex)
against the eager references in complex_eig_chunked.py (per-step + torch-complex
chunked) on REAL random data, fwd + bwd, at T=128/512/1024. Then measures
throughput vs FLA fused GDN-2 and checks the theta=0 / theta=pi reductions.
"""
import math
import sys
import time

import torch

from ndm.triton.complex_eig_chunked import (
    complex_gated_delta_reference,
    complex_gated_delta_chunked,
)
from ndm.triton.complex_eig_chunked_autograd import complex_gated_delta_chunked_triton


def mk(B, T, H, N, V, dev, dtype=torch.float32, seed=0, decay_lo=0.85, decay_hi=1.0):
    g = torch.Generator(device=dev).manual_seed(seed)
    k = torch.randn(B, T, H, N, device=dev, dtype=dtype, generator=g) * 0.5
    q = torch.randn(B, T, H, N, device=dev, dtype=dtype, generator=g) * 0.5
    v = torch.randn(B, T, H, V, device=dev, dtype=dtype, generator=g) * 0.5
    P = N // 2
    r = torch.rand(B, T, H, P, device=dev, dtype=dtype, generator=g) * (decay_hi - decay_lo) + decay_lo
    log_r = r.clamp_min(1e-6).log()
    theta = torch.randn(B, T, H, P, device=dev, dtype=dtype, generator=g) * 0.7
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev, dtype=dtype, generator=g)) * 2.0
    return q, k, v, log_r, theta, beta


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-6)


def check_fwd(dev):
    print("=== forward parity (fused vs eager reference + torch-complex chunked) ===")
    B, H, N, V = 2, 4, 32, 32
    ok = True
    # Fused complex kernel: C<=32 (complex tile doubling exceeds 100KB SMEM at C=64).
    for T, C in [(32, 32), (64, 32), (128, 32), (96, 32), (130, 32), (256, 32), (512, 32), (1024, 32), (130, 16)]:
        q, k, v, log_r, theta, beta = mk(B, T, H, N, V, dev, seed=0)
        ref, refS = complex_gated_delta_reference(q, k, v, log_r, theta, beta)
        chk, chkS = complex_gated_delta_chunked(q, k, v, log_r, theta, beta, chunk_size=C)
        fus, fusS = complex_gated_delta_chunked_triton(q, k, v, log_r, theta, beta, chunk_size=C)
        e_ref = rel(fus, ref)
        e_chk = rel(fus, chk)
        e_s = rel(fusS, refS)
        good = e_ref < 3e-3 and e_s < 3e-3
        ok = ok and good
        print(f"  T={T:5d} C={C}: fused-vs-ref {e_ref:.2e}  fused-vs-chunked {e_chk:.2e}  S {e_s:.2e}  {'OK' if good else 'FAIL'}")
    return ok


def check_bwd(dev):
    print("=== backward parity (fused vs eager reference) ===")
    B, H, N, V = 2, 3, 32, 32
    ok = True
    for T in [128, 512, 1024]:
        q, k, v, log_r, theta, beta = mk(B, T, H, N, V, dev, seed=7)

        def run(fn):
            ts = [t.clone().requires_grad_(True) for t in (q, k, v, log_r, theta, beta)]
            out, _ = fn(*ts)
            (out * out).sum().backward()
            return out.detach(), [t.grad.clone() for t in ts]

        o_ref, g_ref = run(lambda *a: complex_gated_delta_reference(*a))
        o_fus, g_fus = run(lambda *a: complex_gated_delta_chunked_triton(*a, chunk_size=32))
        print(f"  T={T}: fwd {rel(o_fus, o_ref):.2e}")
        for nm, a, b in zip(["q", "k", "v", "log_r", "theta", "beta"], g_ref, g_fus):
            finite = torch.isfinite(b).all().item()
            e = rel(b, a)
            good = finite and e < 2e-2
            ok = ok and good
            print(f"      d{nm:6s} rel {e:.2e}  finite={finite}  {'OK' if good else 'FAIL'}")
    return ok


def check_reductions(dev):
    print("=== reductions ===")
    B, T, H, N, V = 2, 64, 3, 32, 32
    ok = True
    for name, th_val in [("theta=0", 0.0), ("theta=pi", math.pi)]:
        q, k, v, log_r, _, beta = mk(B, T, H, N, V, dev, seed=1)
        theta = torch.full((B, T, H, N // 2), th_val, device=dev)
        ref, _ = complex_gated_delta_reference(q, k, v, log_r, theta, beta)
        fus, _ = complex_gated_delta_chunked_triton(q, k, v, log_r, theta, beta, chunk_size=32)
        e = rel(fus, ref)
        good = e < 3e-3
        ok = ok and good
        print(f"  {name}: fused-vs-ref {e:.2e}  {'OK' if good else 'FAIL'}")
    return ok


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def check_bwd_bf16(dev):
    print("=== backward parity bf16/TF32 (looser tol) ===")
    B, H, N, V, T = 2, 4, 32, 32, 512
    q, k, v, log_r, theta, beta = mk(B, T, H, N, V, dev, seed=7)

    def run(fn, cast):
        ts = [t.clone().requires_grad_(True) for t in (q, k, v, log_r, theta, beta)]
        ins = [t.bfloat16() if cast and t.dtype == torch.float32 else t for t in ts]
        # keep log_r/theta/beta fp32; only q,k,v cast (matches autocast head)
        ins = [ts[i].bfloat16() if (cast and i < 3) else ts[i] for i in range(6)]
        out, _ = fn(*ins)
        (out.float() * out.float()).sum().backward()
        return [t.grad.clone() for t in ts]

    g_ref = run(lambda *a: complex_gated_delta_reference(*a), cast=False)
    g_fus = run(lambda *a: complex_gated_delta_chunked_triton(*a, chunk_size=32, allow_tf32=True), cast=True)
    ok = True
    for nm, a, b in zip(["q", "k", "v", "log_r", "theta", "beta"], g_ref, g_fus):
        e = rel(b, a)
        good = torch.isfinite(b).all().item() and e < 5e-2
        ok = ok and good
        print(f"  d{nm:6s} rel {e:.2e}  {'OK' if good else 'FAIL'}")
    return ok


def check_throughput(dev):
    print("=== throughput: fused-Triton complex (TF32) vs torch-complex vs FLA GDN-2 ===")
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        fla_ok = True
    except Exception as e:
        print(f"  (FLA unavailable: {e})")
        fla_ok = False
    # saturating workload (B*H = 128 programs) so neither kernel is launch-bound.
    B, H, N, V = 8, 16, 32, 32
    for T in [512, 1024, 2048]:
        q, k, v, log_r, theta, beta = mk(B, T, H, N, V, dev, seed=3)
        # fp32 inputs, TF32 matmuls (production autocast computes phases in fp32,
        # heavy dots in TF32) — the fair comparison vs FLA's bf16/TF32 GDN-2.
        t_tf32 = bench(lambda: complex_gated_delta_chunked_triton(q, k, v, log_r, theta, beta, chunk_size=32, allow_tf32=True))
        t_fp32 = bench(lambda: complex_gated_delta_chunked_triton(q, k, v, log_r, theta, beta, chunk_size=32, allow_tf32=False))
        t_torch = bench(lambda: complex_gated_delta_chunked(q, k, v, log_r, theta, beta, chunk_size=32))
        line = (f"  T={T:5d}: fused-TF32 {t_tf32:7.3f} ms  fused-fp32 {t_fp32:7.3f} ms  "
                f"torch-cplx {t_torch:7.3f} ms  vs-torch {t_torch / t_tf32:4.2f}x")
        if fla_ok:
            qg = torch.randn(B, T, H, N, device=dev, dtype=torch.bfloat16)
            kg = torch.nn.functional.normalize(torch.randn(B, T, H, N, device=dev, dtype=torch.bfloat16), dim=-1)
            vg = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16)
            gg = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32))
            bg = torch.sigmoid(torch.randn(B, T, H, device=dev, dtype=torch.bfloat16))
            t_fla = bench(lambda: chunk_gated_delta_rule(qg, kg, vg, gg, bg, head_first=False))
            line += f"  | FLA-GDN2 {t_fla:7.3f} ms  TF32/FLA {t_tf32 / t_fla:4.2f}x"
        print(line)


def main():
    if not torch.cuda.is_available():
        print("no CUDA")
        return 1
    dev = "cuda"
    torch.manual_seed(0)
    of = check_fwd(dev)
    ob = check_bwd(dev)
    obb = check_bwd_bf16(dev)
    orr = check_reductions(dev)
    check_throughput(dev)
    allok = of and ob and obb and orr
    print(f"\n{'ALL PARITY OK' if allok else 'SOME CHECKS FAILED'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
