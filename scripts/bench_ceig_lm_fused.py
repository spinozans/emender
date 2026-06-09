#!/usr/bin/env python3
"""Confirm the FUSED Triton complex kernel engages in the LM hot path, and measure
steady-state fwd+bwd tok/s for the three arms at the real LM config.

This is the crux of the matched-WALL-CLOCK question: the prior complex-eig-lm run
used torch.complex (3x penalty) so its wall-clock numbers were meaningless. Here we
(1) assert complex_gated_delta_chunked_triton is actually called in the complex
head's hot path (NOT the torch.complex reference), and (2) report tok/s so the
matched-wall-clock token budgets are computed from REAL fused throughput.
"""
import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ndm.models import LadderLM
import ndm.models.complex_eig_head as ceh


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(level, dim, depth, n_heads, n_state, mlp_ratio, layer_kwargs=None):
    return LadderLM(
        vocab_size=256, dim=dim, depth=depth, level=level, expansion=1.0,
        n_heads=n_heads, n_state=n_state, use_gate=True, use_conv=True, d_conv=4,
        mlp_ratio=mlp_ratio, mlp_multiple=64, layer_kwargs=layer_kwargs)


# --- instrument the two scan kernels so we can PROVE which path runs ----------
_CALLS = {"fused": 0, "torch_complex": 0, "sequential": 0}
_orig_fused = ceh.complex_gated_delta_chunked_triton
_orig_torch = ceh.complex_gated_delta_chunked
_orig_seq = ceh.complex_gated_delta_reference


def _w_fused(*a, **k):
    _CALLS["fused"] += 1
    return _orig_fused(*a, **k)


def _w_torch(*a, **k):
    _CALLS["torch_complex"] += 1
    return _orig_torch(*a, **k)


def _w_seq(*a, **k):
    _CALLS["sequential"] += 1
    return _orig_seq(*a, **k)


ceh.complex_gated_delta_chunked_triton = _w_fused
ceh.complex_gated_delta_chunked = _w_torch
ceh.complex_gated_delta_reference = _w_seq


def bench(name, level, layer_kwargs, mlp_ratio, dim, depth, n_heads, n_state,
          B, T, iters, warmup, dev):
    torch.manual_seed(0)
    m = build(level, dim, depth, n_heads, n_state, mlp_ratio, layer_kwargs).to(dev)
    m = m.bfloat16()
    m.train()
    nparams = count_params(m)
    x = torch.randint(0, 256, (B, T + 1), device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    for k in _CALLS:
        _CALLS[k] = 0

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = m(x, return_loss=True)
        loss = loss[0] if isinstance(loss, tuple) else loss
        loss.backward()
        opt.step()
        return float(loss)

    for _ in range(warmup):
        step()
    calls_after_warmup = dict(_CALLS)
    torch.cuda.synchronize()
    t0 = time.time()
    last = None
    for _ in range(iters):
        last = step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok = B * T * iters
    toks = tok / dt
    return dict(name=name, params=nparams, tok_s=toks, ms_step=1e3 * dt / iters,
                loss=last, calls=calls_after_warmup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_state', type=int, default=64)
    ap.add_argument('--seq_len', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--mlp_real', type=float, default=2.33)
    ap.add_argument('--mlp_complex', type=float, default=2.0)
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    arms = [
        ("real-eig-gdn", "real-eig-gdn", None, args.mlp_real),
        ("complex-everywhere", "complex-eig-lm", dict(nonlin_subset_frac=0.0),
         args.mlp_complex),
        ("complex+nonlin(1/8)", "complex-eig-lm",
         dict(nonlin_subset_frac=0.125, nonlin_subset_phi='hardtanh'),
         args.mlp_complex),
    ]
    print(f"config dim={args.dim} depth={args.depth} H={args.n_heads} "
          f"N={args.n_state} T={args.seq_len} B={args.batch_size} "
          f"iters={args.iters} bf16 (TF32 matmul on)")
    results = []
    for name, level, lk, mlp in arms:
        r = bench(name, level, lk, mlp, args.dim, args.depth, args.n_heads,
                  args.n_state, args.batch_size, args.seq_len, args.iters,
                  args.warmup, dev)
        results.append(r)
        print(f"\n[{name}] params={r['params']:,} mlp_ratio={mlp}")
        print(f"  tok/s={r['tok_s']:,.0f}  ms/step={r['ms_step']:.1f}  "
              f"loss={r['loss']:.3f}")
        print(f"  kernel calls (during timed warmup window): {r['calls']}")
    base = results[0]['tok_s']
    print("\n=== wall-clock ratios vs real-eig-gdn baseline ===")
    for r in results:
        print(f"  {r['name']:22s} {r['tok_s']:>10,.0f} tok/s   "
              f"{r['tok_s']/base:.3f}x baseline   {base/r['tok_s']:.2f}x slower")
    cx = next(r for r in results if r['name'] == 'complex-everywhere')
    assert cx['calls']['fused'] > 0, "FUSED kernel NOT called — hot path is wrong!"
    assert cx['calls']['torch_complex'] == 0, "torch.complex path used — NOT fused!"
    print("\nHOT-PATH CONFIRMED: complex-everywhere uses fused Triton "
          f"(fused calls={cx['calls']['fused']}, torch.complex calls="
          f"{cx['calls']['torch_complex']})")


if __name__ == '__main__':
    main()
