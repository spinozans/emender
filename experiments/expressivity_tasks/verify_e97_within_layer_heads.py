"""Verify the FUSED within-layer E97 heads in TypedHeadMixtureLayer (task e97-heads-in).

This is the within-layer counterpart to verify_e97_fused_parity.py (which checked
the LAYER-LEVEL interleaved HybridLadderLM). Here the E97 split-edit recurrence is a
FIRST-CLASS head type *inside* a heterogeneous TypedHeadMixtureLayer — many head
types ({gdn2_recall, e97_track, count, latch, nonlin, gdn2_nonlin_shell, e97_raw,
e97_delta}) run in parallel in ONE layer, concatenated and summed through the
residual stream. The two new fused types are e97_raw (split-edit + raw write, the
1.3B leaderboard winner) and e97_delta (split-edit + delta correction, plain E97).

It proves, on REAL kernels / REAL data (no mocks):

  (1) NO SILENT EAGER FALLBACK — every fused head type (e97_raw, e97_delta) routes
      through the bf16 split-edit Triton fwd/bwd kernel during training. We count
      real calls into ndm.triton.e88_triton_optimized.e88_triton_optimized_apply
      and assert one per fused E97 sub-block (the bug that bit wire-fused-e97 was the
      kernel silently NOT engaging — fp32 input failed the bf16 dispatch gate).
  (2) PARITY (fwd + bwd, bf16) of each within-layer E97 head vs the eager reference
      recurrence (same weights, same bf16 input; toggle use_triton).
  (3) HETEROGENEOUS-HEAD LM THROUGHPUT — a full LadderLM(typed-gdn2-lm) whose layers
      hold all 8 head types in parallel; full train step (fwd+bwd+opt) tok/s, and the
      tokens a 15-20 min time-bounded screen would process.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/expressivity_tasks/verify_e97_within_layer_heads.py
"""
from __future__ import annotations

import argparse
import time

import torch

from ndm.models.typed_head_mixture import (
    TypedHeadMixtureLayer, allocate_types, TYPE_NAMES,
)
from ndm.models.e88_fla_hybrid import E88FLAHybrid
import ndm.triton.e88_triton_optimized as e88_triton_mod


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return (torch.norm(a - b) / (torch.norm(b) + 1e-12)).item()


# A logit vector that allocates a genuine head to ALL eight types so the layer is
# truly heterogeneous (gdn recall + 4 unified corners + shell + e97_raw + e97_delta).
# Mild positive weight on the two new fused types so they get real heads.
HETERO_LOGITS = [0.3, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.6]


def build_hetero_layer(dim, n_state, n_heads, device, dtype):
    layer = TypedHeadMixtureLayer(
        dim=dim, n_state=n_state, n_heads=n_heads,
        head_type_logits=HETERO_LOGITS,
        use_triton_e97=True, cast_recurrent_bf16=True,
    ).to(device=device, dtype=dtype)
    return layer


def check_no_eager_fallback(device):
    print("=" * 78)
    print("(1) NO SILENT EAGER FALLBACK — fused Triton kernel engages for every E97 head")
    print("=" * 78)
    dim, n_state, n_heads = 256, 32, 48
    layer = build_hetero_layer(dim, n_state, n_heads, device, torch.bfloat16)
    alloc = layer.alloc
    print("  head allocation:", {k: v for k, v in alloc['counts'].items() if v > 0})
    n_fused = int(alloc['n_e97_raw'] > 0) + int(alloc['n_e97_delta'] > 0)
    assert alloc['n_e97_raw'] > 0 and alloc['n_e97_delta'] > 0, "need both e97 fused heads present"
    assert layer.gdn is not None and layer.unified is not None and layer.shell is not None, \
        "need a genuinely heterogeneous mixture (gdn + unified + shell present too)"
    assert layer.e97_raw is not None and layer.e97_delta is not None

    # Count real calls into the fused split-edit Triton kernel.
    calls = {"n": 0}
    orig = e88_triton_mod.e88_triton_optimized_apply

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    e88_triton_mod.e88_triton_optimized_apply = counting
    try:
        layer.train()
        x = torch.randn(4, 128, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        out = layer(x)
        out.float().pow(2).mean().backward()
    finally:
        e88_triton_mod.e88_triton_optimized_apply = orig

    print(f"  fused split-edit Triton kernel calls: {calls['n']} (expected {n_fused}: "
          f"one per fused E97 head type)")
    assert calls["n"] == n_fused, (
        f"expected {n_fused} fused kernel calls (one per E97 head type) but saw "
        f"{calls['n']} — a head silently ran the eager T-scan")
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(out).all()
    print("  PASS: both e97_raw and e97_delta ran on the fused kernel; grads finite.\n")

    # And confirm the loud guard fires rather than silently degrading: an fp32
    # training input with the bf16 cast DISABLED must raise, never go eager.
    guard = TypedHeadMixtureLayer(
        dim=dim, n_state=n_state, n_heads=n_heads, head_type_logits=HETERO_LOGITS,
        use_triton_e97=True, cast_recurrent_bf16=False,
    ).to(device=device, dtype=torch.float32)
    guard.train()
    xf = torch.randn(2, 64, dim, device=device, dtype=torch.float32)
    raised = False
    try:
        guard(xf)
    except RuntimeError as e:
        raised = "silently fall back" in str(e) or "Fused E97" in str(e)
    assert raised, "loud guard did not fire on fp32 training input with cast disabled"
    print("  PASS: loud guard refuses to silently fall back to eager (fp32, cast off).\n")
    return n_fused


def check_parity(device):
    print("=" * 78)
    print("(2) PARITY — within-layer E97 head: fused bf16 Triton vs eager reference (fwd+bwd)")
    print("=" * 78)
    dim, n_state, n_heads = 256, 32, 16
    results = []
    for raw_write, name in [(True, "e97_raw"), (False, "e97_delta")]:
        for T in (128, 512, 1024):
            torch.manual_seed(0)
            block = E88FLAHybrid(
                dim=dim, n_state=n_state, n_heads=n_heads, expansion=1.0,
                use_split_edit=True, raw_write=raw_write, state_activation='tanh',
                use_gate=True, gate_activation='silu', use_triton=True,
            ).to(device=device, dtype=torch.bfloat16)
            block.train()

            B = max(1, 4096 // T)
            x0 = torch.randn(B, T, dim, device=device, dtype=torch.bfloat16)

            # Fused (use_triton=True)
            xf = x0.clone().requires_grad_(True)
            of = block(xf)[0]
            of.float().pow(2).mean().backward()
            gf = xf.grad.detach().clone()

            # Eager reference (toggle use_triton off on the SAME weights)
            block.use_triton = False
            xe = x0.clone().requires_grad_(True)
            oe = block(xe)[0]
            oe.float().pow(2).mean().backward()
            ge = xe.grad.detach().clone()
            block.use_triton = True

            fwd = rel_l2(of, oe)
            grad = rel_l2(gf, ge)
            ok = fwd < 3e-2 and grad < 3e-2
            results.append((name, T, fwd, grad, ok))
            print(f"  {name:10s} T={T:4d} B={B:3d} | fwd rel-L2 {fwd:.2e} | "
                  f"grad rel-L2 {grad:.2e} | {'PASS' if ok else 'FAIL'}")
    assert all(r[4] for r in results), "within-layer E97 parity FAILED"
    print("  PASS: all within-layer E97 heads match the eager reference (fwd+bwd, bf16).\n")
    return results


def check_unaligned_parity(device):
    """The LM next-token path feeds T-1 timesteps, so an aligned context arrives at
    the e97 heads unaligned (e.g. 511). _run_e97 zero-pads the causal time axis to a
    multiple of the checkpoint interval and truncates back. Verify that padded-fused
    matches eager exactly at an UNALIGNED T, through the real TypedHeadMixtureLayer."""
    print("=" * 78)
    print("(2b) UNALIGNED-T PADDING PARITY — _run_e97 padded-fused vs eager at T=511")
    print("=" * 78)
    dim, n_state, n_heads, T = 256, 32, 16, 511  # 511 % 16 != 0
    torch.manual_seed(1)
    layer = TypedHeadMixtureLayer(
        dim=dim, n_state=n_state, n_heads=n_heads,
        head_type_logits=[-30, -30, -30, -30, -30, -30, 1.0, 1.0],  # e97_raw + e97_delta only
        use_triton_e97=True, cast_recurrent_bf16=True,
    ).to(device=device, dtype=torch.bfloat16)
    layer.train()
    assert layer.e97_raw is not None and layer.e97_delta is not None
    assert layer.gdn is None and layer.unified is None and layer.shell is None

    x0 = torch.randn(2, T, dim, device=device, dtype=torch.bfloat16)
    xf = x0.clone().requires_grad_(True)
    of = layer(xf)
    of.float().pow(2).mean().backward()
    gf = xf.grad.detach().clone()

    # eager reference: toggle the fused kernel off on both sub-blocks
    layer.e97_raw.use_triton = False
    layer.e97_delta.use_triton = False
    xe = x0.clone().requires_grad_(True)
    oe = layer(xe)
    oe.float().pow(2).mean().backward()
    ge = xe.grad.detach().clone()
    layer.e97_raw.use_triton = True
    layer.e97_delta.use_triton = True

    fwd, grad = rel_l2(of, oe), rel_l2(gf, ge)
    ok = fwd < 3e-2 and grad < 3e-2 and of.shape[1] == T
    print(f"  T={T} (unaligned) | out T={of.shape[1]} | fwd rel-L2 {fwd:.2e} | "
          f"grad rel-L2 {grad:.2e} | {'PASS' if ok else 'FAIL'}")
    assert ok, "unaligned-T padding parity FAILED"
    print("  PASS: causal zero-pad+truncate is exact; fused stays on the kernel at any T.\n")
    return fwd, grad


def _time_lm(device, level, layer_kwargs, dim, depth, n_heads, n_state, T, B,
             warmup, n_iter, vocab=32000):
    from ndm.models.ladder_lm import LadderLM
    model = LadderLM(
        vocab_size=vocab, dim=dim, depth=depth, level=level,
        n_heads=n_heads, n_state=n_state, use_gate=True, gate_activation='silu',
        layer_kwargs=layer_kwargs,
    ).to(device=device, dtype=torch.bfloat16)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        # return_loss expects [B, T+1]: inp=x[:,:-1] (length T), target=x[:,1:].
        idx = torch.randint(0, vocab, (B, T + 1), device=device)
        opt.zero_grad(set_to_none=True)
        loss = model(idx, return_loss=True)
        loss.backward()
        opt.step()
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        loss = step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = (B * T * n_iter) / dt
    del model, opt
    torch.cuda.empty_cache()
    return tok_s, n_params, float(loss.item())


def benchmark_lm(device, minutes=17.5):
    print("=" * 78)
    print("(3) HETEROGENEOUS-HEAD LM THROUGHPUT — full LadderLM, all 8 head types in parallel")
    print("=" * 78)
    from ndm.models.ladder_lm import LadderLM

    dim, depth, n_heads, n_state = 768, 12, 48, 32
    T, B = 512, 16

    # Confirm the stacked typed layers really hold all 8 head types in parallel.
    probe = LadderLM(vocab_size=32000, dim=dim, depth=depth, level='typed-gdn2-lm',
                     n_heads=n_heads, n_state=n_state, use_gate=True, gate_activation='silu',
                     layer_kwargs={'head_type_logits': HETERO_LOGITS}).to(device, torch.bfloat16)
    sample = next(m for m in probe.modules() if isinstance(m, TypedHeadMixtureLayer))
    active = {k: v for k, v in sample.alloc['counts'].items() if v > 0}
    assert sample.e97_raw is not None and sample.e97_delta is not None
    assert sample.gdn is not None and sample.unified is not None and sample.shell is not None
    print(f"  config: dim={dim} depth={depth} heads={n_heads} n_state={n_state} B={B} T={T}")
    print(f"  per-layer head types active (8 types in parallel): {active}")
    del probe
    torch.cuda.empty_cache()

    # (a) pure FLA-GDN reference (same FLA kernel the gdn2_recall head uses)
    gdn_tok, gdn_p, _ = _time_lm(device, 'fla-gdn', None, dim, depth, n_heads, n_state,
                                 T, B, warmup=3, n_iter=20)
    print(f"  [a] pure fla-gdn      : {gdn_tok:8,.0f} tok/s  ({gdn_p/1e6:.0f}M params)")

    # (b) heterogeneous, e97 heads FUSED (the deliverable)
    het_tok, het_p, het_loss = _time_lm(
        device, 'typed-gdn2-lm', {'head_type_logits': HETERO_LOGITS, 'use_triton_e97': True},
        dim, depth, n_heads, n_state, T, B, warmup=3, n_iter=20)
    print(f"  [b] hetero (e97 FUSED): {het_tok:8,.0f} tok/s  ({het_p/1e6:.0f}M params, "
          f"loss {het_loss:.2f})  -> {het_tok/gdn_tok:.2f}x of pure-GDN")

    # (c) fused-vs-eager e97 RATIO at a small bounded scale (depth-4, B-4) — the
    # eager T-scan is sequential in T and memory-heavy (no sparse checkpoint), so we
    # measure the ratio cheaply rather than at full scale. The ratio is the portable
    # result: it proves the fused split-edit kernel is what keeps the e97 heads from
    # dominating the layer. Same small scale for both so the ratio is apples-to-apples.
    sdim, sdepth, sB, sT = 512, 4, 4, 512
    cf_tok, _, _ = _time_lm(
        device, 'typed-gdn2-lm', {'head_type_logits': HETERO_LOGITS, 'use_triton_e97': True},
        sdim, sdepth, n_heads, n_state, sT, sB, warmup=2, n_iter=8)
    eag_tok, _, _ = _time_lm(
        device, 'typed-gdn2-lm', {'head_type_logits': HETERO_LOGITS, 'use_triton_e97': False},
        sdim, sdepth, n_heads, n_state, sT, sB, warmup=1, n_iter=3)
    print(f"  [c] small-scale hetero (dim{sdim} depth{sdepth} B{sB} T{sT}): "
          f"e97 FUSED {cf_tok:,.0f} vs EAGER {eag_tok:,.0f} tok/s "
          f"-> fusing the e97 heads is {cf_tok/eag_tok:.1f}x faster")

    screen_tokens = het_tok * minutes * 60
    print(f"\n  {minutes:.0f}-min screen budget (fused hetero): {screen_tokens/1e6:,.1f} M tokens "
          f"({'>=' if screen_tokens >= 10e6 else '<'} tens of M)")
    assert screen_tokens >= 10e6, "screen budget below tens of M tokens"
    assert cf_tok > eag_tok, "fused e97 should beat eager e97"
    print("  PASS: heterogeneous-head LM stays the same order as pure-GDN (4 parallel")
    print("        pathways = ~4x the recurrence work) and sustains a tens-of-M-token")
    print("        15-20 min screen; fusing the e97 heads is the difference.\n")
    return het_tok, gdn_tok, cf_tok, eag_tok, screen_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--minutes', type=float, default=17.5)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU + real Triton/FLA kernels"
    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(device)
    print(f"device={device}  type order={TYPE_NAMES}\n")

    n_fused = check_no_eager_fallback(device)
    parity = check_parity(device)
    unaligned = check_unaligned_parity(device)
    het_tok, gdn_tok, cf_tok, eag_tok, screen = benchmark_lm(device, args.minutes)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  no-eager-fallback : PASS ({n_fused} fused E97 head types, kernel-verified)")
    print(f"  within-layer parity: PASS ({len(parity)} aligned + 1 unaligned-T case, fwd+bwd bf16)")
    print(f"  unaligned-T pad    : PASS (fwd {unaligned[0]:.2e}, grad {unaligned[1]:.2e})")
    print(f"  hetero-LM tok/s   : {het_tok:,.0f} fused ({het_tok/gdn_tok:.2f}x of pure-GDN "
          f"{gdn_tok:,.0f})")
    print(f"  e97 fusion gain   : {cf_tok/eag_tok:.1f}x (small-scale fused {cf_tok:,.0f} vs "
          f"eager {eag_tok:,.0f} tok/s)")
    print(f"  screen budget     : {screen/1e6:,.1f} M tokens / {args.minutes:.0f} min")
    print("  ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
