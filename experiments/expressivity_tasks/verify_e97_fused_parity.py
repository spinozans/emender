"""E97 FUSED-vs-EAGER LM PARITY + THROUGHPUT (task wire-fused-e97).

Verifies that routing the E97 / e97-raw / e97-delta split-edit recurrence through
the fused Triton fwd/bwd kernel (the --use_triton_e88 path) is numerically faithful
to the EAGER PyTorch reference recurrence, END TO END through the actually-wired
HybridLadderLM, at the real bf16 precision and T used by the LM runners.

For each arm we build TWO HybridLadderLMs with bit-identical initialization (same
torch seed before construction) differing ONLY in use_triton_e88:
  * fused = use_triton_e88=True  -> E97 split-edit Triton fwd/bwd kernel
  * eager = use_triton_e88=False -> E88FLAHybrid PyTorch reference T-scan
Same bf16 input -> compare forward logits, the LM loss, and EVERY parameter
gradient. Then a short shared-data training run on each to confirm the loss curves
track. Then a tok/s benchmark of fused vs eager.

REAL model, REAL recurrence, REAL random token batches (no mocks). Tiny footprint
(<2GB) so it co-locates on a low-util GPU without disturbing the running studies.

Usage:
    CUDA_VISIBLE_DEVICES=2 python verify_e97_fused_parity.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ndm.models.hybrid_ladder import HybridLadderLM

# e97-raw arm shape from run_e97_raw_expressivity.py (dim=256, 32 heads, N=V=32).
DIM = 256
DEPTH = 4
N_HEADS = 32
N_STATE = 32
VOCAB = 16

# arm -> e88_kwargs forwarded to the E97 layer (matches run_e97_raw_expressivity).
ARMS = {
    'e97-raw': {'state_activation': 'tanh', 'raw_write': True},   # raw-write (no delta)
    'e97':     {'state_activation': 'tanh', 'raw_write': False},  # delta-correction
    'e97-linear': {'state_activation': 'identity', 'raw_write': False},
}


def build_model(arm_kwargs, use_triton, seed, device, force_bf16=None):
    """Build a HybridLadderLM (layer_pattern=['E97']) with deterministic init.

    force_bf16 overrides cast_recurrent_bf16 so the EAGER reference can also be fed
    bf16 input (bf16-vs-bf16 parity). When None it follows use_triton_e88 (the real
    production default: fused casts to bf16, plain eager stays fp32).
    """
    torch.manual_seed(seed)
    model = HybridLadderLM(
        vocab_size=VOCAB, dim=DIM, depth=DEPTH,
        layer_pattern=['E97'],
        layer_kwargs=[dict(arm_kwargs)],
        n_state=N_STATE, n_heads=N_HEADS, expansion=1.0,
        use_triton_e88=use_triton,
    ).to(device)
    if force_bf16 is not None:
        model.cast_recurrent_bf16 = force_bf16
    return model


def _maxdiff(a, b):
    """Returns (max abs elementwise diff, relative-L2 / Frobenius error).

    Relative-L2 ‖a-b‖₂/‖b‖₂ is the meaningful end-to-end parity metric: per-element
    relative diff is dominated by near-zero denominators (a logit that is ~0 in one
    path and O(1) in the other reports relΔ in the hundreds while contributing
    nothing to the loss), so we report Frobenius relative error instead."""
    a = a.float(); b = b.float()
    abs_d = (a - b).abs()
    rel_l2 = (abs_d.norm() / b.norm().clamp_min(1e-12)).item()
    return abs_d.max().item(), rel_l2


def parity_check(arm, arm_kwargs, args, device):
    B, T = args.batch, args.seq_len
    seed = args.seed

    # Both fed bf16: fused -> Triton split-edit kernel; eager -> bf16 PyTorch
    # reference T-scan. This is the apples-to-apples "fused vs eager at bf16" check.
    fused = build_model(arm_kwargs, use_triton=True, seed=seed, device=device, force_bf16=True)
    eager = build_model(arm_kwargs, use_triton=False, seed=seed, device=device, force_bf16=True)

    # Sanity: identical initialization.
    for (n1, p1), (n2, p2) in zip(fused.named_parameters(), eager.named_parameters()):
        assert torch.equal(p1, p2), f"init mismatch at {n1}"

    fused.train(); eager.train()

    torch.manual_seed(seed + 1)
    x = torch.randint(0, VOCAB, (B, T), device=device)
    y = torch.randint(0, VOCAB, (B, T), device=device)

    # Forward at the real bf16 autocast precision used in training.
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits_f = fused(x)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits_e = eager(x)

    out_abs, out_rel = _maxdiff(logits_f, logits_e)

    loss_f = F.cross_entropy(logits_f.float().view(-1, VOCAB), y.view(-1))
    loss_e = F.cross_entropy(logits_e.float().view(-1, VOCAB), y.view(-1))
    loss_abs = abs(loss_f.item() - loss_e.item())

    # Backward parity on every parameter gradient.
    fused.zero_grad(); eager.zero_grad()
    loss_f.backward(); loss_e.backward()

    # Coalesce None -> zeros. A param can be DISCONNECTED in one path but a
    # zero-gradient Function input in the other — notably the E97 erase gate under
    # raw_write=True, where the delta/read term that consumes it is dropped, so it
    # carries no learning signal either way (eager: grad None, fused: grad 0.0).
    # Both mean "no update"; treating None as zero compares them faithfully.
    grad_abs = 0.0
    worst_param = None
    dead_params = []
    sq_diff = 0.0  # global ‖g_fused - g_eager‖²
    sq_ref = 0.0   # global ‖g_eager‖²
    for (n, pf), (_, pe) in zip(fused.named_parameters(), eager.named_parameters()):
        gf, ge = pf.grad, pe.grad
        if gf is None and ge is None:
            continue
        if gf is None or ge is None:
            # one disconnected; the other must be ~zero for parity to hold.
            present = ge if gf is None else gf
            mag = present.float().abs().max().item()
            dead_params.append((n, mag))
            grad_abs = max(grad_abs, mag)
            sq_diff += present.float().pow(2).sum().item()
            continue
        a, _ = _maxdiff(gf, ge)
        if a > grad_abs:
            grad_abs = a; worst_param = n
        sq_diff += (gf.float() - ge.float()).pow(2).sum().item()
        sq_ref += ge.float().pow(2).sum().item()
    grad_rel = (sq_diff ** 0.5) / max(sq_ref ** 0.5, 1e-12)

    return {
        'out_abs': out_abs, 'out_rel': out_rel,
        'loss_f': loss_f.item(), 'loss_e': loss_e.item(), 'loss_abs': loss_abs,
        'grad_abs': grad_abs, 'grad_rel': grad_rel, 'worst_grad_param': worst_param,
        'dead_params': dead_params,
    }


def loss_curve(arm, arm_kwargs, args, device, use_triton):
    """Short shared-data training run; returns the per-step loss list.

    Both arms fed bf16 (force_bf16=True) so the curves isolate kernel vs reference,
    not a bf16-vs-fp32 precision gap."""
    model = build_model(arm_kwargs, use_triton=use_triton, seed=args.seed,
                        device=device, force_bf16=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    B, T = args.batch, args.seq_len
    g = torch.Generator(device=device).manual_seed(args.seed + 100)
    losses = []
    for _ in range(args.curve_steps):
        x = torch.randint(0, VOCAB, (B, T), device=device, generator=g)
        y = torch.randint(0, VOCAB, (B, T), device=device, generator=g)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, VOCAB), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def benchmark(arm, arm_kwargs, args, device, use_triton):
    model = build_model(arm_kwargs, use_triton=use_triton, seed=args.seed, device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    B, T = args.batch, args.seq_len
    x = torch.randint(0, VOCAB, (B, T), device=device)
    y = torch.randint(0, VOCAB, (B, T), device=device)

    def step():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, VOCAB), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.bench_steps):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    toks = B * T * args.bench_steps
    return toks / dt, dt / args.bench_steps * 1000.0  # tok/s, ms/step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=list(ARMS.keys()), choices=list(ARMS.keys()))
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--curve_steps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--bench_steps', type=int, default=30)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = 'cuda'
    print(f"device={torch.cuda.get_device_name(0)}  "
          f"shape: dim={DIM} depth={DEPTH} heads={N_HEADS} N=V={N_STATE} "
          f"B={args.batch} T={args.seq_len}\n", flush=True)

    # Tolerances on RELATIVE-L2 (Frobenius) error: fused bf16 Triton kernel vs bf16
    # eager reference, end-to-end through 4 layers. bf16 has ~3 decimal digits, so
    # reduction-order differences accumulate to ~1% rel-L2 across the depth-4 stack
    # — far below anything that perturbs training (loss matches to <1e-2 relative,
    # grads to ~1% rel-L2, curves track to the noise floor).
    OUT_REL_TOL = 3e-2     # forward logits relative-L2
    LOSS_REL_TOL = 1e-2    # loss relative diff
    GRAD_REL_TOL = 3e-2    # gradient global relative-L2
    CURVE_REL_TOL = 5e-2   # per-step loss curve relative diff

    all_pass = True
    for arm in args.arms:
        kw = ARMS[arm]
        print(f"=== {arm}  ({kw}) ===", flush=True)
        r = parity_check(arm, kw, args, device)
        loss_rel = r['loss_abs'] / max(abs(r['loss_e']), 1e-12)
        ok_out = r['out_rel'] <= OUT_REL_TOL
        ok_loss = loss_rel <= LOSS_REL_TOL
        ok_grad = r['grad_rel'] <= GRAD_REL_TOL
        print(f"  fwd   logits  rel-L2={r['out_rel']:.3e}  (max|Δ|={r['out_abs']:.3e})  "
              f"[{'PASS' if ok_out else 'FAIL'}]", flush=True)
        print(f"  loss  fused={r['loss_f']:.6f}  eager={r['loss_e']:.6f}  "
              f"relΔ={loss_rel:.3e}  [{'PASS' if ok_loss else 'FAIL'}]", flush=True)
        print(f"  bwd   grad rel-L2={r['grad_rel']:.3e}  (max|Δ|={r['grad_abs']:.3e}, "
              f"worst: {r['worst_grad_param']})  [{'PASS' if ok_grad else 'FAIL'}]", flush=True)
        if r['dead_params']:
            dp = ', '.join(f"{n.split('.')[-2]}.{n.split('.')[-1]}=|{m:.1e}|" for n, m in r['dead_params'])
            print(f"        dead/disconnected (no learning signal in both paths): {dp}", flush=True)

        # Short loss-curve agreement (relative per step, the meaningful measure).
        lc_f = loss_curve(arm, kw, args, device, use_triton=True)
        lc_e = loss_curve(arm, kw, args, device, use_triton=False)
        curve_relmax = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(lc_f, lc_e))
        end_rel = abs(lc_f[-1] - lc_e[-1]) / max(abs(lc_e[-1]), 1e-12)
        ok_curve = curve_relmax <= CURVE_REL_TOL
        print(f"  curve {args.curve_steps} steps  fused[0]={lc_f[0]:.4f}->[-1]={lc_f[-1]:.4f}  "
              f"eager[0]={lc_e[0]:.4f}->[-1]={lc_e[-1]:.4f}  max relΔ={curve_relmax:.3e}  "
              f"endΔ={end_rel:.3e}  [{'PASS' if ok_curve else 'FAIL'}]", flush=True)

        # Throughput.
        tps_f, ms_f = benchmark(arm, kw, args, device, use_triton=True)
        tps_e, ms_e = benchmark(arm, kw, args, device, use_triton=False)
        speedup = tps_f / tps_e
        print(f"  speed fused={tps_f:,.0f} tok/s ({ms_f:.1f} ms/step)  "
              f"eager={tps_e:,.0f} tok/s ({ms_e:.1f} ms/step)  -> {speedup:.1f}x", flush=True)

        arm_pass = ok_out and ok_loss and ok_grad and ok_curve
        all_pass = all_pass and arm_pass
        print(f"  ARM {arm}: {'PASS' if arm_pass else 'FAIL'}\n", flush=True)

    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}", flush=True)
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
