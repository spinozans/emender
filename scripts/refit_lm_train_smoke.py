#!/usr/bin/env python3
"""Real-data smoke TRAIN for the `refit` (TTT inner-opt write) LM head (task ttt-validate).

Builds a LadderLM whose typed-head mixture is forced to be ALL-`refit` (head_type_logits
puts ~all softmax mass on the refit slot, index 8 of TYPE_NAMES), so every recurrent head
in every layer is the momentum-delta inner-optimizer cell, driven by the fused chunked
Triton fwd+bwd kernels (`refit_chunked_triton`, two @triton.jit, no torch fallback). This
runs an actual short optimization loop on REAL text and asserts the loss goes DOWN — i.e.
the fused fwd+bwd carry real learning signal end-to-end through LadderLM, not merely
finite gradients.

REAL DATA: the byte stream is the repository's own source files (genuine UTF-8 text,
deterministic, present in-tree). No synthetic/mock corpus. Next-byte prediction over
sampled windows; AdamW; loss[start] vs loss[end] must show a clear decrease.
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from ndm.models import LadderLM
from ndm.models.typed_head_mixture import TYPE_NAMES

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFIT_IDX = TYPE_NAMES.index('refit')


def load_real_bytes(max_bytes):
    """Concatenate real repo source files into one byte tensor (genuine text)."""
    patterns = ['ndm/**/*.py', 'scripts/*.py', 'tests/*.py', '*.md', 'docs/**/*.md']
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(os.path.join(REPO, p), recursive=True)))
    buf = bytearray()
    for f in files:
        try:
            with open(f, 'rb') as fh:
                buf.extend(fh.read())
                buf.extend(b'\n\n')
        except OSError:
            continue
        if len(buf) >= max_bytes:
            break
    assert len(buf) > 4096, f"not enough real text found (got {len(buf)} bytes)"
    data = torch.tensor(list(buf[:max_bytes]), dtype=torch.long)
    return data, len(files)


def get_batch(data, B, T, dev, gen):
    n = data.numel()
    ix = torch.randint(0, n - T - 1, (B,), generator=gen)
    x = torch.stack([data[i:i + T + 1] for i in ix]).to(dev)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_state', type=int, default=32)
    ap.add_argument('--mlp_ratio', type=float, default=2.0)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--B', type=int, default=8)
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--max_bytes', type=int, default=2_000_000)
    ap.add_argument('--has_mom', type=int, default=1, help='1=full refit (momentum), 0=delta special case')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA (lease a GPU via gpu-broker)"
    dev = args.device

    data, nfiles = load_real_bytes(args.max_bytes)
    print(f"REAL corpus: {data.numel():,} bytes from {nfiles} repo source files")

    # Force the typed-head mixture to be ALL refit: softmax mass on the refit slot.
    logits = [-30.0] * len(TYPE_NAMES)
    logits[REFIT_IDX] = 30.0
    layer_kwargs = dict(
        head_type_logits=logits,
        refit_has_mom=bool(args.has_mom),
        refit_chunk_size=32,
    )
    m = LadderLM(
        vocab_size=256, dim=args.dim, depth=args.depth, level='typed-gdn2-lm',
        expansion=1.0, n_heads=args.n_heads, n_state=args.n_state,
        use_gate=True, use_conv=True, d_conv=4,
        mlp_ratio=args.mlp_ratio, mlp_multiple=64,
        layer_kwargs=layer_kwargs,
    ).to(dev)
    nparams = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"config: dim={args.dim} depth={args.depth} n_heads={args.n_heads} "
          f"n_state={args.n_state} T={args.T} B={args.B} has_mom={bool(args.has_mom)} "
          f"params={nparams:,}  (head-type=ALL refit, idx {REFIT_IDX})")
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    gen = torch.Generator().manual_seed(0)

    losses = []
    for step in range(args.steps):
        x = get_batch(data, args.B, args.T, dev, gen)
        out = m(x, return_loss=True)
        loss = out[0] if isinstance(out, tuple) else out
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        lv = loss.detach().item()
        losses.append(lv)
        assert math.isfinite(lv), f"non-finite loss at step {step}"
        assert torch.isfinite(gnorm), f"non-finite grad-norm at step {step}"
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss {lv:.4f}  bpb {lv/math.log(2):.4f}  gnorm {float(gnorm):.3f}")

    start = sum(losses[:5]) / 5
    end = sum(losses[-5:]) / 5
    drop = start - end
    print(f"\nmean loss first-5={start:.4f}  last-5={end:.4f}  drop={drop:.4f} "
          f"({100*drop/start:.1f}%)  bpb {start/math.log(2):.3f}->{end/math.log(2):.3f}")
    # Real learning signal: byte-LM on 256-vocab starts near ln(256)=5.545; a working
    # kernel must drive it down clearly within a short run.
    assert end < start - 0.3, f"loss did not decrease enough (start {start:.3f} end {end:.3f})"
    assert end < 4.5, f"end loss {end:.3f} implausibly high for a learning byte-LM"
    print("OK: real-data smoke TRAIN learns (loss decreases) through fused refit fwd+bwd.")


if __name__ == '__main__':
    main()
