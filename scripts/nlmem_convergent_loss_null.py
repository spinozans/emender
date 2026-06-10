#!/usr/bin/env python3
"""Convergent-loss-null check for the mlp-mem head (task nlmem-capability,
NONLIN_MEMORY_SPEC.md §8.2).

THE prediction (the lab's standing bar, holding across every exotic head — rot, nonlin,
e97_delta): on natural-language next-byte prediction, the nonlinear MLP-memory head TIES
a linear GDN-2 memory at MATCHED TOKENS (the model-level SwiGLU readout already supplies
read-nonlinearity, so a nonlinear *associative memory* buys nothing on LM loss), and
LOSES at MATCHED WALL-CLOCK (the mlp-mem scan is non-chunkable → slower per token).

This trains BOTH arms on the SAME real repo-byte stream, from the SAME init-seed schedule,
for a matched TOKEN budget, and reports:
  * matched-token  final/mean BPB for each arm + gap   (the convergent-loss-null test)
  * tokens/sec and wall-clock                            (the matched-wall-clock context)

EXACT matched A/B: both arms are the SAME MlpMemHeadLayer shell (identical
projections/conv/gate/o_proj, 0% param diff). The ONLY difference is the recurrent cell —
'mlp-mem-lm' = nonlinear MLP fast-weight memory; 'gdn-matched-lm' = spec §2.3 linear
gated-delta corner. REAL data (repo source bytes), no mocks.

Lease a GPU via the gpu-broker before running.
"""
import argparse
import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from ndm.models import LadderLM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LN2 = math.log(2)


def load_real_bytes(max_bytes):
    patterns = ['ndm/**/*.py', 'scripts/*.py', 'tests/*.py', '*.md', 'docs/**/*.md',
                'paper/**/*.md']
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(os.path.join(REPO, p), recursive=True)))
    buf = bytearray()
    for f in files:
        try:
            with open(f, 'rb') as fh:
                buf.extend(fh.read()); buf.extend(b'\n\n')
        except OSError:
            continue
        if len(buf) >= max_bytes:
            break
    assert len(buf) > 4096, f"not enough real text (got {len(buf)})"
    return torch.tensor(list(buf[:max_bytes]), dtype=torch.long), len(files)


def build(level, args, dev):
    layer_kwargs = dict(mlp_mem_hidden=args.hidden, mlp_mem_eta_max=1.0, mlp_mem_ckpt=16)
    m = LadderLM(
        vocab_size=256, dim=args.dim, depth=args.depth, level=level,
        expansion=1.0, n_heads=args.n_heads, n_state=args.n_state,
        use_gate=True, use_conv=True, d_conv=4,
        mlp_ratio=args.mlp_ratio, mlp_multiple=64,
        layer_kwargs=layer_kwargs,
    ).to(dev)
    return m


def get_batch(data, B, T, dev, gen):
    n = data.numel()
    ix = torch.randint(0, n - T - 1, (B,), generator=gen)
    return torch.stack([data[i:i + T + 1] for i in ix]).to(dev)


def train_arm(level, args, data, dev):
    torch.manual_seed(args.seed)
    m = build(level, args, dev)
    nparams = sum(p.numel() for p in m.parameters() if p.requires_grad)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)  # SAME batch stream for both arms
    losses = []
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(args.steps):
        x = get_batch(data, args.B, args.T, dev, gen)
        out = m(x, return_loss=True)
        loss = out[0] if isinstance(out, tuple) else out
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        lv = loss.detach().item()
        assert math.isfinite(lv), f"{level}: non-finite loss at step {step}"
        losses.append(lv)
        if step % 25 == 0 or step == args.steps - 1:
            print(f"  [{level:15s}] step {step:4d}  bpb {lv/LN2:.4f}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tokens = args.steps * args.B * args.T
    # held-out eval on fresh batches (eval-mode)
    m.eval()
    ev = torch.Generator().manual_seed(args.seed + 99)
    with torch.no_grad():
        ev_losses = []
        for _ in range(args.eval_batches):
            x = get_batch(data, args.B, args.T, dev, ev)
            out = m(x, return_loss=True)
            l = out[0] if isinstance(out, tuple) else out
            ev_losses.append(l.item())
    eval_bpb = (sum(ev_losses) / len(ev_losses)) / LN2
    return {
        'level': level, 'params': nparams,
        'final_bpb': losses[-1] / LN2,
        'last10_bpb': sum(losses[-10:]) / len(losses[-10:]) / LN2,
        'eval_bpb': eval_bpb,
        'elapsed_s': elapsed, 'tokens': tokens, 'tok_per_s': tokens / elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_state', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--mlp_ratio', type=float, default=2.0)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--B', type=int, default=8)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_batches', type=int, default=20)
    ap.add_argument('--max_bytes', type=int, default=3_000_000)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA (lease via gpu-broker)"
    dev = 'cuda'

    data, nfiles = load_real_bytes(args.max_bytes)
    print(f"REAL corpus: {data.numel():,} bytes from {nfiles} repo source files")
    print(f"config: dim={args.dim} depth={args.depth} heads={args.n_heads} n_state={args.n_state} "
          f"hidden={args.hidden} T={args.T} B={args.B} steps={args.steps} "
          f"matched-tokens={args.steps*args.B*args.T:,}\n")

    res = {}
    for level in ('mlp-mem-lm', 'gdn-matched-lm'):
        print(f"=== training {level} ===")
        res[level] = train_arm(level, args, data, dev)
        print(f"    -> {res[level]}\n")

    mm, gd = res['mlp-mem-lm'], res['gdn-matched-lm']
    assert mm['params'] == gd['params'], (mm['params'], gd['params'])
    gap_tok = mm['eval_bpb'] - gd['eval_bpb']     # matched-token: + => mlp-mem worse
    wall_ratio = mm['tok_per_s'] / gd['tok_per_s']  # <1 => mlp-mem slower
    print("=" * 76)
    print("CONVERGENT-LOSS-NULL — matched-token natural-language byte-LM BPB")
    print("=" * 76)
    print(f"params (both arms):     {mm['params']:,}")
    print(f"matched tokens:         {mm['tokens']:,}")
    print(f"mlp-mem   eval BPB:     {mm['eval_bpb']:.4f}   (final {mm['final_bpb']:.4f}, last10 {mm['last10_bpb']:.4f})")
    print(f"gdn-2     eval BPB:     {gd['eval_bpb']:.4f}   (final {gd['final_bpb']:.4f}, last10 {gd['last10_bpb']:.4f})")
    print(f"matched-token gap:      {gap_tok:+.4f} BPB  (mlp-mem - gdn2; |gap|<0.02 => TIE/null)")
    verdict = 'TIE (convergent-loss-null HOLDS)' if abs(gap_tok) < 0.02 else (
        'mlp-mem WORSE' if gap_tok > 0 else 'mlp-mem BETTER')
    print(f"matched-token verdict:  {verdict}")
    print("-" * 76)
    print(f"mlp-mem  throughput:    {mm['tok_per_s']:,.0f} tok/s  ({mm['elapsed_s']:.1f}s)")
    print(f"gdn-2    throughput:    {gd['tok_per_s']:,.0f} tok/s  ({gd['elapsed_s']:.1f}s)")
    print(f"wall-clock ratio:       {wall_ratio:.2f}x  (mlp-mem / gdn2; <1 => mlp-mem slower)")
    print("=" * 76)

    if args.out:
        import json
        json.dump({'config': vars(args), 'mlp_mem': mm, 'gdn2': gd,
                   'matched_token_gap_bpb': gap_tok, 'wall_ratio': wall_ratio,
                   'verdict': verdict}, open(args.out, 'w'), indent=2)
        print(f"saved {args.out}")


if __name__ == '__main__':
    main()
