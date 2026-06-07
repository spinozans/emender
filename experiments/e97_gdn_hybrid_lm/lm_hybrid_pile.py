"""E97-RAW x GDN HYBRID — LM axis on REAL Pile (task e97-gdn-hybrid, axis 2).

Trains ONE arm of the e97-raw:GDN ratio sweep as a HybridLadderLM language model
on REAL Pile tokens (token-matched), recording the train-loss curve and a held-out
loss / tokenizer-invariant BPB. Answers axis 2: does inserting GDN gated-delta
layers into the e97-raw backbone COST LM loss, or match/beat it?

Same data path / tokenizer / chunking / held-out method as the E99 1.3B controls
(experiments/e99_1p3b_controls/e99_lm_controls.py): REAL /home/erikg/elman/data/
pile.txt, p50k_base, chunk 1024, held-out on a disjoint seed, BPB measured by
decoding the exact held-out target tokens back to UTF-8.

Arms are the SAME interleaved patterns as the expressivity sweep
(run_e97_gdn_hybrid.py), tiled to --depth:
  all-e97raw [E97]            h3to1 [E97,E97,E97,gdn]   h1to1 [E97,gdn]
  h1to3 [E97,gdn,gdn,gdn]     all-gdn [gdn]
E97 == E88FLAHybrid(use_split_edit=True) with raw_write=1, state tanh (the e97-raw
LM winner). `gdn` == GatedDeltaNet (study-A recall bar). bf16 for ALL arms (the
GDN chunked-delta kernel rejects fp32; e97-raw's bounded tanh state is faithful in
bf16) — so the token-matched comparison is in one precision.

Token-matching: every arm uses the SAME batch_size x chunk x steps, so equal
--steps == equal tokens. Shape (dim/depth/n_heads/n_state) is identical across
arms (shape-matched, study-A protocol); per-arm param counts are recorded honestly
(GDN layers are ~5x cheaper, so hybrid/all-gdn arms carry fewer params -> any LM
win by the e97-raw-heavy arms is NOT a param advantage to GDN).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)

import numpy as np
import torch

from ndm.models.hybrid_ladder import HybridLadderLM
from ndm.data.tokenized_dataset import TokenizedStreamDataset

DATA = '/home/erikg/elman/data/pile.txt'
TOKENIZER = 'p50k_base'
LN2 = math.log(2.0)
TRAIN_SEED_BASE = 42
HELDOUT_SEED = 7777   # disjoint from train

ARMS: dict[str, list[str]] = {
    'all-e97raw': ['E97'],
    'h3to1':      ['E97', 'E97', 'E97', 'gdn'],
    'h1to1':      ['E97', 'gdn'],
    'h1to3':      ['E97', 'gdn', 'gdn', 'gdn'],
    'all-gdn':    ['gdn'],
}


def build_model(arm: str, dim: int, depth: int, n_heads: int, n_state: int,
                vocab_size: int, device: str) -> HybridLadderLM:
    pattern = ARMS[arm]
    e97kw = {'state_activation': 'tanh', 'raw_write': True}
    layer_kwargs = [dict(e97kw) if lvl == 'E97' else {} for lvl in pattern]
    m = HybridLadderLM(
        vocab_size=vocab_size, dim=dim, depth=depth,
        layer_pattern=pattern, layer_kwargs=layer_kwargs,
        n_state=n_state, n_heads=n_heads, expansion=1.0,
    ).to(device)
    return m


def heldout_eval(model, ds, bs, n_batches, device):
    """Held-out nats/token, measured bytes/token, BPB on a fixed disjoint slice."""
    model.eval()
    total_nll = 0.0
    total_tok = 0
    total_bytes = 0
    with torch.no_grad():
        for _ in range(n_batches):
            chunks, _, _ = ds.get_batch(bs, device=device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                loss = model(chunks, return_loss=True)
                if isinstance(loss, tuple):
                    loss = loss[0]
            n_pred = chunks.shape[0] * (chunks.shape[1] - 1)
            total_nll += float(loss.item()) * n_pred
            total_tok += n_pred
            for row in chunks.tolist():
                total_bytes += len(ds.enc.decode(row[1:]).encode('utf-8'))
    nats = total_nll / total_tok
    bytes_per_tok = total_bytes / total_tok
    bpb = (nats / LN2) / bytes_per_tok
    return nats, bytes_per_tok, bpb, total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARMS.keys()))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_state', type=int, default=64)
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--eval_interval', type=int, default=500)
    ap.add_argument('--heldout_batches', type=int, default=16)
    ap.add_argument('--outdir', default=os.path.join(_THIS, 'results'))
    ap.add_argument('--label', default=None)
    args = ap.parse_args()

    device = 'cuda'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    label = args.label or f"{args.arm}_s{args.seed}"
    os.makedirs(args.outdir, exist_ok=True)

    train_ds = TokenizedStreamDataset(
        data_path=DATA, chunk_size=args.chunk + 1,
        seed=TRAIN_SEED_BASE + args.seed, tokenizer_name=TOKENIZER)
    held_ds = TokenizedStreamDataset(
        data_path=DATA, chunk_size=args.chunk + 1,
        seed=HELDOUT_SEED, tokenizer_name=TOKENIZER)
    vocab_size = train_ds.vocab_size

    model = build_model(args.arm, args.dim, args.depth, args.n_heads,
                        args.n_state, vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_noembed = n_params - model.embed.weight.numel()
    print(f"[{label}] pattern={model.actual_pattern} params={n_params:,} "
          f"(core {n_params_noembed:,}) vocab={vocab_size}", flush=True)

    import schedulefree
    opt = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    tokens_per_step = args.batch_size * args.chunk
    log = {
        'arm': args.arm, 'seed': args.seed, 'pattern': model.actual_pattern,
        'dim': args.dim, 'depth': args.depth, 'n_heads': args.n_heads,
        'n_state': args.n_state, 'chunk': args.chunk, 'batch_size': args.batch_size,
        'steps': args.steps, 'lr': args.lr, 'params': n_params,
        'params_noembed': n_params_noembed, 'tokens_per_step': tokens_per_step,
        'data': DATA, 'tokenizer': TOKENIZER, 'precision': 'bf16',
        'curve': [],
    }

    model.train()
    opt.train()
    t0 = time.time()
    running = []
    for step in range(args.steps):
        chunks, _, _ = train_ds.get_batch(args.batch_size, device=device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(float(loss.item()))

        if step % args.eval_interval == 0 or step == args.steps - 1:
            opt.eval()
            ho_nats, ho_bytes, ho_bpb, _ = heldout_eval(
                model, held_ds, args.batch_size, args.heldout_batches, device)
            opt.train()
            model.train()
            tok = (step + 1) * tokens_per_step
            tr = float(np.mean(running[-args.eval_interval:])) if running else float(loss.item())
            elapsed = time.time() - t0
            print(f"  [{label}] step {step:>5d} tok {tok:>10d} "
                  f"train {tr:.4f} held {ho_nats:.4f} bpb {ho_bpb:.4f} "
                  f"({elapsed:.0f}s, {tok/max(elapsed,1):.0f} tok/s)", flush=True)
            log['curve'].append({
                'step': step, 'tokens': tok, 'train_loss': tr,
                'heldout_nats': ho_nats, 'heldout_bytes_per_tok': ho_bytes,
                'heldout_bpb': ho_bpb, 'elapsed_s': elapsed,
            })

    # Final held-out with more batches for a stable number.
    opt.eval()
    ho_nats, ho_bytes, ho_bpb, ho_tok = heldout_eval(
        model, held_ds, args.batch_size, max(32, args.heldout_batches), device)
    log['final'] = {
        'train_loss_last10pct': float(np.mean(running[-max(1, args.steps // 10):])),
        'heldout_nats': ho_nats, 'heldout_bytes_per_tok': ho_bytes,
        'heldout_bpb': ho_bpb, 'heldout_tokens': ho_tok,
        'total_tokens': args.steps * tokens_per_step,
        'walltime_s': time.time() - t0,
    }
    out_path = os.path.join(args.outdir, f'{label}.json')
    json.dump(log, open(out_path, 'w'), indent=2)
    print(f"[{label}] FINAL held={ho_nats:.4f} bpb={ho_bpb:.4f} "
          f"train10={log['final']['train_loss_last10pct']:.4f} -> {out_path}", flush=True)


if __name__ == '__main__':
    main()
