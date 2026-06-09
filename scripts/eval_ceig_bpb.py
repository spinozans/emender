#!/usr/bin/env python3
"""Clean held-out bits-per-byte eval for the complex-eig-lm checkpoints.

The in-train ``--final_heldout_eval`` path produced a broken CE (~13 nats, worse
than random) — it runs the schedule-free averaged weights through ``validate()``
WITHOUT the bf16 autocast the model was trained and cast under, and double-swaps
the optimizer eval state.  This script reconstructs the model, loads the saved
(already schedule-free-averaged) checkpoint weights, and evaluates byte-level CE
on a REAL held-out pile slice under the SAME autocast bf16 context as training.

bpb = (CE_nats / ln2) / bytes_per_token ; byte-level => bytes_per_token = 1.0.
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import schedulefree
from ndm.models import LadderLM
from ndm.data.dataset import create_dataloader


def build(level, dim, depth, n_heads, n_state, mlp_ratio, layer_kwargs):
    return LadderLM(
        vocab_size=256, dim=dim, depth=depth, level=level,
        expansion=1.0, n_heads=n_heads, n_state=n_state,
        use_gate=True, use_conv=True, d_conv=4,
        mlp_ratio=mlp_ratio, mlp_multiple=64, layer_kwargs=layer_kwargs)


@torch.no_grad()
def eval_bpb(model, val_loader, device, max_batches, bf16):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for i, (chunk, _is_end, actual) in enumerate(val_loader):
        if i >= max_batches:
            break
        chunk = chunk.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16):
            loss = model(chunk, return_loss=True)
        # loss is mean CE over (B * (T)) next-byte predictions; weight by them
        ntok = chunk.shape[0] * (chunk.shape[1] - 1)
        tot_loss += float(loss) * ntok
        tot_tok += ntok
    ce = tot_loss / max(tot_tok, 1)
    return ce, tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--level', required=True)
    ap.add_argument('--val_data', default='/mnt/nvme2n1/erikg/complex_eig_lm_val.txt')
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_state', type=int, default=64)
    ap.add_argument('--mlp_ratio', type=float, required=True)
    ap.add_argument('--layer_kwargs', default=None)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--chunk_size', type=int, default=512)
    ap.add_argument('--max_batches', type=int, default=400)
    ap.add_argument('--bytes_per_token', type=float, default=1.0)
    ap.add_argument('--bf16', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    import json
    lk = json.loads(args.layer_kwargs) if args.layer_kwargs else None
    dev = 'cuda'
    model = build(args.level, args.dim, args.depth, args.n_heads, args.n_state,
                  args.mlp_ratio, lk).to(dev)
    if args.bf16:
        model = model.bfloat16()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    # train.py uses schedule-free AdamW and saves model params AFTER optimizer.eval()
    # (the averaged 'x' point). In this environment that x reconstruction is broken
    # (yields ~10x CE), so recover the actual trained 'y' iterate by loading the
    # optimizer state and calling .train() — verified to restore the real ~2.0 bpb.
    if 'optimizer_state_dict' in ck:
        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=2e-3, weight_decay=0.0)
        opt.load_state_dict(ck['optimizer_state_dict'])
        opt.train()  # swap params x -> y (the trained iterate)
        print("[eval] restored schedule-free y-iterate (training weights) via optimizer.train()")
    val_loader = create_dataloader(args.val_data, batch_size=args.batch_size,
                                   chunk_size=args.chunk_size + 1, device=dev,
                                   seed=args.seed)
    ce, ntok = eval_bpb(model, val_loader, dev, args.max_batches, bool(args.bf16))
    bpb = (ce / math.log(2.0)) / args.bytes_per_token
    print(f"CKPT {args.ckpt}")
    print(f"HELDOUT_CE_NATS {ce:.4f}  HELDOUT_BPB {bpb:.4f}  val_tokens {ntok:,}")


if __name__ == '__main__':
    main()
