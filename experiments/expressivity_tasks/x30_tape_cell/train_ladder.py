"""Ladder reproduction: train ID, select on validation, evaluate frozen bins once.

Warm start from stage1.pt is required, matching the certified curriculum's
shared tape parameters. The intermediate stage-2 occupancy curriculum is
currently omitted and reported as a deviation. No bin influences selection.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .train_stage1 import ROOT, WordModel, artifact_path, device_for, save_json
from experiments.expressivity_tasks.x28e_memory_head.generators import stage3_ladder as data


@torch.no_grad()
def evaluate(model, pool, device):
    model.eval()
    xs, ys, lengths = pool[:3]
    preds = []
    # Keep evaluation memory bounded for long 9x words. This changes no state:
    # every example owns its own independent store.
    for start in range(0, len(xs), 8):
        ln = lengths[start:start+8]
        ids = torch.from_numpy(xs[start:start+8, :int(ln.max())]).to(device)
        logits = model.ladder_logits(ids, torch.from_numpy(ln).to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
    return data.binary_report(ys, np.array(preds))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--budget', type=int, default=736)
    p.add_argument('--updates', type=int, default=30000)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stage1', default=str(ROOT/'runs/stage1.pt'))
    p.add_argument('--checkpoint', default=str(ROOT/'runs/ladder.pt'))
    p.add_argument('--output', default=str(ROOT/'runs/ladder.json'))
    args = p.parse_args()
    torch.set_num_threads(1)
    dev = device_for(args.device)
    torch.manual_seed(args.seed)
    model = WordModel(args.heads, args.budget)
    model.load_state_dict(torch.load(artifact_path(args.stage1), map_location='cpu', weights_only=True))
    model.to(dev)
    model.attach_ladder()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=1e-4)
    train, val = data.ladder_train_pool(), data.ladder_validation_pool()
    rng = np.random.default_rng(0)
    best, best_update, log = -1., None, []
    begin = time.perf_counter()
    for update in range(1, args.updates+1):
        model.train()
        idx = rng.choice(len(train[0]), 64, replace=False)
        opt.zero_grad(set_to_none=True)
        total = 0.
        # Effective batch 64, gradient accumulation; each sample is weighted
        # exactly 1/64. Length sorting limits padding without changing loss.
        idx = idx[np.argsort(train[2][idx])]
        for start in range(0, 64, 8):
            mb = idx[start:start+8]
            lengths = train[2][mb]
            ids = torch.from_numpy(train[0][mb, :int(lengths.max())]).to(dev)
            labels = torch.from_numpy(train[1][mb]).to(dev)
            logits = model.ladder_logits(ids, torch.from_numpy(lengths).to(dev))
            loss = F.binary_cross_entropy_with_logits(logits[:, 1]-logits[:, 0],
                                                       labels.float(), reduction='sum')/64
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if update % 10 == 0:
            print(json.dumps(dict(update=update, loss=total,
                                  seconds=time.perf_counter()-begin)), flush=True)
        if update % args.eval_every == 0 or update == args.updates:
            report = evaluate(model, val, dev)
            metric = report['balanced_accuracy']
            log.append(dict(update=update, validation=report, seconds=time.perf_counter()-begin))
            if metric >= best:
                best, best_update = metric, update
                torch.save(model.state_dict(), artifact_path(args.checkpoint))
            save_json(args.output, dict(config=vars(args), validation=log,
                                       best=best, best_update=best_update, bins_evaluated=False))
            print(json.dumps(log[-1]), flush=True)
    model.load_state_dict(torch.load(artifact_path(args.checkpoint), map_location=dev, weights_only=True))
    bins, hashes = {}, {}
    for name in data.BIN_ORDER:
        pool = data.ladder_bin_pool(name)
        bins[name] = evaluate(model, pool, dev)
        hashes[name] = data.ladder_hash(name, *pool[:3])
    scores = {k: v['balanced_accuracy'] for k, v in bins.items()}
    gate = (scores['bin_id'] >= .99 and scores['bin_2x'] >= .9
            and scores['bin_4x'] >= .9 and scores['bin_9x'] <= .55)
    save_json(args.output, dict(config=vars(args), validation=log,
                               best=best, best_update=best_update, bins=bins,
                               pool_hashes=hashes, gate=gate,
                               physical_gpu=os.environ.get('GPU_LEASE_HELD'),
                               recipe_deviations=['stage-2 occupancy curriculum omitted',
                                                  'H=1 default versus native H=4',
                                                  'random per-update batches versus epoch permutations']))
    return 0 if gate else 2


if __name__ == '__main__':
    raise SystemExit(main())
