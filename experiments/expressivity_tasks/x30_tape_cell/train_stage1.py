"""Reproducible MQAR run. Invoke with python -m ...x30_tape_cell.train_stage1.

CUDA is refused unless the lease wrapper supplied one card in 0..6.
All run artifacts must live in this package. Checkpoints are ignored by git.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cell import TapeCell, TapeHeadConfig
from experiments.expressivity_tasks.x28e_memory_head.generators import stage1_mqar as data

ROOT = Path(__file__).resolve().parent


def device_for(name):
    if name == 'cuda':
        held = os.environ.get('GPU_LEASE_HELD', '')
        if held not in tuple(str(i) for i in range(7)) or os.environ.get('CUDA_VISIBLE_DEVICES') != held:
            raise RuntimeError('CUDA requires a single lease on physical card 0..6; use run_leased.sh')
    return torch.device(name)


def artifact_path(name):
    path = Path(name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('X30 artifacts must stay inside x30_tape_cell/')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class WordModel(nn.Module):
    """Native residual read layer + stage-local embedding and classifier.

    F12f's frozen embedding is intentional: this matches the actual certified
    trainer (its embeddings are frozen despite the task's learned-embedding
    shorthand). The pressure feature is returned by the cell; the baseline
    decoder does not consume it, preserving comparison with native training.
    """
    def __init__(self, heads=4, budget=736, width=256, key=64, value=128):
        super().__init__()
        cfg = TapeHeadConfig(n_heads=heads, budget=budget, key_dim=key, value_dim=value)
        self.cell = TapeCell(width, cfg)
        self.W_o = nn.Linear(heads*value, width)
        self.binary = nn.Linear(width, 2)
        self.embed = nn.Embedding(65, width)
        self.embed.requires_grad_(False)
        self.classifier = nn.Linear(width, 64)

    def features(self, ids):
        x = self.embed(ids)
        read, state, trace, pressure = self.cell(x, ids)
        y = self.W_o(read.flatten(2)) + x
        return y, trace, pressure

    def forward(self, ids):
        return self.classifier(self.features(ids)[0])

    def attach_ladder(self):
        self.embed = nn.Embedding(5, self.embed.embedding_dim).to(self.W_o.weight.device)
        self.embed.requires_grad_(False)

    def ladder_logits(self, ids, lengths):
        y, trace, pressure = self.features(ids)
        return self.binary(y[torch.arange(len(ids), device=ids.device), lengths-1])


def save_json(path, value):
    path = artifact_path(path)
    path.write_text(json.dumps(value, indent=2) + '\n')


def trim_mqar(ids, mask):
    # Trailing pad outputs have no supervision and cannot influence earlier
    # outputs. Keep the registered T=128 pool; omit only its dead suffix from
    # execution. No content token or supervised query is omitted.
    used = np.flatnonzero(np.any((ids != 0) | mask, axis=0))
    return int(used[-1])+1 if len(used) else 1


@torch.no_grad()
def evaluate(model, pool, device, batch=64):
    model.eval()
    xs, ys, mask, classes = pool
    true, pred = [], []
    for start in range(0, len(xs), batch):
        x, y, m = xs[start:start+batch], ys[start:start+batch], mask[start:start+batch]
        t = trim_mqar(x, m)
        logits = model(torch.from_numpy(x[:, :t]).to(device))
        p = logits.argmax(-1).cpu().numpy()+1
        true.extend(y[:, :t][m[:, :t]].tolist())
        pred.extend(p[m[:, :t]].tolist())
    return data.mean_per_class_recall(np.array(pred), np.array(true), classes)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--budget', type=int, default=736)
    p.add_argument('--updates', type=int, default=5000)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--validation-size', type=int, default=20000,
                   help='less than 20000 is a smoke, not acceptance')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default=str(ROOT/'runs/stage1.json'))
    p.add_argument('--checkpoint', default=str(ROOT/'runs/stage1.pt'))
    return p


def main():
    args = parser().parse_args()
    torch.set_num_threads(1)
    device = device_for(args.device)
    torch.manual_seed(args.seed)
    model = WordModel(args.heads, args.budget).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=1e-4)
    train = data.generate_pool(128, 70000, data.STAGE1_SPLITS['train'].seed)
    val = data.generate_pool(128, args.validation_size, data.STAGE1_SPLITS['validation'].seed)
    rng = np.random.default_rng(0)
    log, best, best_update = [], -1., None
    begin = time.perf_counter()
    for update in range(1, args.updates+1):
        model.train()
        idx = rng.choice(len(train[0]), 64, replace=False)
        x, y, mask = (a[idx] for a in train[:3])
        t = trim_mqar(x, mask)
        ids = torch.from_numpy(x[:, :t]).to(device)
        labels = torch.from_numpy(y[:, :t]-1).to(device)
        m = torch.from_numpy(mask[:, :t]).to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(ids)
        loss = F.cross_entropy(logits[m], labels[m])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if update % 10 == 0:
            print(json.dumps({'update': update, 'loss': loss.item(),
                              'seconds': time.perf_counter()-begin}), flush=True)
        if update % args.eval_every == 0 or update == args.updates:
            score = evaluate(model, val, device)
            row = dict(update=update, mean_per_class_recall=score,
                       seconds=time.perf_counter()-begin)
            log.append(row)
            print(json.dumps(row), flush=True)
            if score >= best:
                best, best_update = score, update
                torch.save(model.state_dict(), artifact_path(args.checkpoint))
            result = dict(config=vars(args), validation=log, best=best,
                          best_update=best_update,
                          gate=best >= .99 and args.validation_size == 20000 and update <= 5000,
                          physical_gpu=os.environ.get('GPU_LEASE_HELD'),
                          train_hash=data.stage1_hash('train', *train[:3]),
                          validation_hash=data.stage1_hash('validation', *val[:3]),
                          recipe_deviations=['execution trims unsupervised trailing pads',
                                             'random batches without replacement within update; not epoch permutations'])
            save_json(args.output, result)
            if result['gate']:
                return 0
    return 0 if best >= .99 and args.validation_size == 20000 else 2


if __name__ == '__main__':
    raise SystemExit(main())
