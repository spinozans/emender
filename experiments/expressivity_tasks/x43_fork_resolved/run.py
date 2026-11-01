"""Byte-only shakedown. No index substitutes and no follow-up execution."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Candidate, Policy

ROOT = Path(__file__).resolve().parent


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Control(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.embed = nn.Embedding(256, 256)
        self.position = nn.Embedding(128, 256)
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 256, bias=False)
        # LayerNorms, attention, embeddings, and FF biases included exactly.
        hidden = min(range(1, 4096), key=lambda h: abs(
            256*256*2 + 128*256 + 512 + 2*(4*256*256+4*256+1024+256+h*513)-target))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(256, 4, hidden, dropout=0.,
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(2)])

    def forward(self, ids):
        x = self.embed(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        mask = torch.ones(ids.shape[1], ids.shape[1], device=ids.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)
        return self.head(self.norm(x)).float()


def logits(model, ids, arm):
    if arm == 'emender':
        return F.linear(model.features(ids).float(), model.head.weight)
    return model(ids)


def main():
    assert os.environ.get('GPU_LEASE_HELD') in ('2', '4'), 'broker lease required'
    torch.set_num_threads(1)
    data_bytes = (ROOT/'input.txt').read_bytes()
    data = torch.tensor(list(data_bytes), dtype=torch.long, device='cuda')
    split = int(.9*len(data))
    meta = dict(corpus_bytes=len(data), corpus_sha256=hashlib.sha256(data_bytes).hexdigest(),
                split=split, vocab=256, context=128, batch=16, seed=4301,
                physical_gpu=os.environ['GPU_LEASE_HELD'],
                index_status='blocked: prescribed cargo build failed; see xsa_build.log',
                preregistration_sha256=hashlib.sha256((ROOT/'PREREGISTRATION.md').read_bytes()).hexdigest())
    rows = []
    target = 2_000_000
    for arm in ('emender', 'transformer'):
        torch.manual_seed(4301)
        model = (Candidate(Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.),
                           vocab=256, d=256, layers=2, target=target)
                 if arm == 'emender' else Control(target)).cuda().train()
        if arm == 'emender':
            target = count(model)
        parameters = count(model)
        assert 1_000_000 <= parameters <= 5_000_000
        if arm == 'transformer':
            assert abs(parameters-target)/target < .001
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=.01)
        # Identical sampled batches for the shared prefix of optimizer steps.
        rng = torch.Generator(device='cuda').manual_seed(4301)
        start = time.monotonic()
        steps = 0
        curve = []
        while time.monotonic()-start < 180:
            starts = torch.randint(0, split-128, (16,), device='cuda', generator=rng)
            batch = data[starts[:, None]+torch.arange(129, device='cuda')]
            opt.zero_grad(set_to_none=True)
            if arm == 'emender':
                loss = model.loss(batch[:, :-1], batch[:, 1:])
            else:
                loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1,256), batch[:,1:].reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            steps += 1
            if steps == 1 or steps % 50 == 0:
                curve.append(dict(step=steps, seconds=time.monotonic()-start, nll=float(loss.detach())))
                print(arm, curve[-1], flush=True)
        torch.cuda.synchronize()
        train_seconds = time.monotonic()-start
        torch.save(model.state_dict(), ROOT/f'{arm}.pt')
        model.eval()
        losses, correct, positions = [], [], []
        with torch.no_grad():
            for lo in range(split, len(data)-1, 128):
                hi = min(lo+128, len(data)-1)
                pred = logits(model, data[lo:hi][None], arm)[0]
                truth = data[lo+1:hi+1]
                losses.append(F.cross_entropy(pred,truth,reduction='none').cpu().numpy())
                correct.append((pred.argmax(-1)==truth).cpu().numpy())
                positions.append(np.arange(lo+1,hi+1))
        nll = np.concatenate(losses)
        accuracy = np.concatenate(correct)
        pos = np.concatenate(positions)
        assert len(pos)==len(data)-split-1 and np.all(np.diff(pos)==1)
        np.savez_compressed(ROOT/f'{arm}_per_position.npz', position=pos,nll=nll,correct=accuracy)
        row = dict(arm=arm, trainable_parameters=parameters, total_parameters=sum(p.numel() for p in model.parameters()),
                   steps=steps, train_seconds=train_seconds, heldout_targets=len(pos),
                   heldout_nll=float(nll.mean()), heldout_accuracy=float(accuracy.mean()), curve=curve,
                   fork_nll=None, forced_nll=None, fork_nll_fraction=None, fork_accuracy=None,
                   decomposition_status='blocked: no xsa index or fork labels')
        rows.append(row)
        (ROOT/'receipts/results.json').write_text(json.dumps(dict(metadata=meta, models=rows), indent=2)+'\n')
        print(json.dumps(row), flush=True)
        del opt, model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
