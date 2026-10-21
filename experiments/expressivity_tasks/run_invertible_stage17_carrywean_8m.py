"""X17: scratch T64 E97 with X6 carry slots and a single global million-update wean."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as x10

x7 = x10.x7
ROOT = Path('experiments/expressivity_tasks/results_invertible_stage17_carrywean_8m')
NAMES = ('e97_carrywean_seed0',)
DEPTH, CEILING, EVAL_EVERY = 64, 1000000, 500
PARAMETERS = 9901056
NEUTRAL = 3721


def learning_rate(step):
    if not 0 <= step <= CEILING:
        raise ValueError('update outside registered schedule')
    if step <= 5000:
        return .001 * step / 5000
    return .00001 + .5 * (.001 - .00001) * (1 + math.cos(math.pi * (step - 5000) / (CEILING - 5000)))


def carry_probability(step):
    """One global wean; never reset by stage, validation, or acquisition."""
    if not 0 <= step <= CEILING:
        raise ValueError('update outside registered carry schedule')
    return .5 * (1 - step / CEILING)


def carry_inputs(x, y, probability=0., rng=None):
    """Exact X6 Arm B: P2+ previous-state slots; P1 has no extra slot.

    With probability zero, y is never read and no RNG is consumed.
    """
    if not 0 <= probability <= 1:
        raise ValueError('invalid carry probability')
    depth = x.shape[1] - 2
    positions = np.r_[0, 1, 2 + 2 * np.arange(depth)]
    expanded = np.full((len(x), 2 * depth + 1), NEUTRAL, dtype=x.dtype)
    expanded[:, positions] = x
    if probability and depth > 1:
        selected = rng.random((len(x), depth - 1)) < probability
        expanded[:, 3::2] = np.where(selected, y[:, 2:-1], NEUTRAL)
    return expanded, positions


def build_model():
    import torch
    model = x7.build_model('e97', 'bf16')
    class CarryEmbedding(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.weight = embedding.weight
            self.neutral = torch.nn.Parameter(torch.empty(1, embedding.embedding_dim))
            torch.nn.init.normal_(self.neutral)
        def forward(self, x):
            return torch.nn.functional.embedding(x, torch.cat((self.weight, self.neutral)))
    model.embed = CarryEmbedding(model.embed)
    return model


def forward_batch(model, batch, probability=0., rng=None):
    import torch
    x, y, mask = batch
    x, positions = carry_inputs(x, y, probability, rng)
    x, y, mask = [torch.from_numpy(a).cuda() for a in (x, y, mask)]
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = x7.padded_forward(model, x)
    return logits[:, positions], y, mask


def make_optimizer(model):
    return x10.make_optimizer(model, 'adamw')


def train_update(model, opt, batch, step, rng):
    import torch
    import torch.nn.functional as F
    for group in opt.param_groups:
        group['lr'] = learning_rate(step)
    model.train()
    opt.zero_grad(set_to_none=True)
    logits, y, mask = forward_batch(model, batch, carry_probability(step), rng)
    loss = F.cross_entropy(logits[mask].float(), y[mask])
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return {'loss': loss.item(), 'grad_norm': norm.item(), 'lr': opt.param_groups[0]['lr'],
            'carry_probability': carry_probability(step)}


def evaluate(model, pool):
    """X7's exact per-position instrument, with T64 rather than hard-coded T16."""
    import torch
    import torch.nn.functional as F
    model.eval()
    totals, loss_sum, predictions, losses = {}, 0., [], []
    with torch.no_grad():
        for start in range(0, 512, 32):
            batch = tuple(a[start:start + 32] for a in pool)
            logits, y, mask = forward_batch(model, batch)  # Always neutral; no true feedback.
            if not torch.isfinite(logits).all():
                raise FloatingPointError('nonfinite evaluation logits')
            pred = logits.argmax(-1).cpu().numpy()
            predictions.append(pred)
            per_target = F.cross_entropy(logits[mask].float(), y[mask], reduction='none')
            loss_sum += per_target.sum().item()
            losses.append(per_target.reshape(32, DEPTH).cpu().numpy())
            for key, value in x7.metric_counts(pred, batch[1], batch[2], 61).items():
                totals[key] = totals.get(key, 0) + value
    positions = x7.position_counts(np.concatenate(predictions), pool[1], pool[2], np.concatenate(losses))
    assert len(positions) == DEPTH and all(p['count'] == 512 for p in positions)
    assert sum(p['correct'] for p in positions) == totals['correct']
    return {'aggregate': x7.finish_metrics(totals, loss_sum, 61), 'positions': positions,
            'perfect_prefix_depth': next((p['driver_position'] - 1 for p in positions if p['correct'] != 512), DEPTH)}


def run():
    import torch
    registration = x10.verify_registration(ROOT)
    name = NAMES[0]
    if name not in registration['authorized_runs']:
        raise ValueError('unauthorized run')
    # Direct execution cannot bypass the native worker/lease discipline.
    admission_path = ROOT / 'receipts' / name / 'admission.json'
    admission = json.loads(admission_path.read_text())
    if (admission['registration_sha256'] != x7.digest(ROOT / 'registration.json') or
            admission['gpu'] != int(os.environ['CUDA_VISIBLE_DEVICES'])):
        raise ValueError('invalid card-wait admission')
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=False)
    pool = x7.frozen_pools(0, DEPTH, 61)['val']
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]
    if hashes != registration['pool_array_sha256']['0']:
        raise ValueError('pool identity changed')
    np.savez_compressed(out / 'frozen_validation.npz', **{f'T64_{i}': a for i, a in enumerate(pool)})
    x7.write_json(out / 'data_receipt.json', {'array_sha256': hashes,
                  'archive_sha256': x7.digest(out / 'frozen_validation.npz')})
    torch.manual_seed(0)
    model = build_model().cuda()
    assert len(model.layers) == 4 and sum(p.numel() for p in model.parameters()) == PARAMETERS
    opt = make_optimizer(model)
    assert not opt.state
    rng, task = np.random.default_rng(0), x10.InvertibleQuadraticTask(61)
    config = {'arm': 'adamw', 'architecture': 'e97', 'seed': 0, 'name': name,
              'weights': x10.point('adamw'), 'registration': registration,
              'registration_sha256': x7.digest(ROOT / 'registration.json'),
              'admission': admission,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'host': os.uname().nodename}
    x7.write_json(out / 'config.json', config)
    history, checkpoints, best, entry = [], {}, None, None
    started = time.monotonic()
    with (out / 'training.jsonl').open('a', buffering=1) as log:
        for step in range(CEILING + 1):
            if step:
                metrics = train_update(model, opt, task.generate_batch(32, DEPTH + 2, rng), step, rng)
                log.write(json.dumps({'step': step, **metrics, 'elapsed_s': time.monotonic() - started}) + '\n')
            if step % EVAL_EVERY and step != CEILING:
                continue
            metrics = evaluate(model, pool)
            current = metrics['aggregate']
            is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
            cp = None
            if is_best or step % 1000 == 0 or step == CEILING:
                cp = x10.checkpoint(out, model, opt, rng, step, config)
                checkpoints[cp['path']] = cp
                latest = cp['path']
            if is_best:
                best = {**current, 'step': step, 'checkpoint': cp}
            if step == 0:
                entry = cp['path']
            row = {'step': step, 'carry_probability': carry_probability(step), 'eval_carry_probability': 0., **metrics, 'checkpoint': cp, 'elapsed_s': time.monotonic() - started,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'decision': 'safety_ceiling' if step == CEILING else 'continue'}
            history.append(row)
            x7.write_json(out / 'validation.json', history)
            x7.write_json(out / 'checkpoints.json', checkpoints)
            print(json.dumps({'name': name, **row}), flush=True)
            x10.prune(out, checkpoints, {entry, best['checkpoint']['path'], latest})
    x7.write_json(out / 'result.json', {'name': name, 'status': 'complete', 'total_steps': CEILING,
        'stop_reason': 'safety_ceiling', 'best': best, 'final': history[-1],
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time()})


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
