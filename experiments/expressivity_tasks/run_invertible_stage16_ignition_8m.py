"""X16: explicit-P1-weight T16 ignition sweep and criterion-gated T16 -> T64."""
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

from experiments.expressivity_tasks import run_invertible_stage14_nofreebie_8m as x14

x10, x7 = x14.x10, x14.x7
ROOT = Path('experiments/expressivity_tasks/results_invertible_stage16_ignition_8m')
ARMS = {
    'e97_w64_seed0': ('e97', 1 / 64, 0),
    'e97_w32_seed0': ('e97', 1 / 32, 1),
    'gdn2_w512_seed0': ('gdn2', 1 / 512, 2),
    'gdn2_w2048_seed0': ('gdn2', 1 / 2048, 3),
    'e97_curriculum_seed0': ('e97', None, 6),
}
CEILING, EVAL_EVERY = 500000, 500


def position_loss(logits, y, mask, weight=None):
    """w is the TOTAL P1 share; the other T-1 positions share 1-w equally."""
    import torch.nn.functional as F
    depth = mask.shape[1] - 2
    if mask[:, :2].any() or not mask[:, 2:].all():
        raise ValueError('expected unchanged full driver-target mask')
    losses = F.cross_entropy(logits[mask].float(), y[mask], reduction='none').reshape(-1, depth)
    if weight is None:
        return losses.mean()
    if not 0 <= weight <= 1:
        raise ValueError('P1 share outside [0,1]')
    return weight * losses[:, 0].mean() + (1 - weight) * losses[:, 1:].mean()


def extension_rate(step, entry_lr):
    """Registered choice: warmup-free cosine from attained LR, moments retained."""
    if not 0 <= step <= CEILING:
        raise ValueError('extension update outside registered schedule')
    return 1e-5 + .5 * (entry_lr - 1e-5) * (1 + math.cos(math.pi * step / CEILING))


def train_update(model, opt, batch, step, weight=None, lr=None):
    """X10 AdamW update ordering/clip/instrumentation, changing only scalar loss/LR."""
    import torch
    if lr is None:
        x10.rates(opt, 'adamw', step)
    else:
        for group in opt.param_groups:
            group['lr'] = lr
    model.train()
    opt.zero_grad(set_to_none=True)
    logits, y, mask = x7.forward_batch(model, batch)
    loss = position_loss(logits, y, mask, weight)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    before = [p.detach().clone() for p in model.parameters()] if step % EVAL_EVERY == 0 else None
    torch.cuda.synchronize()
    start = time.monotonic()
    opt.step()
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    displacement = None
    if before is not None:
        displacement = (sum((p.detach().float() - old.float()).square().sum().item()
                            for p, old in zip(model.parameters(), before)) /
                        sum(p.numel() for p in model.parameters())) ** .5
    return {'loss': loss.item(), 'grad_norm': norm.item(), 'lr': opt.param_groups[0]['lr'],
            'optimizer_seconds': seconds, 'full_update_rms': displacement, 'muon_update_rms': None}


def evaluate(model, pool):
    """X7 evaluation, but derive depth from THIS pool; never mutate module globals."""
    import torch
    import torch.nn.functional as F
    depth = pool[2].shape[1] - 2
    if len(pool[0]) != 512 or pool[2][:, :2].any() or not pool[2][:, 2:].all():
        raise ValueError('expected frozen 512-example full-depth pool')
    model.eval()
    totals, loss_sum, predictions, losses = {}, 0., [], []
    with torch.no_grad():
        for start in range(0, 512, 32):
            batch = tuple(a[start:start + 32] for a in pool)
            logits, y, mask = x7.forward_batch(model, batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('nonfinite evaluation logits')
            pred = logits.argmax(-1).cpu().numpy()
            predictions.append(pred)
            per_target = F.cross_entropy(logits[mask].float(), y[mask], reduction='none')
            loss_sum += per_target.sum().item()
            losses.append(per_target.reshape(32, depth).cpu().numpy())
            for key, value in x7.metric_counts(pred, batch[1], batch[2], 61).items():
                totals[key] = totals.get(key, 0) + value
    positions = x7.position_counts(np.concatenate(predictions), pool[1], pool[2], np.concatenate(losses))
    assert len(positions) == depth and sum(p['correct'] for p in positions) == totals['correct']
    return {'aggregate': x7.finish_metrics(totals, loss_sum, 61), 'positions': positions,
            'perfect_prefix_depth': next((p['driver_position'] - 1 for p in positions
                                          if p['correct'] != 512), depth)}


def taught_positions(metrics):
    return [p['driver_position'] for p in metrics['positions'] if p['correct'] >= 508]


def run(name):
    import torch
    architecture, weight, gpu = ARMS[name]
    registration = x10.verify_registration(ROOT)
    if name not in registration['authorized_runs'] or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('unauthorized run/card')
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=False)
    curriculum = weight is None
    depths = (16, 64) if curriculum else (16,)
    pools = {d: x7.frozen_pools(0, d, 61)['val'] for d in depths}
    for d, pool in pools.items():
        hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]
        if hashes != registration['pool_array_sha256'][str(d)]:
            raise ValueError('pool identity changed')
    np.savez_compressed(out / 'frozen_validation.npz',
                        **{f'T{d}_{i}': a for d, pool in pools.items() for i, a in enumerate(pool)})
    x7.write_json(out / 'data_receipt.json', {'archive_sha256': x7.digest(out / 'frozen_validation.npz'),
                   'pool_array_sha256': {str(d): registration['pool_array_sha256'][str(d)] for d in depths}})
    torch.manual_seed(0)
    model = x14.build_model(architecture).cuda()
    assert sum(p.numel() for p in model.parameters()) == {'e97': 9900800, 'gdn2': 9887056}[architecture]
    opt = x14.make_optimizer(model)
    assert not opt.state
    rng, task = np.random.default_rng(0), x10.InvertibleQuadraticTask(61)
    config = {'name': name, 'arm': 'adamw', 'architecture': architecture, 'seed': 0, 'p1_weight': weight,
              'registration_sha256': x7.digest(ROOT / 'registration.json'), 'registration': registration,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'host': os.uname().nodename}
    x7.write_json(out / 'config.json', config)
    history, checkpoints, best = [], {}, None
    entry = latest = transition_cp = None
    transition = None
    started, total, opt_seconds = time.monotonic(), 0, 0.
    bounds = registration['resource_bounds']
    with (out / 'training.jsonl').open('a', buffering=1) as log:
        for depth in depths:
            checkpoint_dir = out / f'T{depth}'
            checkpoint_dir.mkdir()
            best = None  # Do not compare aggregate losses across lengths.
            for phase_step in range(CEILING + 1):
                if time.monotonic() - started > bounds['wall_seconds']:
                    raise TimeoutError('infrastructure wall ceiling; censored')
                if phase_step:
                    total += 1
                    before = time.monotonic()
                    batch = task.generate_batch(32, depth + 2, rng)
                    if curriculum and depth == 16:
                        metrics = x10.train_update(model, opt, 'adamw', batch, phase_step)
                    else:
                        lr = extension_rate(phase_step, transition['entry_lr']) if depth == 64 else None
                        metrics = train_update(model, opt, batch, phase_step, weight, lr)
                    opt_seconds += metrics['optimizer_seconds']
                    log.write(json.dumps({'step': total, 'depth': depth, 'phase_step': phase_step, **metrics,
                                          'update_seconds': time.monotonic() - before,
                                          'elapsed_s': time.monotonic() - started}) + '\n')
                    if torch.cuda.max_memory_allocated() > bounds['peak_allocated_bytes']:
                        raise MemoryError('infrastructure memory ceiling; censored')
                if phase_step % EVAL_EVERY and phase_step != CEILING:
                    continue
                metrics = evaluate(model, pools[depth])
                criterion = curriculum and depth == 16 and len(taught_positions(metrics)) >= 4
                current = metrics['aggregate']
                is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
                cp = None
                if is_best or phase_step % 1000 == 0 or phase_step == CEILING or criterion:
                    cp = x10.checkpoint(checkpoint_dir, model, opt, rng, total,
                                        {**config, 'depth': depth, 'phase_step': phase_step})
                    checkpoints[cp['path']] = cp
                    latest = cp['path']
                if is_best:
                    best = {**current, 'step': total, 'checkpoint': cp}
                if entry is None:
                    entry = cp['path']
                if criterion:
                    transition_cp = cp['path']
                    transition = {'step': total, 'taught_positions': taught_positions(metrics),
                                  'entry_lr': x10.learning_rate(phase_step), 'checkpoint': cp,
                                  't16': metrics}
                    x7.write_json(out / 'transition.json', transition)
                retention = evaluate(model, pools[16]) if depth == 64 else None
                row = {'step': total, 'phase_step': phase_step, 'depth': depth, **metrics,
                       'retention_t16': retention, 'checkpoint': cp,
                       'elapsed_s': time.monotonic() - started, 'optimizer_seconds_total': opt_seconds,
                       'weight_norm': sum(p.detach().float().square().sum().item() for p in model.parameters()) ** .5,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'decision': 'extend' if criterion else ('safety_ceiling' if phase_step == CEILING else 'continue')}
                history.append(row)
                x7.write_json(out / 'validation.json', history)
                x7.write_json(out / 'checkpoints.json', checkpoints)
                print(json.dumps({'name': name, **row}), flush=True)
                keep = {entry, latest, best['checkpoint']['path']}
                if transition_cp:
                    keep.add(transition_cp)
                x10.prune(out, checkpoints, keep)
                if criterion:
                    break
            if curriculum and depth == 16 and transition is None:
                x7.write_json(out / 'result.json', {'name': name, 'status': 'criterion_not_reached',
                    'total_steps': total, 't64_steps': 0, 'stop_reason': 'teaching_ceiling',
                    'final': history[-1], 'finished_at': time.time()})
                return
    x7.write_json(out / 'result.json', {'name': name, 'status': 'complete', 'total_steps': total,
        't64_steps': CEILING if curriculum else 0, 'stop_reason': 'safety_ceiling',
        'best': best, 'transition': transition, 'final': history[-1],
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=ARMS, required=True)
    run(parser.parse_args().name)
