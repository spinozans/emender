"""X14: scratch T16, P1 diagnostic-only, uniform P2..P16 loss; X10 update path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as x10

x7 = x10.x7
ROOT = Path('experiments/expressivity_tasks/results_invertible_stage14_nofreebie_8m')
NAMES = tuple(f'{arm}_seed{seed}' for seed in (0, 1) for arm in ('e97', 'gdn2'))
DEPTH, CEILING, EVAL_EVERY = 16, 500000, 500


def build_model(architecture):
    if architecture == 'e97':
        return x7.build_model('e97', 'bf16')
    if architecture == 'gdn2':
        return x10.stock_model('gdn2', 61)
    raise ValueError('unregistered architecture')


def make_optimizer(model):
    return x10.make_optimizer(model, 'adamw')


def composition_batch(batch):
    """P1 is token index 2, not an input removal; never mutate the evaluation pool."""
    x, y, mask = batch
    if mask.shape[1] != DEPTH + 2 or mask[:, :2].any() or not mask[:, 2:].all():
        raise ValueError('expected full T16 driver-target mask')
    mask = mask.copy()
    mask[:, 2] = False
    return x, y, mask


def train_update(model, opt, batch, step):
    # Only the loss mask differs: actual X10 schedule, CE, clip, optimizer and logs.
    return x10.train_update(model, opt, 'adamw', composition_batch(batch), step)


def evaluate(model, pool):
    metrics = x7.evaluate(model, pool)
    positions = metrics['positions']
    for p in positions:
        p['role'] = 'diagnostic-only non-composition' if p['driver_position'] == 1 else 'composition'
    carry = positions[1:]
    # P1 cannot select best checkpoints or inflate the primary aggregate.
    metrics['all_positions_diagnostic_aggregate'] = metrics.pop('aggregate')
    metrics.pop('perfect_prefix_depth')
    metrics['aggregate'] = {
        'accuracy': sum(p['correct'] for p in carry) / (512 * len(carry)),
        'correct': sum(p['correct'] for p in carry), 'count': 512 * len(carry),
        'loss': sum(p['loss'] for p in carry) / len(carry), 'scope': 'P2..P16 only'}
    metrics['composition_perfect_prefix_length'] = next(
        (i for i, p in enumerate(carry) if p['correct'] != 512), len(carry))
    return metrics


def run(args):
    name = f'{args.arm}_seed{args.seed}'
    if name not in NAMES:
        raise ValueError('unregistered run')
    import torch
    registration = x10.verify_registration(ROOT)
    if name not in registration['authorized_runs']:
        raise ValueError('unauthorized run')
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=False)
    pool = x7.frozen_pools(args.seed, DEPTH, 61)['val']
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]
    if hashes != registration['pool_array_sha256'][str(args.seed)]:
        raise ValueError('pool identity changed')
    np.savez_compressed(out / 'frozen_validation.npz', **{f'T16_{i}': a for i, a in enumerate(pool)})
    x7.write_json(out / 'data_receipt.json', {'array_sha256': hashes,
                  'archive_sha256': x7.digest(out / 'frozen_validation.npz')})
    torch.manual_seed(args.seed)
    model = build_model(args.arm).cuda()
    assert sum(p.numel() for p in model.parameters()) == {'e97': 9900800, 'gdn2': 9887056}[args.arm]
    opt = make_optimizer(model)
    assert not opt.state
    rng, task = np.random.default_rng(args.seed), x10.InvertibleQuadraticTask(61)
    config = {'arm': 'adamw', 'architecture': args.arm, 'seed': args.seed, 'name': name,
              'weights': x10.point('adamw'), 'registration': registration,
              'registration_sha256': x7.digest(ROOT / 'registration.json'),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'host': os.uname().nodename}
    x7.write_json(out / 'config.json', config)
    history, checkpoints, best, entry = [], {}, None, None
    started, opt_seconds = time.monotonic(), 0.
    bounds = registration['resource_bounds']
    with (out / 'training.jsonl').open('a', buffering=1) as log:
        for step in range(CEILING + 1):
            if time.monotonic() - started > bounds['wall_seconds']:
                raise TimeoutError('infrastructure wall ceiling; censored, not optimizer failure')
            if step:
                before = time.monotonic()
                metrics = train_update(model, opt, task.generate_batch(32, DEPTH + 2, rng), step)
                opt_seconds += metrics['optimizer_seconds']
                log.write(json.dumps({'step': step, **metrics, 'update_seconds': time.monotonic() - before,
                                     'elapsed_s': time.monotonic() - started}) + '\n')
                if torch.cuda.max_memory_allocated() > bounds['peak_allocated_bytes']:
                    raise MemoryError('infrastructure memory ceiling; censored')
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
            row = {'step': step, **metrics, 'checkpoint': cp,
                   'elapsed_s': time.monotonic() - started, 'optimizer_seconds_total': opt_seconds,
                   'weight_norm': sum(p.detach().float().square().sum().item() for p in model.parameters()) ** .5,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'weights': x10.point('adamw'),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['e97', 'gdn2'], required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    run(parser.parse_args())
