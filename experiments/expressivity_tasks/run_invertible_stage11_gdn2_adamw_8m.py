"""X11: stock bf16 GDN2, scratch T16, exact X10 ordinary-AdamW recipe."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as x10

x7 = x10.x7
ROOT = Path('experiments/expressivity_tasks/results_invertible_stage11_gdn2_adamw_8m')
NAME = 'gdn2_adamw_seed0'
DEPTH, CEILING, EVAL_EVERY = 16, 500000, 500


def build_model():
    return x10.stock_model('gdn2', 61)


def make_optimizer(model):
    return x10.make_optimizer(model, 'adamw')


def train_update(model, opt, batch, step):
    # Reuse the actual X10 update, including uniform CE, schedule, clip and LR log.
    return x10.train_update(model, opt, 'adamw', batch, step)


def run():
    import torch
    registration = x10.verify_registration(ROOT)
    if NAME not in registration['authorized_runs']:
        raise ValueError('unauthorized run')
    out = ROOT / NAME
    out.mkdir(parents=True, exist_ok=False)
    pool = x7.frozen_pools(0, DEPTH, 61)['val']
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]
    if hashes != registration['pool_array_sha256']['0']:
        raise ValueError('pool identity changed')
    np.savez_compressed(out / 'frozen_validation.npz', **{f'T16_{i}': a for i, a in enumerate(pool)})
    x7.write_json(out / 'data_receipt.json', {'array_sha256': hashes,
                  'archive_sha256': x7.digest(out / 'frozen_validation.npz')})
    torch.manual_seed(0)
    model = build_model().cuda()
    assert sum(p.numel() for p in model.parameters()) == 9887056
    opt = make_optimizer(model)
    assert not opt.state
    rng, task = np.random.default_rng(0), x10.InvertibleQuadraticTask(61)
    config = {'arm': 'adamw', 'architecture': 'gdn2', 'seed': 0, 'name': NAME,
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
            metrics = x7.evaluate(model, pool)
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
            print(json.dumps({'name': NAME, **row}), flush=True)
            x10.prune(out, checkpoints, {entry, best['checkpoint']['path'], latest})
    x7.write_json(out / 'result.json', {'name': NAME, 'status': 'complete', 'total_steps': CEILING,
        'stop_reason': 'safety_ceiling', 'best': best, 'final': history[-1],
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time()})


if __name__ == '__main__':
    run()
