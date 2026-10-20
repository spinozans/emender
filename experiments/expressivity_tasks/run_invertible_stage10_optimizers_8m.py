"""X7b matched stock GDN2 control and X10 seed-matched optimizer instrument.

Standalone single-card research only; no production optimizer or merge changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks import run_invertible_stage7_deepfirst_8m as x7
from experiments.expressivity_tasks.run_invertible_stage3_8m import build_model as stock_model
from experiments.expressivity_tasks.run_invertible_stage9_anneal_8m import learning_rate
from experiments.expressivity_tasks.hybrid_muon import HybridMuon, parameter_routes
from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask

DEPTH, CEILING, EVAL_EVERY = 16, 500000, 500
ARMS = ('muon', 'adamw', 'sf', 'gdn2_uniform')
ROOTS = {arm: Path('experiments/expressivity_tasks/results_invertible_stage' +
         ('7b_stock_8m' if arm == 'gdn2_uniform' else '10_optimizers_8m')) for arm in ARMS}
CARDS = dict(zip(ARMS, (2, 3, 7, 4)))


def build_model(arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    return stock_model('gdn2', 61) if arm == 'gdn2_uniform' else x7.build_model('e97', 'bf16')


def make_optimizer(model, arm):
    import torch
    import schedulefree
    if arm == 'muon':
        return HybridMuon(model)
    if arm == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=0., betas=(.9, .999), eps=1e-8, weight_decay=.01)
    if arm in ('sf', 'gdn2_uniform'):
        return schedulefree.AdamWScheduleFree(model.parameters(), lr=.001, betas=(.9, .95), weight_decay=.01)
    raise ValueError('unregistered optimizer')


def point(arm):
    return 'ScheduleFree eval-x (train-y restored before next update)' if arm in ('sf', 'gdn2_uniform') else 'ordinary live parameters'


def rates(opt, arm, step):
    for group in opt.param_groups:
        group['lr'] = learning_rate(step) if arm in ('muon', 'adamw') else .001


def checkpoint(out, model, opt, rng, step, config):
    import torch
    path = out / f'step_{step:06d}.pt'
    tmp = path.with_suffix('.pt.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step, 'config': config,
                'weights': point(config['arm']), 'optimizer_family': config['arm']}, tmp)
    tmp.replace(path)
    return {'path': str(path), 'sha256': x7.digest(path), 'step': step}


def prune(out, checkpoints, keep):
    """Unconditionally bounded: entry + best + latest/final, plus one atomic staging file."""
    removed = []
    for path, cp in checkpoints.items():
        if path not in keep and Path(path).exists():
            Path(path).unlink()
            removed.append(cp)
    with (out / 'pruning.jsonl').open('a') as log:
        log.write(json.dumps({'removed': removed, 'retained': [checkpoints[p] for p in sorted(keep)]}) + '\n')


def verify_registration(root):
    """Both current worktree and committed blobs must match all bound bytes."""
    path = root / 'registration.json'
    registration = json.loads(path.read_text())
    bound = {str(path): x7.digest(path), **registration['source_sha256'], **registration['dependency_sha256']}
    for name, sha in bound.items():
        if x7.digest(name) != sha:
            raise ValueError('changed registered bytes: ' + name)
        if not Path(name).is_absolute():
            blob = subprocess.check_output(['git', 'show', 'HEAD:' + name])
            if hashlib.sha256(blob).hexdigest() != sha:
                raise ValueError('uncommitted registered source: ' + name)
    return registration


def train_update(model, opt, arm, batch, step):
    import torch
    import torch.nn.functional as F
    rates(opt, arm, step)
    if arm in ('sf', 'gdn2_uniform'):
        opt.train()
    model.train()
    opt.zero_grad(set_to_none=True)
    logits, y, mask = x7.forward_batch(model, batch)
    loss = F.cross_entropy(logits[mask].float(), y[mask])
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    # Full displacement RMS (including decay/auxiliary updates) on validation
    # boundaries; per-matrix Muon direction RMS is additionally recorded there.
    before = [p.detach().clone() for p in model.parameters()] if step % EVAL_EVERY == 0 else None
    torch.cuda.synchronize()
    start = time.monotonic()
    opt.step()
    torch.cuda.synchronize()
    optimizer_seconds = time.monotonic() - start
    displacement_rms = None
    if before is not None:
        displacement_rms = (sum((p.detach().float() - old.float()).square().sum().item()
                               for p, old in zip(model.parameters(), before)) /
                            sum(p.numel() for p in model.parameters())) ** .5
    return {'loss': loss.item(), 'grad_norm': norm.item(), 'lr': opt.param_groups[0]['lr'],
            'optimizer_seconds': optimizer_seconds, 'full_update_rms': displacement_rms,
            'muon_update_rms': getattr(opt, 'last_metrics', {}).get('muon_update_rms')}


def run(args):
    import torch
    registration = verify_registration(ROOTS[args.arm])
    name = f'{args.arm}_seed{args.seed}'
    if name not in registration['authorized_runs']:
        raise ValueError('seed triplet not yet authorized')
    out = ROOTS[args.arm] / name
    out.mkdir(parents=True, exist_ok=False)
    pool = x7.frozen_pools(args.seed, DEPTH, 61)['val']
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]
    if hashes != registration['pool_array_sha256'][str(args.seed)]:
        raise ValueError('pool identity changed')
    np.savez_compressed(out / 'frozen_validation.npz', **{f'T16_{i}': a for i, a in enumerate(pool)})
    x7.write_json(out / 'data_receipt.json', {'array_sha256': hashes, 'archive_sha256': x7.digest(out / 'frozen_validation.npz')})
    torch.manual_seed(args.seed)
    model = build_model(args.arm).cuda()
    assert sum(p.numel() for p in model.parameters()) == (9887056 if args.arm == 'gdn2_uniform' else 9900800)
    if args.arm != 'gdn2_uniform' and parameter_routes(model) != registration['parameter_routes']:
        raise ValueError('parameter routing changed')
    opt = make_optimizer(model, args.arm)
    rng, task = np.random.default_rng(args.seed), InvertibleQuadraticTask(61)
    config = {'arm': args.arm, 'seed': args.seed, 'name': name, 'weights': point(args.arm),
              'registration_sha256': x7.digest(ROOTS[args.arm] / 'registration.json'),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'host': os.uname().nodename, 'registration': registration}
    x7.write_json(out / 'config.json', config)
    history, checkpoints, best, entry = [], {}, None, None
    started = time.monotonic()
    opt_seconds = 0.
    wall_ceiling = registration['resource_bounds'][args.arm]['wall_seconds']
    memory_ceiling = registration['resource_bounds'][args.arm]['peak_allocated_bytes']
    with (out / 'training.jsonl').open('a', buffering=1) as log:
        for step in range(CEILING + 1):
            if time.monotonic() - started > wall_ceiling:
                raise TimeoutError('registered infrastructure wall ceiling; censored, not optimizer failure')
            if step:
                before = time.monotonic()
                metrics = train_update(model, opt, args.arm, task.generate_batch(32, DEPTH + 2, rng), step)
                opt_seconds += metrics['optimizer_seconds']
                log.write(json.dumps({'step': step, **metrics, 'update_seconds': time.monotonic() - before,
                                     'elapsed_s': time.monotonic() - started}) + '\n')
                if torch.cuda.max_memory_allocated() > memory_ceiling:
                    raise MemoryError('registered memory ceiling; infrastructure censoring')
            if step % EVAL_EVERY and step != CEILING:
                continue
            if args.arm in ('sf', 'gdn2_uniform'):
                opt.eval()
            metrics = x7.evaluate(model, pool)
            current = metrics['aggregate']
            is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
            cp = None
            if is_best or step % 1000 == 0 or step == CEILING:
                cp = checkpoint(out, model, opt, rng, step, config)
                checkpoints[cp['path']] = cp
                latest = cp['path']
            if is_best:
                best = {**current, 'step': step, 'checkpoint': cp}
            if step == 0:
                entry = cp['path']
            row = {'step': step, **metrics, 'checkpoint': cp,
                   'elapsed_s': time.monotonic() - started, 'optimizer_seconds_total': opt_seconds,
                   'weight_norm': sum(p.detach().float().square().sum().item() for p in model.parameters()) ** .5,
                   'optimizer_metrics': getattr(opt, 'last_metrics', {}),
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'weights': point(args.arm),
                   'decision': 'safety_ceiling' if step == CEILING else 'continue'}
            history.append(row)
            x7.write_json(out / 'validation.json', history)
            x7.write_json(out / 'checkpoints.json', checkpoints)
            print(json.dumps({'name': name, **row}), flush=True)
            prune(out, checkpoints, {entry, best['checkpoint']['path'], latest})
    x7.write_json(out / 'result.json', {'name': name, 'status': 'complete', 'total_steps': CEILING,
        'stop_reason': 'safety_ceiling', 'best': best, 'final': history[-1],
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    run(parser.parse_args())
