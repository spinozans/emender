"""X15: trusted X10/X11 zero-shot GDN2 and scratch T64 hybrid-Muon probes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
import numpy as np
from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as x10
from experiments.expressivity_tasks import run_invertible_stage11_gdn2_adamw_8m as x11

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage15_probes_8m')
DEPTH, CEILING, EVAL_EVERY = 64, 1000000, 500
NAMES = ('gdn2_zeroshot', 'muon3x_seed0')
CHAMPION = x11.ROOT / x11.NAME / 'step_482500.pt'
CHAMPION_SHA256 = '49a0f96fe7890ca9bb8a860698641ad8407dca412a3b38a01d3ede996c011cd0'
x7 = x10.x7
parameter_routes = x10.parameter_routes
InvertibleQuadraticTask = x10.InvertibleQuadraticTask
verify_registration = x10.verify_registration
checkpoint, prune, point = x10.checkpoint, x10.prune, x10.point

def build_model(arm):
    if arm != 'muon3x':
        raise ValueError('unregistered arm')
    return x10.build_model('muon')

def make_optimizer(model, arm):
    if arm != 'muon3x':
        raise ValueError('unregistered arm')
    return x10.make_optimizer(model, 'muon')

def learning_rate(step):
    if not 0 <= step <= CEILING:
        raise ValueError('update outside registered schedule')
    if step <= 5000:
        return .003 * step / 5000
    return .00003 + .5 * (.003 - .00003) * (1 + math.cos(math.pi * (step - 5000) / (CEILING - 5000)))

def rates(opt, arm, step):
    for group in opt.param_groups:
        group['lr'] = learning_rate(step)

def train_update(model, opt, arm, batch, step):
    with patch.object(x10, 'rates', rates):
        return x10.train_update(model, opt, arm, batch, step)

def evaluate(model, pool):
    # Only expected loss reshape / perfect-prefix fallback changes, not loader.
    with patch.object(x7, 'DEPTH', DEPTH):
        return x7.evaluate(model, pool)

def array_hashes(pool):
    return [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]

def zero_shot():
    import torch
    registration = verify_registration(ROOT)
    out = ROOT / 'gdn2_zeroshot'
    out.mkdir(parents=True, exist_ok=False)
    assert x7.digest(CHAMPION) == CHAMPION_SHA256
    original = json.loads((CHAMPION.parent / 'result.json').read_text())
    assert original['best']['checkpoint']['sha256'] == CHAMPION_SHA256
    historical = json.loads((x11.ROOT / 'registration.json').read_text())
    for name, sha in historical['source_sha256'].items():
        if name.startswith(('ndm/', 'experiments/expressivity_tasks/')) and name.endswith('.py'):
            assert x7.digest(name) == sha, name
    model = x11.build_model().cuda()
    saved = torch.load(CHAMPION, map_location='cpu', weights_only=False)
    assert saved['step'] == 482500
    model.load_state_dict(saved['model'], strict=True)
    del saved
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    pool16 = x7.frozen_pools(0, 16, 61)['val']
    archive = np.load(CHAMPION.parent / 'frozen_validation.npz')
    assert array_hashes(pool16) == array_hashes(tuple(archive[f'T16_{j}'] for j in range(3)))
    t16 = x7.evaluate(model, pool16)
    expected = next(r for r in json.loads((CHAMPION.parent / 'validation.json').read_text()) if r['step'] == 482500)
    assert [p['correct'] for p in t16['positions']] == [p['correct'] for p in expected['positions']]
    assert [p['correct'] for p in t16['positions'][:5]] == [512, 512, 512, 512, 318]
    x7.write_json(out / 't16_control.json', t16)
    pool64 = x7.frozen_pools(0, 64, 61)['val']
    assert array_hashes(pool64) == registration['pool_array_sha256']['0']
    np.savez_compressed(out / 'frozen_validation.npz', **{f'T64_{i}': a for i, a in enumerate(pool64)})
    metrics = evaluate(model, pool64)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert x7.digest(CHAMPION) == CHAMPION_SHA256
    x7.write_json(out / 'result.json', {'name': 'gdn2_zeroshot', 'status': 'complete',
        'total_steps': 0, 'stop_reason': 'zero_shot_complete', 'training_updates': 0,
        'champion_sha256': CHAMPION_SHA256, 'weights_unchanged': True,
        'registration_sha256': x7.digest(ROOT / 'registration.json'),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'pool_array_sha256': array_hashes(pool64), 't16_control': t16, 'final': metrics,
        'finished_at': time.time()})
    print(json.dumps(metrics), flush=True)

def run(args):
    import torch
    registration = verify_registration(ROOT)
    name = f'{args.arm}_seed{args.seed}'
    if name not in registration['authorized_runs']:
        raise ValueError('seed triplet not yet authorized')
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=False)
    pool = x7.frozen_pools(args.seed, DEPTH, 61)['val']
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]
    if hashes != registration['pool_array_sha256'][str(args.seed)]:
        raise ValueError('pool identity changed')
    np.savez_compressed(out / 'frozen_validation.npz', **{f'T64_{i}': a for i, a in enumerate(pool)})
    x7.write_json(out / 'data_receipt.json', {'array_sha256': hashes, 'archive_sha256': x7.digest(out / 'frozen_validation.npz')})
    torch.manual_seed(args.seed)
    model = build_model(args.arm).cuda()
    assert sum(p.numel() for p in model.parameters()) == (9887056 if args.arm == 'gdn2_uniform' else 9900800)
    if args.arm != 'gdn2_uniform' and parameter_routes(model) != registration['parameter_routes']:
        raise ValueError('parameter routing changed')
    opt = make_optimizer(model, args.arm)
    assert not opt.state and not opt.aux.state
    rng, task = np.random.default_rng(args.seed), InvertibleQuadraticTask(61)
    config = {'arm': args.arm, 'seed': args.seed, 'name': name, 'weights': point(args.arm),
              'registration_sha256': x7.digest(ROOT / 'registration.json'),
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
            metrics = evaluate(model, pool)
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
    parser.add_argument('--arm', choices=['muon3x', 'gdn2_zeroshot'], required=True)
    parser.add_argument('--seed', type=int, choices=[0], default=0)
    args = parser.parse_args()
    zero_shot() if args.arm == 'gdn2_zeroshot' else run(args)
