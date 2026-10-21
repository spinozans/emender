#!/usr/bin/env python3
"""X15 preregistration, disposable probe and detached, leased single-card launch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from experiments.expressivity_tasks import run_invertible_stage15_probes_8m as run
from scripts.invertible_stage10_control import identity


def probe():
    """Procedural seed99 only; no confirmatory frozen validation or score selection."""
    architecture = 'muon3x'
    import torch
    torch.manual_seed(99)
    model = run.build_model('muon3x').cuda()
    opt = run.make_optimizer(model, 'muon3x')
    rng, task = run.np.random.default_rng(99), run.x10.InvertibleQuadraticTask(61)
    rows, started = [], time.monotonic()
    for step in range(1, 26):
        rows.append(run.train_update(model, opt, 'muon3x', task.generate_batch(32, 66, rng), step + 4999))
    elapsed = time.monotonic() - started
    path = run.ROOT / f'receipts/probe_{architecture}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, path)
    saved = torch.load(path, weights_only=False)
    restored = run.build_model('muon3x').cuda()
    restored.load_state_dict(saved['model'])
    opt2 = run.make_optimizer(restored, 'muon3x')
    opt2.load_state_dict(saved['optimizer'])
    batch = task.generate_batch(32, 66, rng)
    run.train_update(model, opt, 'muon3x', batch, 5025)
    run.train_update(restored, opt2, 'muon3x', batch, 5025)
    difference = max((p - q).abs().max().item() for p, q in zip(model.parameters(), restored.parameters()))
    if difference > 1e-6:
        raise AssertionError(f'next-update reload mismatch {difference}')
    # Exercise all 64 position counters on a disposable, non-confirmatory pool.
    metrics = run.evaluate(model, task.generate_batch(512, 66, rng))
    assert len(metrics['positions']) == 64 and math.isfinite(metrics['aggregate']['loss'])
    peak = torch.cuda.max_memory_allocated()
    path.unlink()
    receipt = {'architecture': architecture, 'updates': 25, 'seed': 99, 'confirmatory_pool_used': False,
               'elapsed_including_compile_seconds': elapsed,
               'update_seconds_mean_including_compile': elapsed / 25,
               'reload_next_update_max_abs_difference': difference,
               'peak_allocated_bytes_two_models_and_eval': peak,
               'all_64_position_evaluation_finite': True,
               'lr_update1': rows[0]['lr'], 'lr_update25': rows[-1]['lr'],
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'wall_ceiling_seconds': max(172800, math.ceil(elapsed / 25 * run.CEILING * 3)),
               'memory_ceiling_bytes': max(8 * 1024**3, math.ceil(peak * 2))}
    run.x7.write_json(path.with_suffix('.json'), receipt)
    print(json.dumps(receipt), flush=True)


def register():
    import fla
    import torch
    if (run.ROOT / 'registration.json').exists():
        raise FileExistsError('never overwrite registration')
    torch.manual_seed(0)
    model = run.build_model('muon3x')
    run.x11.build_model()
    sources = [Path(__file__).relative_to(Path.cwd()), Path(run.__file__).relative_to(Path.cwd()),
        Path('scripts/invertible_stage10_control.py'), Path('scripts/gpu_lease.sh'),
        Path('scripts/watch_invertible_stage15_probes.py'),
        Path('scripts/watch_invertible_stage10_optimizers.py'),
        Path('tests/test_invertible_stage15_probes_8m.py'), run.ROOT / 'REGISTRATION.md',
        run.ROOT / 'receipts/probe_muon3x.json', run.ROOT / 'receipts/tests_prelaunch.txt',
        run.ROOT / 'receipts/gpu_before.txt', run.ROOT / 'receipts/tests_gpu_prelaunch.txt',
        run.x11.ROOT / 'registration.json']
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(Path.cwd()) and '.venv' not in Path(file).parts:
            path = Path(file).resolve().relative_to(Path.cwd())
            if path.suffix == '.py' and str(path).startswith(('ndm/', 'experiments/expressivity_tasks/')):
                sources.append(path)
    sources.extend(Path('ndm/triton').glob('e88*.py'))
    assert not any('stage12' in str(p) for p in sources)
    dependencies = list(Path(fla.__file__).parent.rglob('*.py'))
    dependencies.append(Path(inspect.getfile(torch.optim.AdamW)))
    probe = json.loads((run.ROOT / 'receipts/probe_muon3x.json').read_text())
    assert run.x7.digest(run.CHAMPION) == run.CHAMPION_SHA256
    registration = {'task': 'X15', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python': sys.version, 'torch': torch.__version__,
        'authorized_runs': list(run.NAMES), 'gpu_binding': dict(zip(run.NAMES, (6, 7))),
        'pool_array_sha256': {'0': run.array_hashes(run.x7.frozen_pools(0, 64, 61)['val'])},
        'source_sha256': {str(p): run.x7.digest(p) for p in sorted(set(sources))},
        'dependency_sha256': {str(p.resolve()): run.x7.digest(p) for p in sorted(set(dependencies))},
        'champion': str(run.CHAMPION), 'champion_sha256': run.CHAMPION_SHA256,
        'parameter_routes': run.parameter_routes(model),
        'recipe': {'architecture': 'E97 legacy bf16, sequential Triton eval, stage10 only',
            'parameters': 9900800, 'drivers': 64, 'batch': 32, 'p': 61, 'seed': 0,
            'optimizer': 'unchanged X10 HybridMuon: full-matrix FP32 NS5, Nesterov .95, RMS .2*sqrt(max(shape)); aux AdamW .9/.999 eps1e-8',
            'decay': 'all parameters .01; no-weight-decay tags ignored',
            'loss': 'uniform CE P1..P64', 'gradient_clip_norm': 1.,
            'schedule': 'linear warmup5K to3e-3, cosine to3e-5 at1M; both groups same LR',
            'ceiling': run.CEILING, 'eval_every': 500, 'eval_examples': 512,
            'initialization': 'scratch torch.manual_seed0, numpy.default_rng0 fresh procedural batches'},
        'resource_bounds': {'muon3x': {'wall_seconds': probe['wall_ceiling_seconds'],
            'peak_allocated_bytes': probe['memory_ceiling_bytes']}, 'wall_seconds': probe['wall_ceiling_seconds']},
        'protocol': (run.ROOT / 'REGISTRATION.md').read_text(),
        'censoring': 'infrastructure is censored, not optimizer failure; no retry, no metric stop',
        'retention': 'entry + aggregate-best + latest/final, at most3 plus atomic staging',
        'scope': 'single-card expressivity; Frontier/DiLoCo requirements not applicable'}
    run.x7.write_json(run.ROOT / 'registration.json', registration)


def worker(name):
    if name not in run.NAMES:
        raise ValueError('unregistered run')
    receipt = run.ROOT / 'receipts' / name
    run.x7.write_json(receipt / 'worker.json', identity(os.getpid()))
    def interrupted(signum, _frame):
        raise SystemExit(f'worker received signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    code, child, error, leased = 1, None, None, False
    env = dict(os.environ)
    try:
        registration = run.x10.verify_registration(run.ROOT)
        gpu = registration['gpu_binding'][name]
        env.update(GPU_LEASE_VISIBLE=str(gpu), GPU_LEASE_PID=str(os.getpid()),
                   CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        with (receipt / 'lease.txt').open('w') as lease:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'],
                           env=env, stdout=lease, stderr=lease, check=True)
        leased = True
        architecture, seed = ('gdn2_zeroshot', '0') if name == 'gdn2_zeroshot' else name.split('_seed')
        cmd = [sys.executable, '-m', 'experiments.expressivity_tasks.run_invertible_stage15_probes_8m',
               '--arm', architecture, '--seed', seed]
        run.x7.write_json(receipt / 'command.json', {'argv': cmd, 'gpu': gpu})
        with (receipt / 'console.txt').open('w') as output:
            child = subprocess.Popen(cmd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            run.x7.write_json(receipt / 'trainer.json', identity(child.pid))
            deadline = time.monotonic() + registration['resource_bounds']['wall_seconds'] + 120
            while True:
                try:
                    code = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    subprocess.run(['bash', 'scripts/gpu_lease.sh', 'heartbeat', str(gpu)], env=env,
                                   check=True, stdout=subprocess.DEVNULL)
                    if time.monotonic() > deadline:
                        raise TimeoutError('external infrastructure wall ceiling')
        if code == 0:
            result = json.loads((run.ROOT / name / 'result.json').read_text())
            if result['total_steps'] != (0 if name == 'gdn2_zeroshot' else run.CEILING) or result['status'] != 'complete':
                raise RuntimeError('invalid terminal result')
    except BaseException as exc:
        error, code = repr(exc), 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
        terminal = {'name': name, 'exit_code': code, 'error': error, 'finished_at': time.time(),
                    'result_sha256': run.x7.digest(run.ROOT / name / 'result.json') if code == 0 else None}
        run.x7.write_json(receipt / 'terminal.json', terminal)
        run.x7.write_json(run.ROOT / 'markers' / (name + ('.DONE' if code == 0 else '.FAIL')), terminal)
        if leased:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'release', str(gpu)], env=env,
                           check=True, stdout=subprocess.DEVNULL)
    return code


def launch(name):
    if name not in run.NAMES:
        raise ValueError('unregistered run')
    registration = run.x10.verify_registration(run.ROOT)
    if name not in registration['authorized_runs'] or (run.ROOT / name).exists():
        raise ValueError('unauthorized or existing run')
    receipt = run.ROOT / 'receipts' / name
    receipt.mkdir(parents=True, exist_ok=False)
    with (receipt / 'worker_console.txt').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage15_control', 'worker', '--name', name],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run.x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'{name} worker={proc.pid} card={registration["gpu_binding"][name]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['probe', 'register', 'launch', 'worker'])
    parser.add_argument('--name', choices=run.NAMES)
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.action == 'probe':
        probe()
    elif args.name is None:
        parser.error('--name required')
    elif args.action == 'worker':
        raise SystemExit(worker(args.name))
    else:
        launch(args.name)
