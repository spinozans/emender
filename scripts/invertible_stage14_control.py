#!/usr/bin/env python3
"""X14 preregistration, disposable probe and detached, leased single-card launch."""
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

from experiments.expressivity_tasks import run_invertible_stage14_nofreebie_8m as run
from scripts.invertible_stage10_control import identity


def probe(architecture):
    """Procedural seed99 only; no confirmatory frozen validation or score selection."""
    import torch
    torch.manual_seed(99)
    model = run.build_model(architecture).cuda()
    opt = run.make_optimizer(model)
    rng, task = run.np.random.default_rng(99), run.x10.InvertibleQuadraticTask(61)
    rows, started = [], time.monotonic()
    for step in range(1, 26):
        rows.append(run.train_update(model, opt, task.generate_batch(32, 18, rng), step))
    elapsed = time.monotonic() - started
    path = run.ROOT / f'receipts/probe_{architecture}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, path)
    saved = torch.load(path, weights_only=False)
    restored = run.build_model(architecture).cuda()
    restored.load_state_dict(saved['model'])
    opt2 = run.make_optimizer(restored)
    opt2.load_state_dict(saved['optimizer'])
    batch = task.generate_batch(32, 18, rng)
    run.train_update(model, opt, batch, 26)
    run.train_update(restored, opt2, batch, 26)
    difference = max((p - q).abs().max().item() for p, q in zip(model.parameters(), restored.parameters()))
    if difference > 1e-6:
        raise AssertionError(f'next-update reload mismatch {difference}')
    # Exercise all 16 position counters on a disposable, non-confirmatory pool.
    metrics = run.evaluate(model, task.generate_batch(512, 18, rng))
    assert len(metrics['positions']) == 16 and math.isfinite(metrics['aggregate']['loss'])
    peak = torch.cuda.max_memory_allocated()
    path.unlink()
    receipt = {'architecture': architecture, 'updates': 25, 'seed': 99, 'confirmatory_pool_used': False,
               'elapsed_including_compile_seconds': elapsed,
               'update_seconds_mean_including_compile': elapsed / 25,
               'reload_next_update_max_abs_difference': difference,
               'peak_allocated_bytes_two_models_and_eval': peak,
               'all_16_position_evaluation_finite': True,
               'lr_update1': rows[0]['lr'], 'lr_update25': rows[-1]['lr'],
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'wall_ceiling_seconds': max(86400, math.ceil(elapsed / 25 * run.CEILING * 3)),
               'memory_ceiling_bytes': max(8 * 1024**3, math.ceil(peak * 2))}
    run.x7.write_json(path.with_suffix('.json'), receipt)
    print(json.dumps(receipt), flush=True)


def register(gpus):
    import fla
    import torch
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError('four distinct physical cards required')
    if (run.ROOT / 'registration.json').exists():
        raise FileExistsError('registration already exists; do not overwrite')
    for architecture in ('e97', 'gdn2'):
        torch.manual_seed(0)
        run.build_model(architecture)
    sources = [Path(__file__).relative_to(Path.cwd()), Path(run.__file__).relative_to(Path.cwd()),
               Path('scripts/invertible_stage10_control.py'), Path('scripts/gpu_lease.sh'),
               Path('scripts/watch_invertible_stage14_nofreebie.py'),
               Path('scripts/watch_invertible_stage10_optimizers.py'),
               Path('tests/test_invertible_stage14_nofreebie_8m.py'),
               run.ROOT / 'REGISTRATION.md',
               run.ROOT / 'receipts/tests_prelaunch.txt', run.ROOT / 'receipts/tests_gpu_prelaunch.txt',
               run.ROOT / 'receipts/gpu_before.txt']
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(Path.cwd()) and '.venv' not in Path(file).parts:
            path = Path(file).resolve().relative_to(Path.cwd())
            if path.suffix == '.py' and str(path).startswith(('ndm/', 'experiments/expressivity_tasks/')):
                sources.append(path)
    sources.extend(Path('ndm/triton').glob('e88*.py'))
    dependencies = list(Path(fla.__file__).parent.rglob('*.py'))
    dependencies.append(Path(inspect.getfile(torch.optim.AdamW)))
    probes = [run.ROOT / f'receipts/probe_{a}.json' for a in ('e97', 'gdn2')]
    sources.extend(probes)
    receipts = [json.loads(p.read_text()) for p in probes]
    pools = {str(seed): [hashlib.sha256(a.tobytes()).hexdigest()
                        for a in run.x7.frozen_pools(seed, 16, 61)['val']] for seed in (0, 1)}
    baseline = run.x10.ROOTS['adamw'] / 'registration.json'
    sources.append(baseline)
    for seed in ('0', '1'):
        if pools[seed] != json.loads(baseline.read_text())['pool_array_sha256'][seed]:
            raise ValueError('pool differs from X10')
    registration = {'task': 'X14', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python': sys.version, 'torch': torch.__version__,
        'fla': getattr(fla, '__version__', None), 'authorized_runs': list(run.NAMES),
        'gpu_binding': dict(zip(run.NAMES, gpus)), 'pool_array_sha256': pools,
        'source_sha256': {str(p): run.x7.digest(p) for p in sorted(set(sources))},
        'dependency_sha256': {str(p.resolve()): run.x7.digest(p) for p in sorted(set(dependencies))},
        'recipe': {'models': 'E97 legacy bf16 recurrent state/sequential Triton eval; stock unmodified bf16 FLA GDN2 chunk, no fp32 adapter',
            'parameters': {'e97': 9900800, 'gdn2': 9887056}, 'dim': 256, 'layers': 4, 'p': 61, 'vocab': 3721,
            'drivers': 16, 'batch': 32, 'seeds': [0, 1],
            'initialization': 'deep-first scratch; torch.manual_seed(seed); no checkpoint loaded',
            'training_rng': 'numpy.default_rng(seed), fresh procedural batches; frozen evaluation never trained on',
            'loss': 'P1 weight ZERO at every update; uniform CE mean on P2..P16 only; inputs unchanged',
            'precision': 'fp32 parameters / bf16 autocast', 'gradient_clip_norm': 1.,
            'optimizer': 'exact X10 torch.optim.AdamW ordinary live parameters',
            'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01,
            'decay_policy': 'ALL parameters .01, including A_log/dt_bias; no_weight_decay tags intentionally ignored, exact X10',
            'schedule': 'actual X10 update: linear warmup update1..5000 to1e-3, cosine to1e-5 at500000',
            'eval_examples': 512, 'eval_every': 500, 'ceiling': run.CEILING,
            'stop': '500K only; no threshold, stagnation or all-perfect stop'},
        'predictions': [
            '(a) NO-FREEBIE CARRY: if E97 acquires P2 materially (>=50/512), carry is learnable without the visible subsidy; P1 scaffolds learning but is not necessary for composition.',
            '(b) FREEBIE-DEPENDENT: if E97 stays at zero on P2+ while GDN2 acquires, E97 composition learning depends on visible-position scaffolding; different learning-mode claim requires receipts at both seeds.',
            '(c) GDN2 columns either way: does the linear-state bootstrap survive carry-only signal?'],
        'interpretation': 'Shares X13 open audit question about stage12 runner, but uses stage10-pattern code paths ONLY. No X12 E97-T64 conclusions imported. Report other/mixed outcomes without forcing (a)/(b); no post-launch tuning.',
        'readouts': 'P1-P16 counts/CE every500; P1 explicitly diagnostic-only NON-COMPOSITION; primary aggregate and best selection P2..P16 only; first50/first512, fixed-final500K composition prefix from P2 and maximum-anywhere separately; transient vs final and both seeds separately',
        'retention': 'entry + composition-best + latest/final, at most3 checkpoints plus atomic staging file; deletion hashes retained',
        'resource_bounds': {'wall_seconds': max(r['wall_ceiling_seconds'] for r in receipts),
                            'peak_allocated_bytes': max(r['memory_ceiling_bytes'] for r in receipts)},
        'censoring': 'infrastructure/time/memory failure censored, not architecture failure; no automatic retry/restart; absent crossings right-censored at observed step, not zero/failure',
        'scope': 'standalone single-card expressivity, not resilient DiLoCo or Frontier; no distributed requirements claimed',
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('NDM_', 'TRITON_'))}}
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
        architecture, seed = name.split('_seed')
        cmd = [sys.executable, '-m', 'experiments.expressivity_tasks.run_invertible_stage14_nofreebie_8m',
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
            if result['total_steps'] != run.CEILING or result['status'] != 'complete':
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
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage14_control', 'worker', '--name', name],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run.x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'{name} worker={proc.pid} card={registration["gpu_binding"][name]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['probe', 'register', 'launch', 'worker'])
    parser.add_argument('--gpus', type=int, nargs=4)
    parser.add_argument('--arm', choices=['e97', 'gdn2'])
    parser.add_argument('--name', choices=run.NAMES)
    args = parser.parse_args()
    if args.action == 'register':
        if args.gpus is None:
            parser.error('--gpus required for registration')
        register(args.gpus)
    elif args.action == 'probe':
        if args.arm is None:
            parser.error('--arm required')
        probe(args.arm)
    elif args.name is None:
        parser.error('--name required')
    elif args.action == 'worker':
        raise SystemExit(worker(args.name))
    else:
        launch(args.name)
