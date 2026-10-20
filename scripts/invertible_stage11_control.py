#!/usr/bin/env python3
"""X11 preregistration, disposable probe and detached, leased single-card launch."""
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

from experiments.expressivity_tasks import run_invertible_stage11_gdn2_adamw_8m as run
from scripts.invertible_stage10_control import identity


def probe():
    """Procedural seed99 only; no confirmatory frozen validation or score selection."""
    import torch
    torch.manual_seed(99)
    model = run.build_model().cuda()
    opt = run.make_optimizer(model)
    rng, task = run.np.random.default_rng(99), run.x10.InvertibleQuadraticTask(61)
    rows, started = [], time.monotonic()
    for step in range(1, 26):
        rows.append(run.train_update(model, opt, task.generate_batch(32, 18, rng), step))
    elapsed = time.monotonic() - started
    path = run.ROOT / 'receipts/probe.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, path)
    saved = torch.load(path, weights_only=False)
    restored = run.build_model().cuda()
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
    metrics = run.x7.evaluate(model, task.generate_batch(512, 18, rng))
    assert len(metrics['positions']) == 16 and math.isfinite(metrics['aggregate']['loss'])
    peak = torch.cuda.max_memory_allocated()
    path.unlink()
    receipt = {'updates': 25, 'seed': 99, 'confirmatory_pool_used': False,
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


def register(gpu):
    import fla
    import torch
    torch.manual_seed(0)
    run.build_model()
    sources = [Path(__file__).relative_to(Path.cwd()), Path(run.__file__).relative_to(Path.cwd()),
               Path('scripts/invertible_stage10_control.py'), Path('scripts/gpu_lease.sh'),
               Path('scripts/watch_invertible_stage11_gdn2_adamw.py'),
               Path('scripts/watch_invertible_stage10_optimizers.py'),
               Path('tests/test_invertible_stage11_gdn2_adamw_8m.py'),
               run.ROOT / 'REGISTRATION.md', run.ROOT / 'receipts/probe.json',
               run.ROOT / 'receipts/tests_prelaunch.txt', run.ROOT / 'receipts/gpu_before.txt']
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(Path.cwd()) and '.venv' not in Path(file).parts:
            path = Path(file).resolve().relative_to(Path.cwd())
            if path.suffix == '.py' and str(path).startswith(('ndm/', 'experiments/expressivity_tasks/')):
                sources.append(path)
    baselines = {name: root / 'registration.json' for name, root in (
        ('X7b', run.x10.ROOTS['gdn2_uniform']), ('X10', run.x10.ROOTS['adamw']))}
    sources.extend(baselines.values())
    pools = {'0': [hashlib.sha256(a.tobytes()).hexdigest() for a in run.x7.frozen_pools(0, 16, 61)['val']]}
    for path in baselines.values():
        if pools['0'] != json.loads(path.read_text())['pool_array_sha256']['0']:
            raise ValueError('pool differs from matched baseline')
    dependencies = list(Path(fla.__file__).parent.rglob('*.py'))
    dependencies.append(Path(inspect.getfile(torch.optim.AdamW)))
    probe_receipt = json.loads((run.ROOT / 'receipts/probe.json').read_text())
    registration = {'task': 'X11', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python': sys.version, 'torch': torch.__version__,
        'fla': getattr(fla, '__version__', None), 'authorized_runs': [run.NAME], 'gpu_binding': {run.NAME: gpu},
        'pool_array_sha256': pools,
        'source_sha256': {str(p): run.x7.digest(p) for p in sorted(set(sources))},
        'dependency_sha256': {str(p.resolve()): run.x7.digest(p) for p in sorted(set(dependencies))},
        'recipe': {'model': 'real fla GDN2, unmodified stock bf16 chunk kernel; fp32 parameters/bf16 autocast',
                   'parameters': 9887056, 'dim': 256, 'layers': 4, 'p': 61, 'vocab': 3721,
                   'drivers': 16, 'batch': 32, 'initialization': 'torch.manual_seed(0), scratch; no checkpoint loaded',
                   'training_rng': 'numpy.default_rng(0), fresh procedural batches; frozen pool never trained on',
                   'loss': 'uniform target cross entropy', 'gradient_clip_norm': 1.,
                   'optimizer': 'torch.optim.AdamW, ordinary live parameters',
                   'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01,
                   'decay_policy': 'ALL parameters .01, including A_log/dt_bias; _no_weight_decay tags intentionally ignored, exact X10-adamw policy',
                   'schedule': 'exact X9/X10: update1..5000 linear to1e-3, then cosine to1e-5 at500000; LR logged every update',
                   'eval_examples': 512, 'eval_every': 500, 'ceiling': run.CEILING,
                   'stop': '500K safety ceiling only; no stagnation, threshold or all-perfect stop'},
        'predictions': [
            '(a) If GDN2+AdamW reaches 5+ perfect positions, architecture gains depth from the optimizer too and the P5+ question opens. Any P5>=50/512 is material; 512/512 is perfect. Report material acquisition even without five perfect positions.',
            '(b) If GDN2+AdamW matches X7b four perfect positions with no P5 gain, depth at this scale is bounded near4 under both regimes; E97/GDN2 reads parity-at4 with different optimizer sensitivity. Other outcomes are reported as such, not forced into (a)/(b).',
            '(c) Either way this completes the matched-recipe E97-SF/E97-AdamW/E97-Muon/GDN2-SF/GDN2-AdamW table for the paper rewrite.'],
        'P5_question': 'X7b P5=3/512 at500K: a longer ceiling may be needed later. This run reports only what500K buys; no extension or tuning in this registration.',
        'readouts': 'P1-P16 counts/CE every500; first50/first512, final contiguous perfect prefix and maximum-anywhere separately; fixed-final500K primary, transients descriptive',
        'retention': 'entry + aggregate-best + latest/final, at most3 checkpoints plus one atomic staging file; deletion hashes retained',
        'resource_bounds': {'wall_seconds': probe_receipt['wall_ceiling_seconds'],
                            'peak_allocated_bytes': probe_receipt['memory_ceiling_bytes']},
        'censoring': 'infrastructure/time/memory failure is censored, not optimizer failure; no automatic retry or restart',
        'scope': 'standalone single-card expressivity experiment, not resilient DiLoCo or Frontier; no distributed requirements claimed',
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('NDM_', 'TRITON_'))}}
    run.x7.write_json(run.ROOT / 'registration.json', registration)


def worker():
    receipt = run.ROOT / 'receipts' / run.NAME
    run.x7.write_json(receipt / 'worker.json', identity(os.getpid()))
    def interrupted(signum, _frame):
        raise SystemExit(f'worker received signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    code, child, error, leased = 1, None, None, False
    env = dict(os.environ)
    try:
        registration = run.x10.verify_registration(run.ROOT)
        gpu = registration['gpu_binding'][run.NAME]
        env.update(GPU_LEASE_VISIBLE=str(gpu), GPU_LEASE_PID=str(os.getpid()),
                   CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        with (receipt / 'lease.txt').open('w') as lease:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'],
                           env=env, stdout=lease, stderr=lease, check=True)
        leased = True
        cmd = [sys.executable, '-m', 'experiments.expressivity_tasks.run_invertible_stage11_gdn2_adamw_8m']
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
            result = json.loads((run.ROOT / run.NAME / 'result.json').read_text())
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
        terminal = {'name': run.NAME, 'exit_code': code, 'error': error, 'finished_at': time.time(),
                    'result_sha256': run.x7.digest(run.ROOT / run.NAME / 'result.json') if code == 0 else None}
        run.x7.write_json(receipt / 'terminal.json', terminal)
        run.x7.write_json(run.ROOT / 'markers' / (run.NAME + ('.DONE' if code == 0 else '.FAIL')), terminal)
        if leased:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'release', str(gpu)], env=env,
                           check=True, stdout=subprocess.DEVNULL)
    return code


def launch():
    registration = run.x10.verify_registration(run.ROOT)
    if run.NAME not in registration['authorized_runs'] or (run.ROOT / run.NAME).exists():
        raise ValueError('unauthorized or existing run')
    receipt = run.ROOT / 'receipts' / run.NAME
    receipt.mkdir(parents=True, exist_ok=False)
    with (receipt / 'worker_console.txt').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage11_control', 'worker'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run.x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'{run.NAME} worker={proc.pid} card={registration["gpu_binding"][run.NAME]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['probe', 'register', 'launch', 'worker'])
    parser.add_argument('--gpu', type=int, help='physical card to bind at registration')
    args = parser.parse_args()
    if args.action == 'register':
        if args.gpu is None:
            parser.error('--gpu required for registration')
        register(args.gpu)
    elif args.action == 'worker':
        raise SystemExit(worker())
    else:
        globals()[args.action]()
