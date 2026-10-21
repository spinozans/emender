"""X16 preregistration, disposable preflight and native detached leased launches."""
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

from experiments.expressivity_tasks import run_invertible_stage16_ignition_8m as run
from scripts.invertible_stage10_control import identity


def probe(architecture):
    """Disposable seed99, including T64 shape, serialization and depth-safe evaluation."""
    import torch
    torch.manual_seed(99)
    model = run.x14.build_model(architecture).cuda()
    opt = run.x14.make_optimizer(model)
    rng, task = run.np.random.default_rng(99), run.x10.InvertibleQuadraticTask(61)
    started = time.monotonic()
    for step in range(1, 26):
        run.train_update(model, opt, task.generate_batch(32, 18, rng), step, 1 / 64)
    depth = 64 if architecture == 'e97' else 16
    batch = task.generate_batch(32, depth + 2, rng)
    run.train_update(model, opt, batch, 26, lr=run.extension_rate(1, .0005))
    path = run.ROOT / f'receipts/probe_{architecture}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, path)
    restored = run.x14.build_model(architecture).cuda()
    saved = torch.load(path, weights_only=False)
    restored.load_state_dict(saved['model'])
    opt2 = run.x14.make_optimizer(restored)
    opt2.load_state_dict(saved['optimizer'])
    batch = task.generate_batch(32, depth + 2, rng)
    run.train_update(model, opt, batch, 27, lr=run.extension_rate(2, .0005))
    run.train_update(restored, opt2, batch, 27, lr=run.extension_rate(2, .0005))
    difference = max((p - q).abs().max().item() for p, q in zip(model.parameters(), restored.parameters()))
    assert difference <= 1e-6
    for d in ((16, 64, 16) if architecture == 'e97' else (16,)):
        metrics = run.evaluate(model, task.generate_batch(512, d + 2, rng))
        assert len(metrics['positions']) == d and math.isfinite(metrics['aggregate']['loss'])
    receipt = {'architecture': architecture, 'seed': 99, 'confirmatory_pool_used': False,
               'reload_next_update_max_abs_difference': difference, 'evaluated_depths': [16, depth],
               'elapsed_seconds': time.monotonic() - started,
               'peak_allocated_bytes_two_models_and_eval': torch.cuda.max_memory_allocated(),
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    path.unlink()
    run.x7.write_json(path.with_suffix('.json'), receipt)
    print(json.dumps(receipt), flush=True)


def register():
    import fla
    import torch
    if (run.ROOT / 'registration.json').exists():
        raise FileExistsError('immutable registration already exists')
    for architecture in ('e97', 'gdn2'):
        torch.manual_seed(0)
        run.x14.build_model(architecture)
    sources = [Path(__file__).relative_to(Path.cwd()), Path(run.__file__).relative_to(Path.cwd()),
               Path('scripts/invertible_stage10_control.py'), Path('scripts/gpu_lease.sh'),
               Path('scripts/watch_invertible_stage10_optimizers.py'),
               Path('scripts/watch_invertible_stage16_ignition.py'),
               Path('tests/test_invertible_stage16_ignition_8m.py'), run.ROOT / 'REGISTRATION.md']
    sources.extend(run.ROOT / 'receipts' / p for p in (
        'probe_e97.json', 'probe_gdn2.json', 'tests_prelaunch.txt', 'tests_gpu_prelaunch.txt', 'gpu_before.txt'))
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(Path.cwd()) and '.venv' not in Path(file).parts:
            path = Path(file).resolve().relative_to(Path.cwd())
            if path.suffix == '.py' and str(path).startswith(('ndm/', 'experiments/expressivity_tasks/')):
                sources.append(path)
    sources.extend(Path('ndm/triton').glob('e88*.py'))
    dependencies = list(Path(fla.__file__).parent.rglob('*.py'))
    dependencies.append(Path(inspect.getfile(torch.optim.AdamW)))
    pools = {str(d): [hashlib.sha256(a.tobytes()).hexdigest() for a in run.x7.frozen_pools(0, d, 61)['val']]
             for d in (16, 64)}
    baseline = run.x10.ROOTS['adamw'] / 'registration.json'
    sources.append(baseline)
    assert pools['16'] == json.loads(baseline.read_text())['pool_array_sha256']['0']
    registration = {
        'task': 'X16', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python': sys.version, 'torch': torch.__version__,
        'fla': getattr(fla, '__version__', None), 'authorized_runs': list(run.ARMS),
        'gpu_binding': {n: a[2] for n, a in run.ARMS.items()},
        'p1_weight': {n: a[1] for n, a in run.ARMS.items()}, 'pool_array_sha256': pools,
        'source_sha256': {str(p): run.x7.digest(p) for p in sorted(set(sources))},
        'dependency_sha256': {str(p.resolve()): run.x7.digest(p) for p in sorted(set(dependencies))},
        'recipe': 'Exact X10 AdamW: scratch seed0, p61/vocab3721, dim256/layers4, E97 9900800 legacy bf16 recurrence/sequential Triton eval; GDN2 9887056 unmodified stock bf16 FLA chunk; batch32, fp32 params/bf16 autocast, clip1, betas(.9,.999),eps1e-8, ALL parameters wd.01 (including tagged A_log/dt_bias); linear5K warmup to1e-3, cosine to1e-5 at500K; procedural numpy.default_rng(0), no frozen-pool training; 512-example eval every500; no stagnation/ignition stopping',
        'sweep': 'Fixed T16; explicit CE w*mean(P1)+(1-w)*mean(P2..P16); unchanged inputs and full mask; loss weight is the sole within-architecture variable; 500K fixed-final primary',
        'curriculum': {
            'criterion': 'first scheduled T16 evaluation with at least4 positions >=508/512; P1 included; max500K teaching updates, otherwise criterion_not_reached and NO T64 launch',
            'design_choice': 'ONE choice: preserve same live model, AdamW moments, torch/CUDA/numpy RNG; warmup-free cosine from attained teaching LR to1e-5 over500K T64 updates; no optimizer reset or LR jump',
            'extension_loss': 'uniform full P1..P64 CE, no replay/mixing of T16 training',
            'readout': 'T64 baseline before extension and every500 updates; independent frozen T16 retention at every T64 evaluation; taught-position counts/drop from transition and new P>=50 beyond transition T64 frontier; final500K distinct from transients'},
        'predictions': [
            '(sweep a) E97 ignition (any P>=50/512) bracketed by ladder: w=0 fails X14, w=1/16 ignites X10; measure 1/64 and1/32, not assume monotonicity.',
            '(sweep b) GDN2 failure point, if in ladder, measured with1/512 and1/2048; existing1/136 ignites X8 and0 fails X14. Historical X8 differs in optimizer/weight distribution: contextual, not a matched causal endpoint.',
            '(sweep c) Assemble architecture-specific any-position ignition vs explicit w and first50 timing, plus every-position first50/first508/first512 and final/max counts. Historical T64 E971/64 is context, not a controlled T16 point.',
            '(curriculum a) TEACH-SHORT-THEN-EXTEND: any T64 position>=50 beats cold-start zero; distinguish zero-shot baseline transfer from actual acquisition of new material positions during extension.',
            '(curriculum b) If T64 never extends beyond taught/zero-shot levels, extension has its own ignition barrier: partial fix, not full success. Report maximum material-position index and new positions relative to transition baseline.',
            '(curriculum c) Track retention on frozen T16 throughout; report all taught-position counts and drops from transition, first loss of >=508, final versus transient damage.'
        ],
        'retention': 'sweep entry+best+latest <=3 checkpoints; curriculum entry+criterion+phase-best+latest <=4; atomic staging; hash receipts for deleted checkpoints; T16/T64 checkpoint namespaces separate',
        'resource_bounds': {'wall_seconds': 14 * 86400, 'peak_allocated_bytes': 16 * 1024**3},
        'censoring': 'No automatic retry/restart. Infrastructure, wall or memory termination is censored, not architecture failure. Missing crossings right-censored at observed phase update. Teaching criterion not reached is not T64 failure. Terminal finite500K no-ignition means only no observed ignition by500K.',
        'future_arms': 'NOT authorized/auto-launched: E97 1/128, GDN2 1/1024 etc. require later preregistration and free cards; 4/5/7 forbidden for this launch',
        'scope': 'standalone single-host expressivity; no resilient DiLoCo, Frontier or system-scale claim',
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('NDM_', 'TRITON_'))}}
    run.x7.write_json(run.ROOT / 'registration.json', registration)


def valid_result(name, result):
    if result.get('name') != name:
        return False
    if name == 'e97_curriculum_seed0':
        if result.get('status') == 'criterion_not_reached':
            return (result.get('total_steps') == run.CEILING and result.get('t64_steps') == 0
                    and result.get('stop_reason') == 'teaching_ceiling')
        return (result.get('status') == 'complete' and result.get('t64_steps') == run.CEILING
                and result.get('stop_reason') == 'safety_ceiling'
                and result.get('total_steps') == result['transition']['step'] + run.CEILING)
    return (result.get('status') == 'complete' and result.get('total_steps') == run.CEILING
            and result.get('stop_reason') == 'safety_ceiling')


def worker(name):
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
        if gpu != run.ARMS[name][2] or gpu not in (0, 1, 2, 3, 6):
            raise ValueError('forbidden GPU binding')
        env.update(GPU_LEASE_VISIBLE=str(gpu), GPU_LEASE_PID=str(os.getpid()),
                   CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        with (receipt / 'lease.txt').open('w') as lease:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'],
                           env=env, stdout=lease, stderr=lease, check=True)
        leased = True
        cmd = [sys.executable, '-m', 'experiments.expressivity_tasks.run_invertible_stage16_ignition_8m', '--name', name]
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
        if code == 0 and not valid_result(name, json.loads((run.ROOT / name / 'result.json').read_text())):
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
    registration = run.x10.verify_registration(run.ROOT)
    if name not in registration['authorized_runs'] or (run.ROOT / name).exists():
        raise ValueError('unauthorized or existing run')
    receipt = run.ROOT / 'receipts' / name
    receipt.mkdir(parents=True, exist_ok=False)
    with (receipt / 'worker_console.txt').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage16_control', 'worker', '--name', name],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run.x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'{name} worker={proc.pid} card={registration["gpu_binding"][name]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['probe', 'register', 'launch', 'worker'])
    parser.add_argument('--arm', choices=['e97', 'gdn2'])
    parser.add_argument('--name', choices=run.ARMS)
    args = parser.parse_args()
    if args.action == 'register':
        register()
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
