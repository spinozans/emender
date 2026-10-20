#!/usr/bin/env python3
"""X7b/X10 bounded probe, preregistration, detached single-card worker and readout."""
from __future__ import annotations

import argparse
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

# Invoked as -m scripts.invertible_stage10_control from repository root.
from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as run
from experiments.expressivity_tasks import hybrid_muon


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return {'pid': pid, 'starttime': stat[19], 'host': socket.gethostname(), 'ppid': int(stat[1])}


def probe(arm):
    """Disposable synthetic/procedural updates; no frozen validation scores."""
    import torch
    torch.manual_seed(0)
    model = run.build_model(arm).cuda()
    opt = run.make_optimizer(model, arm)
    rng, task = run.np.random.default_rng(0), run.InvertibleQuadraticTask(61)
    rows = []
    start = time.monotonic()
    for step in range(1, 26):
        rows.append(run.train_update(model, opt, arm, task.generate_batch(32, 18, rng), step))
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    # Full-model serialization/next-update parity, without task-score selection.
    path = run.ROOTS[arm] / 'receipts' / f'probe_{arm}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, path)
    saved = torch.load(path, weights_only=False)
    restored = run.build_model(arm).cuda()
    restored.load_state_dict(saved['model'])
    opt2 = run.make_optimizer(restored, arm)
    opt2.load_state_dict(saved['optimizer'])
    batch = task.generate_batch(32, 18, rng)
    run.train_update(model, opt, arm, batch, 26)
    run.train_update(restored, opt2, arm, batch, 26)
    max_difference = max((p - q).abs().max().item() for p, q in zip(model.parameters(), restored.parameters()))
    if max_difference > 1e-6:
        raise AssertionError(f'next-update reload mismatch {max_difference}')
    peak = torch.cuda.max_memory_allocated()
    path.unlink()
    result = {'arm': arm, 'updates': 25,
              'source_sha256': {p: run.x7.digest(p) for p in [str(Path(__file__).relative_to(Path.cwd())),
                  str(Path(run.__file__).relative_to(Path.cwd())), str(Path(hybrid_muon.__file__).relative_to(Path.cwd()))]},
              'elapsed_including_compile_seconds': elapsed,
              'steady_optimizer_seconds_mean': sum(r['optimizer_seconds'] for r in rows[5:]) / 20,
              'peak_allocated_bytes_two_models': peak, 'reload_next_update_max_abs_difference': max_difference,
              'frozen_task_evaluation': False, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'wall_ceiling_seconds': max(86400, math.ceil(elapsed / 25 * 500000 * 3)),
              'memory_ceiling_bytes': max(8 * 1024**3, math.ceil(peak * 2))}
    run.x7.write_json(path.with_suffix('.json'), result)
    print(json.dumps(result), flush=True)


def register():
    import torch
    import fla
    import schedulefree
    from datetime import datetime, timezone
    # Construct before enumerating modules: model wrappers/cores are lazy imports.
    torch.manual_seed(0)
    routes = hybrid_muon.parameter_routes(run.build_model('muon'))
    run.build_model('gdn2_uniform')
    # All operational code and imported project model/kernel/task implementations.
    sources = [Path(__file__).relative_to(Path.cwd()), Path(run.__file__).relative_to(Path.cwd()),
               Path(hybrid_muon.__file__).relative_to(Path.cwd()),
               Path('scripts/watch_invertible_stage10_optimizers.py'), Path('scripts/watch_invertible_stage7b_stock.py'),
               Path('scripts/gpu_lease.sh'), Path('tests/test_invertible_stage10_optimizers_8m.py'),
               Path('docs/OPTIMIZERS-2026-report.md')]
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(Path.cwd()) and '.venv' not in Path(file).parts:
            path = Path(file).resolve().relative_to(Path.cwd())
            if path.suffix == '.py' and str(path).startswith(('ndm/', 'experiments/expressivity_tasks/')):
                sources.append(path)
    sources.extend(Path('ndm/triton').glob('e88*.py'))
    for arm in run.ARMS:
        sources.append(run.ROOTS[arm] / 'receipts' / f'probe_{arm}.json')
    sources.extend(run.ROOTS['muon'] / 'receipts' / name for name in ('tests_prelaunch.txt', 'tests_gpu_prelaunch.txt', 'packages.txt', 'gpu_before.txt'))
    dependencies = list(Path(fla.__file__).parent.rglob('*.py'))
    dependencies += [Path(inspect.getfile(schedulefree.AdamWScheduleFree)), Path(inspect.getfile(torch.optim.AdamW))]
    pools = {str(seed): [hashlib.sha256(a.tobytes()).hexdigest() for a in run.x7.frozen_pools(seed, 16, 61)['val']] for seed in range(3)}
    common = {'registered_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
        'python': sys.version, 'torch': torch.__version__, 'fla': getattr(fla, '__version__', None),
        'pool_array_sha256': pools, 'parameter_routes': routes,
        'source_sha256': {str(p): run.x7.digest(p) for p in sorted(set(sources))},
        'dependency_sha256': {str(p.resolve()): run.x7.digest(p) for p in sorted(set(dependencies))},
        'shared_recipe': 'p61/vocab3721, scratch T16, E97 9900800 / GDN2 9887056, dim256/depth4, batch32, uniform target CE, clip1, fp32 parameters/bf16 autocast, E97 legacy state + sequential Triton eval, GDN2 unmodified stock bf16 chunk recurrence; eval512 every500, frozen validation never trained on; procedural numpy RNG(seed), torch initialization(seed); no ladder/carry/threshold/stagnation stop; final500000 primary',
        'decay_policy': 'Supervisor-approved exact historical recipe: all parameters .01, including A_log/dt_bias _no_weight_decay tags IGNORED in ALL arms. No knob LR multipliers exist. Decay-policy sensitivity is a registered FOLLOW-UP, not this experiment variable. Moonshot per-parameter RMS shape scaling is A-specific as part of Muon itself.',
        'retention': 'always entry + aggregate-best + latest/final, at most one staging checkpoint; deletion hashes retained; no historical deletion',
        'resilience_scope': 'R16 evidence discipline by analogy only; no DiLoCo/K-boundary/Frontier/system-scale claim; no retry',
        'recipes': {'muon': {'family': hybrid_muon.VERSION, 'momentum': .95, 'nesterov': True,
            'ns_coefficients': list(hybrid_muon.COEFFICIENTS), 'ns_iterations': 5, 'ns_normalization': 'FP32 Frobenius norm + 1e-7; tall transpose; zero maps to zero', 'dtype': 'FP32 parameters/momentum/NS; bf16 model autocast',
            'scaling': '.2*sqrt(max(rows,cols)); approximate AdamW update RMS matching, not exact .2',
            'auxiliary_adamw': 'betas(.9,.999),eps1e-8,wd.01', 'schedule': 'X9 linear5K warmup1e-3 then cosine1e-5 at500K, both groups once/update'},
            'adamw': 'exact fresh X9: betas(.9,.999),eps1e-8,wd.01; same5K/1e-3/1e-5 schedule',
            'sf': 'exact X7: AdamWScheduleFree,betas(.9,.95),eps1e-8,wd.01,constant1e-3,no warmup; installed implementation/defaults source hashed',
            'gdn2_uniform': 'exact X7 SF recipe above; stock X8 GDN2 bf16 kernel, uniform loss only'},
        'point_semantics': {a: run.point(a) for a in run.ARMS},
        'seeds': [0, 1, 2], 'ceiling': 500000, 'readouts': 'full P1-P16 joint counts/CE, first50/first512, fixed-final contiguous perfect prefix, separate max-anywhere; update/grad/weight norms, optimizer and end-to-end seconds including eval/checkpoint/compilation; peak allocation',
        'analysis_command': '.venv/bin/python -m scripts.invertible_stage10_control analyze --arm ARM',
        'resource_bounds': {a: {'wall_seconds': json.loads((run.ROOTS[a] / 'receipts' / f'probe_{a}.json').read_text())['wall_ceiling_seconds'],
                                'peak_allocated_bytes': json.loads((run.ROOTS[a] / 'receipts' / f'probe_{a}.json').read_text())['memory_ceiling_bytes']} for a in run.ARMS},
        'censoring': 'resource/time/infrastructure failure censored, not optimizer loss; no automatic retry or restart',
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('NDM_', 'TRITON_'))}}
    predictions = [
        'R1: B final perfect-prefix>=3 in >=2/3 seeds; C<=1 in >=2/3. Failed replication makes historical-pathology transfer inconclusive.',
        'R2: A final depth>=3 in >=2/3 seeds; fewer falsifies. Paired P3 acquisition/depth reported; matching not a speed win.',
        'R3: conservative prediction no robust P4+ perfect-composition gain. Higher bar A final>=4 in >=2/3, relative advantage only if A exceeds paired B in >=2/3. Transient50/512 exploratory, not robust final composition.',
        'R4: C flat only tests retained SF recipe, not all SF repairs; beta2/decay/schedule/averaging mechanisms not isolated.',
        '(a) Muon composes: any P2+ >=50/512 material,512 perfect. Compare contemporaneous B material/perfect acquisition times; earlier material versus historical B P2 first512~245500 is a descriptive benchmark, not a like-threshold speed win. Better paired basin-dynamics evidence requires same-threshold and measured wall times.',
        '(b) P4 question: does ANY arm reach P4>=50/512, the position no run materially cracked? Report all arms and final vs transient separately.',
        '(c) Muon flat like C is exonerated-or-indicted by its own numerical/routing/source receipts; instrument reports what happens, no post-result tuning.'
    ]
    for kind, arms in [('X10', ('muon', 'adamw', 'sf')), ('X7b', ('gdn2_uniform',))]:
        root = run.ROOTS[arms[0]]
        reg = {**common, 'task': kind, 'authorized_runs': [f'{a}_seed0' for a in arms],
               'gpu_binding': {f'{a}_seed0': run.CARDS[a] for a in arms},
               'remaining_seeds': 'X10 seeds1/2 require clean implementation/receipt review, NOT seed0 score selection; not auto-launched',
               'predictions': predictions if kind == 'X10' else [
                   'If uniform stock-bf16 GDN2 reaches P1-P3 perfect and P4 materially above X8 9/512 (>=50/512 like X7-fp32 trajectory), X8 GDN2 weighting refutation upgrades descriptive to causal on matched evidence; otherwise do not upgrade. Same seed0 initialization/pool/task/500K; no fp32 adapter.']}
        run.x7.write_json(root / 'registration.json', reg)


def worker(arm):
    root, name = run.ROOTS[arm], f'{arm}_seed0'
    receipt = root / 'receipts' / name
    receipt.mkdir(parents=True, exist_ok=True)
    run.x7.write_json(receipt / 'worker.json', identity(os.getpid()))
    def interrupted(signum, _frame):
        raise SystemExit(f'worker received signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    env = {**os.environ, 'GPU_LEASE_VISIBLE': str(run.CARDS[arm]), 'GPU_LEASE_PID': str(os.getpid()),
           'CUDA_VISIBLE_DEVICES': str(run.CARDS[arm]), 'PYTHONUNBUFFERED': '1'}
    code, child = 1, None
    error = None
    try:
        registration = run.verify_registration(root)
        with (receipt / 'lease.txt').open('w') as lease:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'], env=env, stdout=lease, stderr=lease, check=True)
        cmd = [sys.executable, '-m', 'experiments.expressivity_tasks.run_invertible_stage10_optimizers_8m', '--arm', arm, '--seed', '0']
        run.x7.write_json(receipt / 'command.json', {'argv': cmd, 'gpu': run.CARDS[arm]})
        with (receipt / 'console.txt').open('w') as output:
            child = subprocess.Popen(cmd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            run.x7.write_json(receipt / 'trainer.json', identity(child.pid))
            deadline = time.monotonic() + registration['resource_bounds'][arm]['wall_seconds'] + 120
            while True:
                try:
                    code = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    subprocess.run(['bash', 'scripts/gpu_lease.sh', 'heartbeat', str(run.CARDS[arm])], env=env, check=True, stdout=subprocess.DEVNULL)
                    if time.monotonic() > deadline:
                        raise TimeoutError('external wall ceiling')
        if code == 0 and not (root / name / 'result.json').exists():
            raise RuntimeError('missing result')
    except BaseException as exc:
        error = repr(exc)
        code = 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
        terminal = {'name': name, 'exit_code': code, 'error': error, 'finished_at': time.time(),
                    'result_sha256': run.x7.digest(root / name / 'result.json') if code == 0 else None}
        run.x7.write_json(receipt / 'terminal.json', terminal)
        run.x7.write_json(root / 'markers' / (name + ('.DONE' if code == 0 else '.FAIL')), terminal)
        subprocess.run(['bash', 'scripts/gpu_lease.sh', 'release', str(run.CARDS[arm])], env=env, check=True, stdout=subprocess.DEVNULL)
    return code


def launch(arm):
    root = run.ROOTS[arm]
    registration = run.verify_registration(root)
    name = f'{arm}_seed0'
    if name not in registration['authorized_runs'] or (root / name).exists():
        raise ValueError('unauthorized or existing run')
    receipt = root / 'receipts' / name
    receipt.mkdir(parents=True, exist_ok=False)
    with (receipt / 'worker_console.txt').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage10_control', 'worker', '--arm', arm],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run.x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'{name} worker={proc.pid} card={run.CARDS[arm]}', flush=True)


def analyze(arm):
    rows = json.loads((run.ROOTS[arm] / f'{arm}_seed0' / 'validation.json').read_text())
    crossings = [{'position': p + 1, 'first50': next((r['step'] for r in rows if r['positions'][p]['correct'] >= 50), None),
                  'first512': next((r['step'] for r in rows if r['positions'][p]['correct'] == 512), None),
                  'maximum': max(r['positions'][p]['correct'] for r in rows),
                  'final': rows[-1]['positions'][p]['correct']} for p in range(16)]
    print(json.dumps({'arm': arm, 'observed_through_step': rows[-1]['step'],
                      'fixed_final_valid': rows[-1]['step'] == 500000,
                      'final_prefix': rows[-1]['perfect_prefix_depth'], 'positions': crossings}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['probe', 'register', 'launch', 'worker', 'analyze'])
    parser.add_argument('--arm', choices=run.ARMS)
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.arm is None:
        parser.error('--arm required')
    else:
        result = globals()[args.action](args.arm)
        if args.action == 'worker':
            raise SystemExit(result)
