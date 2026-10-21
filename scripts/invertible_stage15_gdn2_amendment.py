#!/usr/bin/env python3
"""Attended X15 GDN2-only amendment: supervisor corrected best/final conflation.

Original registered source and running Muon are immutable. One authorized retry,
no T64 score existed before this amendment. Failed original attempt retained.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from experiments.expressivity_tasks import run_invertible_stage15_probes_8m as original
from scripts.invertible_stage10_control import identity

ROOT = original.ROOT / 'gdn2_amendment'
NAMES = ('gdn2_zeroshot',)
CEILING = 0
x7, x10, x11 = original.x7, original.x10, original.x11
CHAMPION, CHAMPION_SHA256 = original.CHAMPION, original.CHAMPION_SHA256
array_hashes, evaluate, verify_registration = original.array_hashes, original.evaluate, original.verify_registration
# Existing launch/lease protocol is copied verbatim below with only module/root binding changed.
run = sys.modules[__name__]


def register():
    base = original.verify_registration(original.ROOT)
    if (ROOT / 'registration.json').exists():
        raise FileExistsError('never overwrite amendment')
    sources = dict(base['source_sha256'])
    paths = [Path(__file__).relative_to(Path.cwd()),
        Path('scripts/watch_invertible_stage15_amended.py'),
        Path('tests/test_invertible_stage15_amendment.py'),
        original.ROOT / 'registration.json', ROOT / 'AMENDMENT.md',
        ROOT / 'receipts/tests_prelaunch.txt']
    failed = original.ROOT / 'receipts/gdn2_zeroshot'
    paths.extend(p for p in failed.glob('*') if p.is_file())
    paths.append(original.ROOT / 'markers/gdn2_zeroshot.FAIL')
    sources.update({str(p): x7.digest(p) for p in paths})
    registration = {'task': 'X15 GDN2 attended amendment1',
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'authorized_runs': list(NAMES), 'gpu_binding': {'gdn2_zeroshot': 6},
        'source_sha256': sources, 'dependency_sha256': base['dependency_sha256'],
        'pool_array_sha256': base['pool_array_sha256'], 'champion_sha256': CHAMPION_SHA256,
        'expected_t16_p1_p5': [512,512,512,512,333],
        'original_registration_sha256': x7.digest(original.ROOT / 'registration.json'),
        'resource_bounds': {'wall_seconds': 3600},
        'amendment': (ROOT / 'AMENDMENT.md').read_text()}
    x7.write_json(ROOT / 'registration.json', registration)


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
    assert [p['correct'] for p in t16['positions'][:5]] == [512, 512, 512, 512, 333]
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
        cmd = [sys.executable, '-m', 'scripts.invertible_stage15_gdn2_amendment', 'evaluate']
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
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage15_gdn2_amendment', 'worker', '--name', name],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run.x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'{name} worker={proc.pid} card={registration["gpu_binding"][name]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'launch', 'worker', 'evaluate'])
    parser.add_argument('--name', choices=NAMES, default='gdn2_zeroshot')
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.action == 'evaluate':
        zero_shot()
    elif args.action == 'worker':
        raise SystemExit(worker(args.name))
    else:
        launch(args.name)
