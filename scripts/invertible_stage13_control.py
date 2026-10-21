#!/usr/bin/env python3
"""X13 CPU registration and audited, leased, detached seed-replication workers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from experiments.expressivity_tasks import run_invertible_stage13_zerocheck_8m as run
from scripts.invertible_stage10_control import identity


def register():
    import torch
    run.require_clean_audit(run.ROOT)
    if (run.ROOT / 'registration.json').exists():
        raise FileExistsError('registration is immutable')
    torch.manual_seed(1)
    model = run.build_model('e97')  # CPU only, no validation or update.
    assert len(model.layers) == 4 and sum(p.numel() for p in model.parameters()) == 9900800
    sources = [Path(__file__).relative_to(Path.cwd()), Path(run.__file__).relative_to(Path.cwd()),
        Path('scripts/watch_invertible_stage13_zerocheck.py'), Path('scripts/audit_invertible_stage13_zerocheck.py'),
        Path('scripts/gpu_lease.sh'), Path('scripts/invertible_stage10_control.py'),
        Path('scripts/watch_invertible_stage10_optimizers.py'), Path('tests/test_invertible_stage13_zerocheck.py'),
        run.ROOT / 'REGISTRATION.md', run.ROOT / 'receipts/tests_prelaunch.txt',
        run.x12.ROOT / 'registration.json', *sorted((run.ROOT / 'audit').glob('*.json'))]
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(Path.cwd()) and '.venv' not in Path(file).parts:
            path = Path(file).resolve().relative_to(Path.cwd())
            if path.suffix == '.py' and str(path).startswith(('ndm/', 'experiments/expressivity_tasks/')):
                sources.append(path)
    registration = {'task': 'X13', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python_executable': sys.executable, 'python': sys.version,
        'torch': torch.__version__, 'authorized_runs': list(run.NAMES),
        'audit_verdict_sha256': run.x7.digest(run.ROOT / 'audit/verdict.json'),
        'source_sha256': {str(p): run.x7.digest(p) for p in sorted(set(sources))},
        'dependency_sha256': {inspect.getfile(torch.optim.AdamW): run.x7.digest(inspect.getfile(torch.optim.AdamW))},
        'pool_array_sha256': {str(seed): run.array_hashes(run.x7.frozen_pools(seed, 64, 61)['val']) for seed in run.SEEDS},
        'recipe': {**json.loads((run.x12.ROOT / 'registration.json').read_text())['recipe'],
            'initialization': 'scratch torch.manual_seed(seed) and numpy.default_rng(seed), seeds1 and2',
            'pool_discipline': 'exact historical frozen seed-specific T64 val512; never sampled for training; no test scores'},
        'predictions': ['supervisor prelaunch ruling: ANY seed with ANY P1..P64 >=50/512 => seed-0 learned-nothing zero was lottery; re-scope',
                        'both seeds with NO P1..P64 >=50/512 => learned-nothing zero replicates as3/3',
                        'P1-only marginal escape is NOT composition: re-scope to learns visible position but not carry at T64'],
        'criteria': {'marginal_escape': 'any position P1..P64 >=50/512 at any validation',
                     'composition': 'any position P2..P64 >=50/512 at any validation'},
        'readout': 'fixed-final1M prefix; full P1-P64 latest/max/first50/first512 per seed; P1 and P2+ classified separately',
        'stop_policy': '1M ceiling only; no stagnation, threshold, all-perfect or wall-time stop; failures censored; no automatic retries',
        'retention': 'entry + aggregate-best + latest/final, at most3 checkpoints plus atomic staging file; deletion hashes',
        'scope': 'NO mechanism claims; X12 interpretation suspended pending audit; standalone single-card expressivity, not resilient DiLoCo or Frontier',
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('NDM_', 'TRITON_', 'OMP_', 'MKL_', 'OPENBLAS_'))}}
    run.x7.write_json(run.ROOT / 'registration.json', registration)


def launch():
    run.require_clean_audit(run.ROOT)
    run.x10.verify_registration(run.ROOT)
    free = subprocess.check_output(['bash', 'scripts/gpu_lease.sh', 'list-free'], text=True, timeout=30)
    cards = [int(c) for c in free.strip().split(',') if c]
    if len(cards) < 2:
        raise RuntimeError('two grantable cards required; no workers launched')
    for name in run.NAMES:
        if (run.ROOT / name).exists() or (run.ROOT / 'receipts' / name).exists():
            raise FileExistsError('never restart existing seed: ' + name)
    launched = []
    for seed, name, gpu in zip(run.SEEDS, run.NAMES, cards[:2]):
        receipt = run.ROOT / 'receipts' / name
        receipt.mkdir(parents=True, exist_ok=False)
        run.x7.write_json(receipt / 'admission.json', {'gpu': gpu,
            'registration_sha256': run.x7.digest(run.ROOT / 'registration.json'),
            'audit_verdict_sha256': run.x7.digest(run.ROOT / 'audit/verdict.json')})
        with (receipt / 'worker_console.txt').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage13_control', 'worker', '--seed', str(seed)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'name': name, 'gpu': gpu, 'worker': identity(proc.pid)}
        run.x7.write_json(receipt / 'detached.json', record)
        launched.append(record)
        print(json.dumps(record), flush=True)
    run.x7.write_json(run.ROOT / 'receipts/launch.json', {'runs': launched, 'time': time.time()})


def worker(seed):
    name = f'e97_bf16_seed{seed}'
    receipt = run.ROOT / 'receipts' / name
    run.x7.write_json(receipt / 'worker.json', identity(os.getpid()))
    def interrupted(signum, _frame):
        raise SystemExit(f'worker received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    code, child, error, leased, gpu = 1, None, None, False, None
    env = dict(os.environ)
    try:
        run.require_clean_audit(run.ROOT)
        registration = run.x10.verify_registration(run.ROOT)
        if name not in registration['authorized_runs']:
            raise ValueError('unauthorized worker')
        admission = json.loads((receipt / 'admission.json').read_text())
        if (admission['registration_sha256'] != run.x7.digest(run.ROOT / 'registration.json')
                or admission['audit_verdict_sha256'] != run.x7.digest(run.ROOT / 'audit/verdict.json')):
            raise ValueError('admission changed')
        gpu = admission['gpu']
        env.update(GPU_LEASE_VISIBLE=str(gpu), GPU_LEASE_PID=str(os.getpid()),
                   CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        with (receipt / 'lease.txt').open('x') as lease:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'],
                           env=env, stdout=lease, stderr=lease, check=True)
        leased = True
        cmd = [sys.executable, '-m', 'experiments.expressivity_tasks.run_invertible_stage13_zerocheck_8m', '--seed', str(seed)]
        run.x7.write_json(receipt / 'command.json', {'argv': cmd, 'gpu': gpu,
            'audit_verdict_sha256': admission['audit_verdict_sha256']})
        with (receipt / 'console.txt').open('x') as output:
            child = subprocess.Popen(cmd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            run.x7.write_json(receipt / 'trainer.json', identity(child.pid))
            while True:
                try:
                    code = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    subprocess.run(['bash', 'scripts/gpu_lease.sh', 'heartbeat', str(gpu)], env=env,
                                   check=True, stdout=subprocess.DEVNULL)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'launch', 'worker'])
    parser.add_argument('--seed', type=int, choices=run.SEEDS)
    args = parser.parse_args()
    if args.action == 'worker':
        if args.seed is None:
            parser.error('--seed required')
        raise SystemExit(worker(args.seed))
    globals()[args.action]()
