#!/usr/bin/env python3
"""X17 CPU registration, fail-closed first-free-card waiting and native leased worker."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from experiments.expressivity_tasks import run_invertible_stage17_carrywean_8m as run
from scripts.invertible_stage10_control import identity


PREDICTIONS = [
    "(a) WEAN-TRANSFER: the run acquires carry positions materially (any P2+ >=50/512 during or by end) with the crutch annealed to zero - examples-of-working training transfers to the no-crutch regime at the horizon where cold learning fails; the fix works and the paper gets its training-technology result.",
    "(b) CRUTCH-DEPENDENT-AT-DEPTH: composition appears while p is high but decays as p anneals (X6's cycling signature, flattened) - the carry remains crutch-bound at T64 and the wean problem persists at long horizons.",
    "(c) The P1 diagnostic rides free: does the supervised carry slot also ignite P1-like visible learning earlier than Muon/AdamW cold runs? Logged either way."
]


def register():
    import fla
    import torch
    if (run.ROOT / 'registration.json').exists():
        raise FileExistsError('registration is immutable')
    torch.manual_seed(0)
    model = run.build_model()  # CPU only; no GPU lease, transfer, or evaluation.
    assert len(model.layers) == 4 and sum(p.numel() for p in model.parameters()) == run.PARAMETERS
    sources = [Path(__file__).relative_to(Path.cwd()), Path(run.__file__).relative_to(Path.cwd()),
        Path('scripts/watch_invertible_stage17_carrywean.py'), Path('scripts/gpu_lease.sh'),
        Path('scripts/invertible_stage10_control.py'), Path('scripts/watch_invertible_stage10_optimizers.py'),
        Path('tests/test_invertible_stage17_carrywean_8m.py'), run.ROOT / 'REGISTRATION.md',
        run.ROOT / 'receipts/tests_prelaunch.txt']
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(Path.cwd()) and '.venv' not in Path(file).parts:
            path = Path(file).resolve().relative_to(Path.cwd())
            if path.suffix == '.py' and str(path).startswith(('ndm/', 'experiments/expressivity_tasks/')):
                sources.append(path)
    sources.extend(Path('ndm/triton').glob('e88*.py'))
    dependencies = list(Path(fla.__file__).parent.rglob('*.py')) + [Path(inspect.getfile(torch.optim.AdamW))]
    registration = {'task': 'X17', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python_executable': sys.executable, 'python': sys.version,
        'torch': torch.__version__, 'authorized_runs': list(run.NAMES),
        'source_sha256': {str(p): run.x7.digest(p) for p in sorted(set(sources))},
        'dependency_sha256': {str(p.resolve()): run.x7.digest(p) for p in sorted(set(dependencies))},
        'pool_array_sha256': {'0': [hashlib.sha256(a.tobytes()).hexdigest() for a in run.x7.frozen_pools(0, 64, 61)['val']]},
        'recipe': {'initialization': 'scratch torch.manual_seed(0), numpy.default_rng(0); carry Bernoulli draws share training RNG as in X6',
            'architecture': 'exact X10 E97 legacy bf16 recurrence, sequential Triton at train/eval; dim256, four layers; fp32 parameters/bf16 autocast',
            'parameters': run.PARAMETERS, 'input_vocab': 3722, 'output_vocab': 3721, 'batch': 32, 'drivers': 64,
            'format_decision': 'Supervisor approved exact X6 Arm B: slots before P2..P64 ONLY; no slot before P1. 129 tokens right-padded to144; logits gathered back to original66 positions. Train/eval identical format.',
            'carry': 'independent Bernoulli(p) per example and P2+ slot: TRUE previous joint-state token or neutral3721; never current/future state; EVAL ALWAYS NEUTRAL',
            'wean': 'ONE design: p=.5*(1-update/1000000), full1M linear global wean; never reset; actual p logged every update; p=0 at1M',
            'optimizer': 'exact X12 ordinary AdamW live parameters, betas(.9,.999), eps1e-8, ALL parameters wd.01, clip1; uniform masked-target CE',
            'lr': 'X12: updates1..5000 linear to1e-3; cosine to1e-5 at1M; actual LR logged every update',
            'evaluation': 'exact frozen historical seed0 T64 validation512; neutral only; update0 then every500 through1M; P1..P64 count/CE; never used in training; no test selection'},
        'gpu_binding': 'poll broker list-free every60s; first grantable card only; no lease while waiting; atomic no-wait acquire before trainer; lost race fails closed without retry',
        'predictions': PREDICTIONS,
        'readout': 'first50 and first512 per position; latest/final and independent maxima; single-validation perfect prefix; P1 and P2+ separate; early composition does NOT establish zero-crutch survival',
        'stop_policy': 'fixed1M only; no threshold, stagnation, all-perfect or wall stop; infrastructure failures censored; no automatic retry/restart',
        'retention': 'entry + aggregate-best + latest/final <=3 checkpoints plus atomic staging; deletion hashes',
        'scope': 'standalone single-card expressivity; no resilient DiLoCo or Frontier requirement applicable',
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('NDM_', 'TRITON_', 'OMP_', 'MKL_', 'OPENBLAS_'))}}
    run.x7.write_json(run.ROOT / 'registration.json', registration)


def arm():
    run.x10.verify_registration(run.ROOT)
    receipt = run.ROOT / 'receipts/scheduler'
    receipt.mkdir(parents=True, exist_ok=False)
    with (receipt / 'console.txt').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage17_control', 'schedule'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run.x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'ARMED first-free scheduler PID={proc.pid}; no GPU lease taken by arming', flush=True)


def schedule(interval=60):
    receipt = run.ROOT / 'receipts/scheduler'
    run.x7.write_json(receipt / 'worker.json', identity(os.getpid()))
    try:
        run.x10.verify_registration(run.ROOT)
        with (receipt / 'polls.jsonl').open('a', buffering=1) as log:
            while True:
                try:
                    free = subprocess.check_output(['bash', 'scripts/gpu_lease.sh', 'list-free'], text=True, timeout=30)
                    free = free.strip()
                    if free and not re.fullmatch(r'[0-7](,[0-7])*', free):
                        raise ValueError('invalid broker card list')
                    cards = [int(c) for c in free.split(',')] if free else []
                    if len(set(cards)) != len(cards):
                        raise ValueError('duplicate broker cards')
                    state = {'time': time.time(), 'phase': 'WAIT_CARDS', 'grantable': cards}
                except (subprocess.SubprocessError, OSError, ValueError) as exc:
                    cards = []  # Broker failure is never permission to lease.
                    state = {'time': time.time(), 'phase': 'WAIT_CARDS', 'error': repr(exc)}
                log.write(json.dumps(state) + '\n')
                print(json.dumps(state), flush=True)
                if cards:
                    break
                time.sleep(interval)
        run.x10.verify_registration(run.ROOT)
        name, gpu = run.NAMES[0], cards[0]
        directory = run.ROOT / 'receipts' / name
        if (run.ROOT / name).exists():
            raise FileExistsError('never restart an existing run')
        directory.mkdir(parents=True, exist_ok=False)
        run.x7.write_json(directory / 'admission.json', {'gpu': gpu,
            'registration_sha256': run.x7.digest(run.ROOT / 'registration.json')})
        with (directory / 'worker_console.txt').open('x') as output:
            child = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage17_control', 'worker'],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'name': name, 'gpu': gpu, 'worker': identity(child.pid)}
        run.x7.write_json(directory / 'detached.json', record)
        run.x7.write_json(receipt / 'terminal.json', {'status': 'launched', 'runs': [record], 'time': time.time()})
        return 0
    except BaseException as exc:
        run.x7.write_json(receipt / 'terminal.json', {'status': 'failed', 'error': repr(exc), 'time': time.time()})
        return 1


def worker():
    name = run.NAMES[0]
    receipt = run.ROOT / 'receipts' / name
    run.x7.write_json(receipt / 'worker.json', identity(os.getpid()))
    def interrupted(signum, _frame):
        raise SystemExit(f'worker received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    code, child, error, leased, gpu = 1, None, None, False, None
    env = dict(os.environ)
    try:
        registration = run.x10.verify_registration(run.ROOT)
        if name not in registration['authorized_runs']:
            raise ValueError('unauthorized worker')
        admission = json.loads((receipt / 'admission.json').read_text())
        if admission['registration_sha256'] != run.x7.digest(run.ROOT / 'registration.json'):
            raise ValueError('admission changed')
        gpu = admission['gpu']
        if type(gpu) is not int or gpu not in range(8):
            raise ValueError('invalid GPU admission')
        env.update(GPU_LEASE_VISIBLE=str(gpu), GPU_LEASE_PID=str(os.getpid()),
                   CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        with (receipt / 'lease.txt').open('x') as lease:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'],
                           env=env, stdout=lease, stderr=lease, check=True)
        leased = True
        cmd = [sys.executable, '-m', 'experiments.expressivity_tasks.run_invertible_stage17_carrywean_8m']
        run.x7.write_json(receipt / 'command.json', {'argv': cmd, 'gpu': gpu})
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
            if (result['name'] != name or result['total_steps'] != run.CEILING or result['status'] != 'complete'
                    or result['stop_reason'] != 'safety_ceiling' or result['final']['step'] != run.CEILING
                    or result['final']['carry_probability'] != 0. or result['final']['eval_carry_probability'] != 0.
                    or len(result['final']['positions']) != 64):
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
    parser.add_argument('action', choices=['register', 'arm', 'schedule', 'worker'])
    args = parser.parse_args()
    if args.action in ('schedule', 'worker'):
        raise SystemExit(globals()[args.action]())
    globals()[args.action]()
