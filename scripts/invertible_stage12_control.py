#!/usr/bin/env python3
"""CPU-only preregistration/arming; fail-closed post-W3 launcher and leased workers."""
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

from experiments.expressivity_tasks import run_invertible_stage12_separation_8m as run
from scripts.invertible_stage10_control import identity

RECEIPTS = Path('receipts/w3-matcheddose')
PREDICTIONS = [
    "(a) SEPARATION HYPOTHESIS (the operator's framing + the mechanism expectation): GDN2's affine carry is depth-robust by construction; E97's bounded tanh state must re-encode exact state per step. Prediction: as T grows past the depth-4 parity point, GDN2's perfect-prefix depth EXCEEDS E97's materially under the matched working recipe (e.g., GDN2 >=2 positions beyond E97's perfect prefix at T64's ceiling). Report each run's perfect-prefix depth and per-position maxima as the primary readout.",
    "(b) PARITY-AT-DEPTH alternative: both reach the same prefix depth at T64/1M - then the architectures are equivalent on this task family at 8M under working recipes, and the composition-depth framing was always a training-stack artifact; register this reading plainly if it holds.",
    "(c) The P5-first acquisition timing at T64 (before deeper positions can matter) - if neither learns P5 at the longer ceiling, register that the ceiling is still the binding constraint and the separation question needs either longer ceilings or curriculum staging under the working recipe.",
    "(d) GDN2 should not be able to do more nonlinear steps than it has layers",
]


def git(*args):
    return subprocess.check_output(['git', *args], text=True, timeout=120)


def completed_gate(ref='origin/main'):
    """Only a remote-committed final report AND remote-committed read receipts admit GPUs.

    A launcher, arm-training DONE, local supervision terminal or idle board is not
    completion. Unknown receipt formats remain pending for attended clarification.
    """
    report = git('show', f'{ref}:W3-report.md')
    if not re.search(r'^##+\s+.*(?:gate verdict|final.verdict)', report, re.I | re.M):
        return None
    paths = git('ls-tree', '-r', '--name-only', ref, str(RECEIPTS)).splitlines()
    receipts = []
    for path in paths:
        name = Path(path).name.lower()
        if path.endswith('.json') and (name.startswith('gate-read') or ('gate' in name and 'verdict' in name)):
            content = git('show', f'{ref}:{path}')
            try:
                row = json.loads(content)
            except ValueError:
                continue
            if row.get('schema') == 'emender-w3-verdict-v1' and row.get('status') == 'measurements-complete':
                receipts.append({'path': path, 'sha256': hashlib.sha256(content.encode()).hexdigest()})
    if not receipts:
        return None
    return {'origin_commit': git('rev-parse', ref).strip(),
            'report_commit': git('log', '-1', '--format=%H', ref, '--', 'W3-report.md').strip(),
            'report_sha256': hashlib.sha256(report.encode()).hexdigest(), 'receipts': receipts,
            'verified_at': time.time()}


def register():
    import fla
    import torch
    if (run.ROOT / 'registration.json').exists():
        raise FileExistsError('registration is immutable')
    architectures = {}
    for arm in run.ARMS:
        torch.manual_seed(0)
        model = run.build_model(arm)  # CPU only; never .cuda() or evaluate.
        architectures[arm] = {'layers': len(model.layers), 'parameters': sum(p.numel() for p in model.parameters())}
        assert architectures[arm] == {'layers': 4, 'parameters': run.PARAMETERS[arm]}
        del model
    sources = [Path(__file__).relative_to(Path.cwd()), Path(run.__file__).relative_to(Path.cwd()),
               Path('scripts/watch_invertible_stage12_separation.py'), Path('scripts/gpu_lease.sh'),
               Path('scripts/invertible_stage10_control.py'), Path('scripts/watch_invertible_stage10_optimizers.py'),
               Path('tests/test_invertible_stage12_separation_8m.py'), run.ROOT / 'REGISTRATION.md',
               run.ROOT / 'receipts/tests_prelaunch.txt']
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(Path.cwd()) and '.venv' not in Path(file).parts:
            path = Path(file).resolve().relative_to(Path.cwd())
            if path.suffix == '.py' and str(path).startswith(('ndm/', 'experiments/expressivity_tasks/')):
                sources.append(path)
    dependencies = list(Path(fla.__file__).parent.rglob('*.py')) + [Path(inspect.getfile(torch.optim.AdamW))]
    registration = {
        'task': 'X12', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python_executable': sys.executable, 'python': sys.version,
        'torch': torch.__version__, 'fla': getattr(fla, '__version__', None),
        'authorized_runs': list(run.NAMES), 'architectures': architectures,
        'gpu_binding': 'post-W3 only: first two broker-grantable cards, one per run; no leases while waiting',
        'pool_array_sha256': {'0': [hashlib.sha256(a.tobytes()).hexdigest() for a in run.x7.frozen_pools(0, 64, 61)['val']]},
        'source_sha256': {str(p): run.x7.digest(p) for p in sorted(set(sources))},
        'dependency_sha256': {str(p.resolve()): run.x7.digest(p) for p in sorted(set(dependencies))},
        'recipe': {'initialization': 'scratch torch.manual_seed(0) independently for both; numpy.default_rng(0) procedural training',
            'architectures': 'X10 E97 legacy bf16 recurrence; real unmodified stock fla-gdn chunk GDN2; fp32 parameters/bf16 autocast',
            'p': 61, 'vocab': 3721, 'dim': 256, 'drivers': 64, 'batch': 32,
            'padding': '66 tokens right-padded to80 by unchanged X3 padded_forward; unpadded targets',
            'loss': 'uniform target cross entropy; no carry supervision or curriculum',
            'optimizer': 'ordinary torch.optim.AdamW live parameters', 'betas': [.9, .999], 'eps': 1e-8,
            'weight_decay': .01, 'decay_policy': 'ALL parameters, including A_log/dt_bias; exact X10 policy; tags ignored',
            'gradient_clip_norm': 1., 'schedule': 'updates1..5000 linear to1e-3; cosine to1e-5 at1000000; LR logged every update',
            'eval_every': 500, 'eval_examples': 512, 'positions': list(range(1, 65)), 'ceiling': 1000000,
            'stop': 'ceiling only, no stagnation rule, no threshold or all-perfect stop'},
        'predictions': PREDICTIONS,
        'competing_predictions': '(a) and (d) contradict each other; let the T64 columns decide. GDN2 observed four-perfect plus budding P5 must be reconciled with whichever survives.',
        'prediction_d_mechanism': 'GDN2 recurrent carry is linear in state (delta-rule update linear-in-S with token-dependent coefficients); state-dependent nonlinearity (squaring the carried coordinate) is available only via the readout->MLP->write-back path, once per layer per timestep; E97 tanh acts directly on state each step.',
        'prediction_d_readout': 'GDN2 perfect-prefix depth caps at or near its four layers; E97 prefix extends beyond its own four layers if depth-robust.',
        'readouts': 'fixed-final1M contiguous perfect prefix primary; maximum prefix at a single validation and independent per-position maxima descriptive; P1-P64 counts/CE every500; first50 and first512 acquisition steps per position, P5-first timing specifically',
        'pool_discipline': 'exact X3 frozen seed0 T64 validation; never sampled for training; no test or held-out performance inspected',
        'retention': 'entry + aggregate-best + latest/final, at most3 checkpoints plus one atomic staging file per run; deletion hashes retained',
        'resource_bounds': 'one GPU each; 1M update ceiling; no metric-based or wall-based early stop; OOM/disk/process failures censored, no automatic retries',
        'gate_policy': 'origin/main final Gate verdict heading in W3-report.md PLUS committed gate-read*/gate-verdict JSON with emender-w3-verdict-v1 measurements-complete; unknown formats fail closed pending attended clarification',
        'scope': 'standalone single-card expressivity only; not resilient DiLoCo or Frontier; no distributed requirements claimed',
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('NDM_', 'TRITON_', 'OMP_', 'MKL_', 'OPENBLAS_'))}}
    run.x7.write_json(run.ROOT / 'registration.json', registration)


def arm():
    run.x10.verify_registration(run.ROOT)
    receipt = run.ROOT / 'receipts/scheduler'
    receipt.mkdir(parents=True, exist_ok=False)
    with (receipt / 'console.txt').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage12_control', 'schedule'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run.x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'ARMED post-W3 scheduler PID={proc.pid}; no GPU lease taken', flush=True)


def schedule(interval=300):
    receipt = run.ROOT / 'receipts/scheduler'
    run.x7.write_json(receipt / 'worker.json', identity(os.getpid()))
    try:
        run.x10.verify_registration(run.ROOT)
        with (receipt / 'polls.jsonl').open('x', buffering=1) as log:
            while True:
                try:
                    git('fetch', 'origin', 'main')
                    gate = completed_gate()
                    state = {'time': time.time(), 'phase': 'WAIT_GATE', 'gate': gate}
                    if gate:
                        free = subprocess.check_output(['bash', 'scripts/gpu_lease.sh', 'list-free'], text=True, timeout=30).strip()
                        cards = [int(c) for c in free.split(',') if c]
                        state.update(phase='WAIT_CARDS', grantable=cards)
                        if len(cards) >= 2:
                            break
                except (subprocess.SubprocessError, OSError, ValueError) as exc:
                    state = {'time': time.time(), 'phase': 'WAIT_GATE', 'error': repr(exc)}
                log.write(json.dumps(state) + '\n')
                print(json.dumps(state), flush=True)
                time.sleep(interval)
            log.write(json.dumps({**state, 'phase': 'GATE_COMPLETED'}) + '\n')
        run.x10.verify_registration(run.ROOT)
        run.x7.write_json(receipt / 'gate_completed.json', gate)
        launched = []
        for name, gpu in zip(run.NAMES, cards[:2]):
            directory = run.ROOT / 'receipts' / name
            directory.mkdir(parents=True, exist_ok=False)
            if (run.ROOT / name).exists():
                raise FileExistsError('never restart an existing run')
            run.x7.write_json(directory / 'admission.json', {'gpu': gpu, 'gate': gate,
                'registration_sha256': run.x7.digest(run.ROOT / 'registration.json')})
            with (directory / 'worker_console.txt').open('x') as output:
                child = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage12_control', 'worker',
                                          '--name', name], stdin=subprocess.DEVNULL, stdout=output,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            record = {'name': name, 'gpu': gpu, 'worker': identity(child.pid)}
            run.x7.write_json(directory / 'detached.json', record)
            launched.append(record)
        run.x7.write_json(receipt / 'terminal.json', {'status': 'launched', 'runs': launched, 'time': time.time()})
        return 0
    except BaseException as exc:
        run.x7.write_json(receipt / 'terminal.json', {'status': 'failed', 'error': repr(exc), 'time': time.time()})
        return 1


def worker(name):
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
        gate = completed_gate(admission['gate']['origin_commit'])
        if gate is None or gate['report_sha256'] != admission['gate']['report_sha256']:
            raise ValueError('W3 gate completion not verified; no lease')
        if admission['registration_sha256'] != run.x7.digest(run.ROOT / 'registration.json'):
            raise ValueError('admission registration changed')
        gpu = admission['gpu']
        env.update(GPU_LEASE_VISIBLE=str(gpu), GPU_LEASE_PID=str(os.getpid()),
                   CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        with (receipt / 'lease.txt').open('x') as lease:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'],
                           env=env, stdout=lease, stderr=lease, check=True)
        leased = True
        cmd = [sys.executable, '-m', 'experiments.expressivity_tasks.run_invertible_stage12_separation_8m',
               '--arm', name.split('_')[0]]
        run.x7.write_json(receipt / 'command.json', {'argv': cmd, 'gpu': gpu, 'gate': gate})
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
    parser.add_argument('action', choices=['register', 'arm', 'schedule', 'worker'])
    parser.add_argument('--name', choices=run.NAMES)
    args = parser.parse_args()
    if args.action == 'worker':
        if not args.name:
            parser.error('--name required')
        raise SystemExit(worker(args.name))
    if args.action == 'schedule':
        raise SystemExit(schedule())
    globals()[args.action]()
