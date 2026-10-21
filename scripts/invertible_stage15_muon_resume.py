#!/usr/bin/env python3
"""X15b attended infrastructure resume; original X15 artifacts remain immutable."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
from experiments.expressivity_tasks import run_invertible_stage15_probes_8m as original
from scripts.invertible_stage10_control import identity

ROOT = original.ROOT / 'muon_resume'
OUT = ROOT / 'muon3x_seed0'
SOURCE = original.ROOT / 'muon3x_seed0'
CHECKPOINT = SOURCE / 'step_433000.pt'
SHA256 = '373a7ece3a97790e81d9fa3d0888afdedd74e8eaf8e9b68bbc834c8cf2e882c0'
CEILING = original.CEILING
NAMES = ('muon3x_seed0',)
x7, x10 = original.x7, original.x10
DISK_MIN_BYTES = 200 * 1024**3  # Conservative interpretation of 200G: 200 GiB.
DIAGNOSTICS = ('loss', 'grad_norm', 'lr', 'full_update_rms', 'muon_update_rms')


class DiskGuardError(RuntimeError):
    pass


def disk_guard():
    free = shutil.disk_usage('/').free
    if free < DISK_MIN_BYTES:
        raise DiskGuardError(f'root free {free} < {DISK_MIN_BYTES}; retain last committed checkpoint')
    return free


def load_checkpoint(path=CHECKPOINT, expected=SHA256):
    import torch
    if x7.digest(path) != expected:
        raise ValueError('resume checkpoint hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['step'] != 433000 or saved['optimizer_family'] != 'muon3x':
        raise ValueError('wrong resume checkpoint')
    for family in ('muon', 'adamw'):
        if not saved['optimizer'][family]['state']:
            raise ValueError('missing optimizer state: ' + family)
    if not saved['numpy_rng'] or not saved['torch_rng'].numel() or len(saved['cuda_rng']) != 1:
        raise ValueError('missing RNG state')
    return saved


def assert_state_equal(actual, expected):
    """Assert exact restored state, ignoring only the tensor storage device."""
    import torch
    if isinstance(expected, torch.Tensor):
        if actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError('restored tensor mismatch')
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError('restored keys mismatch')
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError('restored sequence mismatch')
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    elif actual != expected:
        raise ValueError('restored value mismatch')


def replay_reference(path):
    rows = {}
    with path.open() as stream:
        for line in stream:
            match = re.match(r'\{"step": (\d+),', line)
            if not match or not 433001 <= int(match[1]) <= 433296:
                continue
            step = int(match[1])
            try:
                row = json.loads(line)
            except ValueError:
                if step != 433296:
                    raise
                # Only complete numeric/null fields from the truncated final line.
                row = {k: json.loads(v) for k, v in re.findall(
                    r'"([a-z_]+)": (null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=,)', line)}
            if not all(k in row for k in DIAGNOSTICS):
                raise ValueError(f'incomplete replay diagnostics at {step}')
            rows[step] = {k: row[k] for k in DIAGNOSTICS}
    if set(rows) != set(range(433001, 433297)):
        raise ValueError('missing replay updates')
    return rows


def compare_replay(step, metrics, reference):
    for key, expected in reference[step].items():
        if metrics[key] != expected:
            raise ValueError(f'replay mismatch step={step} {key}: {metrics[key]} != {expected}')


def register():
    base = original.verify_registration(original.ROOT)
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'registration.json').exists():
        raise FileExistsError('never overwrite resume registration')
    saved = load_checkpoint()
    manifest = json.loads((SOURCE / 'checkpoints.json').read_text())
    if manifest[str(CHECKPOINT)]['sha256'] != SHA256 or manifest[str(CHECKPOINT)]['step'] != 433000:
        raise ValueError('original checkpoint manifest mismatch')
    reference = replay_reference(SOURCE / 'training.jsonl')
    receipt = {'checkpoint': str(CHECKPOINT), 'sha256': SHA256, 'step': saved['step'],
        'optimizer_state_entries': {k: len(saved['optimizer'][k]['state']) for k in ('muon', 'adamw')},
        'rng_keys_verified': ['numpy_rng', 'torch_rng', 'cuda_rng'],
        'replay_updates': len(reference), 'proof_scope': 'diagnostics, not bitwise full-state proof'}
    x7.write_json(ROOT / 'receipts/checkpoint_preflight.json', receipt)
    sources = dict(base['source_sha256'])
    paths = [Path(__file__).relative_to(Path.cwd()), Path('scripts/watch_invertible_stage15_muon_resume.py'),
        Path('tests/test_invertible_stage15_muon_resume.py'), ROOT / 'AMENDMENT-2026-10-21.md',
        ROOT / 'registration-supersession.json', ROOT / 'receipts/checkpoint_preflight.json',
        ROOT / 'receipts/tests_prelaunch.txt', original.ROOT / 'registration.json',
        ]
    artifacts = [SOURCE / name for name in ('training.jsonl', 'validation.json', 'checkpoints.json',
        'config.json', 'frozen_validation.npz')]
    sources.update({str(p): x7.digest(p) for p in paths})
    registration = {**base, 'task': 'X15b Muon infrastructure resume',
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'authorized_runs': list(NAMES), 'gpu_binding': {'muon3x_seed0': 7}, 'source_sha256': sources,
        'resume': receipt, 'disk_min_bytes': DISK_MIN_BYTES,
        'artifact_sha256': {str(p): x7.digest(p) for p in artifacts},
        'original_registration_sha256': x7.digest(original.ROOT / 'registration.json'),
        'protocol': (ROOT / 'AMENDMENT-2026-10-21.md').read_text()}
    x7.write_json(ROOT / 'registration.json', registration)


def train():
    import torch
    registration = x10.verify_registration(ROOT)
    for path, sha in registration['artifact_sha256'].items():
        if x7.digest(path) != sha:
            raise ValueError('changed original artifact: ' + path)
    disk_guard()
    saved = load_checkpoint()
    reference = replay_reference(SOURCE / 'training.jsonl')
    OUT.mkdir(parents=True, exist_ok=False)
    pool = original.x7.frozen_pools(0, 64, 61)['val']
    if original.array_hashes(pool) != registration['pool_array_sha256']['0']:
        raise ValueError('frozen pool changed')
    model = original.build_model('muon3x').cuda()
    model.load_state_dict(saved['model'], strict=True)
    opt = original.make_optimizer(model, 'muon3x')
    opt.load_state_dict(saved['optimizer'])
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    assert_state_equal(model.state_dict(), saved['model'])
    assert_state_equal(opt.state_dict(), saved['optimizer'])
    assert_state_equal(rng.bit_generator.state, saved['numpy_rng'])
    assert_state_equal(torch.get_rng_state(), saved['torch_rng'])
    assert_state_equal(torch.cuda.get_rng_state_all(), saved['cuda_rng'])
    task = original.InvertibleQuadraticTask(61)
    config = {**saved['config'], 'resume_registration_sha256': x7.digest(ROOT / 'registration.json'),
        'resume_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'provenance': 'checkpoint-433000 + verified-diagnostic replay + fresh updates'}
    history = json.loads((SOURCE / 'validation.json').read_text())
    if history[-1]['step'] != 433000:
        raise ValueError('unexpected historical validation boundary')
    checkpoints = json.loads((SOURCE / 'checkpoints.json').read_text())
    best_row = min(history, key=lambda r: (-r['aggregate']['accuracy'], r['aggregate']['loss']))
    best = {**best_row['aggregate'], 'step': best_row['step'], 'checkpoint': best_row['checkpoint']}
    latest = checkpoints[str(CHECKPOINT)]
    disk_guard()
    x7.write_json(OUT / 'config.json', config)
    started, opt_seconds = time.monotonic(), history[-1]['optimizer_seconds_total']
    bounds = registration['resource_bounds']['muon3x']
    try:
        with (OUT / 'training.jsonl').open('a', buffering=1) as log:
            for step in range(433001, CEILING + 1):
                disk_guard()  # Includes per-update log writes, not only large writes.
                if time.monotonic() - started > bounds['wall_seconds']:
                    raise TimeoutError('registered infrastructure wall ceiling')
                before = time.monotonic()
                metrics = original.train_update(model, opt, 'muon3x', task.generate_batch(32, 66, rng), step)
                opt_seconds += metrics['optimizer_seconds']
                if step <= 433296:
                    compare_replay(step, metrics, reference)
                row = {'step': step, **metrics, 'update_seconds': time.monotonic() - before,
                    'elapsed_s': time.monotonic() - started,
                    'provenance': 'verified_diagnostic_replay' if step <= 433296 else 'resumed'}
                disk_guard()
                log.write(json.dumps(row) + '\n')
                if step == 433296:
                    x7.write_json(OUT / 'replay_verified.json', {'through_step': step, 'updates': 296,
                        'diagnostics': list(DIAGNOSTICS), 'bitwise_full_state_proven': False})
                if torch.cuda.max_memory_allocated() > bounds['peak_allocated_bytes']:
                    raise MemoryError('registered memory ceiling')
                if step % original.EVAL_EVERY and step != CEILING:
                    continue
                disk_guard()
                metrics = original.evaluate(model, pool)
                current = metrics['aggregate']
                is_best = (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
                cp = None
                if is_best or step % 1000 == 0 or step == CEILING:
                    disk_guard()
                    cp = original.checkpoint(OUT, model, opt, rng, step, config)
                    checkpoints[cp['path']] = cp
                    latest = cp
                if is_best:
                    best = {**current, 'step': step, 'checkpoint': cp}
                row = {'step': step, **metrics, 'checkpoint': cp, 'provenance': 'resumed',
                    'elapsed_s': time.monotonic() - started, 'optimizer_seconds_total': opt_seconds,
                    'optimizer_metrics': getattr(opt, 'last_metrics', {}),
                    'decision': 'safety_ceiling' if step == CEILING else 'continue'}
                history.append(row)
                disk_guard()
                x7.write_json(OUT / 'validation.json', history)
                disk_guard()
                x7.write_json(OUT / 'checkpoints.json', checkpoints)
                print(json.dumps(row), flush=True)
                # Never prune the immutable original files, only new resume checkpoints.
                keep = {best['checkpoint']['path'], latest['path']}
                for path in OUT.glob('step_*.pt'):
                    if str(path) not in keep:
                        path.unlink()
        disk_guard()
        x7.write_json(OUT / 'result.json', {'name': 'muon3x_seed0', 'status': 'complete',
            'total_steps': CEILING, 'stop_reason': 'safety_ceiling', 'best': best,
            'final': history[-1], 'finished_at': time.time(), 'provenance': config['provenance']})
    except BaseException as exc:
        # Never replace/delete a committed checkpoint on guard/ENOSPC/replay failure.
        # Emergency metadata is tiny; concurrent external exhaustion can still defeat writes.
        x7.write_json(OUT / 'interrupted.json', {'status': 'INFRA-INTERRUPTED',
            'error': repr(exc), 'resumable_checkpoint': latest, 'finished_at': time.time()})
        raise


def launch():
    x10.verify_registration(ROOT)
    disk_guard()
    receipt = ROOT / 'receipts/muon3x_seed0'
    receipt.mkdir(exist_ok=False)
    with (receipt / 'worker_console.txt').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', __spec__.name, 'worker'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'muon3x_seed0 worker={proc.pid} card=7', flush=True)


def worker():
    name = 'muon3x_seed0'
    if name not in NAMES:
        raise ValueError('unregistered run')
    receipt = ROOT / 'receipts' / name
    x7.write_json(receipt / 'worker.json', identity(os.getpid()))
    def interrupted(signum, _frame):
        raise SystemExit(f'worker received signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    code, child, error, leased = 1, None, None, False
    env = dict(os.environ)
    try:
        registration = x10.verify_registration(ROOT)
        gpu = registration['gpu_binding'][name]
        env.update(GPU_LEASE_VISIBLE=str(gpu), GPU_LEASE_PID=str(os.getpid()),
                   CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        with (receipt / 'lease.txt').open('w') as lease:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'],
                           env=env, stdout=lease, stderr=lease, check=True)
        leased = True
        cmd = [sys.executable, '-m', 'scripts.invertible_stage15_muon_resume', 'train']
        x7.write_json(receipt / 'command.json', {'argv': cmd, 'gpu': gpu})
        with (receipt / 'console.txt').open('w') as output:
            child = subprocess.Popen(cmd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            x7.write_json(receipt / 'trainer.json', identity(child.pid))
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
            result = json.loads((ROOT / name / 'result.json').read_text())
            if result['total_steps'] != (0 if name == 'gdn2_zeroshot' else CEILING) or result['status'] != 'complete':
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
        interrupted_path = ROOT / name / 'interrupted.json'
        terminal = {'name': name, 'exit_code': code, 'error': error, 'finished_at': time.time(),
                    'status': 'complete' if code == 0 else 'INFRA-INTERRUPTED',
                    'interruption': json.loads(interrupted_path.read_text()) if interrupted_path.exists() else
                        {'resumable_checkpoint': {'path': str(CHECKPOINT), 'sha256': SHA256, 'step': 433000}},
                    'result_sha256': x7.digest(ROOT / name / 'result.json') if code == 0 else None}
        x7.write_json(receipt / 'terminal.json', terminal)
        x7.write_json(ROOT / 'markers' / (name + ('.DONE' if code == 0 else '.FAIL')), terminal)
        if leased:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'release', str(gpu)], env=env,
                           check=True, stdout=subprocess.DEVNULL)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'launch', 'worker', 'train'])
    args = parser.parse_args()
    if args.action == 'worker':
        raise SystemExit(worker())
    globals()[args.action]()
