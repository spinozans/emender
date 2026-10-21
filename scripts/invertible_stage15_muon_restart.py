#!/usr/bin/env python3
"""X15b amendment-2: restart from verified checkpoint, no at-death replay claim."""
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
from scripts.invertible_stage15_muon_resume import (
    disk_guard, load_checkpoint, assert_state_equal,
)

ROOT = original.ROOT / 'muon_restart'
OUT = ROOT / 'muon3x_seed0'
SOURCE = original.ROOT / 'muon3x_seed0'
CHECKPOINT = SOURCE / 'step_433000.pt'
SHA256 = '373a7ece3a97790e81d9fa3d0888afdedd74e8eaf8e9b68bbc834c8cf2e882c0'
CEILING = original.CEILING
NAMES = ('muon3x_seed0',)
x7, x10 = original.x7, original.x10
DISK_MIN_BYTES = 200 * 1024**3  # Conservative interpretation of 200G: 200 GiB.
PROVENANCE = ('restart-from-checkpoint; historical 433,001-433,296 superseded '
              'by re-execution from checkpoint state')


def register():
    base = original.verify_registration(original.ROOT)
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'registration.json').exists():
        raise FileExistsError('never overwrite resume registration')
    saved = load_checkpoint()
    manifest = json.loads((SOURCE / 'checkpoints.json').read_text())
    if manifest[str(CHECKPOINT)]['sha256'] != SHA256 or manifest[str(CHECKPOINT)]['step'] != 433000:
        raise ValueError('original checkpoint manifest mismatch')
    receipt = {'checkpoint': str(CHECKPOINT), 'sha256': SHA256, 'step': saved['step'],
        'optimizer_state_entries': {k: len(saved['optimizer'][k]['state']) for k in ('muon', 'adamw')},
        'rng_keys_verified': ['numpy_rng', 'torch_rng', 'cuda_rng'],
        'fresh_updates': 567000, 'proof_scope': 'exact checkpoint restoration only; no at-death state claim'}
    x7.write_json(ROOT / 'receipts/checkpoint_preflight.json', receipt)
    sources = dict(base['source_sha256'])
    paths = [Path(__file__).relative_to(Path.cwd()), Path('scripts/watch_invertible_stage15_muon_restart.py'),
        Path('tests/test_invertible_stage15_muon_restart.py'), ROOT / 'AMENDMENT-2-2026-10-21.md',
        ROOT / 'AMENDMENT-3-2026-10-21.md',
        ROOT / 'registration-supersession.json', ROOT / 'receipts/checkpoint_preflight.json',
        ROOT / 'receipts/tests_prelaunch.txt', original.ROOT / 'registration.json',
        Path('scripts/invertible_stage15_muon_resume.py'),
        original.ROOT / 'muon_resume/registration.json',
        original.ROOT / 'muon_resume/AMENDMENT-2026-10-21.md',
        original.ROOT / 'muon_resume/receipts/muon3x_seed0/terminal.json',
        original.ROOT / 'muon_resume/receipts/muon3x_seed0/console.txt',
        original.ROOT / 'muon_resume/muon3x_seed0/interrupted.json',
        original.ROOT / 'muon_resume/LAUNCH-INTERRUPTION-2026-10-21.md']
    artifacts = [SOURCE / name for name in ('training.jsonl', 'validation.json', 'checkpoints.json',
        'config.json', 'frozen_validation.npz')]
    sources.update({str(p): x7.digest(p) for p in paths})
    registration = {**base, 'task': 'X15b amendment-2 restart from checkpoint',
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'authorized_runs': list(NAMES), 'gpu_binding': {'muon3x_seed0': 7}, 'source_sha256': sources,
        'resume': receipt, 'disk_min_bytes': DISK_MIN_BYTES,
        'artifact_sha256': {str(p): x7.digest(p) for p in artifacts},
        'original_registration_sha256': x7.digest(original.ROOT / 'registration.json'),
        'protocol': (ROOT / 'AMENDMENT-2-2026-10-21.md').read_text()}
    x7.write_json(ROOT / 'registration.json', registration)


def prune_checkpoints(keep, checkpoints):
    """Durably receipt verified bytes before unlink; a receipt is not unlink success."""
    for path in sorted(OUT.glob('step_*.pt')):
        if str(path) in keep:
            continue
        disk_guard()
        checkpoint = checkpoints[str(path)]
        sha256 = x7.digest(path)
        if checkpoint['path'] != str(path) or sha256 != checkpoint['sha256']:
            raise ValueError('refuse to prune unverified checkpoint: ' + str(path))
        receipt = {'path': str(path), 'sha256': sha256, 'step': checkpoint['step'],
            'timestamp': datetime.now(timezone.utc).isoformat(), 'event': 'pre_unlink'}
        disk_guard()
        with (OUT / 'pruning.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        path.unlink()


def train():
    import torch
    registration = x10.verify_registration(ROOT)
    for path, sha in registration['artifact_sha256'].items():
        if x7.digest(path) != sha:
            raise ValueError('changed original artifact: ' + path)
    disk_guard()
    saved = load_checkpoint()
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
        'provenance': PROVENANCE, 'historical_training_superseded_after_step': 433000}
    history = json.loads((SOURCE / 'validation.json').read_text())
    if history[-1]['step'] != 433000:
        raise ValueError('unexpected historical validation boundary')
    checkpoints = json.loads((SOURCE / 'checkpoints.json').read_text())
    best_row = min(history, key=lambda r: (-r['aggregate']['accuracy'], r['aggregate']['loss']))
    best = {**best_row['aggregate'], 'step': best_row['step'], 'checkpoint': best_row['checkpoint']}
    latest = checkpoints[str(CHECKPOINT)]
    disk_guard()
    x7.write_json(OUT / 'config.json', config)
    disk_guard()
    x7.write_json(OUT / 'checkpoint_restored.json', {'step': 433000, 'sha256': SHA256,
        'model_optimizer_rng_exact': True, 'at_death_restoration_claimed': False,
        'fresh_updates_to_execute': 567000, 'provenance': PROVENANCE})
    started, opt_seconds = time.monotonic(), history[-1]['optimizer_seconds_total']
    bounds = registration['resource_bounds']['muon3x']
    try:
        with (OUT / 'training.jsonl').open('a', buffering=1) as log:
            disk_guard()
            log.write(json.dumps({'step': 433000, 'event': 'restart_from_checkpoint',
                'checkpoint_sha256': SHA256, 'provenance': PROVENANCE}) + '\n')
            for step in range(433001, CEILING + 1):
                disk_guard()  # Includes per-update log writes, not only large writes.
                if time.monotonic() - started > bounds['wall_seconds']:
                    raise TimeoutError('registered infrastructure wall ceiling')
                before = time.monotonic()
                metrics = original.train_update(model, opt, 'muon3x', task.generate_batch(32, 66, rng), step)
                opt_seconds += metrics['optimizer_seconds']
                row = {'step': step, **metrics, 'update_seconds': time.monotonic() - before,
                    'elapsed_s': time.monotonic() - started,
                    'provenance': PROVENANCE}
                disk_guard()
                log.write(json.dumps(row) + '\n')
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
                row = {'step': step, **metrics, 'checkpoint': cp, 'provenance': PROVENANCE,
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
                prune_checkpoints(keep, checkpoints)
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
        cmd = [sys.executable, '-m', 'scripts.invertible_stage15_muon_restart', 'train']
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
