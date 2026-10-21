"""X16E: separately registered T64 extension from the terminal three-perfect T16 state."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from scripts import invertible_stage16_17_restart as prior
from scripts.invertible_stage10_control import identity
from scripts.watch_invertible_stage10_optimizers import process_alive

x16, x10, x7 = prior.x16, prior.x10, prior.x7
ROOT = x16.ROOT / 'curriculum_extension'
NAME = 'e97_extension_seed0'
SOURCE = prior.root_for('e97_curriculum_seed0') / 'e97_curriculum_seed0'
CHECKPOINT = SOURCE / 'T16/step_500000.pt'
CHECKPOINT_SHA256 = '870c2dcf5e863dcab817e3c6b0150d054e91cff8456d231378d3748676e06b1f'
CEILING, EVAL_EVERY, GPU = 500000, 500, 6
write_json, disk_guard = prior.write_json, prior.disk_guard


def learning_rate(step):
    if not 0 <= step <= CEILING:
        raise ValueError('extension update outside registered schedule')
    return 3e-6 + .5 * (3e-4 - 3e-6) * (1 + math.cos(math.pi * step / CEILING))


def register():
    base = prior.verify('e97_curriculum_seed0')
    if (ROOT / 'registration.json').exists():
        raise FileExistsError('immutable extension registration exists')
    if x7.digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError('terminal teach checkpoint changed')
    result = prior.read_json(SOURCE / 'result.json')
    terminal = prior.read_json(SOURCE.parent / 'receipts/e97_curriculum_seed0/terminal.json')
    marker = prior.read_json(SOURCE.parent / 'markers/e97_curriculum_seed0.DONE')
    if not (terminal == marker and terminal['exit_code'] == 0
            and terminal['result_sha256'] == x7.digest(SOURCE / 'result.json')
            and result['status'] == 'criterion_not_reached' and result['total_steps'] == 500000
            and result['t64_steps'] == 0 and result['final']['checkpoint']['sha256'] == CHECKPOINT_SHA256):
        raise ValueError('invalid terminal teaching evidence')
    paths = [Path(__file__).relative_to(Path.cwd()), Path('tests/test_invertible_stage16_extension.py'),
             ROOT / 'AMENDMENT.md', ROOT / 'tests_prelaunch.txt', ROOT / 'tests_gpu_prelaunch.txt',
             SOURCE.parent / 'registration.json',
             x16.ROOT / 'registration.json']
    artifacts = [SOURCE / 'result.json', SOURCE / 'restart_admitted.json',
                 SOURCE.parent / 'receipts/e97_curriculum_seed0/terminal.json',
                 SOURCE.parent / 'markers/e97_curriculum_seed0.DONE',
                 x16.ROOT / 'infra_resume/e97_curriculum_seed0/interrupted.json']
    write_json(ROOT / 'registration.json', {**base, 'task': 'X16E curriculum extension',
        'registered_utc': datetime.now(timezone.utc).isoformat(), 'authorized_runs': [NAME],
        'gpu_binding': {NAME: GPU}, 'source_checkpoint': {'path': str(CHECKPOINT), 'sha256': CHECKPOINT_SHA256},
        'teach_counts': [p['correct'] for p in result['final']['positions']],
        'source_sha256': {**base['source_sha256'], **{str(p): x7.digest(p) for p in paths}},
        'artifact_sha256': {str(p): x7.digest(p) for p in artifacts},
        'recipe': 'Terminal three-perfect E97 seed0; preserve AdamW/RNG; batch32, fp32 params/bf16 autocast, legacy recurrence/sequential Triton eval, clip1, betas(.9,.999),eps1e-8, all-parameter wd.01; fresh warmup-free cosine3e-4 to3e-6 over500K T64-only updates; frozen512 eval every500.',
        'curriculum': 'This amendment governs X16E only; original criterion-gated X16 remains unchanged and never reached T64.',
        'extension': {'budget': CEILING, 'peak_lr': 3e-4, 'floor_lr': 3e-6,
            'schedule': 'fresh warmup-free cosine; phase_step0 is T64 baseline, updates1..500000 T64 only',
            'state': 'preserve terminal model, AdamW moments, NumPy/torch/CUDA RNG; no optimizer reset',
            'loss': 'uniform full P1..P64 CE; no T16 replay or mixing',
            'retention': 'independent frozen T16 every validation including step0; all positions exact counts',
            'checkpoints': 'extension entry+best+latest <=3 plus atomic staging; source immutable'},
        'predictions': [
            '(a) TRANSFER: any T64 P2+ >=50/512 with P1-P3 retention (>=508/512 on T16) beats cold-start zero. Separate step0 runtime transfer from new acquisition beyond step0 material positions; only new acquisition supports learning transfer.',
            '(b) NO-TRANSFER: T64 columns remain zero/chance while T16 retention holds: horizon-specific ignition barrier; partial curriculum fix. Report exact counts and any transferred-but-not-new positions rather than forcing a binary outcome.',
            '(c) RETENTION DAMAGE: taught T16 positions decay under T64 training; report all exact trajectories, drops from baseline, first below508, transient and final damage.'],
        'interpretation': 'Original criterion not reached; new authorized arm, NOT original gated extension. Three-perfect taught state. Task P4=8 differs from hash-bound final P4=5. Restart divergence at252204 may have damaged budding P4 OR may be irrelevant; neither causal reading claimed.',
        'censoring': 'Fixed-final500K primary; transient crossings descriptive; missing crossings right-censored at observed phase_step. Infrastructure/disk/wall/memory termination censored, not architecture failure. No automatic retries.',
        'scope': 'single-host standalone expressivity, no Frontier/Slurm/resilient DiLoCo claim'})


def verify():
    reg = x10.verify_registration(ROOT)
    if reg['authorized_runs'] != [NAME] or reg['gpu_binding'] != {NAME: GPU}:
        raise ValueError('unauthorized extension/card')
    for path, sha in reg['artifact_sha256'].items():
        if x7.digest(path) != sha:
            raise ValueError('changed teaching artifact: ' + path)
    if reg['source_checkpoint'] != {'path': str(CHECKPOINT), 'sha256': CHECKPOINT_SHA256} or x7.digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError('changed terminal checkpoint')
    return reg


def train():
    import torch
    reg = verify()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(GPU):
        raise ValueError('wrong visible card')
    disk_guard()
    out = ROOT / NAME
    out.mkdir(exist_ok=False)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if saved['step'] != 500000 or saved['config']['depth'] != 16 or saved['optimizer_family'] != 'adamw':
        raise ValueError('wrong terminal teaching state')
    pools = {d: x7.frozen_pools(0, d, 61)['val'] for d in (16, 64)}
    for d, pool in pools.items():
        if [hashlib.sha256(a.tobytes()).hexdigest() for a in pool] != reg['pool_array_sha256'][str(d)]:
            raise ValueError('changed frozen pool')
    model = x16.x14.build_model('e97').cuda()
    model.load_state_dict(saved['model'], strict=True)
    opt = x16.x14.make_optimizer(model)
    opt.load_state_dict(saved['optimizer'])
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for actual, expected in ((model.state_dict(), saved['model']), (opt.state_dict(), saved['optimizer']),
            (rng.bit_generator.state, saved['numpy_rng']), (torch.get_rng_state(), saved['torch_rng']),
            (torch.cuda.get_rng_state_all(), saved['cuda_rng'])):
        prior.assert_state_equal(actual, expected)
    config = {'name': NAME, 'arm': 'adamw', 'depth': 64, 'seed': 0,
        'registration_sha256': x7.digest(ROOT / 'registration.json'),
        'source_checkpoint': reg['source_checkpoint'], 'preserved_state_exactly_compared': True,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    write_json(out / 'config.json', config)
    history, checkpoints, best, entry, latest = [], {}, None, None, None
    started, opt_seconds = time.monotonic(), 0.
    task = x10.InvertibleQuadraticTask(61)
    disk_guard()
    (out / 'T64').mkdir()
    try:
        with (out / 'training.jsonl').open('x', buffering=1) as log:
            for step in range(CEILING + 1):
                disk_guard()
                if time.monotonic() - started > reg['resource_bounds']['wall_seconds']:
                    raise TimeoutError('registered wall ceiling; censored')
                if step:
                    before = time.monotonic()
                    metrics = x16.train_update(model, opt, task.generate_batch(32, 66, rng), step,
                                               weight=None, lr=learning_rate(step))
                    opt_seconds += metrics['optimizer_seconds']
                    disk_guard()
                    log.write(json.dumps({'step': step, 'phase_step': step, 'depth': 64,
                        'teach_step': 500000, **metrics, 'update_seconds': time.monotonic()-before,
                        'elapsed_s': time.monotonic()-started}) + '\n')
                if torch.cuda.max_memory_allocated() > reg['resource_bounds']['peak_allocated_bytes']:
                    raise MemoryError('registered memory ceiling; censored')
                if step % EVAL_EVERY and step != CEILING:
                    continue
                metrics = x16.evaluate(model, pools[64])
                retention = x16.evaluate(model, pools[16])
                if torch.cuda.max_memory_allocated() > reg['resource_bounds']['peak_allocated_bytes']:
                    raise MemoryError('registered evaluation memory ceiling; censored')
                if step == 0 and [p['correct'] for p in retention['positions']] != reg['teach_counts']:
                    raise ValueError('restored T16 retention differs from hash-bound terminal validation')
                current = metrics['aggregate']
                is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
                cp = None
                if is_best or step % 1000 == 0 or step == CEILING:
                    disk_guard()
                    cp = x10.checkpoint(out / 'T64', model, opt, rng, step, {**config, 'phase_step': step})
                    checkpoints[cp['path']] = cp
                    latest = cp['path']
                if entry is None:
                    entry = cp['path']
                if is_best:
                    best = {**current, 'step': step, 'checkpoint': cp}
                row = {'step': step, 'phase_step': step, 'depth': 64, **metrics,
                    'retention_t16': retention, 'checkpoint': cp, 'lr': learning_rate(step),
                    'elapsed_s': time.monotonic()-started, 'optimizer_seconds_total': opt_seconds,
                    'decision': 'safety_ceiling' if step == CEILING else 'continue'}
                history.append(row)
                write_json(out / 'validation.json', history)
                write_json(out / 'checkpoints.json', checkpoints)
                prior.prune_new(out, checkpoints, {entry, latest, best['checkpoint']['path']})
                print(json.dumps({'name': NAME, **row}), flush=True)
        write_json(out / 'result.json', {'name': NAME, 'status': 'complete', 'total_steps': CEILING,
            't64_steps': CEILING, 'teach_steps': 500000, 'stop_reason': 'safety_ceiling',
            'best': best, 'final': history[-1], 'finished_at': time.time()})
    except BaseException as exc:
        x7.write_json(out / 'interrupted.json', {'status': 'INFRA-INTERRUPTED', 'error': repr(exc),
            'latest_checkpoint': checkpoints.get(latest), 'finished_at': time.time()})
        raise


def valid_result(result):
    final = result.get('final', {})
    return (result.get('name') == NAME and result.get('status') == 'complete'
        and result.get('total_steps') == CEILING and result.get('t64_steps') == CEILING
        and result.get('stop_reason') == 'safety_ceiling' and final.get('phase_step') == CEILING
        and final.get('depth') == 64 and len(final.get('positions', [])) == 64
        and len(final.get('retention_t16', {}).get('positions', [])) == 16)


def worker():
    receipt = ROOT / 'receipts' / NAME
    def interrupted(signum, _frame):
        raise SystemExit(f'worker received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    child, leased, code, error = None, False, 1, None
    env = dict(os.environ, GPU_LEASE_VISIBLE=str(GPU), GPU_LEASE_PID=str(os.getpid()),
               CUDA_VISIBLE_DEVICES=str(GPU), PYTHONUNBUFFERED='1')
    try:
        verify()
        write_json(receipt / 'worker.json', identity(os.getpid()))
        with (receipt / 'lease.txt').open('x') as log:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'],
                           env=env, stdout=log, stderr=log, check=True)
        leased = True
        cmd = [sys.executable, '-m', 'scripts.invertible_stage16_extension', 'train']
        write_json(receipt / 'command.json', {'argv': cmd, 'gpu': GPU})
        with (receipt / 'console.txt').open('x') as log:
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(receipt / 'trainer.json', identity(child.pid))
            deadline = time.monotonic() + 14*86400 + 120
            while True:
                try:
                    code = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    subprocess.run(['bash', 'scripts/gpu_lease.sh', 'heartbeat', str(GPU)],
                                   env=env, check=True, stdout=subprocess.DEVNULL)
                    if time.monotonic() > deadline:
                        raise TimeoutError('external wall ceiling')
        if code == 0 and not valid_result(prior.read_json(ROOT / NAME / 'result.json')):
            raise ValueError('invalid extension result')
    except BaseException as exc:
        code, error = 1, repr(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
        terminal = {'name': NAME, 'exit_code': code, 'error': error, 'finished_at': time.time(),
            'status': 'complete' if code == 0 else 'INFRA-INTERRUPTED',
            'result_sha256': x7.digest(ROOT / NAME / 'result.json') if code == 0 else None}
        try:
            x7.write_json(receipt / 'terminal.json', terminal)
            x7.write_json(ROOT / 'markers' / (NAME + ('.DONE' if code == 0 else '.FAIL')), terminal)
        finally:
            if leased:
                subprocess.run(['bash', 'scripts/gpu_lease.sh', 'release', str(GPU)], env=env,
                               check=True, stdout=subprocess.DEVNULL)
    return code


def launch():
    verify()
    disk_guard()
    receipt = ROOT / 'receipts' / NAME
    if (ROOT / NAME).exists():
        raise FileExistsError('no retry/restart')
    receipt.mkdir(parents=True, exist_ok=False)
    with (receipt / 'worker_console.txt').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage16_extension', 'worker'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'{NAME}: worker={proc.pid}, card={GPU}', flush=True)


def status():
    receipt = ROOT / 'receipts' / NAME
    if (ROOT / 'markers' / f'{NAME}.FAIL').exists():
        return 1, 'INFRA-INTERRUPTED; censored, inspect terminal receipt'
    if (ROOT / 'markers' / f'{NAME}.DONE').exists():
        try:
            terminal = prior.read_json(receipt / 'terminal.json')
            valid = (terminal == prior.read_json(ROOT / 'markers' / f'{NAME}.DONE')
                and terminal['name'] == NAME and terminal['exit_code'] == 0
                and terminal['result_sha256'] == x7.digest(ROOT / NAME / 'result.json')
                and valid_result(prior.read_json(ROOT / NAME / 'result.json')))
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        return (0, 'verified fixed-final500K') if valid else (1, 'invalid terminal receipt')
    for role in ('worker', 'trainer'):
        path = receipt / f'{role}.json'
        if not path.exists():
            if role == 'worker':
                return 1, 'missing worker; censored'
            continue
        if not process_alive(prior.read_json(path)):
            time.sleep(2)
            if any((ROOT / 'markers' / f'{NAME}.{s}').exists() for s in ('DONE', 'FAIL')):
                return status()
            return 1, f'dead {role}; censored'
    return None, 'RUNNING; fixed-final pending'


def growth_report():
    path = ROOT / NAME / 'validation.json'
    rows = prior.read_json(path) if path.exists() else []
    lines = ['# X16E T64 extension and T16 retention', '', status()[1],
        'Fixed-final500K primary; transients descriptive; absent crossings right-censored at latest observed phase_step.',
        'Task P4=8; hash-bound teaching final P4=5. Both restart-divergence readings remain unproven.', '']
    if rows:
        baseline = {p['driver_position'] for p in rows[0]['positions'] if p['correct'] >= 50}
        seen = {p['driver_position'] for r in rows for p in r['positions'] if p['correct'] >= 50}
        final = {p['driver_position'] for p in rows[-1]['positions'] if p['correct'] >= 50}
        lines += [f'Observed through extension update {rows[-1]["phase_step"]}.',
            f'T64 step0 material={sorted(baseline)}; new transient={sorted(seen-baseline)}; new latest={sorted(final-baseline)}.',
            '|Position|T64 baseline|Latest|Max|First50|First508|First512|', '|---|---|---|---|---|---|---|']
        for j in range(64):
            counts = [r['positions'][j]['correct'] for r in rows]
            first = [next((r['phase_step'] for r in rows if r['positions'][j]['correct'] >= t), None) for t in (50,508,512)]
            lines.append(f'|P{j+1}|{counts[0]}|{counts[-1]}|{max(counts)}|{first[0]}|{first[1]}|{first[2]}|')
        lines += ['', '## retention_t16 (exact full trajectories in validation.json)',
            '|Position|Baseline|Latest|Min|Latest drop|First below508|', '|---|---|---|---|---|---|']
        for j in range(16):
            counts = [r['retention_t16']['positions'][j]['correct'] for r in rows]
            first = next((r['phase_step'] for r,c in zip(rows,counts) if c < 508), None)
            lines.append(f'|P{j+1}|{counts[0]}|{counts[-1]}|{min(counts)}|{counts[0]-counts[-1]}|{first}|')
    prior.write_text(ROOT / 'GROWTH_REPORT.md.tmp', '\n'.join(lines)+'\n')
    (ROOT / 'GROWTH_REPORT.md.tmp').replace(ROOT / 'GROWTH_REPORT.md')


def watch(interval, once=False):
    if interval <= 0:
        raise ValueError('positive interval required')
    while True:
        code, message = status()
        row = {'time': time.time(), 'exit_code': code, 'message': message}
        disk_guard()
        with (ROOT / 'receipts/watcher/polls.jsonl').open('a') as log:
            log.write(json.dumps(row)+'\n')
        growth_report()
        print(json.dumps(row), flush=True)
        if code is not None:
            write_json(ROOT / 'receipts/watcher/terminal.json', row)
            return code
        if once:
            return 2
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'launch', 'worker', 'train', 'watch'])
    parser.add_argument('--interval', type=float, default=900)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.action == 'watch':
        raise SystemExit(watch(args.interval, args.once))
    result = globals()[args.action]()
    if args.action == 'worker':
        raise SystemExit(result)
