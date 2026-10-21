#!/usr/bin/env python3
"""Attended X16R/X17R infrastructure recovery; never mutate the original evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from experiments.expressivity_tasks import run_invertible_stage16_ignition_8m as x16
from experiments.expressivity_tasks import run_invertible_stage17_carrywean_8m as x17
from scripts.invertible_stage10_control import identity

x10, x7 = x16.x10, x16.x7
BINDINGS = {**{n: a[2] for n, a in x16.ARMS.items()}, x17.NAMES[0]: 4}
DISK_MIN_BYTES = 200 * 1024**3
SCOPE = 'Exact replayed diagnostics are evidence, not proof of bitwise full-state trajectory identity.'
DIAGNOSTICS = ('loss', 'grad_norm', 'lr', 'full_update_rms', 'muon_update_rms', 'carry_probability')


class DiskGuardError(RuntimeError):
    pass


def disk_guard():
    free = shutil.disk_usage('/').free
    if free < DISK_MIN_BYTES:
        raise DiskGuardError(f'root free {free} < {DISK_MIN_BYTES}; retain last committed checkpoint')
    return free


def write_json(path, data):
    disk_guard()
    x7.write_json(path, data)


def write_text(path, text):
    disk_guard()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def module_for(name):
    return x16 if name in x16.ARMS else x17


def root_for(name):
    return module_for(name).ROOT / 'infra_resume'


def read_json(path):
    return json.loads(path.read_text())


def checkpoint_preflight(source):
    """Pruning receipts must explain every absent registered checkpoint."""
    import torch
    checkpoints = read_json(source / 'checkpoints.json')
    removed = {}
    retained = []
    for line in (source / 'pruning.jsonl').read_text().splitlines():
        receipt = json.loads(line)
        removed.update({c['path']: c for c in receipt['removed']})
        retained = receipt['retained']
    live = []
    for path, cp in checkpoints.items():
        if cp['path'] != path:
            raise ValueError('checkpoint key/path mismatch')
        if Path(path).exists():
            if x7.digest(path) != cp['sha256']:
                raise ValueError('checkpoint hash mismatch: ' + path)
            live.append(cp)
        elif removed.get(path) != cp:
            raise ValueError('checkpoint missing without matching pruning receipt: ' + path)
    if {c['path'] for c in live} != {c['path'] for c in retained}:
        raise ValueError('live checkpoints disagree with final pruning receipt')
    cp = max(live, key=lambda c: c['step'])
    saved = torch.load(cp['path'], map_location='cpu', weights_only=False)
    if saved['step'] != cp['step'] or saved['optimizer_family'] != 'adamw':
        raise ValueError('wrong resume checkpoint')
    if not saved['optimizer']['state'] or not saved['optimizer']['param_groups']:
        raise ValueError('missing AdamW state')
    if not saved['numpy_rng'] or not saved['torch_rng'].numel() or len(saved['cuda_rng']) != 1:
        raise ValueError('missing RNG state')
    return cp, saved


def replay_reference(path, checkpoint_step):
    """Only full rows count as completed updates; a final prefix is diagnostic only."""
    rows, prefix, last_full, partial_step = {}, None, None, None
    with path.open() as stream:
        for line in stream:
            if prefix is not None:
                raise ValueError('truncated row is not final')
            try:
                row = json.loads(line)
                if last_full is not None and row['step'] != last_full + 1:
                    raise ValueError('nonconsecutive full training rows')
                last_full = row['step']
                if row['step'] > checkpoint_step:
                    rows[row['step']] = {k: row[k] for k in DIAGNOSTICS if k in row}
                    if not all(k in row for k in ('loss', 'grad_norm', 'lr')):
                        raise ValueError('missing full-row diagnostics')
            except json.JSONDecodeError:
                match = re.match(r'\{"step": (\d+),', line)
                if not match:
                    raise ValueError('unreadable final step')
                partial_step = int(match[1])
                prefix = {k: json.loads(v) for k, v in re.findall(
                    r'"([a-z_]+)": (null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=,)', line)
                    if k in DIAGNOSTICS}
    if last_full is None or set(rows) != set(range(checkpoint_step + 1, last_full + 1)):
        raise ValueError('missing/duplicate replay updates')
    if partial_step is not None and partial_step != last_full + 1:
        raise ValueError('unexpected truncated update')
    return rows, {'last_full_step': last_full, 'partial_step': partial_step,
                  'prefix_diagnostics': prefix, 'completed_replay_updates': len(rows)}


def compare_replay(step, metrics, expected):
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f'replay mismatch step={step} {key}: {metrics.get(key)} != {value}')


def assert_state_equal(actual, expected):
    import torch
    if isinstance(expected, torch.Tensor):
        if actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError('restored tensor mismatch')
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError('restored keys mismatch')
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise ValueError('restored sequence mismatch')
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    elif actual != expected:
        raise ValueError('restored value mismatch')


def register(stage):
    mod = x16 if stage == 16 else x17
    base = x10.verify_registration(mod.ROOT)
    root = mod.ROOT / 'infra_resume'
    disk_guard()
    root.mkdir(exist_ok=False)
    resumes, artifacts, sources = {}, {}, dict(base['source_sha256'])
    for name in base['authorized_runs']:
        source = mod.ROOT / name
        cp, saved = checkpoint_preflight(source)
        _, replay = replay_reference(source / 'training.jsonl', cp['step'])
        if stage == 16 and (saved['config']['depth'] != 16 or (source / 'transition.json').exists()):
            raise ValueError('this attended recovery expects pre-transition T16 checkpoint')
        resume = {**replay, 'checkpoint': cp, 'gpu': BINDINGS[name],
                  'optimizer_state_entries': len(saved['optimizer']['state']),
                  'rng_keys_verified': ['numpy_rng', 'torch_rng', 'cuda_rng'],
                  'original_ceiling': mod.CEILING, 'proof_scope': SCOPE}
        resumes[name] = resume
        amendment = root / f'AMENDMENT-2026-10-21-{name}.md'
        supersession = root / f'registration-supersession-{name}.json'
        write_text(amendment, f'''# {name}: 2026-10-21 attended infrastructure amendment
Original registration and interrupted run artifacts remain immutable.
The approximately 21:00 UTC ENOSPC window is an INFRA-INTERRUPTION, not architecture failure.
Resume checkpoint: {cp['path']} at {cp['step']}, SHA256 {cp['sha256']}.
Last FULL training row: {replay['last_full_step']}; partial death row: {replay['partial_step']}.
Restore model, AdamW moments, NumPy, torch and CUDA RNG exactly; replay all full rows,
then check only readable diagnostic fields of the partial row. Abort on ANY diagnostic mismatch.
{SCOPE}
Continue ORIGINAL global LR/carry schedules to {mod.CEILING}; no warmup or wean reset.
X16 curriculum retains its original criterion, live moments/RNG and 500K T64 extension,
with the original attained-LR extension cosine; no T64 launch without the criterion.
Original evaluation, stopping, predictions, memory/wall limits and checkpoint selection apply.
New checkpoints alone are pruned with deletion hashes; original retained evidence is never deleted.
Card {BINDINGS[name]} only, atomic no-wait gpu_lease; no automatic restart or lease retry.
Before every normal runner write, require root-free >=200 GiB. On guard failure retain the
last hash-receipted checkpoint, abort nonzero, and attempt tiny emergency interruption/terminal
metadata (the sole below-floor write exception). Atomic checkpoint staging remains required.
This pre-write guard cannot guarantee writes against simultaneous external disk exhaustion.
All zero-copy/full-state/DiLoCo/Frontier/scale/overlap conformance claims remain unclaimed.
''')
        write_json(supersession, {'name': name, 'original_registration': str(mod.ROOT / 'registration.json'),
            'original_registration_sha256': x7.digest(mod.ROOT / 'registration.json'),
            'amendment': str(amendment), 'amendment_sha256': x7.digest(amendment),
            'supersedes': 'interrupted execution only, not original scientific registration', **resume})
        sources.update({str(p): x7.digest(p) for p in (amendment, supersession)})
        for filename in ('training.jsonl', 'validation.json', 'checkpoints.json', 'pruning.jsonl',
                         'config.json', 'frozen_validation.npz'):
            p = source / filename
            artifacts[str(p)] = x7.digest(p)
        ledger = source / 'INFRA-INTERRUPTION-2026-10-21.md'
        write_text(ledger, f'''# INFRA-INTERRUPTION
Window: 2026-10-21 approximately 21:00 UTC (root ENOSPC, external churn).
Last full/death-prefix step: {replay['last_full_step']} / {replay['partial_step']}.
Resume checkpoint: {cp['path']} SHA256 {cp['sha256']}.
Replay verification: pending; authoritative result will be {root / name / 'replay_verified.json'}.
{SCOPE}
Admission amendment: {amendment}; original schedule/ceiling unchanged.
''')
        sources[str(ledger)] = x7.digest(ledger)
    paths = [Path(__file__).relative_to(Path.cwd()), Path('tests/test_invertible_stage16_17_resume.py'),
             mod.ROOT / 'registration.json', mod.ROOT / 'receipts/infra_recovery_tests_prelaunch.txt']
    sources.update({str(p): x7.digest(p) for p in paths})
    write_json(root / 'registration.json', {**base, 'task': f'X{stage}R infrastructure recovery',
        'registered_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sources,
        'artifact_sha256': artifacts, 'resume': resumes, 'disk_min_bytes': DISK_MIN_BYTES,
        'gpu_binding': {n: BINDINGS[n] for n in resumes}, 'proof_scope': SCOPE})


def verify(name):
    root = root_for(name)
    registration = x10.verify_registration(root)
    if name not in registration['authorized_runs'] or registration['gpu_binding'][name] != BINDINGS[name]:
        raise ValueError('unauthorized run/card')
    for path, sha in registration['artifact_sha256'].items():
        if x7.digest(path) != sha:
            raise ValueError('changed original artifact: ' + path)
    cp = registration['resume'][name]['checkpoint']
    if x7.digest(cp['path']) != cp['sha256']:
        raise ValueError('changed checkpoint')
    return registration


def prune_new(out, checkpoints, keep):
    removed = []
    for path, cp in checkpoints.items():
        if Path(path).is_relative_to(out) and path not in keep and Path(path).exists():
            disk_guard()
            if x7.digest(path) != cp['sha256']:
                raise ValueError('refuse to prune unverified checkpoint')
            Path(path).unlink()
            removed.append(cp)
    disk_guard()
    with (out / 'pruning.jsonl').open('a') as stream:
        stream.write(json.dumps({'removed': removed, 'retained': [checkpoints[p] for p in sorted(keep)]}) + '\n')


def train(name):
    import torch
    mod, root = module_for(name), root_for(name)
    registration = verify(name)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(BINDINGS[name]):
        raise ValueError('wrong visible card')
    resume = registration['resume'][name]
    source, out = mod.ROOT / name, root / name
    disk_guard()
    out.mkdir(exist_ok=False)
    saved = torch.load(resume['checkpoint']['path'], map_location='cpu', weights_only=False)
    reference, summary = replay_reference(source / 'training.jsonl', saved['step'])
    if summary != {k: resume[k] for k in summary}:
        raise ValueError('replay reference changed')
    is16, curriculum = mod is x16, name == 'e97_curriculum_seed0'
    architecture, weight = x16.ARMS[name][:2] if is16 else ('e97', None)
    depths = (16, 64) if curriculum else ((16,) if is16 else (64,))
    pools = {d: x7.frozen_pools(0, d, 61)['val'] for d in depths}
    for d, pool in pools.items():
        key = str(d) if is16 else '0'
        if [hashlib.sha256(a.tobytes()).hexdigest() for a in pool] != registration['pool_array_sha256'][key]:
            raise ValueError('frozen pool changed')
    model = (x16.x14.build_model(architecture) if is16 else x17.build_model()).cuda()
    model.load_state_dict(saved['model'], strict=True)
    opt = x16.x14.make_optimizer(model) if is16 else x17.make_optimizer(model)
    opt.load_state_dict(saved['optimizer'])
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for actual, expected in ((model.state_dict(), saved['model']), (opt.state_dict(), saved['optimizer']),
            (rng.bit_generator.state, saved['numpy_rng']), (torch.get_rng_state(), saved['torch_rng']),
            (torch.cuda.get_rng_state_all(), saved['cuda_rng'])):
        assert_state_equal(actual, expected)
    config = {**saved['config'], 'resume_registration_sha256': x7.digest(root / 'registration.json'),
        'resume_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'proof_scope': SCOPE}
    history = [r for r in read_json(source / 'validation.json') if r['step'] <= saved['step']]
    checkpoints = read_json(source / 'checkpoints.json')
    best_row = min(history, key=lambda r: (-r['aggregate']['accuracy'], r['aggregate']['loss']))
    best = {**best_row['aggregate'], 'step': best_row['step'], 'checkpoint': best_row['checkpoint']}
    entry = history[0]['checkpoint']['path']
    latest = resume['checkpoint']['path']
    transition = transition_cp = None
    total, started = saved['step'], time.monotonic()
    opt_seconds = history[-1].get('optimizer_seconds_total', 0.)
    bounds = registration.get('resource_bounds')
    task = x10.InvertibleQuadraticTask(61)
    write_json(out / 'config.json', config)
    write_json(out / 'validation.json', history)
    write_json(out / 'checkpoints.json', checkpoints)
    write_text(out / 'INFRA-INTERRUPTION.md', f"# INFRA-INTERRUPTION\n2026-10-21 ~21:00 UTC ENOSPC.\n{json.dumps(resume, indent=2)}\nReplay pending. {SCOPE}\n")
    try:
        disk_guard()
        with (out / 'training.jsonl').open('x', buffering=1) as log:
            for depth in depths:
                start = saved['config']['phase_step'] + 1 if is16 and depth == 16 else (saved['step'] + 1 if not is16 else 0)
                if is16 and depth == 64:
                    best = None
                checkpoint_dir = out / f'T{depth}' if is16 else out
                disk_guard()
                checkpoint_dir.mkdir(exist_ok=True)
                for phase_step in range(start, mod.CEILING + 1):
                    disk_guard()
                    if bounds and time.monotonic() - started > bounds['wall_seconds']:
                        raise TimeoutError('registered infrastructure wall ceiling')
                    if phase_step:
                        total += 1
                        before = time.monotonic()
                        batch = task.generate_batch(32, depth + 2, rng)
                        if not is16:
                            metrics = x17.train_update(model, opt, batch, phase_step, rng)
                        elif curriculum and depth == 16:
                            metrics = x10.train_update(model, opt, 'adamw', batch, phase_step)
                        else:
                            lr = x16.extension_rate(phase_step, transition['entry_lr']) if depth == 64 else None
                            metrics = x16.train_update(model, opt, batch, phase_step, weight, lr)
                        opt_seconds += metrics.get('optimizer_seconds', 0.)
                        if total in reference:
                            compare_replay(total, metrics, reference[total])
                        if total == summary['partial_step']:
                            compare_replay(total, metrics, summary['prefix_diagnostics'])
                        row = {'step': total, **metrics, 'elapsed_s': time.monotonic() - started}
                        if is16:
                            row.update(depth=depth, phase_step=phase_step, update_seconds=time.monotonic() - before)
                        row['provenance'] = 'diagnostic_replay' if total <= (summary['partial_step'] or summary['last_full_step']) else 'resumed'
                        disk_guard()
                        log.write(json.dumps(row) + '\n')
                        if total == (summary['partial_step'] or summary['last_full_step']):
                            evidence = {**summary, 'status': 'verified', 'bitwise_full_state_proven': False,
                                'proof_scope': SCOPE, 'checkpoint': resume['checkpoint']}
                            write_json(out / 'replay_verified.json', evidence)
                            write_text(out / 'INFRA-INTERRUPTION.md', f'# INFRA-INTERRUPTION\n2026-10-21 ~21:00 UTC ENOSPC.\n{json.dumps(evidence, indent=2)}\n')
                            print(json.dumps({'name': name, 'replay': 'verified', 'through': total}), flush=True)
                        if bounds and torch.cuda.max_memory_allocated() > bounds['peak_allocated_bytes']:
                            raise MemoryError('registered infrastructure memory ceiling')
                    if phase_step % mod.EVAL_EVERY and phase_step != mod.CEILING:
                        continue
                    disk_guard()
                    metrics = mod.evaluate(model, pools[depth])
                    criterion = curriculum and depth == 16 and len(x16.taught_positions(metrics)) >= 4
                    current = metrics['aggregate']
                    is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
                    cp = None
                    if is_best or phase_step % 1000 == 0 or phase_step == mod.CEILING or criterion:
                        disk_guard()
                        cp = x10.checkpoint(checkpoint_dir, model, opt, rng, total,
                            {**config, 'depth': depth, 'phase_step': phase_step} if is16 else config)
                        checkpoints[cp['path']] = cp
                        latest = cp['path']
                    if is_best:
                        best = {**current, 'step': total, 'checkpoint': cp}
                    if criterion:
                        transition_cp = cp['path']
                        transition = {'step': total, 'taught_positions': x16.taught_positions(metrics),
                            'entry_lr': x10.learning_rate(phase_step), 'checkpoint': cp, 't16': metrics}
                        write_json(out / 'transition.json', transition)
                    row = {'step': total, **metrics, 'checkpoint': cp,
                        'elapsed_s': time.monotonic() - started,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'decision': 'extend' if criterion else ('safety_ceiling' if phase_step == mod.CEILING else 'continue')}
                    if is16:
                        row.update(phase_step=phase_step, depth=depth,
                            retention_t16=x16.evaluate(model, pools[16]) if depth == 64 else None,
                            optimizer_seconds_total=opt_seconds,
                            weight_norm=sum(p.detach().float().square().sum().item() for p in model.parameters()) ** .5)
                    else:
                        row.update(carry_probability=x17.carry_probability(phase_step), eval_carry_probability=0.)
                    history.append(row)
                    write_json(out / 'validation.json', history)
                    write_json(out / 'checkpoints.json', checkpoints)
                    keep = {entry, latest, best['checkpoint']['path']}
                    if transition_cp:
                        keep.add(transition_cp)
                    prune_new(out, checkpoints, keep)
                    if criterion:
                        break
                if curriculum and depth == 16 and transition is None:
                    write_json(out / 'result.json', {'name': name, 'status': 'criterion_not_reached',
                        'total_steps': total, 't64_steps': 0, 'stop_reason': 'teaching_ceiling', 'final': history[-1],
                        'finished_at': time.time()})
                    return
        write_json(out / 'result.json', {'name': name, 'status': 'complete', 'total_steps': total,
            't64_steps': mod.CEILING if curriculum else 0, 'stop_reason': 'safety_ceiling',
            'best': best, 'transition': transition, 'final': history[-1],
            'wall_seconds': time.monotonic() - started, 'finished_at': time.time()})
    except BaseException as exc:
        # Emergency metadata is the only below-floor exception. Concurrent exhaustion can defeat it.
        x7.write_json(out / 'interrupted.json', {'status': 'INFRA-INTERRUPTED', 'error': repr(exc),
            'resumable_checkpoint': checkpoints[latest], 'replay_verified': (out / 'replay_verified.json').exists(),
            'proof_scope': SCOPE, 'finished_at': time.time()})
        raise


def valid_result(name, result):
    if name in x16.ARMS:
        from scripts.invertible_stage16_control import valid_result as valid16
        return valid16(name, result)
    return (result.get('name') == name and result.get('status') == 'complete'
        and result.get('total_steps') == x17.CEILING and result.get('stop_reason') == 'safety_ceiling'
        and result['final']['step'] == x17.CEILING and result['final']['carry_probability'] == 0.
        and result['final']['eval_carry_probability'] == 0. and len(result['final']['positions']) == 64)


def worker(name):
    root, receipt = root_for(name), root_for(name) / 'receipts' / name
    def interrupted(signum, _frame):
        raise SystemExit(f'worker received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    child, leased, code, error = None, False, 1, None
    env = dict(os.environ)
    gpu = BINDINGS[name]
    try:
        verify(name)
        write_json(receipt / 'worker.json', identity(os.getpid()))
        env.update(GPU_LEASE_VISIBLE=str(gpu), GPU_LEASE_PID=str(os.getpid()),
                   CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        disk_guard()
        with (receipt / 'lease.txt').open('x') as lease:
            subprocess.run(['bash', 'scripts/gpu_lease.sh', 'acquire', '1', '--no-wait'],
                env=env, stdout=lease, stderr=lease, check=True)
        leased = True
        cmd = [sys.executable, '-m', 'scripts.invertible_stage16_17_resume', 'train', '--name', name]
        write_json(receipt / 'command.json', {'argv': cmd, 'gpu': gpu})
        disk_guard()
        with (receipt / 'console.txt').open('x') as output:
            child = subprocess.Popen(cmd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(receipt / 'trainer.json', identity(child.pid))
            while True:
                try:
                    code = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    subprocess.run(['bash', 'scripts/gpu_lease.sh', 'heartbeat', str(gpu)], env=env,
                        check=True, stdout=subprocess.DEVNULL)
        if code == 0 and (not (root / name / 'replay_verified.json').exists()
                          or not valid_result(name, read_json(root / name / 'result.json'))):
            raise ValueError('invalid terminal/replay result')
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
            'status': 'complete' if code == 0 else 'INFRA-INTERRUPTED',
            'result_sha256': x7.digest(root / name / 'result.json') if code == 0 else None}
        # Tiny emergency terminal evidence must still be attempted below the disk floor.
        try:
            x7.write_json(receipt / 'terminal.json', terminal)
            x7.write_json(root / 'markers' / (name + ('.DONE' if code == 0 else '.FAIL')), terminal)
        finally:
            if leased:
                subprocess.run(['bash', 'scripts/gpu_lease.sh', 'release', str(gpu)], env=env,
                    check=True, stdout=subprocess.DEVNULL)
    return code


def launch(name):
    verify(name)
    root, receipt = root_for(name), root_for(name) / 'receipts' / name
    if (root / name).exists():
        raise FileExistsError('no automatic retry/restart')
    disk_guard()
    receipt.mkdir(parents=True, exist_ok=False)
    with (receipt / 'worker_console.txt').open('x') as output:
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage16_17_resume', 'worker', '--name', name],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(receipt / 'detached.json', {'worker': identity(proc.pid), **identity(proc.pid)})
    if name == x17.NAMES[0]:
        write_json(root / 'receipts/scheduler/terminal.json', {'status': 'launched',
            'admission': 'attended fixed-card4 infra recovery, no card-wait scheduler', 'time': time.time()})
    print(f'{name}: worker={proc.pid}, card={BINDINGS[name]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'launch', 'worker', 'train'])
    parser.add_argument('--stage', type=int, choices=[16, 17])
    parser.add_argument('--name', choices=BINDINGS)
    args = parser.parse_args()
    if args.action == 'register':
        if args.stage is None:
            parser.error('--stage required')
        register(args.stage)
    else:
        if args.name is None:
            parser.error('--name required')
        result = globals()[args.action](args.name)
        if args.action == 'worker':
            raise SystemExit(result)
