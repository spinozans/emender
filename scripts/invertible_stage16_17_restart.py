#!/usr/bin/env python3
"""Approved checkpoint re-execution after a failed exact-replay gate.

The frozen replay runner remains unchanged for the three successfully verified arms.
This separate admission never claims restoration of the historical at-death state.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from scripts import invertible_stage16_17_resume as replay
from scripts.invertible_stage10_control import identity

x16, x17, x10, x7 = replay.x16, replay.x17, replay.x10, replay.x7
BINDINGS = {n: replay.BINDINGS[n] for n in ('e97_w32_seed0', 'e97_curriculum_seed0', 'e97_carrywean_seed0')}
DISK_MIN_BYTES = replay.DISK_MIN_BYTES
SCOPE = 'Restart from exact checkpoint state; historical post-checkpoint updates superseded. NO at-death-restoration or exact replay claim.'
disk_guard, write_json, write_text = replay.disk_guard, replay.write_json, replay.write_text
assert_state_equal, read_json = replay.assert_state_equal, replay.read_json
module_for, prune_new, valid_result = replay.module_for, replay.prune_new, replay.valid_result
X15_DIVERGENCE = Path('experiments/expressivity_tasks/results_invertible_stage15_probes_8m/muon_resume/LAUNCH-INTERRUPTION-2026-10-21.md')


def root_for(name):
    return module_for(name).ROOT / 'infra_restart'


def provenance(resume):
    return {'mode': 'restart_from_checkpoint', 'checkpoint_step': resume['checkpoint']['step'],
        'historical_tail_superseded_from': resume['checkpoint']['step'] + 1,
        'historical_last_full_step': resume['last_full_step'],
        'historical_partial_death_step': resume['partial_step'],
        'restored_at_death_state': False, 'replay_verified': False,
        'divergence_receipt': resume['divergence_receipt']}


def register(stage):
    mod = x16 if stage == 16 else x17
    original = mod.ROOT / 'infra_resume'
    base = x10.verify_registration(original)
    root = mod.ROOT / 'infra_restart'
    disk_guard()
    # The prelaunch test receipt is written before registration; no existing admission allowed.
    if (root / 'registration.json').exists():
        raise FileExistsError('immutable restart registration already exists')
    root.mkdir(exist_ok=True)
    names = [n for n in BINDINGS if module_for(n) is mod]
    sources = dict(base['source_sha256'])
    artifacts = dict(base['artifact_sha256'])
    resumes = {}
    for name in names:
        replay.verify(name)
        failed = original / name / 'interrupted.json'
        failure = read_json(failed)
        if failure['replay_verified'] or 'replay mismatch' not in failure['error']:
            raise ValueError('restart requires an actual failed exact-replay receipt')
        if not (original / 'markers' / (name + '.FAIL')).exists():
            raise ValueError('missing fail-closed terminal marker')
        cp, _ = replay.checkpoint_preflight(mod.ROOT / name)
        if cp != base['resume'][name]['checkpoint']:
            raise ValueError('checkpoint selection changed')
        resume = {**base['resume'][name], 'mode': 'restart_from_checkpoint',
            'proof_scope': SCOPE, 'divergence_receipt': str(failed),
            'divergence_receipt_sha256': x7.digest(failed), 'divergence': failure['error']}
        resumes[name] = resume
        amendment = root / f'AMENDMENT-2026-10-21-restart-{name}.md'
        supersession = root / f'registration-supersession-{name}.json'
        write_text(amendment, f'''# {name}: approved checkpoint restart amendment, 2026-10-21
The original registration and the exact-replay admission remain immutable.
The supervisor explicitly approved this separate re-admission after the exact gate failed.
Path: RESTART-FROM-CHECKPOINT, not replay-verified resume.
Checkpoint {cp['path']}, step {cp['step']}, SHA256 {cp['sha256']}.
Model, AdamW moments, NumPy/torch/CUDA RNG restored and exactly compared before training.
Historical full updates {cp['step']+1}..{resume['last_full_step']} and partial death
row {resume['partial_step']} are SUPERSEDED by re-execution; training.jsonl marks this boundary
in EVERY row. No historical post-checkpoint validation is copied into this continuation.
{SCOPE}
Exact replay divergence: {failure['error']}
Receipt: {failed}, SHA256 {x7.digest(failed)}.
Independent X15 divergence: {X15_DIVERGENCE}, SHA256 {x7.digest(X15_DIVERGENCE)}:
Muon's update433044 muon_update_rms0.11409317095890362 vs0.11409317090657782.
CUDA floating-point nondeterminism is a plausible justification for checkpoint restart,
not an independently isolated cause or proof that the trajectories coincide.
No tolerance was weakened; failed exact-replay launches are retained as failed evidence.
Continue original schedules and ORIGINAL ceiling {mod.CEILING}, no LR/carry reset.
X16 curriculum retains original teaching criterion and live-moment/RNG cosine extension
for500K T64 updates only after that criterion; no new scientific arm or readout.
Card{BINDINGS[name]} only with atomic no-wait gpu_lease; no automatic retry.
The registered200GiB fail-closed write guard, atomic checkpoints, original evaluation,
retention and stopping rules remain in force; original checkpoints never pruned.
Tiny emergency terminal/interruption metadata is the below-floor write exception.
Concurrent external disk exhaustion can still defeat writes; no absolute ENOSPC guarantee.
''')
        write_json(supersession, {'name': name, 'prior_registration': str(original / 'registration.json'),
            'prior_registration_sha256': x7.digest(original / 'registration.json'),
            'amendment': str(amendment), 'amendment_sha256': x7.digest(amendment), **resume})
        paths = [amendment, supersession, failed, original / 'receipts' / name / 'terminal.json',
                 original / 'markers' / (name + '.FAIL')]
        sources.update({str(p): x7.digest(p) for p in paths})
    paths = [Path(__file__).relative_to(Path.cwd()), Path('tests/test_invertible_stage16_17_restart.py'),
        Path('scripts/watch_invertible_stage16_17_recovery.py'),
        original / 'registration.json', root / 'tests_prelaunch.txt', X15_DIVERGENCE]
    sources.update({str(p): x7.digest(p) for p in paths})
    write_json(root / 'registration.json', {**base, 'task': f'X{stage}R checkpoint restart',
        'registered_utc': datetime.now(timezone.utc).isoformat(), 'authorized_runs': names,
        'source_sha256': sources, 'artifact_sha256': artifacts, 'resume': resumes,
        'gpu_binding': {n: BINDINGS[n] for n in names}, 'proof_scope': SCOPE})


def verify(name):
    root = root_for(name)
    reg = x10.verify_registration(root)
    if name not in reg['authorized_runs'] or reg['gpu_binding'][name] != BINDINGS[name]:
        raise ValueError('unauthorized checkpoint restart')
    resume = reg['resume'][name]
    if resume['mode'] != 'restart_from_checkpoint':
        raise ValueError('wrong restart provenance')
    for path, sha in reg['artifact_sha256'].items():
        if x7.digest(path) != sha:
            raise ValueError('changed original artifact: ' + path)
    if x7.digest(resume['checkpoint']['path']) != resume['checkpoint']['sha256']:
        raise ValueError('changed restart checkpoint')
    return reg


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
        'proof_scope': SCOPE, 'provenance': provenance(resume)}
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
    write_json(out / 'restart_admitted.json', provenance(resume))
    write_text(out / 'INFRA-INTERRUPTION.md', f"# INFRA-INTERRUPTION\n2026-10-21 ~21:00 UTC ENOSPC.\n{json.dumps(resume, indent=2)}\n{SCOPE}\n")
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
                        row = {'step': total, **metrics, 'elapsed_s': time.monotonic() - started}
                        if is16:
                            row.update(depth=depth, phase_step=phase_step, update_seconds=time.monotonic() - before)
                        row['provenance'] = provenance(resume)
                        disk_guard()
                        log.write(json.dumps(row) + '\n')
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
            'resumable_checkpoint': checkpoints[latest], 'replay_verified': False, 'restart_from_checkpoint': True,
            'proof_scope': SCOPE, 'finished_at': time.time()})
        raise


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
        cmd = [sys.executable, '-m', 'scripts.invertible_stage16_17_restart', 'train', '--name', name]
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
        if code == 0 and (not (root / name / 'restart_admitted.json').exists()
                          or not valid_result(name, read_json(root / name / 'result.json'))):
            raise ValueError('invalid terminal/restart result')
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
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage16_17_restart', 'worker', '--name', name],
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
