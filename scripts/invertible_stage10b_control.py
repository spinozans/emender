#!/usr/bin/env python3
"""X10b preregistration, numerical probe, detached leased launch and readout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

from experiments.expressivity_tasks import run_invertible_stage10b_muonlr_8m as run
from experiments.expressivity_tasks import hybrid_muon

BASELINE_ROOT = Path('experiments/expressivity_tasks/results_invertible_stage10_optimizers_8m')

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
        cmd = [sys.executable, '-m', 'experiments.expressivity_tasks.run_invertible_stage10b_muonlr_8m', '--arm', arm, '--seed', '0']
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
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.invertible_stage10b_control', 'worker', '--arm', arm],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run.x7.write_json(receipt / 'detached.json', identity(proc.pid))
    print(f'{name} worker={proc.pid} card={run.CARDS[arm]}', flush=True)


def baseline_summary(arm):
    path = BASELINE_ROOT / f'{arm}_seed0'
    rows = json.loads((path / 'validation.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    terminal = json.loads((BASELINE_ROOT / 'receipts' / f'{arm}_seed0' / 'terminal.json').read_text())
    done = json.loads((BASELINE_ROOT / 'markers' / f'{arm}_seed0.DONE').read_text())
    if (terminal != done or terminal['exit_code'] != 0 or
            terminal['result_sha256'] != run.x7.digest(path / 'result.json') or
            result['total_steps'] != 500000 or result['final'] != rows[-1] or
            [r['step'] for r in rows] != list(range(0, 500001, 500))):
        raise ValueError('invalid historical baseline receipts')
    return {'label': f'X10-{arm}@1e-3', 'artifact_sha256': {
        f: run.x7.digest(path / f) for f in ('validation.json', 'result.json')},
        'rows': rows}


def register():
    """Bind unchanged X10 bytes and prelaunch hypotheses; never overwrite."""
    import torch
    if (run.ROOT / 'registration.json').exists():
        raise FileExistsError('registration is immutable')
    baseline = run.verify_registration(BASELINE_ROOT)
    torch.manual_seed(0)
    model = run.build_model(run.ARMS[0])
    routes = hybrid_muon.parameter_routes(model)
    if routes != baseline['parameter_routes']:
        raise ValueError('not the exact X10 parameter routing')
    pools = [hashlib.sha256(a.tobytes()).hexdigest() for a in run.x7.frozen_pools(0, 16, 61)['val']]
    if pools != baseline['pool_array_sha256']['0']:
        raise ValueError('not the X10 frozen pool')
    extra = [Path(__file__).relative_to(Path.cwd()), Path(run.__file__).relative_to(Path.cwd()),
             Path('scripts/watch_invertible_stage10b_muonlr.py'),
             Path('tests/test_invertible_stage10b_muonlr_8m.py'),
             run.ROOT / 'REGISTRATION.md']
    extra += [run.ROOT / 'receipts' / f'probe_{a}.json' for a in run.ARMS]
    extra += [run.ROOT / 'receipts' / f for f in ('tests_prelaunch.txt', 'tests_gpu_prelaunch.txt', 'gpu_before.txt')]
    summaries = {a: baseline_summary(a) for a in ('muon', 'adamw')}
    for arm, summary in summaries.items():
        path = run.ROOT / 'baselines' / f'x10_{arm}.json'
        run.x7.write_json(path, summary)
        extra.append(path)
    bounds = {}
    for arm in run.ARMS:
        probe = json.loads((run.ROOT / 'receipts' / f'probe_{arm}.json').read_text())
        if probe['reload_next_update_max_abs_difference'] != 0 or probe['frozen_task_evaluation']:
            raise ValueError('probe did not pass bit-exact no-score gate')
        for p, sha in probe['source_sha256'].items():
            if run.x7.digest(p) != sha:
                raise ValueError('probe source changed: ' + p)
        bounds[arm] = {'wall_seconds': max(baseline['resource_bounds']['muon']['wall_seconds'], probe['wall_ceiling_seconds']),
                       'peak_allocated_bytes': max(baseline['resource_bounds']['muon']['peak_allocated_bytes'], probe['memory_ceiling_bytes'])}
    registration = {**baseline, 'task': 'X10b', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'parent_registration_sha256': run.x7.digest(BASELINE_ROOT / 'registration.json'),
        'source_sha256': {**baseline['source_sha256'],
                          str(BASELINE_ROOT / 'registration.json'): run.x7.digest(BASELINE_ROOT / 'registration.json'),
                          **{str(p): run.x7.digest(p) for p in extra}},
        'authorized_runs': [f'{a}_seed0' for a in run.ARMS], 'seeds': [0],
        'gpu_binding': {f'{a}_seed0': run.CARDS[a] for a in run.ARMS},
        'remaining_seeds': 'none authorized; one seed per arm',
        'recipes': {a: {**baseline['recipes']['muon'], 'lr_multiplier': run.MULTIPLIERS[a],
                       'schedule': f'linear warmup updates1..5000 to {run.learning_rate(a,5000):g}; cosine floor {run.learning_rate(a,500000):g} at500000; both groups once/update; actual LR logged each update'} for a in run.ARMS},
        'predictions': [
            '(a) LR-UNDERSCCALED: if either higher-LR arm consolidates P4 materially beyond X10 Muon final448/512 (e.g. >=508/512 at500K), or accelerates same-threshold P4 acquisition, the X10 lag supports an LR-scale artifact for this learned solution; record which multiplier wins. Report transient crossings separately from final retention.',
            '(b) SCALE-RIGHT: if 3x/10x show P1-P3 degradation or numerical instability (any of these prefix positions below512 at final), this supports 1e-3-class for this hybrid at8M and a method-intrinsic P4 lag at this tested scale; do not generalize beyond this one-seed screen. Infrastructure censoring is not optimizer instability.',
            '(c) For EACH higher-LR arm, print a table with THREE outcome columns: higher-LR arm | X10-muon@1e-3 | X10-adamw@1e-3, plus row labels P1-P16 first>=50/512, first512/512 (update/elapsed seconds), maxima and fixed-final counts. Muon baseline P3-perfect281500 vs AdamW306500 (25000 earlier), P4-material348500 vs309000 (39500 later), P4-perfect censored vs391500. No fixed-final verdict before500K; mixed/neither outcomes inconclusive, not forced into (a)/(b).'],
        'resource_bounds': bounds, 'point_semantics': {a: run.point(a) for a in run.ARMS},
        'analysis_command': '.venv/bin/python -m scripts.invertible_stage10b_control analyze --output X10b-growth-report.md',
        'readouts': baseline['readouts'] + '; joint marginal1/3721 and suffix7 floor printed; exact counts, learned-solution scope only',
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('NDM_', 'TRITON_'))}}
    run.x7.write_json(run.ROOT / 'registration.json', registration)
    print('registered X10b before launch', flush=True)


def crossing(rows, position, threshold):
    row = next((r for r in rows if r['positions'][position]['correct'] >= threshold), None)
    return 'not reached (censored)' if row is None else f"{row['step']:,} / {row['elapsed_s']:.3f}s"


def analyze(output=None):
    """Read CPU receipts only; paired comparison columns for each LR arm."""
    registration = run.verify_registration(run.ROOT)
    text = ['# X10b higher-LR hybrid-Muon growth report', '',
            'One seed0 per arm; exact joint-state counts/512, learned-solution scope only.',
            'First crossings on the500-update grid are transient, not retained acquisition.',
            'Uniform joint marginal1/3721 =0.0268744961%; CE ln3721 =8.221747728 nats.',
            'Suffix7-restricted P9-P16 floor applies to observers missing >=2 iid drivers, not prefix-aware recurrent models.', '']
    for arm in run.ARMS:
        path = run.ROOT / f'{arm}_seed0'
        rows = json.loads((path / 'validation.json').read_text())
        complete = (path / 'result.json').exists() and rows[-1]['step'] == 500000
        if complete:
            result = json.loads((path / 'result.json').read_text())
            terminal = json.loads((run.ROOT / 'receipts' / f'{arm}_seed0' / 'terminal.json').read_text())
            done = json.loads((run.ROOT / 'markers' / f'{arm}_seed0.DONE').read_text())
            if (terminal != done or terminal['exit_code'] != 0 or result['final'] != rows[-1] or
                    terminal['result_sha256'] != run.x7.digest(path / 'result.json')):
                raise ValueError('invalid higher-LR terminal receipts')
        baselines = [json.loads((run.ROOT / 'baselines' / f'x10_{a}.json').read_text()) for a in ('muon', 'adamw')]
        columns = [rows] + [b['rows'] for b in baselines]
        text += [f'## {arm}: observed through{rows[-1]["step"]:,}; ' + ('fixed-final500K' if complete else 'LIVE, fixed-final pending'), '',
                 f'| Position/readout | {arm} | X10-muon@1e-3 | X10-adamw@1e-3 |',
                 '|---|---:|---:|---:|']
        for p in range(16):
            for threshold in (50, 512):
                text.append(f'| P{p+1} first>={threshold} update/seconds | ' + ' | '.join(crossing(v,p,threshold) for v in columns) + ' |')
            text.append(f'| P{p+1} max-anywhere | ' + ' | '.join(str(max(r['positions'][p]['correct'] for r in v))+'/512' for v in columns) + ' |')
            text.append(f'| P{p+1} ' + ('fixed-final' if complete else 'latest, not final') + ' | ' + ' | '.join(str(v[-1]['positions'][p]['correct'])+'/512' for v in columns) + ' |')
        final = rows[-1]
        text += ['', f"Latest contiguous perfect prefix={final['perfect_prefix_depth']}; terminal P16={final['aggregate']['terminal_correct']}/512; suffix7-restricted={final['aggregate']['post_suffix7_correct']}/4096.", '']
        if complete:
            p4 = final['positions'][3]['correct']
            prefix = all(final['positions'][p]['correct'] == 512 for p in range(3))
            baseline = columns[1]
            earlier = any(next((r['step'] for r in rows if r['positions'][3]['correct'] >= t), 500001) <
                          next((r['step'] for r in baseline if r['positions'][3]['correct'] >= t), 500001) for t in (50,512))
            text += [f'Registered diagnostics: (a) P4>=508={p4>=508}, same-threshold P4 acceleration={earlier}; (b) P1-P3 end degradation={not prefix}. Mixed outcomes remain inconclusive; optimizer numerical failures require separate receipt audit.', '']
    text += ['## Registered predictions (pre-launch)', *registration['predictions'], '',
             'Winner and interpretation require full per-update finite/LR audit, source/checkpoint/terminal review and independent acceptance; no cross-seed robustness or production promotion.']
    report = '\n'.join(text) + '\n'
    if output:
        Path(output).write_text(report)
    else:
        print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['probe', 'register', 'launch', 'worker', 'analyze'])
    parser.add_argument('--arm', choices=run.ARMS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.action == 'analyze':
        analyze(args.output)
    elif args.arm is None:
        parser.error('--arm required')
    else:
        result = globals()[args.action](args.arm)
        if args.action == 'worker':
            raise SystemExit(result)
