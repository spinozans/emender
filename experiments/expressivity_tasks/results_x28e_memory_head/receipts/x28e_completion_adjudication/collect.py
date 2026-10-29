"""Read-only adjudication; no training, GPU access, or frozen-package writes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
PACKAGE = ROOT / 'experiments/expressivity_tasks/x28e_memory_head'
RESULTS = PACKAGE.parent / 'results_x28e_memory_head'
OUT = Path(__file__).resolve().parent


def payload_hash(row):
    return hashlib.sha256(json.dumps(
        {k: v for k, v in row.items() if k != 'payload_sha256'},
        sort_keys=True, separators=(',', ':'), default=int).encode()).hexdigest()


def evidence(path):
    data = path.read_bytes()
    item = {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    if path.suffix == '.json':
        row = json.loads(data)
        item['payload'] = row
        item['payload_hash_valid'] = row.get('payload_sha256') == payload_hash(row)
        assert item['payload_hash_valid'], path
    return item


def write(name, row):
    row['payload_sha256'] = payload_hash(row)
    (OUT / name).write_text(json.dumps(row, indent=2) + '\n')


assert sys.executable == '/home/erikg/emender/.venv/bin/python'
launcher = PACKAGE / 'launcher.py'
checkpoint = RESULTS / 'tape_B0x16_seed0/checkpoint.json'
command = [sys.executable, str(launcher), '--eval', '--arm', 'tape_B0x16',
           '--checkpoint', str(RESULTS / 'tape_B0x16_seed0/best.pt')]
probe = subprocess.run(command, capture_output=True, text=True)
assert probe.returncode == 2 and 'unrecognized arguments' in probe.stderr
write('eval_entrypoint_failure.json', {
    'command': command, 'exit_code': probe.returncode,
    'stdout': probe.stdout, 'stderr': probe.stderr,
    'classification': 'unsupported CLI, not a registered scientific/resource gate',
    'consequence': 'STOP: no frozen launcher eval subcommand exists; no evaluation executed',
    'selected_checkpoint': evidence(checkpoint),
    'weights': evidence(RESULTS / 'tape_B0x16_seed0/best.pt'),
    'launcher': evidence(launcher),
})

corners = []
freeze = json.loads((PACKAGE / 'freeze_manifest.json').read_text())['manifest_hash']
for arm, slots in [('tape_B0', 736), ('tape_B0x4', 2944), ('tape_B0x16', 11776),
                   ('tape_unbounded', None), ('tape_appendonly', None)]:
    run = RESULTS / f'{arm}_seed0'
    receipts = RESULTS / 'receipts' / f'{arm}_seed0'
    row = {'arm': arm, 'slots': slots, 'freeze_manifest_hash': freeze,
           'launch': evidence(receipts / 'launch.json'),
           'checkpoint': evidence(run / 'checkpoint.json')}
    launch = row['launch']['payload']
    assert launch['arm'] == arm and launch['freeze_manifest_hash'] == freeze
    training = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
    row['last_training_record'] = training[-1]
    row['training_source'] = evidence(run / 'training.jsonl')
    final = run / 'final_eval.json'
    if final.exists():
        ev = evidence(final)
        row['final_eval_source'] = {k: v for k, v in ev.items() if k != 'payload'}
        row['bins'] = {k: v for k, v in ev['payload'].items()
                       if k.startswith('bin_')}
        # Keep the complete numerical reports; lengthy per-example meta is hash-bound above.
        for item in row['bins'].values():
            item.pop('meta', None)
        row['result'] = evidence(run / 'result.json')
        row['status'] = 'complete; existing final evaluation'
    else:
        guard = evidence(receipts / 'guard.json')
        row['guard'] = guard
        g = guard['payload']
        assert g['arm'] == arm
        assert g['free_bytes'] < 400 * 1024**3
        assert g['run_bytes'] <= 40 * 1024**3 and g['wall_seconds'] <= 7 * 86400
        row['status'] = 'infrastructure right-censored; disk policy gate fired correctly'
        row['gate'] = {
            'source': 'launcher.py:632-634',
            'condition': 'size > RUN_BYTES or wall > WALL_SECONDS or free < FREE_BYTES',
            'fired_clause': 'free < FREE_BYTES',
            'free_bytes': g['free_bytes'], 'FREE_BYTES': 429496729600,
            'deficit_bytes': 429496729600 - g['free_bytes'],
            'run_bytes_clause': False, 'wall_seconds_clause': False,
            'classification_source': 'launcher.py:210-220; validated identity-bound guard => censored',
        }
        row['resume_disposition'] = (
            'Unavailable: launcher main only accepts --launch/--once/--terminal/--exit-code; '
            'watch only launches queued arms, and guard keeps this arm censored. '
            'Trainer has no resume CLI or optimizer-state reload. No receipts deleted or gate bypassed. '
            'Stage-2 impossibility is NOT established by this infrastructure censor.')
        if arm == 'tape_B0x16':
            row['eval_disposition'] = 'u9000 metric 0.98 selected; final evaluation missing; unsupported launcher eval CLI (exit 2), not a scientific gate'
    write(arm + '.json', row)
    corners.append({'arm': arm, 'slots': slots, 'status': row['status'],
                    'receipt': evidence(OUT / (arm + '.json'))['sha256']})

log = Path('/tmp/pi-processes-8OXa9B/proc_3b2b-stdout.log')
polls = RESULTS / 'receipts/watcher_polls.jsonl'
write('phase_diagram.json', {
    'corners': corners,
    'certification': 'INCOMPLETE: only B0 and B0x4 have final evaluations among the five requested corners',
    'watcher_terminal': evidence(RESULTS / 'receipts/watcher_terminal.json'),
    'watcher_status': evidence(RESULTS / 'receipts/watcher_status.json'),
    'watcher_wiring': evidence(RESULTS / 'receipts/watcher_wiring.json'),
    'watcher_polls': evidence(polls),
    'last_two_polls': [json.loads(s) for s in polls.read_text().splitlines()[-2:]],
    'launcher_stdout': evidence(log),
    'launcher_stdout_tail': log.read_text().splitlines()[-2:],
    'launcher_source': evidence(launcher), 'trainer_source': evidence(PACKAGE / 'trainer.py'),
    'residual_risks': [
        'No B0x16 final evaluation; selected validation score is not ID/OOD certification.',
        'No supported launcher eval/resume paths in frozen package; three corners remain unresolved.',
        'Infrastructure censoring cannot establish a scientific stage-2 boundary.',
        'contact_supervisor tool unavailable in this session; no supervisor decision could be requested.',
    ],
})
print('Validated source receipt hashes and identity bindings; wrote five hash-bound corner receipts.')
print('B0x16 eval attempt: exit 2, unrecognized arguments; no GPU touched, no frozen code changed.')
