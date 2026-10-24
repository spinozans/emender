"""X23E persistent launch/watch chain with identity receipts and disk censoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from experiments.expressivity_tasks.run_merrill26_separators import (
    ROOT, FREE_BYTES, RUN_BYTES, WALL_SECONDS, digest, write_json, verify_registration,
)
import shutil


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return {'pid': pid, 'starttime': stat[19], 'host': socket.gethostname()}


def alive(record):
    if record['host'] != socket.gethostname():
        raise ValueError('watcher must run on launch host')
    try:
        stat = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(') ', 1)[1].split()
        return stat[0] not in ('Z', 'X') and stat[19] == record['starttime']
    except FileNotFoundError:
        return False


def terminal(name, rc, override=None):
    path = ROOT / name / 'result.json'
    guard = ROOT / 'receipts' / name / 'guard.json'
    if guard.exists():
        override = 'infrastructure resource guard; right-censored'
    if override or not path.exists():
        infrastructure = override is not None or rc in (137, 143, -9, -15)
        write_json(path, {'name': name, 'status': 'censored' if infrastructure else 'error',
            'stop_reason': override or f'process exit {rc} without trainer result',
            'scientific_verdict': 'not adjudicated; right-censored' if infrastructure else 'not adjudicated',
            'finished_at': time.time()})
    result = json.loads(path.read_text())
    state = result['status'] if rc == 0 or result['status'] == 'censored' else 'error'
    row = {'name': name, 'exit_code': rc, 'status': state, 'finished_at': time.time(),
           'result_sha256': digest(path), 'stop_reason': result['stop_reason']}
    write_json(ROOT / 'receipts' / name / 'terminal.json', row)
    marker = 'DONE' if state == 'complete' else ('CENSORED' if state == 'censored' else 'FAIL')
    write_json(ROOT / 'markers' / f'{name}.{marker}', row)


def rung_status(name):
    receipt = ROOT / 'receipts' / name
    path = receipt / 'terminal.json'
    if path.exists():
        row = json.loads(path.read_text())
        result_path = ROOT / name / 'result.json'
        marker = 'DONE' if row['status'] == 'complete' else ('CENSORED' if row['status'] == 'censored' else 'FAIL')
        result = json.loads(result_path.read_text())
        if (row['name'] != name or digest(result_path) != row['result_sha256']
                or json.loads((ROOT / 'markers' / f'{name}.{marker}').read_text()) != row
                or result['name'] != name
                or (row['status'] == 'complete' and (row['exit_code'] != 0 or result['status'] != 'complete'))):
            raise ValueError('invalid terminal receipt: ' + name)
        return row['status']
    record = receipt / 'worker.json'
    if not record.exists():
        return 'queued'
    if not alive(json.loads(record.read_text())):
        # Allow shell's terminal writer to finish; no fabricated scientific failure.
        time.sleep(2)
        if path.exists():
            return rung_status(name)
        terminal(name, -9, 'worker disappeared; infrastructure censoring')
        return 'censored'
    return 'running'


def spawn(name, card):
    receipt = ROOT / 'receipts' / name
    receipt.mkdir(parents=True, exist_ok=False)
    arm, task, _ = name.split('_')
    command = ['bash', 'scripts/launch_merrill26_separators.sh', '--worker', arm, task, str(card)]
    stream = (receipt / 'worker_console.txt').open('w')
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    stream.close()
    write_json(receipt / 'worker.json', {**identity(process.pid), 'card': card, 'command': command,
                                        'launched_at': time.time()})
    return process


def launch(*, restartable_x27=False):
    reg = verify_registration()
    if restartable_x27 and reg['task'] != 'X27':
        raise ValueError('restartable launch is X27-only')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    upstream = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', '@{upstream}'], text=True).strip()
    remote, branch = upstream.split('/', 1)
    remote_row = subprocess.check_output(['git', 'ls-remote', remote, 'refs/heads/' + branch], text=True).strip()
    if not remote_row or remote_row.split()[0] != commit:
        raise ValueError('registration must be pushed before launching')
    launch_exists = (ROOT / 'receipts' / 'launch.json').exists()
    if launch_exists and not restartable_x27:
        raise ValueError('launch already exists; do not repeat mutations')
    for name in reg['runs']:
        arm, task, _ = name.split('_')
        path = ROOT / 'receipts' / f'smoke_{arm}_{task}.json'
        if json.loads(path.read_text())['status'] != 'passed':
            raise ValueError('smoke gate failed: ' + name)
    if launch_exists:
        # X27 reserves each arm's receipt directory exclusively before spawning.
        return reg
    write_json(ROOT / 'receipts' / 'launch.json', {'commit': commit, 'remote': remote_row,
        'registration_sha256': digest(ROOT / 'registration.json'), 'launched_at': time.time(),
        'watcher': identity(os.getpid()), 'gpu_before': subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'], text=True)})
    return reg


def watch(reg, initial=False, once=False):
    children = {}
    names = reg['runs']
    if initial:
        for name in names:
            if name == 'gdn2_mod97_seed0':
                continue
            children[name] = spawn(name, reg['cards'][name])
    while True:
        states = {name: rung_status(name) for name in names}
        # The secondary arms deliberately share card 6 and may never overlap.
        if states['gdn2_mod97_seed0'] == 'queued' and states['e97_mod97_seed0'] in ('complete', 'censored', 'error'):
            children['gdn2_mod97_seed0'] = spawn('gdn2_mod97_seed0', 6)
            states['gdn2_mod97_seed0'] = 'running'
        free = shutil.disk_usage(ROOT).free
        for name, state in states.items():
            if state != 'running':
                continue
            record = json.loads((ROOT / 'receipts' / name / 'worker.json').read_text())
            out = ROOT / name
            size = 0
            for path in out.rglob('*'):
                try:
                    if path.is_file():
                        size += path.stat().st_size
                except FileNotFoundError:
                    pass  # Atomic checkpoint staging file was just renamed.
            guard_path = ROOT / 'receipts' / name / 'guard.json'
            if (guard_path.exists() or free < FREE_BYTES or size > RUN_BYTES
                    or time.time() - record['launched_at'] > WALL_SECONDS):
                if not guard_path.exists():
                    write_json(guard_path, {'free_bytes': free, 'rung_bytes': size, 'at': time.time(),
                        'decision': 'infrastructure censoring; TERM then KILL own process group'})
                elapsed = time.time() - json.loads(guard_path.read_text())['at']
                if alive(record):
                    os.killpg(record['pid'], signal.SIGKILL if elapsed >= 30 else signal.SIGTERM)
        for name, child in list(children.items()):
            if child.poll() is not None:
                del children[name]
        row = {'at': time.time(), 'states': states, 'free_bytes': free}
        with (ROOT / 'receipts' / 'watcher_polls.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        write_json(ROOT / 'receipts' / 'watcher_status.json', row)
        lines = ['# X23E observed rungs', '', 'Missing threshold crossings are right-censored, never evidence of inability.', '']
        for name, state in states.items():
            path = ROOT / name / 'training.jsonl'
            latest = None
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 4096))
                    rows = stream.read().splitlines()
                    if rows:
                        try:
                            latest = json.loads(rows[-1]).get('step')
                        except ValueError:
                            pass
            lines.append(f'- {name}: {state}; latest observed update {latest}; unobserved crossings censored.')
        (ROOT / 'GROWTH_REPORT.md').write_text('\n'.join(lines) + '\n')
        print(json.dumps(row), flush=True)
        if all(state in ('complete', 'censored', 'error') for state in states.values()):
            write_json(ROOT / 'receipts' / 'watcher_terminal.json', row)
            return 1 if 'error' in states.values() else 0
        if once:
            return 2
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--terminal')
    parser.add_argument('--exit-code', type=int, default=0)
    args = parser.parse_args()
    if args.terminal:
        terminal(args.terminal, args.exit_code)
    else:
        reg = launch() if args.launch else json.loads((ROOT / 'registration.json').read_text())
        raise SystemExit(watch(reg, initial=args.launch, once=args.once))
