#!/usr/bin/env python3
"""Supervisor-side X9b ScheduleFree anneal completion watcher; zero only for the successful terminal receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import time

NAMES = ['e97_bf16_seed0']
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'experiments/expressivity_tasks/results_invertible_stage9b_sf_anneal_8m'


def process_alive(record):
    if record['host'] != socket.gethostname():
        raise ValueError('watcher must run on the launch host')
    try:
        stat = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(') ', 1)[1].split()
        return stat[0] not in ('Z', 'X') and stat[19] == str(record['starttime'])
    except FileNotFoundError:
        return False


def status(root):
    pending = []
    for name in NAMES:
        if (root / 'markers' / f'{name}.FAIL').exists():
            return 1, f'FAIL {name}; inspect receipts/{name}/terminal.json'
        done = root / 'markers' / f'{name}.DONE'
        if done.exists():
            try:
                row = json.loads(done.read_text())
                terminal = json.loads((root / 'receipts' / name / 'terminal.json').read_text())
                result_path = root / name / 'result.json'
                result_bytes = result_path.read_bytes()
                result = json.loads(result_bytes)
                valid = (row == terminal and row.get('name') == name and row.get('exit_code') == 0
                    and row.get('result_sha256') == hashlib.sha256(result_bytes).hexdigest()
                    and result.get('status') == 'complete' and result.get('total_steps') == 500000
                    and result.get('stop_reason') == 'safety_ceiling')
            except (OSError, ValueError):
                valid = False
            if not valid:
                return 1, f'FAIL invalid DONE receipt {name}'
            continue
        for role in ('worker', 'trainer'):
            record = root / 'receipts' / name / f'{role}.json'
            if not record.exists():
                if role == 'worker':
                    return 1, f'FAIL missing worker identity {name}'
                continue  # The leased worker may still be waiting for its GPU.
            if not process_alive(json.loads(record.read_text())):
                # Allow the worker to publish the terminal receipt after wait().
                time.sleep(2)
                if done.exists() or (root / 'markers' / f'{name}.FAIL').exists():
                    return status(root)
                return 1, f'FAIL {name} {role} died without a terminal marker'
        pending.append(name)
    return (None, 'remaining=' + ','.join(pending)) if pending else (0, 'DONE X9b ScheduleFree anneal run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--interval', type=float, default=900, help='poll seconds (default 15 minutes)')
    parser.add_argument('--log', type=Path, help='default ROOT/receipts/completion_watcher.log')
    parser.add_argument('--once', action='store_true', help='one status check; 2 means still running')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    log = args.log or args.root / 'receipts/completion_watcher.log'
    while True:
        code, message = status(args.root)
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {message}'
        with log.open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)
        if code is not None:
            return code
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
