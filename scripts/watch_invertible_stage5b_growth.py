#!/usr/bin/env python3
"""Supervisor-side X5c completion watcher; zero only for six successful receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

NAMES = [f'{arm}_seed{seed}' for seed in range(3) for arm in ('e97', 'gdn2')]
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'experiments/expressivity_tasks/results_invertible_stage5c_growth_8m'


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
            row = json.loads(done.read_text())
            if row.get('name') != name or row.get('exit_code') != 0 or not (root / name / 'result.json').exists():
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
    return (None, 'remaining=' + ','.join(pending)) if pending else (0, 'DONE all six growth runs')


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
