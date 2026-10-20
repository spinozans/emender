#!/usr/bin/env python3
"""Native supervisor-armable X11 watcher: 0 complete, 1 failed, 2 pending (--once)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.watch_invertible_stage10_optimizers import process_alive

NAME = 'gdn2_adamw_seed0'
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'experiments/expressivity_tasks/results_invertible_stage11_gdn2_adamw_8m'


def status(root):
    if (root / 'markers' / f'{NAME}.FAIL').exists():
        return 1, f'FAIL {NAME}; inspect receipts/{NAME}/terminal.json'
    done = root / 'markers' / f'{NAME}.DONE'
    if done.exists():
        try:
            marker = json.loads(done.read_text())
            terminal = json.loads((root / 'receipts' / NAME / 'terminal.json').read_text())
            path = root / NAME / 'result.json'
            result = json.loads(path.read_text())
            valid = (marker == terminal and marker['name'] == NAME and marker['exit_code'] == 0
                     and marker['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
                     and result['name'] == NAME and result['status'] == 'complete'
                     and result['total_steps'] == 500000 and result['stop_reason'] == 'safety_ceiling')
        except (OSError, ValueError, KeyError):
            valid = False
        return (0, 'DONE X11 stock GDN2 AdamW') if valid else (1, 'FAIL invalid X11 terminal receipt')
    for role in ('worker', 'trainer'):
        record = root / 'receipts' / NAME / f'{role}.json'
        if not record.exists():
            if role == 'worker':
                return 1, 'FAIL missing X11 worker identity'
            continue  # Lease acquisition may still be in progress.
        if not process_alive(json.loads(record.read_text())):
            time.sleep(2)  # Allow worker wait()/terminal publication.
            if done.exists() or (root / 'markers' / f'{NAME}.FAIL').exists():
                return status(root)
            return 1, f'FAIL X11 {role} died without terminal marker'
    return None, f'RUNNING {NAME}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--interval', type=float, default=900)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    log = args.log or args.root / 'receipts/completion_watcher.log'
    log.parent.mkdir(parents=True, exist_ok=True)
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
