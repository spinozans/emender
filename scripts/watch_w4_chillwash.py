#!/usr/bin/env python3
"""Detached W4 observer; records occupancy/liveness, never launches or repeats work."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.w3_matcheddose_preflight import ROOT
from scripts.watch_w3_gate_window import process_fields, snapshot, belongs_to


def status(root):
    terminal = root / 'terminal.json'
    if terminal.exists():
        code = json.loads(terminal.read_text())['exit_code']
        return code, 'COMPLETE exploratory observations' if code == 0 else 'STOPPED inspect receipts; no retry'
    identity = json.loads((root / 'launcher.json').read_text())
    try:
        if process_fields(identity['pid'])[1] != str(identity['starttime']):
            return 1, 'STOPPED launcher identity is no longer live'
    except FileNotFoundError:
        return 1, 'STOPPED launcher exited without terminal receipt'
    return None, 'WAIT fixed W4 pipeline; no partial behavioral read'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'w4-chillwash')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('positive interval required')
    root = args.root
    fields = process_fields(os.getpid())
    with (root / 'watcher.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), ppid=fields[0], starttime=fields[1], interval=args.interval), stream)
    identity = json.loads((root / 'launcher.json').read_text())
    with (root / 'occupancy.jsonl').open('x', buffering=1) as log:
        while True:
            state = snapshot()
            selected = next(row[1] for row in state['gpus'] if row[0] == identity['physical_gpu'])
            state['foreign_on_selected_gpu'] = [p for p in state['processes'] if p['uuid'] == selected
                and not belongs_to(p['pid'], identity['pid'], str(identity['starttime']))]
            log.write(json.dumps(state, sort_keys=True) + '\n')
            code, message = status(root)
            print(state['time'], message, flush=True)
            if code is not None:
                return code
            time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
