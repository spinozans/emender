#!/usr/bin/env python3
"""Supervisor-side W3 arm watcher: no launch, retry, gate read or git mutation."""
import argparse
import json
from pathlib import Path
import time

from scripts.watch_invertible_stage9b_sf_anneal import process_alive
from scripts.w3_matcheddose_preflight import ROOT


def status(root):
    if (root / 'FAIL').exists():
        return 1, 'FAIL W3 arms; inspect *-terminal.json and consoles; no retry authorized'
    if (root / 'DONE').exists():
        for arm in ('A', 'B'):
            terminal = json.loads((root / f'{arm}-terminal.json').read_text())
            receipt = json.loads((root / arm / 'complete-receipt.json').read_text())
            if (terminal['exit_code'] != 0 or receipt['status'] != 'complete'
                    or (receipt['updates'], receipt['inputs'], receipt['targets']) != (512, 3571182, 154386)):
                return 1, 'FAIL invalid W3 completion receipt'
        return 0, 'DONE W3 fixed u512 arms; frozen gate read pending (eight free GPUs required)'
    identity_path = root / 'launcher.json'
    if not identity_path.exists():
        return 1, 'FAIL missing W3 launcher identity; arms not launched'
    if not process_alive(json.loads(identity_path.read_text())):
        return 1, 'FAIL W3 launcher died without terminal marker; inspect durable evidence'
    return None, 'WAIT W3 serial arms; no partial gate read'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'w3-matcheddose/arms')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=300)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('positive interval required')
    while True:
        code, message = status(args.root)
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)
        if code is not None:
            return code
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
