"""X25 five-rung guard; reuse X23E identity, push gate and terminal receipts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from experiments.expressivity_tasks.run_recipe_controlled_separators import ROOT, CARDS, configure
from experiments.expressivity_tasks.run_merrill26_separators import (
    FREE_BYTES, RUN_BYTES, WALL_SECONDS, digest, write_json,
)
from scripts import watch_merrill26_separators as protocol


def configure_watcher():
    configure()
    protocol.ROOT = ROOT


def spawn(name, card):
    if CARDS.get(name) != card:
        raise ValueError('unauthorized X25 card binding')
    receipt = ROOT / 'receipts' / name
    receipt.mkdir(parents=True, exist_ok=False)
    arm, task, _ = name.split('_')
    command = ['bash', 'scripts/launch_recipe_controlled_separators.sh', '--worker', arm, task, str(card)]
    with (receipt / 'worker_console.txt').open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    write_json(receipt / 'worker.json', {**protocol.identity(process.pid), 'card': card,
        'command': command, 'launched_at': time.time()})
    return process


def watch(initial=False, once=False):
    configure_watcher()
    children = []
    if initial:
        if shutil.disk_usage(ROOT).free < FREE_BYTES:
            raise OSError('disk guard: less than 400 GiB free before launch')
        reg = protocol.launch()  # hashes, smoke, HEAD and remote push checked before spawn
        assert set(reg['runs']) == set(CARDS) and reg['cards'] == CARDS
        children = [spawn(name, card) for name, card in CARDS.items()]
        write_json(ROOT / 'receipts' / 'watcher_wiring.json', {
            'watcher': protocol.identity(os.getpid()),
            'registration_sha256': digest(ROOT / 'registration.json'),
            'status_path': str(ROOT / 'receipts' / 'watcher_status.json'),
            'terminal_path': str(ROOT / 'receipts' / 'watcher_terminal.json'),
            'poll_seconds': 60,
            'supervisor_probe': [sys.executable, '-m', 'scripts.watch_recipe_controlled_separators', '--once'],
            'native_arming': 'supervisor-owned; persistent resource guard already running'})
    while True:
        states = {name: protocol.rung_status(name) for name in CARDS}
        free = shutil.disk_usage(ROOT).free
        sizes = {}
        for name, state in states.items():
            size = 0
            for path in (ROOT / name).rglob('*'):
                try:
                    if path.is_file():
                        size += path.stat().st_size
                except FileNotFoundError:
                    pass  # atomic checkpoint rename
            sizes[name] = size
            if state != 'running':
                continue
            record = json.loads((ROOT / 'receipts' / name / 'worker.json').read_text())
            guard = ROOT / 'receipts' / name / 'guard.json'
            if (guard.exists() or free < FREE_BYTES or size > RUN_BYTES
                    or time.time() - record['launched_at'] > WALL_SECONDS):
                if not guard.exists():
                    write_json(guard, {'at': time.time(), 'free_bytes': free, 'rung_bytes': size,
                        'decision': 'infrastructure censoring; TERM then KILL own process group'})
                if protocol.alive(record):
                    elapsed = time.time() - json.loads(guard.read_text())['at']
                    os.killpg(record['pid'], signal.SIGKILL if elapsed >= 30 else signal.SIGTERM)
        children = [child for child in children if child.poll() is None]
        row = {'at': time.time(), 'states': states, 'free_bytes': free, 'rung_bytes': sizes}
        with (ROOT / 'receipts' / 'watcher_polls.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        write_json(ROOT / 'receipts' / 'watcher_status.json', row)
        print(json.dumps(row), flush=True)
        if all(state in ('complete', 'censored', 'error') for state in states.values()):
            write_json(ROOT / 'receipts' / 'watcher_terminal.json', row)
            return int('error' in states.values())
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
    configure_watcher()
    if args.terminal:
        if args.terminal not in CARDS:
            raise ValueError('unauthorized rung')
        protocol.terminal(args.terminal, args.exit_code)
    else:
        raise SystemExit(watch(args.launch, args.once))
