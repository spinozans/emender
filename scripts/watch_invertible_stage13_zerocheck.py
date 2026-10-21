#!/usr/bin/env python3
"""Native X13 watcher: 0 terminal success, 1 failure, 2 pending (--once)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.watch_invertible_stage10_optimizers import process_alive

NAMES = ('e97_bf16_seed1', 'e97_bf16_seed2')
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'experiments/expressivity_tasks/results_invertible_stage13_zerocheck_8m'


def read(path):
    return json.loads(path.read_text())


def status(root):
    pending = []
    for name in NAMES:
        receipt = root / 'receipts' / name
        if (root / 'markers' / f'{name}.FAIL').exists():
            return 1, f'FAIL {name}; infrastructure censoring, no automatic retry'
        done = root / 'markers' / f'{name}.DONE'
        if done.exists():
            try:
                marker = read(done)
                path = root / name / 'result.json'
                result = read(path)
                valid = (marker == read(receipt / 'terminal.json') and marker['name'] == name
                         and marker['exit_code'] == 0
                         and marker['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
                         and result['name'] == name and result['status'] == 'complete'
                         and result['total_steps'] == 1000000 and result['stop_reason'] == 'safety_ceiling'
                         and result['final']['step'] == 1000000
                         and len(result['final']['positions']) == 64)
            except (OSError, ValueError, KeyError):
                valid = False
            if not valid:
                return 1, f'FAIL invalid terminal receipt {name}'
            continue
        for role in ('worker', 'trainer'):
            record = receipt / f'{role}.json'
            if not record.exists():
                if role == 'worker':
                    detached = receipt / 'detached.json'
                    if not detached.exists() or not process_alive(read(detached)['worker']):
                        return 1, f'FAIL missing worker {name}'
                continue
            if not process_alive(read(record)):
                time.sleep(2)
                if done.exists() or (root / 'markers' / f'{name}.FAIL').exists():
                    return status(root)
                return 1, f'FAIL {name} {role} died without terminal marker'
        pending.append(name)
    return (None, 'RUNNING ' + ','.join(pending)) if pending else (0, 'DONE X13 seeds1/2 matched1M')


def growth_report(root):
    histories = []
    lines = ['# X13 seed replication growth receipt', '',
             'Counts correct/512. Maxima can occur at different updates; no mechanism claims.',
             'Marginal escape: ANY P1..P64 >=50 anywhere. Composition: ANY P2+ >=50 anywhere.',
             'Partial zero histories are right-censored, not completed replications.', '']
    for name in NAMES:
        path = root / name / 'validation.json'
        rows = read(path) if path.exists() else []
        histories.append(rows)
        if not rows:
            lines.append(f'- {name}: not started.')
            continue
        maximum = max(p['correct'] for row in rows for p in row['positions'])
        composed = max(p['correct'] for row in rows for p in row['positions'][1:])
        lines.append(f'- {name}: through {rows[-1]["step"]}; latest perfect prefix {rows[-1]["perfect_prefix_depth"]}; '
                     f'maximum single-validation prefix {max(r["perfect_prefix_depth"] for r in rows)}; '
                     f'marginal escape observed={maximum >= 50}; composition observed={composed >= 50}.')
    lines += ['', '| Position | seed1 latest | seed2 latest | seed1 max | seed2 max | seed1 first50 / first512 | seed2 first50 / first512 |',
              '|---|---:|---:|---:|---:|---|---|']
    for j in range(64):
        latest, maxima, timing = [], [], []
        for rows in histories:
            latest.append(str(rows[-1]['positions'][j]['correct']) if rows else '—')
            maxima.append(str(max(r['positions'][j]['correct'] for r in rows)) if rows else '—')
            timing.append(' / '.join(next((str(r['step']) for r in rows if r['positions'][j]['correct'] >= t), '—')
                                     for t in (50, 512)))
        lines.append(f'| P{j+1} | {latest[0]} | {latest[1]} | {maxima[0]} | {maxima[1]} | {timing[0]} | {timing[1]} |')
    escaped = any(p['correct'] >= 50 for rows in histories for row in rows for p in row['positions'])
    if escaped:
        lines += ['', 'Registered marginal-escape criterion met: seed0 learned-nothing zero was lottery; re-scope. P1-only escape is not composition.']
    elif all(histories) and status(root)[0] == 0:
        lines += ['', 'Neither replication seed reaches50 at any position: learned-nothing zero replicates as3/3 under this finite recipe/ceiling.']
    else:
        lines += ['', 'Replication verdict PENDING; no completed-zero claim from partial histories.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--interval', type=float, default=900)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    log = args.root / 'receipts/completion_watcher.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        code, message = status(args.root)
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {message}'
        with log.open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)
        if args.report or code == 0:
            (args.root / 'GROWTH_REPORT.md').write_text(growth_report(args.root))
        if code is not None:
            return code
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
