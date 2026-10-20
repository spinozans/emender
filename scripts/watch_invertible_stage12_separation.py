#!/usr/bin/env python3
"""Native X12 watcher: 0 complete, 1 failed, 2 pending (--once); CPU-only growth tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.watch_invertible_stage10_optimizers import process_alive

NAMES = ('e97_bf16_seed0', 'gdn2_bf16_seed0')
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'experiments/expressivity_tasks/results_invertible_stage12_separation_8m'


def read(path):
    return json.loads(path.read_text())


def status(root):
    scheduler = root / 'receipts/scheduler'
    terminal = scheduler / 'terminal.json'
    if not terminal.exists():
        record = scheduler / 'worker.json'
        if not record.exists():
            record = scheduler / 'detached.json'
        if not record.exists() or not process_alive(read(record)):
            return 1, 'FAIL post-W3 scheduler missing/dead; no automatic retry'
        return None, 'WAIT post-W3 admission; no GPU leases until committed W3 gate completion'
    if read(terminal)['status'] != 'launched':
        return 1, 'FAIL post-W3 scheduler; inspect receipts/scheduler/terminal.json'
    pending = []
    for name in NAMES:
        if (root / 'markers' / f'{name}.FAIL').exists():
            return 1, f'FAIL {name}; infrastructure censoring, not architecture evidence'
        done = root / 'markers' / f'{name}.DONE'
        if done.exists():
            try:
                marker = read(done)
                path = root / name / 'result.json'
                result = read(path)
                valid = (marker == read(root / 'receipts' / name / 'terminal.json')
                         and marker['name'] == name and marker['exit_code'] == 0
                         and marker['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
                         and result['name'] == name and result['status'] == 'complete'
                         and result['total_steps'] == 1000000 and result['stop_reason'] == 'safety_ceiling'
                         and len(result['final']['positions']) == 64)
            except (OSError, ValueError, KeyError):
                valid = False
            if not valid:
                return 1, f'FAIL invalid terminal receipt {name}'
            continue
        for role in ('worker', 'trainer'):
            record = root / 'receipts' / name / f'{role}.json'
            if not record.exists():
                if role == 'worker':
                    detached = root / 'receipts' / name / 'detached.json'
                    if not detached.exists() or not process_alive(read(detached)['worker']):
                        return 1, f'FAIL missing worker {name}'
                continue
            if not process_alive(read(record)):
                time.sleep(2)
                if done.exists() or (root / 'markers' / f'{name}.FAIL').exists():
                    return status(root)
                return 1, f'FAIL {name} {role} died without terminal marker'
        pending.append(name)
    return (None, 'RUNNING ' + ','.join(pending)) if pending else (0, 'DONE X12 T64 matched1M')


def growth_report(root):
    """Two full positional columns; never select a checkpoint from validation maxima."""
    histories = []
    for name in NAMES:
        path = root / name / 'validation.json'
        histories.append(read(path) if path.exists() else [])
    lines = ['# X12 T64 growth report', '', 'Counts are correct/512; maxima may occur at different updates.',
             'Only fixed-final 1M perfect prefixes decide the registered comparison; partial reads are censored.', '']
    for name, rows in zip(NAMES, histories):
        if rows:
            lines.append(f'- {name}: latest update {rows[-1]["step"]}; latest perfect prefix {rows[-1]["perfect_prefix_depth"]}; maximum single-validation prefix {max(r["perfect_prefix_depth"] for r in rows)}.')
        else:
            lines.append(f'- {name}: not started; waiting for post-W3 admission.')
    lines += ['', '| Position | E97 latest | GDN2 latest | E97 max | GDN2 max | E97 first50 / first512 | GDN2 first50 / first512 |',
              '|---|---:|---:|---:|---:|---|---|']
    for j in range(64):
        values, maxima, timing = [], [], []
        for rows in histories:
            values.append(str(rows[-1]['positions'][j]['correct']) if rows else '—')
            maxima.append(str(max(r['positions'][j]['correct'] for r in rows)) if rows else '—')
            times = [next((str(r['step']) for r in rows if r['positions'][j]['correct'] >= threshold), '—') for threshold in (50, 512)]
            timing.append(' / '.join(times))
        lines.append(f'| P{j + 1} | {values[0]} | {values[1]} | {maxima[0]} | {maxima[1]} | {timing[0]} | {timing[1]} |')
    lines += ['', 'P5 first50 is material acquisition; first512 is perfect acquisition; absent times are right-censored.',
              'Competing (a)/(d): affine-carry separation versus layer-count cap; both models have four layers.',
              'Reconcile the earlier four-perfect plus budding-P5 observation with whichever survives.']
    complete = all((root / name / 'result.json').exists() and rows and rows[-1]['step'] == 1000000
                   for name, rows in zip(NAMES, histories))
    if complete and status(root)[0] == 0:
        e, g = [rows[-1]['perfect_prefix_depth'] for rows in histories]
        if g >= e + 2:
            lines.append('(a) GDN2 exceeds E97 by at least two perfect-prefix positions at the registered ceiling.')
        elif e == g:
            lines.append('(b) PARITY-AT-DEPTH: both reach the same prefix depth at T64/1M - then the architectures are equivalent on this task family at 8M under working recipes, and the composition-depth framing was always a training-stack artifact.')
        else:
            lines.append('The fixed-final outcome is neither the registered >=2 GDN2 separation nor exact parity; report it without forcing a category.')
        if all(max(r['positions'][4]['correct'] for r in rows) < 50 for rows in histories):
            lines.append('(c) Neither materially acquired P5: the ceiling remains binding; separation needs longer ceilings or curriculum staging under the working recipe.')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--interval', type=float, default=900)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--report', action='store_true', help='write current side-by-side growth table, including partial/censored state')
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
