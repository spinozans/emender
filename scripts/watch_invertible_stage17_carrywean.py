#!/usr/bin/env python3
"""Native X17 watcher: 0 complete, 1 failed, 2 pending (--once); CPU-only growth tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.watch_invertible_stage10_optimizers import process_alive

NAMES = ('e97_carrywean_seed0',)
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'experiments/expressivity_tasks/results_invertible_stage17_carrywean_8m'


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
            return 1, 'FAIL card-wait scheduler missing/dead; no automatic retry'
        return None, 'WAIT first free card; no GPU lease while waiting'
    if read(terminal)['status'] != 'launched':
        return 1, 'FAIL card-wait scheduler; inspect receipts/scheduler/terminal.json'
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
                         and result['final']['step'] == 1000000
                         and result['final']['carry_probability'] == 0.
                         and result['final']['eval_carry_probability'] == 0.
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
    return (None, 'RUNNING ' + ','.join(pending)) if pending else (0, 'DONE X17 T64 global wean1M')


def growth_report(root):
    """All observations are neutral evaluation, even when training p is high."""
    path = root / NAMES[0] / 'validation.json'
    rows = read(path) if path.exists() else []
    lines = ['# X17 carry-supervised T64 growth receipt', '',
             'All evaluation slots ALWAYS neutral; p below is the training schedule, NOT eval feedback.',
             'Counts correct/512. Maxima from different updates are not a jointly acquired prefix.',
             'Missing crossings and partial histories are right-censored; no completed1M claim before hash-bound DONE.', '']
    if rows:
        last = rows[-1]
        lines.append(f'Latest update {last["step"]}; training p={last["carry_probability"]:.6f}; '
                     f'perfect prefix {last["perfect_prefix_depth"]}; '
                     f'maximum single-validation prefix {max(r["perfect_prefix_depth"] for r in rows)}.')
        maximum = max(p['correct'] for row in rows for p in row['positions'][1:])
        final_max = max(p['correct'] for p in last['positions'][1:])
        lines.append(f'P2+ material composition ever observed={maximum >= 50} (max {maximum}); '
                     f'latest material composition={final_max >= 50} (max {final_max}).')
    else:
        lines.append('Not started; awaiting first broker-grantable card. Predictions unresolved.')
    lines += ['', '| Position | Latest | Max | Peak step / p | First50 step / p | First512 step / p |',
              '|---|---:|---:|---|---|---|']
    for j in range(64):
        def stamp(row):
            return f'{row["step"]} / {row["carry_probability"]:.6f}' if row else '—'
        peak = max(rows, key=lambda row: row['positions'][j]['correct']) if rows else None
        first = [next((r for r in rows if r['positions'][j]['correct'] >= t), None) for t in (50, 512)]
        latest = rows[-1]['positions'][j]['correct'] if rows else '—'
        maximum = peak['positions'][j]['correct'] if peak else '—'
        lines.append(f'| P{j+1} | {latest} | {maximum} | {stamp(peak)} | {stamp(first[0])} | {stamp(first[1])} |')
    lines += ['', '## Registered interpretations',
              '(a) WEAN-TRANSFER requires distinguishing transient P2+ >=50 from survival at the zero-crutch ceiling.',
              '(b) CRUTCH-DEPENDENT-AT-DEPTH: inspect peak step/p versus latest/final counts per position for decay; no causal diagnosis from partial histories.',
              '(c) P1 first50/first512 reported above, whether or not composition ignites.', '',
              '### Cold P1 timing context (historical, not controlled comparisons)',
              '| Cold run | Observed through | P1 first50 | P1 first512 |', '|---|---:|---:|---:|']
    for label, relative in (
        ('X12 AdamW E97 T64 seed0', 'results_invertible_stage12_separation_8m/e97_bf16_seed0'),
        ('X10 Muon E97 T16 seed0 (different horizon)', 'results_invertible_stage10_optimizers_8m/muon_seed0'),
        ('X15 Muon3x E97 T16 seed0 (different horizon/LR)', 'results_invertible_stage15_probes_8m/muon3x_seed0')):
        path = root.parent / relative / 'validation.json'
        cold = read(path) if path.exists() else []
        first = [next((str(r['step']) for r in cold if r['positions'][0]['correct'] >= t), '—') for t in (50, 512)]
        lines.append(f'| {label} | {cold[-1]["step"] if cold else "—"} | {first[0]} | {first[1]} |')
    if rows and status(root)[0] == 0:
        survived = any(p['correct'] >= 50 for p in rows[-1]['positions'][1:])
        ever = any(p['correct'] >= 50 for row in rows for p in row['positions'][1:])
        lines += ['', f'Completed1M, p=0: material P2+ survives={survived}; ever observed={ever}.',
                  'If ever=true but survives=false, the material composition did not survive the registered wean. '
                  'If both false, composition never ignited: neither successful transfer nor observed crutch-dependence.']
    else:
        lines += ['', 'Zero-crutch fixed-final verdict PENDING (or infrastructure-censored on FAIL).']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--interval', type=float, default=900)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--report', action='store_true', help='write full current positional table, including partial/censored state')
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
        if args.report or code is not None:
            (args.root / 'GROWTH_REPORT.md').write_text(growth_report(args.root))
        if code is not None:
            return code
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
