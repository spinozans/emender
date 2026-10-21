#!/usr/bin/env python3
"""Native supervisor-armable X14 watcher: 0 complete, 1 failed, 2 pending (--once)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.watch_invertible_stage10_optimizers import process_alive

NAMES = tuple(f'{arm}_seed{seed}' for seed in (0, 1) for arm in ('e97', 'gdn2'))
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'experiments/expressivity_tasks/results_invertible_stage14_nofreebie_8m'


def arm_status(root, name):
    if (root / 'markers' / f'{name}.FAIL').exists():
        return 1, f'FAIL {name}; inspect receipts/{name}/terminal.json'
    done = root / 'markers' / f'{name}.DONE'
    if done.exists():
        try:
            marker = json.loads(done.read_text())
            terminal = json.loads((root / 'receipts' / name / 'terminal.json').read_text())
            path = root / name / 'result.json'
            result = json.loads(path.read_text())
            valid = (marker == terminal and marker['name'] == name and marker['exit_code'] == 0
                     and marker['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
                     and result['name'] == name and result['status'] == 'complete'
                     and result['total_steps'] == 500000 and result['stop_reason'] == 'safety_ceiling')
        except (OSError, ValueError, KeyError):
            valid = False
        return (0, f'DONE {name}') if valid else (1, 'FAIL invalid X14 terminal receipt')
    for role in ('worker', 'trainer'):
        record = root / 'receipts' / name / f'{role}.json'
        if not record.exists():
            if role == 'worker':
                return 1, 'FAIL missing X14 worker identity'
            continue  # Lease acquisition may still be in progress.
        if not process_alive(json.loads(record.read_text())):
            time.sleep(2)  # Allow worker wait()/terminal publication.
            if done.exists() or (root / 'markers' / f'{name}.FAIL').exists():
                return arm_status(root, name)
            return 1, f'FAIL X14 {role} died without terminal marker'
    return None, f'RUNNING {name}'


def status(root):
    results = [arm_status(root, name) for name in NAMES]
    code = 1 if any(c == 1 for c, _ in results) else (0 if all(c == 0 for c, _ in results) else None)
    return code, '; '.join(message for _, message in results)


def growth_report(root):
    lines = ['# X14 no-freebie growth report', '',
             'P1 is diagnostic-only NON-COMPOSITION; loss and primary aggregate exclude P1.',
             'No X12 E97-T64 conclusion imported; stage10-pattern path, X13 audit remains open.',
             'Final500K primary; transients descriptive. Missing crossings are right-censored, not failures.', '']
    for name in NAMES:
        code, message = arm_status(root, name)
        lines.extend([f'## {name}', message, ''])
        path = root / name / 'validation.json'
        if not path.exists():
            lines.append('No evaluation receipt yet; pending/censored.'); continue
        rows = json.loads(path.read_text())
        last = rows[-1]
        lines.extend([f"Observed through step {last['step']}; fixed-final valid: {code == 0}.",
                      '|Position|Role|Latest /512|Max /512|First50|First512|', '|---|---|---|---|---|---|'])
        for j in range(16):
            counts = [(r['step'], r['positions'][j]['correct']) for r in rows]
            first50 = next((s for s, c in counts if c >= 50), None)
            first512 = next((s for s, c in counts if c == 512), None)
            role = 'diagnostic-only non-composition' if j == 0 else 'composition'
            lines.append(f'|P{j+1}|{role}|{counts[-1][1]}|{max(c for _, c in counts)}|{first50}|{first512}|')
        lines.extend([f"Composition perfect-prefix length from P2: {last['composition_perfect_prefix_length']}", ''])
    path = root / 'GROWTH_REPORT.md'
    tmp = path.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--interval', type=float, default=900)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--report', action='store_true', help='write current growth report (also on terminal wake)')
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
        if args.report or code is not None:
            growth_report(args.root)
        if code is not None:
            return code
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
