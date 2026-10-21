"""Native X16 supervisor watcher; monitor ALL arms, write growth each wake, never retry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.invertible_stage16_control import run, valid_result
from scripts.watch_invertible_stage10_optimizers import process_alive


def arm_status(root, name):
    if (root / 'markers' / f'{name}.FAIL').exists():
        return 1, f'FAIL {name}: infrastructure/error censored; inspect terminal.json'
    done = root / 'markers' / f'{name}.DONE'
    if done.exists():
        try:
            marker = json.loads(done.read_text())
            terminal = json.loads((root / 'receipts' / name / 'terminal.json').read_text())
            path = root / name / 'result.json'
            result = json.loads(path.read_text())
            valid = (marker == terminal and marker['name'] == name and marker['exit_code'] == 0
                     and marker['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
                     and valid_result(name, result))
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        return (0, f'TERMINAL {name}: {result["status"]}') if valid else (1, f'FAIL invalid terminal {name}')
    for role in ('worker', 'trainer'):
        path = root / 'receipts' / name / f'{role}.json'
        if not path.exists():
            if role == 'worker':
                return 1, f'FAIL missing worker {name}'
            continue
        if not process_alive(json.loads(path.read_text())):
            time.sleep(2)
            if done.exists() or (root / 'markers' / f'{name}.FAIL').exists():
                return arm_status(root, name)
            return 1, f'FAIL dead {role} {name}; unreceipted termination is censored'
    return None, f'RUNNING {name}'


def status(root):
    results = [arm_status(root, name) for name in run.ARMS]
    # A single failed arm must NOT disarm monitoring of the other four.
    code = None if any(c is None for c, _ in results) else max(c for c, _ in results)
    return code, '; '.join(m for _, m in results)


def first(rows, position, threshold):
    return next((r['phase_step'] for r in rows if r['positions'][position]['correct'] >= threshold), None)


def growth_report(root):
    lines = ['# X16 ignition sweep and curriculum growth', '',
             'Fixed-final500K primary; transient crossings descriptive. None = right-censored at observed phase step.',
             'Historical endpoints are context, not matched causal points; seed0 only, no monotonicity assumed.', '']
    for name, (_, weight, _) in run.ARMS.items():
        code, message = arm_status(root, name)
        lines.extend([f'## {name} (explicit P1 share {weight})', message, ''])
        path = root / name / 'validation.json'
        if not path.exists():
            lines.append('No evaluation yet; pending/censored.'); continue
        history = json.loads(path.read_text())
        for depth in sorted({r['depth'] for r in history}):
            rows = [r for r in history if r['depth'] == depth]
            last = rows[-1]
            ignition = next((r['phase_step'] for r in rows
                             if any(p['correct'] >= 50 for p in r['positions'])), None)
            lines.extend([f'T{depth}: observed through phase update {last["phase_step"]}; any-position first50={ignition}; fixed-final={code == 0 and last["phase_step"] == run.CEILING}.',
                          '|Position|Latest /512|Max /512|First50|First508|First512|', '|---|---|---|---|---|---|'])
            for j in range(depth):
                counts = [r['positions'][j]['correct'] for r in rows]
                lines.append(f'|P{j+1}|{counts[-1]}|{max(counts)}|{first(rows,j,50)}|{first(rows,j,508)}|{first(rows,j,512)}|')
            lines.append('')
        t64 = [r for r in history if r['depth'] == 64]
        transition_path = root / name / 'transition.json'
        if not t64 or not transition_path.exists():
            continue
        transition = json.loads(transition_path.read_text())
        baseline = {p['driver_position'] for p in t64[0]['positions'] if p['correct'] >= 50}
        seen = {p['driver_position'] for r in t64 for p in r['positions'] if p['correct'] >= 50}
        final = {p['driver_position'] for p in t64[-1]['positions'] if p['correct'] >= 50}
        lines.extend([f'Transition teaching update {transition["step"]}; attained LR {transition["entry_lr"]}; taught {transition["taught_positions"]}.',
                      f'T64 zero-shot material positions: {sorted(baseline)}; new transient: {sorted(seen-baseline)}; new latest: {sorted(final-baseline)}.',
                      f'Material frontier baseline={max(baseline, default=0)}, transient={max(seen, default=0)}, latest={max(final, default=0)}.',
                      '### Retention on independent frozen T16 pool',
                      '|Position|Taught|At transition /512|Latest /512|Min /512|Latest drop|First below508 (T64 update)|',
                      '|---|---|---|---|---|---|---|'])
        for j in range(16):
            counts = [r['retention_t16']['positions'][j]['correct'] for r in t64]
            old = transition['t16']['positions'][j]['correct']
            drop = next((r['phase_step'] for r, c in zip(t64, counts) if c < 508), None)
            lines.append(f'|P{j+1}|{j+1 in transition["taught_positions"]}|{old}|{counts[-1]}|{min(counts)}|{old-counts[-1]}|{drop}|')
        lines.append('')
    path = root / 'GROWTH_REPORT.md'
    tmp = path.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=run.ROOT)
    parser.add_argument('--interval', type=float, default=900)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('positive interval required')
    log = args.root / 'receipts/completion_watcher.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        code, message = status(args.root)
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {message}'
        with log.open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)
        growth_report(args.root)
        if code is not None:
            run.x7.write_json(args.root / 'receipts/watcher/terminal.json',
                              {'exit_code': code, 'message': message, 'finished_at': time.time()})
            return code
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
