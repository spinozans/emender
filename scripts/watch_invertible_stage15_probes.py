#!/usr/bin/env python3
"""Native supervisor-armable X15 watcher: 0 complete, 1 failed, 2 pending (--once)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from scripts.watch_invertible_stage10_optimizers import process_alive

NAMES = ('gdn2_zeroshot', 'muon3x_seed0')
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'experiments/expressivity_tasks/results_invertible_stage15_probes_8m'


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
                     and result['total_steps'] == (0 if name == 'gdn2_zeroshot' else 1000000)
                     and result['stop_reason'] == ('zero_shot_complete' if name == 'gdn2_zeroshot' else 'safety_ceiling'))
        except (OSError, ValueError, KeyError):
            valid = False
        return (0, f'DONE {name}') if valid else (1, 'FAIL invalid X15 terminal receipt')
    for role in ('worker', 'trainer'):
        record = root / 'receipts' / name / f'{role}.json'
        if not record.exists():
            if role == 'worker':
                return 1, 'FAIL missing X15 worker identity'
            continue  # Lease acquisition may still be in progress.
        if not process_alive(json.loads(record.read_text())):
            time.sleep(2)  # Allow worker wait()/terminal publication.
            if done.exists() or (root / 'markers' / f'{name}.FAIL').exists():
                return arm_status(root, name)
            return 1, f'FAIL X15 {role} died without terminal marker'
    return None, f'RUNNING {name}'


def status(root):
    results = [arm_status(root, name) for name in NAMES]
    code = 1 if any(c == 1 for c, _ in results) else (0 if all(c == 0 for c, _ in results) else None)
    return code, '; '.join(message for _, message in results)


def marginal_signature(rows):
    """Descriptive exact-marginal reference, never a statistical no-learning test."""
    return [{'step': r['step'], 'ce': r['aggregate']['loss'],
             'ce_minus_ln3721': r['aggregate']['loss'] - math.log(3721),
             'max_position_correct': max(p['correct'] for p in r['positions']),
             'material_positions': [p['driver_position'] for p in r['positions'] if p['correct'] >= 50],
             'position_correct': [p['correct'] for p in r['positions']],
             'position_ce': [p['loss'] for p in r['positions']]} for r in rows]


def growth_report(root):
    lines = ['# X15 probe growth report', '',
        'Single-seed conditional probes; transients descriptive; fixed-final1M primary for Muon.',
        'Missing crossings right-censored at last observed step, not zero/failure.',
        'Marginal reference CE ln3721=8.221747728; joint chance1/3721; expected0.1376/512 per position.',
        'Near-marginal CE is NOT proof of no learning. Materiality >=50/512; all64 columns carry verdict.', '']
    zero = root / 'gdn2_zeroshot/result.json'
    if zero.exists():
        result = json.loads(zero.read_text())
        code, message = arm_status(root, 'gdn2_zeroshot')
        t16, t64 = result['t16_control']['positions'], result['final']['positions']
        lines += ['## GDN2 zero-shot', message,
            'T16 champion four-perfect plus P5=318; same hash, strict stock X11 loader, zero updates.',
            '|Position|T16 /512|T64 zero-shot /512|T64 CE|', '|---|---|---|---|']
        for j, p in enumerate(t64):
            control = str(t16[j]['correct']) if j < 16 else 'not trained'
            lines.append(f"|P{j+1}|{control}|{p['correct']}|{p['loss']:.6f}|")
        trained = [p['correct'] for p in t64[:4]]
        if code == 0:
            lines.append(f'Prediction(a)/(b) trained P1-P4 runtime readout: {trained}; compare to512 each, without forcing mixed outcomes into a binary verdict.')
            lines.append(f"Prediction(c): P5 T16=318 -> T64={t64[4]['correct']}; P5+ always reported above.")
    path = root / 'muon3x_seed0/validation.json'
    if path.exists():
        rows = json.loads(path.read_text())
        code, message = arm_status(root, 'muon3x_seed0')
        last = rows[-1]
        sig = marginal_signature(rows)
        tmp = root / 'marginal_signature.json.tmp'
        tmp.write_text(json.dumps({'reference_ce': math.log(3721), 'chance': 1/3721,
            'expected_correct_per_position': 512/3721, 'rows': sig}, indent=2) + '\n')
        tmp.replace(root / 'marginal_signature.json')
        lines += ['', '## Muon@3x scratch T64', message,
            f"Observed through{last['step']}; complete fixed-final1M: {code == 0}.",
            '|Position|Latest /512|Max /512|First>=50|First512|Latest CE|', '|---|---|---|---|---|---|']
        for j in range(64):
            counts = [(r['step'], r['positions'][j]['correct']) for r in rows]
            first50 = next((s for s,c in counts if c >= 50), f"censored@{last['step']}")
            first512 = next((s for s,c in counts if c == 512), f"censored@{last['step']}")
            lines.append(f"|P{j+1}|{counts[-1][1]}|{max(c for _,c in counts)}|{first50}|{first512}|{last['positions'][j]['loss']:.6f}|")
        lines += ['', '### Loss trajectory / exact-marginal signature (full500-step series in marginal_signature.json)',
                  '|Step|Validation CE|CE - ln3721|Max position /512|Material positions|', '|---|---|---|---|---|']
        for r in sig:
            if r['step'] in (0, 5000, 50000, 100000, 250000, 500000, 750000, 1000000, last['step']):
                lines.append(f"|{r['step']}|{r['ce']:.9f}|{r['ce_minus_ln3721']:+.9f}|{r['max_position_correct']}|{r['material_positions']}|")
        acquired = [(r['step'], r['material_positions']) for r in sig if r['material_positions']]
        lines.append(f'Prediction(a) material crossing observed: {acquired[0] if acquired else "none, right-censored"}.')
        lines.append('Prediction(b) optimizer-robust horizon failure remains conditional on full1M completion and loss/count signature; near marginal alone is insufficient. AdamW comparison is historical, not a new matched-seed causal trial.')
        training = root / 'muon3x_seed0/training.jsonl'
        if training.exists():
            first, latest = None, None
            with training.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # Last live write can be incomplete.
                    if first is None:
                        first = row
                    latest = row
            if latest:
                lines.append(f"Training loss: update{first['step']}={first['loss']:.9f}; update{latest['step']}={latest['loss']:.9f}; full per-update trajectory in training.jsonl.")
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
