#!/usr/bin/env python3
"""Native X15b completion watcher; proc identities and hash-bound terminal receipts."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
from scripts import watch_invertible_stage15_probes as original

DEFAULT_ROOT = original.DEFAULT_ROOT / 'muon_resume'


def status(root):
    results = [original.arm_status(root.parent / 'gdn2_amendment', 'gdn2_zeroshot'),
               original.arm_status(root, 'muon3x_seed0')]
    code = 1 if any(c == 1 for c, _ in results) else (0 if all(c == 0 for c, _ in results) else None)
    return code, '; '.join(m for _, m in results)


def growth_report(root):
    path = root / 'muon3x_seed0/validation.json'
    if not path.exists():
        path = root.parent / 'muon3x_seed0/validation.json'
    rows = json.loads(path.read_text())
    sig = original.marginal_signature(rows)
    code, message = original.arm_status(root, 'muon3x_seed0')
    last = rows[-1]
    acquired = [(r['step'], r['material_positions']) for r in sig if r['material_positions']]
    final_material = sig[-1]['material_positions']
    complete = code == 0 and last['step'] == 1000000
    lines = ['# X15G growth-report addendum — X15b Muon resume', '', message,
        'Provenance: checkpoint-433000 + verified-diagnostic replay + fresh updates.',
        'Replay agreement is not bitwise full-state proof. Original artifacts preserved.',
        'Concession #9: original 1M ceiling/cosine retained; task text incorrectly said 500K.',
        f"Observed through {last['step']}; fixed-final1M complete: {complete}.",
        f'Any-position >=50/512 escape crossing: {acquired[0] if acquired else "none observed (right-censored until completion)"}.',
        f'Latest material positions: {final_material}.',
        f"Latest CE {last['aggregate']['loss']:.9f}; CE-ln3721={sig[-1]['ce_minus_ln3721']:+.9f}.",
        'Near-marginal CE is NOT proof of no learning; no post-score equivalence tolerance invented.',
        'Full per-position exact-marginal signature in marginal_signature.json.', '']
    if complete:
        lines += [f'Terminal registered escape verdict: {"ESCAPED" if acquired else "NO REGISTERED ESCAPE"}.',
            'Fixed-final retained material positions: ' + str(final_material) + '.',
            'Exact-marginal/optimizer-robust failure is not established by the escape bar alone; inspect all64 counts/CE and the full trajectory.']
    else:
        lines += ['Terminal verdict pending/censored; no 1M conclusion.']
    lines += ['', '|Position|Latest /512|Max /512|First>=50|Latest CE|', '|---|---|---|---|---|']
    for j in range(64):
        counts = [(r['step'], r['positions'][j]['correct']) for r in rows]
        first = next((s for s, c in counts if c >= 50), f"censored@{last['step']}")
        lines.append(f"|P{j+1}|{counts[-1][1]}|{max(c for _, c in counts)}|{first}|{last['positions'][j]['loss']:.9f}|")
    for filename, text in [('X15G-growth-report-addendum.md', '\n'.join(lines) + '\n'),
        ('marginal_signature.json', json.dumps({'reference_ce': math.log(3721), 'rows': sig}, indent=2) + '\n')]:
        target = root / filename
        tmp = target.with_suffix(target.suffix + '.tmp')
        tmp.write_text(text)
        tmp.replace(target)


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
        print(line, flush=True)
        with log.open('a') as stream:
            stream.write(line + '\n')
        if args.report or code is not None:
            growth_report(args.root)
        if code is not None:
            return code
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
