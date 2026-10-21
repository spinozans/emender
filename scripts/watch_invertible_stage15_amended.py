#!/usr/bin/env python3
"""X15 combined native wake after approved GDN2-only best/final correction."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from scripts import watch_invertible_stage15_probes as original

DEFAULT_ROOT = original.DEFAULT_ROOT


def status(root):
    results = [original.arm_status(root / 'gdn2_amendment', 'gdn2_zeroshot'),
               original.arm_status(root, 'muon3x_seed0')]
    code = 1 if any(c == 1 for c,_ in results) else (0 if all(c == 0 for c,_ in results) else None)
    return code, '; '.join(m for _,m in results)


def growth_report(root):
    original.growth_report(root)  # Original Muon sources/registration remain immutable.
    result_path = root / 'gdn2_amendment/gdn2_zeroshot/result.json'
    lines = ['\n## GDN2 attended amendment1 — aggregate-best482500',
             'Original fail-closed attempt preserved; supervisor task conflated best P5=333 with final P5=318.',
             'Amendment was committed/pushed before T64 evaluation; no tuning or checkpoint change.']
    if result_path.exists():
        result = json.loads(result_path.read_text())
        code, message = original.arm_status(root / 'gdn2_amendment', 'gdn2_zeroshot')
        t16, t64 = result['t16_control']['positions'], result['final']['positions']
        lines += [message, 'Zero training updates; all historical T16 counts reproduced exactly; model state unchanged.',
                  '|Position|T16 champion /512|T64 zero-shot /512|T64 CE|', '|---|---|---|---|']
        for j,p in enumerate(t64):
            control = str(t16[j]['correct']) if j < 16 else 'not trained'
            lines.append(f"|P{j+1}|{control}|{p['correct']}|{p['loss']:.6f}|")
        if code == 0:
            counts = [p['correct'] for p in t64[:4]]
            if counts == [512]*4:
                lines.append('Prediction(a) supported: GDN2 retains four perfect trained positions on T64; shared runtime generalization with E97.')
            else:
                lines.append(f'Prediction(a)/(b): trained P1-P4={counts}, compared with512 each. Report mixed/noncollapse outcomes without forcing a binary verdict.')
            lines.append(f"Prediction(c): near-learned P5=333 -> {t64[4]['correct']}/512 zero-shot; P5+ columns reported, not a new training result.")
    else:
        lines.append('Corrected GDN2 evaluation pending/censored; no runtime verdict.')
    path = root / 'GROWTH_REPORT.md'
    tmp = path.with_suffix('.md.tmp')
    tmp.write_text(path.read_text() + '\n'.join(lines) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--interval', type=float, default=900)
    p.add_argument('--once', action='store_true')
    p.add_argument('--report', action='store_true')
    args = p.parse_args()
    if args.interval <= 0:
        p.error('--interval must be positive')
    log = args.root / 'receipts/amended_completion_watcher.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        code,message = status(args.root)
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
