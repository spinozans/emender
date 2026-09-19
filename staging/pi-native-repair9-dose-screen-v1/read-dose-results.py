#!/usr/bin/env python3
"""Collect the dose-screen probe readings into one table (STAGED; run by the
orchestrator after the three probes are evaluated).

Reads, per dose probe dir:
  stage-b/summary.json            -> valid_first_frames / 14, correct_first_actions / 14
  execution-slice/summary.json    -> per-model successes (dose-probe-y/x out of 24;
                                     bridge controls = same-run window reference)
and prints them next to the bracketing evidence (v2-prep v10 u256 gate;
v3-prep u256 gate) and the frozen gate thresholds. Evidence only; no verdicts.
"""
import json
from pathlib import Path

P = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')
PROBES = {d: P / f'pi-native-repair9-dose-screen-probe-{d}-v1' for d in ('d25', 'd50', 'd75')}


def stage_b(probe):
    s = json.loads((probe / 'stage-b' / 'summary.json').read_text())
    return s['valid_first_frames'], s['correct_first_actions'], s['checkpoint_sha256'][:12]


def execution(probe):
    s = json.loads((probe / 'execution-slice' / 'summary.json').read_text())
    out = {}
    for name, m in s['models'].items():
        out[name] = (m['successes'], m['episodes'])
    return out


def main():
    rows = []
    for d, probe in PROBES.items():
        try:
            v, c, ck = stage_b(probe)
            ex = execution(probe)
            rows.append((d, ck, v, c, ex.get('dose-probe-y'), ex.get('dose-probe-x'),
                         ex.get('bridge-y-control'), ex.get('bridge-x-control')))
        except FileNotFoundError as e:
            print(f'pending: {d} ({e.filename} missing)')
    print()
    print('dose  | ckpt        | Stage-B valid/correct (of 14) | slice y (of 24) | slice x (of 24) | bridge-y | bridge-x')
    for d, ck, v, c, y, x, by, bx in rows:
        print(f'{d:5} | {ck} | {v:2d} valid / {c:2d} correct          | {y[0]:2d}/{y[1]:<2d}          | {x[0]:2d}/{x[1]:<2d}          | {by[0]}/{by[1]} | {bx[0]}/{bx[1]}')
    print()
    print('Bracketing evidence (frozen 96-case panel dual gates):')
    print('  v2 prep (v10 arc u256; restored pools at 0%):   Stage-B 9/14 valid 6/14 correct; execution 0/96')
    print('  v3 prep (v3 arc u256; restored pools at 100%):  Stage-B 4/14 valid 4/14 correct; execution 59/60 of 96 (bridge control 67/96)')
    print('Frozen gate thresholds (unchanged): Stage-B >= 12/14 valid AND >= 10/14 correct; execution >= 64/96 (full panel).')
    print('The 24-case slice is a screen: slice successes are NOT gate readings; the frozen')
    print('execution gate remains bound to the full 96-case panel.')


if __name__ == '__main__':
    main()
