#!/usr/bin/env python3
"""Monitor the explicit X16R/X17R per-arm admissions; never launch/retry training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from scripts import invertible_stage16_17_resume as recovery
from scripts import invertible_stage16_17_restart as restart
from scripts import watch_invertible_stage16_ignition as w16
from scripts import watch_invertible_stage17_carrywean as w17

run, x17 = recovery.x16, recovery.x17
NAMES, read, first = x17.NAMES, recovery.read_json, w16.first


def active_root(name):
    return restart.root_for(name) if name in restart.BINDINGS else recovery.root_for(name)


def stage_status(stage):
    if stage == 17:
        return w17.status(active_root(NAMES[0]))
    results = [w16.arm_status(active_root(name), name) for name in run.ARMS]
    code = None if any(c is None for c, _ in results) else max(c for c, _ in results)
    return code, '; '.join(m for _, m in results)


def growth16(root):
    lines = ['# X16 ignition sweep and curriculum growth', '',
             'Fixed-final500K primary; transient crossings descriptive. None = right-censored at observed phase step.',
             'Historical endpoints are context, not matched causal points; seed0 only, no monotonicity assumed.', '']
    for name, (_, weight, _) in run.ARMS.items():
        arm_root = active_root(name)
        code, message = w16.arm_status(arm_root, name)
        lines.extend([f'## {name} (explicit P1 share {weight})', message, ''])
        path = arm_root / name / 'validation.json'
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
        transition_path = arm_root / name / 'transition.json'
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


def growth17(root):
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
        path = x17.ROOT.parent / relative / 'validation.json'
        cold = read(path) if path.exists() else []
        first = [next((str(r['step']) for r in cold if r['positions'][0]['correct'] >= t), '—') for t in (50, 512)]
        lines.append(f'| {label} | {cold[-1]["step"] if cold else "—"} | {first[0]} | {first[1]} |')
    if rows and w17.status(root)[0] == 0:
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
    parser.add_argument('--stage', type=int, choices=[16, 17], required=True)
    parser.add_argument('--interval', type=float, default=900)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('positive interval required')
    mod = run if args.stage == 16 else x17
    receipt = mod.ROOT / 'infra_watch'
    recovery.disk_guard()
    receipt.mkdir(exist_ok=True)
    while True:
        code, message = stage_status(args.stage)
        row = {'time': time.time(), 'stage': args.stage, 'status': message, 'exit_code': code}
        recovery.disk_guard()
        with (receipt / 'polls.jsonl').open('a') as log:
            log.write(json.dumps(row) + '\n')
        recovery.disk_guard()
        if args.stage == 16:
            growth16(receipt)
        else:
            recovery.write_text(receipt / 'GROWTH_REPORT.md', growth17(active_root(NAMES[0])))
        print(json.dumps(row), flush=True)
        if code is not None:
            recovery.write_json(receipt / 'terminal.json', row)
            return code
        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
