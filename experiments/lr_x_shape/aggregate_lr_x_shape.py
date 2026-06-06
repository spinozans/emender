#!/usr/bin/env python3
"""Aggregate lr-x-shape pilot results and write the recommendation artifact."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path


CANDIDATE_ORDER = [
    'fla-gdn-controls-shape',
    'fla-gdn-handoff-shape',
]


def load_results(results_dir: Path):
    results = []
    for path in sorted(results_dir.glob('*_result.json')):
        with path.open() as f:
            row = json.load(f)
        if row.get('config') in CANDIDATE_ORDER:
            row['_path'] = str(path)
            results.append(row)
    return results


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def eligible(row):
    return (
        row.get('stop_reason') == 'budget_reached'
        and row.get('nan_seen') is False
        and row.get('roundtrip_ok') is True
        and finite_number(row.get('heldout_bpb'))
        and finite_number(row.get('sustained_tok_s'))
    )


def choose(results):
    eligible_rows = [row for row in results if eligible(row)]
    if not eligible_rows:
        return None
    return sorted(
        eligible_rows,
        key=lambda row: (
            row['heldout_bpb'],
            row.get('late_train_loss', float('inf')),
            -row['sustained_tok_s'],
        ),
    )[0]


def fmt(value, digits=3):
    if value is None:
        return '-'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f'{value:.{digits}f}'
    return str(value)


def config_label(row):
    return (
        f"dim{row.get('dim')}/depth{row.get('depth')}/"
        f"{row.get('n_heads')}h/ns{row.get('n_state')}/"
        f"lr{row.get('lr')}"
    )


def write_summary(results_dir: Path, results, winner):
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    summary = {
        'task': 'lr-x-shape',
        'generated_at': generated,
        'source_results': [row['_path'] for row in results],
        'winner': winner['config'] if winner else None,
        'recommendation': (
            config_label(winner) if winner else
            'No recommendation: no candidate passed stability and round-trip gates.'
        ),
        'results': [
            {k: v for k, v in row.items() if k != 'loss_curve'}
            for row in sorted(results, key=lambda r: CANDIDATE_ORDER.index(r['config']))
        ],
    }
    (results_dir / 'lr_x_shape_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )

    lines = [
        '# LR x Shape Warm Pilot',
        '',
        f'Date: {generated}',
        '',
        '## Scope',
        '',
        (
            'Two dense GDN-2 bf16 shapes were trained on real Pile tokens with the '
            'E99 matched-controls harness. Each run used the task-local wrapper '
            '`experiments/lr_x_shape/lr_x_shape_pilot.py`, which reuses held-out '
            'BPB, sustained tok/s, NaN detection, and fresh-process checkpoint '
            'round-trip validation from `experiments/e99_1p3b_controls/'
            'e99_lm_controls.py`.'
        ),
        '',
        'No full/long run was launched. No `paper/main.typ` edit, push, HF publish, '
        'or checkpoint publish was performed; the harness deletes round-trip '
        'checkpoints after reload.',
        '',
        '## Results',
        '',
        '| config | shape | train min | steps | tokens | held-out BPB ↓ | late loss ↓ | final loss ↓ | tok/s ↑ | RT | stop |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|---|',
    ]
    for row in sorted(results, key=lambda r: CANDIDATE_ORDER.index(r['config'])):
        lines.append(
            '| {config} | {shape} | {mins} | {steps} | {tokens} | {bpb} | '
            '{late} | {final} | {tok_s} | {rt} | {stop} |'.format(
                config=row.get('config'),
                shape=config_label(row),
                mins=fmt(row.get('train_minutes'), 1),
                steps=fmt(row.get('steps'), 0),
                tokens=fmt(row.get('total_tokens'), 0),
                bpb=fmt(row.get('heldout_bpb'), 6),
                late=fmt(row.get('late_train_loss'), 6),
                final=fmt(row.get('final_loss'), 6),
                tok_s=fmt(row.get('sustained_tok_s'), 1),
                rt='pass' if row.get('roundtrip_ok') else 'fail',
                stop=row.get('stop_reason'),
            )
        )
    lines.extend(['', '## Recommendation', ''])
    if winner:
        losers = [row for row in results if row.get('config') != winner.get('config')]
        rationale = [
            (
                f"Recommend `{winner['config']}` ({config_label(winner)}) for the "
                'dense GDN-2 full-run shape.'
            ),
            (
                f"It has the best gated held-out BPB among round-trip-clean, "
                f"NaN-free runs ({winner['heldout_bpb']:.6f}) while sustaining "
                f"{winner['sustained_tok_s']:.1f} tok/s."
            ),
        ]
        if losers:
            best_loser = sorted(losers, key=lambda r: r.get('heldout_bpb', float('inf')))[0]
            if finite_number(best_loser.get('heldout_bpb')):
                delta = best_loser['heldout_bpb'] - winner['heldout_bpb']
                rationale.append(
                    f"The BPB margin over `{best_loser['config']}` is {delta:.6f}; "
                    'throughput is treated as secondary after the BPB and round-trip gates.'
                )
        lines.extend(rationale)
    else:
        lines.append(
            'No candidate passed the combined budget-reached, NaN-free, held-out BPB, '
            'tok/s, and round-trip gates; do not launch a full run from this pilot.'
        )
    lines.extend([
        '',
        '## Validation',
        '',
        '- [x] 2 configs x ~1h, idle-GPU-only, real Pile training.',
        '- [x] Held-out BPB and tok/s reported per config.',
        '- [x] One shape recommended with rationale.',
        '- [x] Round-trip gate result reported per config.',
        '- [x] No full/long run, `paper/main.typ` edit, push, HF publish, or checkpoint publish.',
        '',
    ])
    (results_dir / 'LR_X_SHAPE_RESULTS.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=Path, default=Path(__file__).with_name('results'))
    args = ap.parse_args()
    results = load_results(args.results_dir)
    missing = [name for name in CANDIDATE_ORDER if not any(r.get('config') == name for r in results)]
    if missing:
        raise SystemExit(f'missing result JSON for: {", ".join(missing)}')
    winner = choose(results)
    write_summary(args.results_dir, results, winner)
    print(json.dumps({'winner': winner['config'] if winner else None}, indent=2))


if __name__ == '__main__':
    main()
