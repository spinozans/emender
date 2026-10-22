"""Read-only X21E artifact audit; checks receipts without rerunning evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def audit(root):
    reg = load(root / 'registration.json')
    reg_sha = digest(root / 'registration.json')
    gen = load(root / 'generation.json')
    gen_sha = digest(root / 'generation.json')
    assert gen['registration_sha256'] == reg_sha
    for path, sha in reg['source_sha256'].items():
        assert digest(path) == sha, path
    assert digest(root / 'REGISTRATION.md') == reg['predictions_sha256']
    pools = {}
    for depth, info in gen['pools'].items():
        assert digest(info['path']) == info['archive_sha256']
        with np.load(info['path']) as archive:
            x, target, mask = (archive[f'T{depth}_{i}'] for i in range(3))
            pools[depth] = (x, target, mask)
            assert [hashlib.sha256(a.tobytes()).hexdigest() for a in (x, target, mask)] == info['array_sha256']
            assert len({row.tobytes() for row in x}) == 512
            assert not mask[:, :2].any() and mask[:, 2:].all()
            a, b = x[:, 0] - 61, x[:, 1] - 122
            for i in range(int(depth)):
                a, b = b, (a + b*b + x[:, i + 2]) % 61
                np.testing.assert_array_equal(target[:, i + 2], 61*a + b)
    counts = {'champions': 0, 'rungs': 0, 'control_batches': 0, 'evaluation_batches': 0,
              'infrastructure_censored': 0, 'training_updates': 0}
    for r in reg['champions']:
        if not r['usable']:
            assert r['unusable_reason']
            continue
        counts['champions'] += 1
        for path in (r['checkpoint']['path'], r['snapshot_path']):
            assert digest(path) == r['checkpoint']['sha256']
        assert digest(r['receipt_source']) == r['receipt_source_sha256']
        assert digest(r['frozen_pool']) == r['frozen_pool_sha256']
        control = root / 'controls' / r['name']
        receipt = load(control / 'result.json')
        assert receipt['integer_counts_exact'] is True
        for a, b in zip(receipt['metrics']['positions'], r['frozen_metrics']['positions']):
            assert all(a[k] == b[k] for k in ('correct', 'count', 'x_correct', 'y_correct', 'out_of_range'))
        control_batches = sorted(control.glob('batch_*.json'))
        assert len(control_batches) == 16
        counts['control_batches'] += len(control_batches)
        for precision in reg['precisions']:
            assert load(root / f'runtime_{precision}.json')['training_updates'] == 0
            for m in reg['multipliers']:
                directory = root / 'evaluations' / r['name'] / precision / f'{m:02d}x'
                result = load(directory / 'result.json')
                assert result['registration_sha256'] == reg_sha
                assert result['generation_sha256'] == gen_sha
                assert result['champion_sha256'] == r['checkpoint']['sha256']
                assert result['training_updates'] == 0
                counts['rungs'] += 1
                if result['status'] != 'complete':
                    counts['infrastructure_censored'] += 1
                    assert result['error']
                    continue
                assert result['weights_unchanged'] is True
                assert result['depth'] == r['depth'] * m
                info = gen['pools'][str(result['depth'])]
                assert result['pool_archive_sha256'] == info['archive_sha256']
                metrics = result['metrics']
                positions, agg = metrics['positions'], metrics['aggregate']
                assert len(positions) == result['depth']
                assert all(p['count'] == 512 for p in positions)
                for key in ('correct', 'count', 'x_correct', 'y_correct', 'out_of_range'):
                    assert sum(p[key] for p in positions) == agg[key]
                assert positions[-1]['correct'] == agg['terminal_correct']
                assert sum(p['correct'] for p in positions[8:]) == agg['post_suffix7_correct']
                assert abs(sum(p['loss'] for p in positions) / len(positions) - agg['loss']) < 1.e-10
                batches = sorted(directory.glob('batch_*.json'))
                assert len(batches) == 16
                assert {p.name: digest(p) for p in batches} == result['batch_receipt_sha256']
                counts['evaluation_batches'] += len(batches)
                rows = [load(p) for p in batches]
                for i, row in enumerate(rows):
                    assert row['first_example'] == 32*i and row['examples'] == 32
                    assert row['status'] == 'complete' and row['training_updates'] == 0
                    assert row['registration_sha256'] == reg_sha
                    assert row['generation_sha256'] == gen_sha
                    assert row['champion_sha256'] == r['checkpoint']['sha256']
                    assert row['pool_archive_sha256'] == info['archive_sha256']
                    hashes = [hashlib.sha256(a[32*i:32*(i+1)].tobytes()).hexdigest()
                              for a in pools[str(result['depth'])]]
                    assert row['input_target_mask_sha256'] == hashes
                for key in ('correct', 'count', 'x_correct', 'y_correct', 'out_of_range',
                            'terminal_correct', 'terminal_count', 'post_suffix7_correct', 'post_suffix7_count'):
                    assert sum(b['aggregate'][key] for b in rows) == agg[key]
                assert abs(sum(b['aggregate']['loss'] * b['aggregate']['count'] for b in rows) /
                           agg['count'] - agg['loss']) < 1.e-6
    assert counts['rungs'] == counts['champions'] * 6 * 6
    summary = load(root / 'summary.json')
    assert summary['registration_sha256'] == reg_sha
    for r in reg['champions']:
        if not r['usable']:
            continue
        base = load(root / 'evaluations' / r['name'] / 'native' / '01x' / 'result.json')['metrics']['positions']
        base_survivors = {p['driver_position'] for p in base if p['correct'] >= 480}
        for precision in reg['precisions']:
            first_loss = None
            for m in reg['multipliers']:
                result = load(root / 'evaluations' / r['name'] / precision / f'{m:02d}x' / 'result.json')
                if result['status'] != 'complete':
                    continue
                ps = result['metrics']['positions']
                survivors = [p['driver_position'] for p in ps if p['correct'] >= 480]
                curve = summary['curves'][r['name']][precision][str(m)]
                assert curve['surviving_columns'] == survivors
                assert curve['accuracy'] == result['metrics']['aggregate']['accuracy']
                if first_loss is None and any(ps[p-1]['correct'] < 480 for p in base_survivors):
                    first_loss = m
                if precision == 'native':
                    for row in summary['alignment'][r['name']][str(m)]:
                        p = row['position']
                        start, suffix = p in base_survivors, p - (len(ps) - len(base)) in base_survivors
                        expected = 'ambiguous' if start and suffix else 'start-anchored' if start else 'suffix-relative' if suffix else 'unclassified'
                        assert row['classification'] == expected
            failure = summary['horizon_at_failure'][r['name']][precision]
            assert failure['status'] == ('right-censored' if first_loss is None else 'already-failed-1x' if first_loss == 1 else 'observed-failure')
            if first_loss:
                assert failure['multiplier'] == first_loss
    print(json.dumps({'status': 'PASS', **counts, 'registration_sha256': reg_sha,
                      'generation_sha256': gen_sha}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/expressivity_tasks/results_stage21e_horizonladder'))
    audit(parser.parse_args().root)
