"""Independently attest completed receipts, rates, failures and immutable sources."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
R = ROOT / 'receipts/part6'
PYTHON = '/home/erikg/emender/.venv/bin/python'
LAYERS = {'22M': 6, '150M': 12, '1.3B': 24}
CORE = ('model.py', 'forward.py', 'backward.py', 'kernels.py',
        'part6_checkpoint.py', 'part6_journal.py', 'scale_surface.py')


def read(path):
    return json.loads(path.read_text())


def check_point(point, *, updates):
    assert point['physical_gpu'] in ('1', '5', '6')
    assert point['python'] == PYTHON
    assert point['peak_allocated'] > 0
    for name in CORE:
        assert point['sources'][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), name
    if point['status'] == 'failed':
        assert 'OutOfMemoryError' in point['error'], point['error']
        return
    assert point['status'] == 'passed'
    assert len(point['seconds']) == updates
    assert all(math.isfinite(t) and t > 0 for t in point['seconds'])
    assert all(math.isfinite(v) for v in point['losses'])
    tokens = point['batch'] * point['context']
    assert point['processed_microbatch_tokens'] == tokens
    assert math.isclose(point['tokens_per_second'], tokens * updates / sum(point['seconds']), rel_tol=1e-12)
    expected = 4 * LAYERS[point['scale']] * point['accumulation_steps'] if point['arm'] == 'pure' else 0
    assert point['replays_per_update'] == [expected] * updates
    assert abs(point['parameters']['total'] / point['pure_parameters']['total'] - 1) < .001


def main():
    summary = read(R / 'summary.json')
    for scale, context, batch, arm in itertools.product(LAYERS, (512, 2048), (64, 128), ('pure', 'transformer')):
        point = read(R / 'surface' / f'{scale}_{arm}_B{batch}_T{context}.json')
        assert point['status'] == 'passed' and point['warmups'] >= 3
        check_point(point, updates=20)
    assert summary['sustained_arms'] == 24 and summary['matched_ratios'] == 12
    for row in summary['surface']:
        assert math.isclose(row['ratio_to_transformer'], row['pure']['tokens_per_second'] / row['transformer']['tokens_per_second'], rel_tol=1e-12)
    baseline = [read(p) for p in (R / 'baseline/surface').glob('*.json')]
    assert len(baseline) == 18 and all(p['status'] == 'failed' and 'OutOfMemoryError' in p['error'] for p in baseline)
    controls = [read(p) for directory in R.glob('memory_*') if directory.is_dir() for p in directory.glob('*.json')]
    assert len(controls) == 58, len(controls)
    for point in controls:
        assert point['warmups'] == 0 and point['measured_updates'] == 2 and point['allocator_expandable']
        check_point(point, updates=2)
    assert summary['surface_status'] == 'passed'
    assert all(v == 'passed' for v in summary['production_parity'].values())
    assert len(summary['checkpoint_byte_parity']) == 8
    assert all(v == 'passed' for v in summary['checkpoint_byte_parity'].values())
    executions = 0
    for prefix, suite in itertools.product(('', 'fused_pool_'), ('x31_tape_cell_fused', 'x36_tape_perf')):
        expected = 109 if suite.startswith('x31') else 70
        nodes = list(ET.parse(R / f'{prefix}{suite}.xml').getroot().iter('testsuite'))
        assert sum(int(n.get('tests', 0)) for n in nodes) == expected
        assert all(int(n.get(key, 0)) == 0 for n in nodes for key in ('failures', 'errors', 'skipped'))
        executions += expected
    assert summary['local_tests']['status'] == 'passed' and summary['local_tests']['passed'] == 15
    assert summary['curve']['status'] == 'passed' and summary['curve']['tokens'] >= 2_000_000
    assert summary['pilots'] and not summary['unqualified_pilots']
    for point in summary['pilots']:
        raw = read(R / (point['label'] + '.json'))
        assert raw['status'] == 'passed' and raw['warmups'] == 3 and len(raw['seconds']) == 20
        assert raw['physical_gpu'] in ('1', '5', '6')
        assert all(math.isfinite(t) and t > 0 for t in raw['seconds'])
        assert len(raw['losses']) == 20 and all(math.isfinite(v) for v in raw['losses'])
        assert math.isclose(raw['tokens_per_second'], raw['batch'] * raw['context'] * 20 / sum(raw['seconds']), rel_tol=1e-12)
        for name in CORE + (('part6_pool_fusion.py',) if raw['fused_pool'] else ()):
            assert raw['sources'][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), name
        curve = read(R / ('matched_' + point['label']) / 'summary.json')
        assert curve['status'] == 'passed' and curve['tokens'] >= 2_000_000 and curve['max_loss_difference'] <= 1e-5
    best = summary['best_sustained_pure']
    expected_gate = 'passed' if best['tokens_per_second'] >= 100000 else 'failed'
    assert summary['status'] == expected_gate
    assert not list(R.rglob('*.pt')) and not list(R.rglob('*.pth'))
    result = dict(status='passed', surface_arms=24, matched_ratios=12,
                  historical_oom_reproductions=18, two_update_controls=len(controls),
                  controls_passed=sum(p['status'] == 'passed' for p in controls),
                  controls_oom=sum(p['status'] == 'failed' for p in controls),
                  strict_byte_gates=8, native_fixture_executions=executions, local_tests=15,
                  immutable_core_sources=list(CORE), pilot_curves=len(summary['pilots']),
                  performance_gate=expected_gate, best_tokens_per_second=best['tokens_per_second'],
                  manual_notes='Rates recomputed from synchronized samples; failed controls verified as actual OOMs. All 24 surface source bindings match current native/checkpoint/journal code. No .pt/.pth artifacts.')
    (R / 'part7_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
