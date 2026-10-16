"""Source-named checks for X3 completeness, paired grid, and validation-only LR."""
import json

import pytest

from experiments.expressivity_tasks.aggregate_invertible_stage3_8m import aggregate, build
from experiments.expressivity_tasks.run_invertible_stage3_8m import LRS, matrix, write_json


def fake_result(row):
    # Deliberately opposite validation and test LR rankings.
    val = {1e-3: .10, 3e-4: .20, 1e-4: .15}[row['lr']]
    acc = {1e-3: .8, 3e-4: .4, 1e-4: .6}[row['lr']] + row['seed']*.01
    metric = {'accuracy': acc, 'x_accuracy': .9, 'y_accuracy': .5,
              'post_suffix7_accuracy': acc, 'loss': 1.}
    panels = {'test_1x': metric}
    if row['drivers'] == 128:
        panels.update({'test_4x': metric, 'test_8x': metric})
    return {'best_validation': {'accuracy': val, 'loss': 2., 'step': 5000},
            'step': 5000, 'stop_step': 10000, 'stop_reason': 'plateau',
            'test': panels, 'suffix7': {k: {'accuracy': None if row['drivers']==1 else 1/row['p']**2} for k in panels},
            'trainable_parameters': 9900800, 'total_training_tokens': 41600000,
            'training_tokens_per_second': 1., 'wall_seconds': 1.}


def test_missing_matrix_fails_closed_and_does_not_impute(tmp_path):
    report = tmp_path / 'report.md'
    assert not build(tmp_path, report)
    s = json.loads((tmp_path / 'results_summary.json').read_text())
    assert s['status'] == 'incomplete_fail_closed'
    assert len(s['runs']) == len(s['errors']) == 36
    assert s['best_lrs_validation_only'] == s['grid'] == s['matched_lr'] == {}
    assert all(r['status'] == 'unavailable' for r in s['runs'].values())
    assert 'support no final architecture conclusion' in report.read_text()


def test_all_grid_lrs_and_seeds_reported_and_lr_choice_never_uses_test(tmp_path):
    for row in matrix():
        write_json(tmp_path / row['name'] / 'result.json', fake_result(row))
    report = tmp_path / 'report.md'
    assert build(tmp_path, report, audit=False)
    s = json.loads((tmp_path / 'results_summary.json').read_text())
    assert s['status'] == 'complete_unaudited'  # Synthetic fixtures never count as receipts.
    assert s['best_lrs_validation_only'] == {'e97': 3e-4, 'gdn2': 3e-4}
    assert len(s['grid']) == 6 and len(s['matched_lr']) == 3
    for arm in ('e97', 'gdn2'):
        for lr in LRS:
            stats = s['grid'][f'{arm}_{lr:g}']['test_1x']['accuracy']
            assert stats['n'] == 3
            assert stats['sample_sd'] == pytest.approx(.01)
            assert stats['sem'] == pytest.approx(.01/3**.5)
    for contrast in s['matched_lr'].values():
        assert contrast['per_seed'] == [0., 0., 0.]
    text = report.read_text()
    assert all(row['name'] in text for row in matrix())
    assert 'Both grid readings' in text and 'Matched LR=' in text


def test_partial_seed_cannot_pick_architecture_lr(tmp_path):
    for row in matrix():
        if row['kind'] == 'primary' and row['arm'] == 'e97' and row['seed'] == 2 and row['lr'] == 1e-3:
            continue
        write_json(tmp_path / row['name'] / 'result.json', fake_result(row))
    assert not build(tmp_path, tmp_path / 'report.md', audit=False)
    s = json.loads((tmp_path / 'results_summary.json').read_text())
    assert 'e97' not in s['best_lrs_validation_only']
    assert 'e97_0.001' not in s['grid'] and '0.001' not in s['matched_lr']


def test_sample_uncertainty_is_not_population_sd():
    stats = aggregate([1., 2., 3.])
    assert stats['mean'] == 2. and stats['sample_sd'] == 1.
    assert stats['sem'] == pytest.approx(1/3**.5)
