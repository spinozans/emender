"""CPU accounting gates for the X5cP single-pass positional evaluation."""
import numpy as np
import pytest

from experiments.expressivity_tasks.eval_invertible_stage5c_positions import position_counts, reconcile
from experiments.expressivity_tasks.run_invertible_stage2 import metric_counts
from experiments.expressivity_tasks.run_invertible_stage3_8m import finish_metrics


def fixture_arrays():
    target = np.full((512, 6), 62, dtype=np.int64)  # (x,y)=(1,1)
    pred = target.copy()
    pred[:, 3] = 63  # x only
    pred[:, 4] = 123  # y only
    pred[:, 5] = 3783  # invalid but modulo-y matches: exclude it nonetheless
    mask = np.ones_like(target, dtype=bool)
    mask[:, :2] = False
    losses = np.broadcast_to(np.arange(1., 5.), (512, 4))
    return pred, target, mask, losses


def test_position_counts_driver_indices_and_invalid_coordinates():
    rows = position_counts(*fixture_arrays())
    assert [r['driver_position'] for r in rows] == [1, 2, 3, 4]
    assert [r['token_index'] for r in rows] == [2, 3, 4, 5]
    assert [r['count'] for r in rows] == [512] * 4
    assert [r['correct'] for r in rows] == [512, 0, 0, 0]
    assert [r['x_correct'] for r in rows] == [512, 512, 0, 0]
    assert [r['y_correct'] for r in rows] == [512, 0, 512, 0]
    assert [r['out_of_range'] for r in rows] == [0, 0, 0, 512]
    assert [r['loss'] for r in rows] == [1, 2, 3, 4]


def test_reconcile_preserves_best_final_differences_without_patching():
    pred, target, mask, losses = fixture_arrays()
    rows = position_counts(pred, target, mask, losses)
    aggregate = finish_metrics(metric_counts(pred, target, mask), losses.sum(), 61)
    final = {**aggregate, 'correct': 511, 'terminal_correct': 1}
    result = reconcile(aggregate, rows, aggregate.copy(), final)
    assert result['best_counts_match']
    assert result['best_loss_matches_at_1e-6']
    assert result['best_minus_final_counts']['correct'] == 1
    assert result['best_minus_final_counts']['terminal_correct'] == -1
    assert result['position_mean_loss_delta'] == 0
    assert final['correct'] == 511
    mismatch = reconcile(aggregate, rows, final, final)
    assert mismatch['best_count_differences'] == {'correct': 1, 'terminal_correct': -1}
    assert not mismatch['best_counts_match']
    rows[0]['correct'] -= 1
    with pytest.raises(ValueError, match='position totals'):
        reconcile(aggregate, rows, final, final)


def test_position_counts_rejects_prefix_or_missing_driver_mask():
    pred, target, mask, losses = fixture_arrays()
    mask[:, 0] = True
    with pytest.raises(ValueError, match='driver targets'):
        position_counts(pred, target, mask, losses)
    with pytest.raises(ValueError, match='shape'):
        position_counts(pred[:1], target, mask, losses)


def test_six_frozen_positional_receipts_and_composition_reading():
    import hashlib
    import json
    from pathlib import Path
    from experiments.expressivity_tasks.eval_invertible_stage5c_positions import ROOT

    expected = {
        'e97_seed0': [512, 10, 2, 0],
        'e97_seed1': [512, 0, 0, 1],
        'e97_seed2': [511, 1, 0, 1],
        'gdn2_seed0': [512, 512, 512, 0],
        'gdn2_seed1': [512, 512, 508, 0],
        'gdn2_seed2': [512, 512, 4, 0],
    }
    data_by_seed = {}
    for name, joints in expected.items():
        receipt = json.loads((ROOT / 'receipts' / name / 'positional_receipts.json').read_text())
        rows = receipt['positions']
        assert [row['correct'] for row in rows] == joints
        assert [row['count'] for row in rows] == [512] * 4
        assert [row['out_of_range'] for row in rows] == [0] * 4
        comparison = receipt['reconciliation']
        assert reconcile(receipt['aggregate'], rows, comparison['recorded_best'],
                         comparison['recorded_final']) == comparison
        assert comparison['best_counts_match']
        assert comparison['best_loss_delta'] == 0
        assert abs(comparison['position_mean_loss_delta']) < 1e-6
        assert receipt['aggregate']['terminal_correct'] == rows[-1]['correct']
        runtime = receipt['runtime']
        assert runtime['deterministic_algorithms']
        assert runtime['forward_batches'] == 16
        for source in ('experiments/expressivity_tasks/eval_invertible_stage5c_positions.py',
                       'experiments/expressivity_tasks/run_invertible_stage3_8m.py',
                       'experiments/expressivity_tasks/run_invertible_stage2.py',
                       'ndm/models/hybrid_ladder.py'):
            recorded_hash = next(value for key, value in receipt['sources'].items()
                                 if key == source or key.endswith('/' + source))
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == recorded_hash
        seed = name[-1]
        data = receipt['data']
        if seed in data_by_seed:
            assert data['sha256'] == data_by_seed[seed]['sha256']
            assert data['T4_array_sha256'] == data_by_seed[seed]['T4_array_sha256']
        data_by_seed[seed] = data
