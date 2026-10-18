"""CPU gates for X6D variable-depth accounting and immutable checkpoint copies."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.expressivity_tasks import eval_invertible_stage6_positions as diagnostic
from experiments.expressivity_tasks.run_invertible_stage2 import metric_counts
from experiments.expressivity_tasks.run_invertible_stage3_8m import finish_metrics


@pytest.mark.parametrize('depth', [1, 2, 4, 8, 16])
def test_position_counts_and_reconciliation_at_every_registered_depth(depth):
    target = np.full((512, depth + 2), 62, dtype=np.int64)
    pred = target.copy()
    mask = np.ones_like(pred, dtype=bool)
    mask[:, :2] = False
    if depth >= 4:
        pred[:, 3] = 63  # x only
        pred[:, 4] = 123  # y only
        pred[:, 5] = 3783  # modulo y matches, but invalid
    losses = np.broadcast_to(np.arange(1., depth + 1), (512, depth))
    rows = diagnostic.position_counts(pred, target, mask, losses)
    assert [r['driver_position'] for r in rows] == list(range(1, depth + 1))
    assert [r['token_index'] for r in rows] == list(range(2, depth + 2))
    assert all(r['count'] == 512 for r in rows)
    if depth >= 4:
        assert [r['correct'] for r in rows[:4]] == [512, 0, 0, 0]
        assert [r['x_correct'] for r in rows[:4]] == [512, 512, 0, 0]
        assert [r['y_correct'] for r in rows[:4]] == [512, 0, 512, 0]
        assert [r['out_of_range'] for r in rows[:4]] == [0, 0, 0, 512]
    aggregate = finish_metrics(metric_counts(pred, target, mask), losses.sum(), 61)
    comparison = diagnostic.reconcile(aggregate, rows, aggregate)
    assert comparison['recorded_counts_match']
    assert comparison['recorded_loss_delta'] == comparison['position_mean_loss_delta'] == 0
    recorded = {**aggregate, 'correct': aggregate['correct'] - 1}
    assert diagnostic.reconcile(aggregate, rows, recorded)['recorded_count_differences'] == {'correct': 1}
    assert recorded['correct'] == aggregate['correct'] - 1
    rows[0]['correct'] -= 1
    with pytest.raises(ValueError, match='position totals'):
        diagnostic.reconcile(aggregate, rows, recorded)


def test_position_counts_rejects_scored_prefix_and_bad_shape():
    x = np.zeros((512, 3), dtype=np.int64)
    mask = np.ones_like(x, dtype=bool)
    with pytest.raises(ValueError, match='driver targets'):
        diagnostic.position_counts(x, x, mask, np.ones((512, 1)))
    with pytest.raises(ValueError, match='512'):
        diagnostic.position_counts(x[:1], x, mask, np.ones((512, 1)))


def test_safe_copy_hashes_before_after_and_rejects_live_destination(tmp_path):
    live, copy = tmp_path / 'live.pt', tmp_path / 'scratch/copy.pt'
    live.write_bytes(b'unchanging retained checkpoint')
    expected = diagnostic.digest(live)
    diagnostic.safe_copy(live, copy, expected)
    assert copy.read_bytes() == live.read_bytes()
    with pytest.raises(ValueError, match='overwrite source'):
        diagnostic.safe_copy(live, live, expected)
    with pytest.raises(ValueError, match='pre-copy hash'):
        diagnostic.safe_copy(live, copy, 'bad')
    assert live.read_bytes() == b'unchanging retained checkpoint'


def test_capture_lines_excludes_partial_live_append(tmp_path):
    path = tmp_path / 'transitions.jsonl'
    path.write_bytes(b'{"event": "entry"}\n{"event":')
    lines, sha = diagnostic.capture_lines(path)
    assert lines == [{'event': 'entry'}]
    assert sha == hashlib.sha256(b'{"event": "entry"}\n').hexdigest()


def test_registered_receipts_are_hashed_and_reconcile_without_gpu():
    evidence = diagnostic.EVIDENCE
    summary = diagnostic.read_json(evidence / 'evaluation_summary.json')
    inventory = diagnostic.read_json(evidence / 'inventory.json')
    assert diagnostic.digest(evidence / 'registration.md') == inventory['registration_sha256']
    assert set(inventory['runs']) == set(diagnostic.NAMES)
    assert summary['runtime']['co_resident'] and not summary['runtime']['exclusive_lease']
    receipts = {}
    for item in summary['receipts']:
        path = Path(item['path'])
        assert diagnostic.digest(path) == item['sha256']
        receipt = diagnostic.read_json(path)
        comparison = diagnostic.reconcile(receipt['aggregate'], receipt['positions'], receipt['recorded'])
        assert comparison == receipt['reconciliation']
        assert comparison['recorded_counts_match'] and abs(comparison['recorded_loss_delta']) <= 1e-6
        assert all(r['count'] == 512 and r['out_of_range'] == 0 for r in receipt['positions'])
        # Per-position fp64 means versus canonical fp32 batch reductions.
        assert np.isclose(comparison['position_mean_loss_delta'], 0.,
                          atol=1e-6 + 1e-7 * receipt['aggregate']['loss'], rtol=0.)
        assert receipt['carry_probability_at_evaluation'] == 0
        for source, sha in receipt['sources'].items():
            assert diagnostic.digest(source) == sha
        receipts[str(path)] = receipt
    for alias in summary['aliases']:
        receipt = receipts[alias['receipt']]
        comparison = diagnostic.reconcile(receipt['aggregate'], receipt['positions'], alias['recorded'])
        assert comparison == alias['reconciliation']
        assert comparison['recorded_counts_match'] and abs(comparison['recorded_loss_delta']) <= 1e-6
    for item in inventory['runs'].values():
        path = Path(item['metadata'])
        assert diagnostic.digest(path) == item['sha256']
        meta = diagnostic.read_json(path)
        for stage in diagnostic.DEPTHS:
            assert any(j['role'] == 'best' and j['stage_depth'] == stage for j in meta['jobs'])
            assert any(j['role'] == 'exit' and j['stage_depth'] == stage for j in meta['jobs'])
        for missing in meta['missing_entry_checkpoints']:
            assert missing['pruning_events']
        for boundary in meta['boundaries']:
            assert boundary['recorded_metrics_exactly_equal']
