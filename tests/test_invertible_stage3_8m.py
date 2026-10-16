"""CPU protocol checks for X3 amendment #4 (no training side effects)."""
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from experiments.expressivity_tasks import run_invertible_stage3_8m as runner


def test_matrix_is_symmetric_complete_and_has_no_duplicate_names():
    rows = runner.matrix()
    assert len(rows) == len({r['name'] for r in rows}) == 36
    assert sum(r['kind'] == 'primary' for r in rows) == 18
    assert sum(r['kind'] == 'difficulty' for r in rows) == 12
    assert sum(r['kind'] == 'one_step' for r in rows) == 6
    assert [{k: v for k, v in r.items() if k not in ('name', 'arm')} for r in rows if r['arm'] == 'e97'] == [
        {k: v for k, v in r.items() if k not in ('name', 'arm')} for r in rows if r['arm'] == 'gdn2']


def test_plateau_calibrated_thresholds_strict_and_eligible_at_10k():
    h = [{'step': s, 'best_accuracy': .2, 'train_loss_window_first250': 10.,
          'train_loss_window_final250': 9.995} for s in range(0, 10250, 250)]
    assert not runner.plateau(h[:-1])
    assert runner.plateau(h)
    h[-1]['train_loss_window_final250'] = 9.99
    assert not runner.plateau(h)  # Exactly0.1% is not stagnant.
    h[-1]['train_loss_window_final250'] = 10.1
    assert runner.plateau(h)
    h[-1]['best_accuracy'] = .201
    assert not runner.plateau(h)  # Exactly0.1pp is not stagnant.
    h[-1]['best_accuracy'] = .2009
    assert runner.plateau(h)
    h[-1]['step'] += 1
    with pytest.raises(ValueError):
        runner.plateau(h)


@pytest.mark.parametrize('drivers', [1, 128])
def test_p61_pools_have_exact_x2_array_and_serialized_hash_identity(tmp_path, drivers):
    pools = runner.frozen_pools(0, drivers, 61)
    prior = runner.x2_pools(0, drivers)
    paths = []
    for i, obj in enumerate([pools, prior]):
        path = tmp_path / f'{i}.npz'
        np.savez_compressed(path, **{f'{name}_{j}': a for name, pool in obj.items() for j, a in enumerate(pool)})
        paths.append(path)
    assert runner.digest(paths[0]) == runner.digest(paths[1])
    for name in pools:
        for a, b in zip(pools[name], prior[name]):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('p', [13, 31])
def test_difficulty_pools_are_frozen_disjoint_and_use_correct_field(p):
    a, b = runner.frozen_pools(0, 128, p), runner.frozen_pools(0, 128, p)
    seen = set()
    for name in a:
        np.testing.assert_array_equal(a[name][0], b[name][0])
        rows = {tuple(r) for r in a[name][0]}
        assert len(rows) == len(a[name][0]) and not rows & seen
        seen.update(rows)
        assert a[name][1].max() < p*p


@pytest.mark.parametrize('p', [13, 31, 61])
def test_field_metrics_and_suffix7_reference(p):
    pool = runner.InvertibleQuadraticTask(p).generate_batch(8, 14, np.random.default_rng(8))
    counts = runner.metric_counts(pool[1].copy(), pool[1], pool[2], p)
    out = runner.finish_metrics(counts, 0., p)
    assert out['accuracy'] == out['x_accuracy'] == out['y_accuracy'] == 1.
    assert out['suffix_floor'] == 1 / p**2
    suffix = runner.suffix7(pool, p)
    assert suffix['count'] == 8*4
    assert suffix['initial_state_used'] is False
    # Changing hidden initial tokens cannot change the suffix-only predictions.
    changed = (pool[0].copy(), pool[1], pool[2])
    changed[0][:, :2] = 0
    assert runner.suffix7(changed, p) == suffix


def test_one_step_suffix_is_explicitly_not_applicable():
    pool = runner.frozen_pools(0, 1, 61)['test_1x']
    assert runner.suffix7(pool, 61)['accuracy'] is None
    assert runner.suffix7(pool, 61)['count'] == 0


def test_pruning_preserves_current_best_and_10k_and_audits_hashes(tmp_path, monkeypatch):
    cps = {}
    for step in [0, 250, 1000, 1250, 10000, 10750]:
        path = tmp_path / f'{step}.pt'
        path.write_bytes(str(step).encode())
        cps[str(step)] = {'path': str(path), 'sha256': runner.digest(path)}
    usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: usage(1000, 900, 100))
    runner.prune(tmp_path, cps, 1250, 10750)
    receipt = json.loads((tmp_path / 'pruning.jsonl').read_text())
    assert receipt['pruned_steps'] == [250, 1000]
    assert set(receipt['retained_hashes']) == {'0', '1250', '10000', '10750'}
    assert all(Path(cps[k]['path']).exists() for k in receipt['retained_hashes'])


def test_real_architecture_types_parameter_counts_and_state_flags():
    e, g = runner.build_model('e97', 61), runner.build_model('gdn2', 61)
    assert sum(p.numel() for p in e.parameters()) == 9900800
    assert sum(p.numel() for p in g.parameters()) == 9887056
    assert e.embed.weight is e.out_proj.weight and g.embed.weight is g.out_proj.weight
    assert e.mlps is not None and g.mlps is not None
    for layer in e.layers:
        assert layer.use_split_edit and not layer.raw_write and layer.state_activation == 'tanh'
    for layer in g.layers:
        assert type(layer).__name__ == 'FLAGatedDeltaNetLayer'
        assert layer.allow_neg_eigval and layer.gdn.allow_neg_eigval
        assert (layer.gdn.num_v_heads, layer.gdn.head_k_dim, layer.gdn.head_v_dim) == (2, 128, 1104)
