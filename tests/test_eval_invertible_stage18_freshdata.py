"""CPU guards for the X18 generation, novelty replay and frozen eval scope."""
import json

import numpy as np
import pytest

from experiments.expressivity_tasks import eval_invertible_stage18_freshdata as x18


@pytest.mark.parametrize('depth', (16, 64))
def test_x2b_generator_reproducible_independent_scalar_targets(depth):
    task = x18.x12.x10.InvertibleQuadraticTask(61)
    pool = x18.fresh_training_batch(task, 512, depth, np.random.default_rng(123 + depth))
    repeat = task.generate_batch(512, depth + 2, np.random.default_rng(123 + depth))
    assert x18.hashes(pool) == x18.hashes(repeat)
    assert len(x18.row_set(pool[0])) == 512
    x18.validate_pool(pool, depth)
    bad = [a.copy() for a in pool]
    bad[1][0, -1] = (bad[1][0, -1] + 1) % 3721
    with pytest.raises(ValueError, match='arithmetic'):
        x18.validate_pool(bad, depth)


@pytest.mark.parametrize('depth', (16, 64))
@pytest.mark.parametrize('seed', (0, 2))
def test_replay_inputs_exact_original_batch_order_and_rng_state(depth, seed):
    original_rng = np.random.default_rng(seed)
    replay_rng = np.random.default_rng(seed)
    task = x18.x12.x10.InvertibleQuadraticTask(61)
    inputs = np.concatenate([task.generate_batch(32, depth + 2, original_rng)[0] for _ in range(7)])
    replayed = np.concatenate([x18.replay_inputs(replay_rng, depth, 3),
                               x18.replay_inputs(replay_rng, depth, 4)])
    np.testing.assert_array_equal(inputs, replayed)
    assert original_rng.bit_generator.state == replay_rng.bit_generator.state


@pytest.mark.parametrize('depth', (16, 64))
def test_training_overlap_detects_exact_sequence_not_shared_prefix(depth):
    rng = np.random.default_rng(2)
    task = x18.x12.x10.InvertibleQuadraticTask(61)
    batches = [task.generate_batch(32, depth + 2, rng) for _ in range(7)]
    pool = tuple(np.concatenate([b[i] for b in batches]) for i in range(3))
    overlaps, state = x18.training_overlap(pool, 2, depth, 7, chunk=3)
    assert overlaps == 224
    assert state == rng.bit_generator.state
    inputs = pool[0].copy()
    inputs[:, -1] = (inputs[:, -1] + 1) % 61
    overlaps, _ = x18.training_overlap((inputs, pool[1], pool[2]), 2, depth, 7, chunk=2)
    assert overlaps == 0


def test_evaluate_pool_restores_scoped_depth_on_error(monkeypatch):
    original = x18.x12.DEPTH
    def fail(model, pool):
        assert x18.x12.DEPTH == 16
        raise RuntimeError('expected')
    monkeypatch.setattr(x18.x12, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='expected'):
        x18.evaluate_pool(None, (np.zeros((512, 18)),))
    assert x18.x12.DEPTH == original


def test_register_refuses_reselection(tmp_path, monkeypatch):
    monkeypatch.setattr(x18, 'ROOT', tmp_path)
    (tmp_path / 'registration.json').write_text('{}')
    with pytest.raises(FileExistsError, match='frozen'):
        x18.register()


def test_registration_detects_changed_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(x18, 'ROOT', tmp_path)
    source = tmp_path / 'source.py'
    source.write_text('before')
    (tmp_path / 'registration.json').write_text(json.dumps({'source_sha256': {str(source): x18.x7.digest(source)}}))
    assert x18.registration()
    source.write_text('after')
    with pytest.raises(ValueError, match='hash mismatch'):
        x18.registration()


def test_generate_requires_registration_before_any_draw(tmp_path, monkeypatch):
    monkeypatch.setattr(x18, 'ROOT', tmp_path)
    def no_draw(*args):
        pytest.fail('unregistered generator draw')
    monkeypatch.setattr(x18, 'fresh_training_batch', no_draw)
    with pytest.raises(FileNotFoundError):
        x18.generate()


def test_generate_once_hash_bound_with_frozen_overlap_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(x18, 'ROOT', tmp_path)
    reg = {'source_sha256': {}, 'champions': [], 'fresh_seeds': {'16': 12316, '64': 12364}}
    (tmp_path / 'registration.json').write_text(json.dumps(reg))
    x18.generate()
    receipt = json.loads((tmp_path / 'generation.json').read_text())
    assert receipt['registration_sha256'] == x18.x7.digest(tmp_path / 'registration.json')
    for depth in (16, 64):
        info = receipt['pools'][str(depth)]
        assert info['unique_sequences'] == 512
        assert info['scalar_targets_verified'] == depth * 512
        x18.require_hash(info['path'], info['archive_sha256'])
        assert x18.hashes(x18.load_pool(info['path'], depth)) == info['array_sha256']
    with pytest.raises(FileExistsError, match='already frozen'):
        x18.generate()


def test_generate_rejects_existing_frozen_sequences(tmp_path, monkeypatch):
    monkeypatch.setattr(x18, 'ROOT', tmp_path)
    pool = x18.fresh_training_batch(x18.x12.x10.InvertibleQuadraticTask(61), 512, 16,
                                   np.random.default_rng(12316))
    archive = tmp_path / 'old.npz'
    np.savez_compressed(archive, **{f'T16_{i}': a for i, a in enumerate(pool)})
    reg = {'source_sha256': {}, 'fresh_seeds': {'16': 12316, '64': 12364},
           'champions': [{'name': 'old', 'depth': 16, 'frozen_pool': str(archive),
                          'frozen_pool_sha256': x18.x7.digest(archive)}]}
    (tmp_path / 'registration.json').write_text(json.dumps(reg))
    with pytest.raises(ValueError, match='fresh/frozen overlap'):
        x18.generate()
    assert not (tmp_path / 'generation.json').exists()
