"""Source-named X23E task/oracle/protocol and receipt regression tests."""
import json
import itertools

import numpy as np
import pytest

from experiments.expressivity_tasks.tasks import merrill26_separators as task
from experiments.expressivity_tasks import run_merrill26_separators as run
from scripts import watch_merrill26_separators as watcher


def test_sorted_deterministic_unary_graph_and_single_pass_oracle():
    edges = [(0, 2), (1, 3), (2, 4)]
    assert task.reachable(0, edges, 4) == 1
    assert task.reachable(0, edges, 3) == 0
    assert task.reachable(2, [], 2) == 1
    assert task.unary_graph(0, [(0, 2)], 2).tolist() == [7, 2, 3, 4, 1, 1, 5, 1, 1, 6]
    for bad in ([(1, 2), (0, 3)], [(0, 1), (0, 2)], [(2, 1)]):
        with pytest.raises(ValueError):
            task.reachable(0, bad, 3)


def test_all_small_bucket_graphs_match_independent_dfs():
    for n in range(1, 7):
        for bits in itertools.product((0, 1), repeat=n + 1):
            edges = task.bucket_edges(np.asarray(bits))
            graph = dict(edges)
            seen, stack = set(), [0]
            while stack:
                v = stack.pop()
                if v in seen:
                    continue
                seen.add(v)
                if v in graph:
                    stack.append(graph[v])
            assert task.reachable(0, edges, n) == int(n in seen) == int(bits[0] == bits[-1])


def test_integer_product_order_exactness_and_positivity_not_zero():
    a = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)
    b = a.T
    assert not np.array_equal(a @ b, b @ a)
    assert np.array_equal(task.matrix_product(np.stack([a, b])), a.astype(object) @ b)
    huge = task.matrix_product(np.ones((200, 3, 3), dtype=np.int8))
    assert huge[0, 0] == 3**199 and huge[0, 0] > 2**63
    assert task.matrix_product(np.zeros((1, 3, 3), dtype=np.int8))[0, 0] == 0
    with pytest.raises(ValueError):
        task.matrix_product(np.zeros((1, 2, 2)))


def test_mod97_exact_reduction_invertible_inputs_and_nine_token_blocks():
    payload, y = task.sample('mod97', 20, np.random.default_rng(13))
    a = payload.reshape(-1, 3, 3)
    assert all(task.determinant(m) % 97 for m in a)
    assert np.array_equal(task.matrix_product(a, 97), task.matrix_product(a) % 97)
    assert y == task.matrix_product(a, 97)[0, 0]
    assert len(task.encode('mod97', payload)) == 180
    assert np.array_equal(task.encode('integer', payload) - 1, payload)


@pytest.mark.parametrize('name', task.TASKS)
def test_dataset_reproducible_balanced_binary_and_roundtrip(name, tmp_path):
    a = task.make_dataset(name, 12, 1, 7, 231111)
    b = task.make_dataset(name, 12, 1, 7, 231111)
    assert np.array_equal(a.payload, b.payload)
    if name != 'mod97':
        assert np.bincount(a.labels).tolist() == [6, 6]
    for i, n in enumerate(a.lengths):
        payload = a.payload[a.offsets[i]:a.offsets[i + 1]]
        tokens, y = a.example(i)
        if name == 'graph':
            assert y == int(payload[0] == payload[-1])
        elif name == 'control':
            assert y == int(payload.sum()) % 2 and tokens[-1] == 2
        elif name == 'integer':
            assert y == int(task.matrix_product(payload.reshape(-1, 3, 3))[0, 0] > 0)
        assert len(tokens) >= n
    path = tmp_path / 'pool.npz'
    a.save(path)
    restored = task.Dataset.load(name, path)
    assert np.array_equal(restored.example(3)[0], a.example(3)[0])


def test_max_graph_and_recipe_boundaries():
    tokens = task.encode('graph', np.zeros(301, dtype=np.int8))
    assert len(tokens) == 90904
    assert run.learning_rate(0) == 0
    assert run.learning_rate(5000) == .001
    assert run.learning_rate(60000) == pytest.approx(.00001)
    with pytest.raises(ValueError):
        run.learning_rate(60001)
    assert task.SPLITS['train'][:3] == (70000, 1, 100)
    assert [task.SPLITS[k][:3] for k in ('test_id', 'test_ood1', 'test_ood2')] == [
        (10000, 1, 100), (10000, 101, 200), (10000, 201, 300)]


def test_microbatch_budget_preserves_all_examples_and_targets():
    pool = task.make_dataset('control', 64, 1, 100, 236666)
    rows = list(run.batches(pool, range(64), 200))
    assert sum(map(len, rows)) == 64
    assert all(max(len(x) for x, _ in batch) * len(batch) <= 200 for batch in rows)
    assert [y for batch in rows for _, y in batch] == pool.labels.tolist()


def test_censor_receipt_identity_hash_and_marker_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    name = 'e97_graph_seed0'
    watcher.terminal(name, 143)
    assert watcher.rung_status(name) == 'censored'
    result = tmp_path / name / 'result.json'
    row = json.loads(result.read_text())
    assert 'right-censored' in row['scientific_verdict']
    row['status'] = 'complete'
    result.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        watcher.rung_status(name)


@pytest.mark.parametrize('arm', ['e97', 'gdn2'])
def test_only_vocabulary_changes_from_x_series_stock_recurrent_shapes(arm):
    from experiments.expressivity_tasks.run_invertible_stage3_8m import build_model as stock
    model, original = run.build_model(arm, 'graph'), stock(arm, 61)
    assert model.embed.weight is model.out_proj.weight
    shapes = lambda m: {n: tuple(p.shape) for n, p in m.named_parameters() if n != 'embed.weight'}
    assert shapes(model) == shapes(original)
    assert sum(p.numel() for p in model.parameters()) == {'e97': 8950272, 'gdn2': 8936528}[arm]


def test_disk_guard_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'FREE_BYTES', 2**90)
    with pytest.raises(OSError, match='disk guard'):
        run.safety(tmp_path, 0)
