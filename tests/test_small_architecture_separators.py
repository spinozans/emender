"""X27 counts, size-only protocol, zero-update gates and queued resource guard."""
import json
import signal
from types import SimpleNamespace

import pytest
import torch

from experiments.expressivity_tasks import run_small_architecture_separators as run
from scripts import watch_small_architecture_separators as watcher


def test_depth_one_counts_and_effective_shapes():
    for arm, expected in [('e97', 570464), ('e97softplus', 570464), ('gdn2', 563218)]:
        model = run.build_model(arm, 'graph')
        assert model.depth == len(model.layers) == 1
        assert model.vocab_size == 8 and model.mlp_ratio == 2.
        assert sum(p.numel() for p in model.parameters()) == expected
        assert 200000 <= expected <= 800000
        layer = model.layers[0]
        if arm != 'gdn2':
            assert model.dim == 192 and model.mlp_hidden_dim == 384
            assert (layer.n_heads, layer.n_state) == (16, 16)
            assert layer.recurrent_state_precision == 'legacy' and layer.fused_inference
            assert layer.use_split_edit and layer.use_triton and not layer.linear_state
            assert layer.state_activation == ('tanh' if arm == 'e97' else 'softplus')
        else:
            assert model.dim == 128 and model.mlp_hidden_dim == 256
            assert (layer.num_heads, layer.actual_head_dim, layer.expansion) == (1, 128, 8.625)
            assert layer.allow_neg_eigval and layer.gdn.use_short_conv
            assert layer.gdn.v_proj.out_features == 1104


def test_activation_pair_has_identical_seed_zero_tensors():
    torch.manual_seed(0)
    tanh = run.build_model('e97', 'graph').state_dict()
    torch.manual_seed(0)
    softplus = run.build_model('e97softplus', 'graph').state_dict()
    assert tanh.keys() == softplus.keys()
    assert all(torch.equal(v, softplus[k]) for k, v in tanh.items())


def test_x25_recipe_and_loop_unchanged(monkeypatch):
    for attr in ('ROOT', 'build_model', 'learning_rate', 'WEIGHT_DECAY'):
        monkeypatch.setattr(run.protocol, attr, getattr(run.protocol, attr))
    forward, batches, evaluate = run.protocol.forward_rows, run.protocol.batches, run.protocol.evaluate
    run.configure()
    assert run.protocol.ROOT == run.ROOT and run.protocol.build_model is run.build_model
    assert (run.protocol.BATCH, run.protocol.CEILING, run.protocol.EVAL_EVERY) == (64, 60000, 500)
    assert run.protocol.WEIGHT_DECAY == 1e-4
    assert [run.protocol.learning_rate(i) for i in (0, 1, 5000, 60000)] == [1e-4] * 4
    assert (run.protocol.forward_rows, run.protocol.batches, run.protocol.evaluate) == (forward, batches, evaluate)


def test_graph_only_authorization_and_fixed_queue():
    assert run.CARDS == {'e97_graph_seed0': 6, 'e97softplus_graph_seed0': 0, 'gdn2_graph_seed0': 1}
    for arm, task in [('e97', 'integer'), ('gdn2', 'mod97'), ('rnn', 'graph')]:
        with pytest.raises(ValueError):
            run.build_model(arm, task)
    with pytest.raises(ValueError, match='card binding'):
        watcher.spawn('e97_graph_seed0', 0)


def test_shared_dataset_link_never_creates_a_replica(tmp_path, monkeypatch):
    root, comparison = tmp_path / 'x27', tmp_path / 'x23e'
    root.mkdir(); (comparison / 'datasets').mkdir(parents=True)
    monkeypatch.setattr(run, 'ROOT', root)
    monkeypatch.setattr(run, 'COMPARISON_ROOT', comparison)
    run.link_datasets(); run.link_datasets()
    assert (root / 'datasets').resolve() == (comparison / 'datasets').resolve()
    (root / 'datasets').unlink(); (root / 'datasets').mkdir()
    with pytest.raises(ValueError, match='X23E dataset'):
        run.link_datasets()


def test_queue_uses_only_bound_idle_cards_and_rechecks_registration(monkeypatch):
    states = {name: 'queued' for name in run.CARDS}
    monkeypatch.setattr(watcher.subprocess, 'check_output', lambda *a, **kw: '2\n6\n')
    checked, spawned = [], []
    monkeypatch.setattr(watcher.protocol, 'verify_registration', lambda: checked.append(True))
    monkeypatch.setattr(watcher, 'spawn', lambda name, card: spawned.append((name, card)) or name)
    assert watcher.launch_queued(states) == ['e97_graph_seed0']
    assert spawned == [('e97_graph_seed0', 6)] and checked == [True]
    assert states == {'e97_graph_seed0': 'running', 'e97softplus_graph_seed0': 'queued', 'gdn2_graph_seed0': 'queued'}
    assert watcher.launch_queued(states) == []  # never relaunch a running arm
    assert len(checked) == 1


def test_queue_fails_closed_if_registration_changed(monkeypatch):
    monkeypatch.setattr(watcher.subprocess, 'check_output', lambda *a, **kw: '6\n')
    def bad_registration():
        raise ValueError('registered bytes changed')
    monkeypatch.setattr(watcher.protocol, 'verify_registration', bad_registration)
    monkeypatch.setattr(watcher, 'spawn', lambda *a: pytest.fail('spawn before registered-byte gate'))
    with pytest.raises(ValueError, match='registered bytes changed'):
        watcher.launch_queued({name: 'queued' for name in run.CARDS})


def test_initial_disk_guard_precedes_push_and_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    monkeypatch.setattr(watcher, 'configure_watcher', lambda: None)
    monkeypatch.setattr(watcher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(watcher.protocol, 'launch', lambda: pytest.fail('mutation before disk guard'))
    with pytest.raises(OSError, match='disk guard'):
        watcher.watch(initial=True)


@pytest.mark.parametrize('alive', [False, True])
@pytest.mark.parametrize('elapsed', [0, 31])
def test_guard_identity_and_term_kill_escalation(tmp_path, monkeypatch, alive, elapsed):
    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    monkeypatch.setattr(watcher, 'configure_watcher', lambda: None)
    monkeypatch.setattr(watcher.protocol, 'rung_status', lambda name: 'running')
    monkeypatch.setattr(watcher.protocol, 'alive', lambda record: alive)
    monkeypatch.setattr(watcher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(watcher.time, 'time', lambda: 1000)
    for name in run.CARDS:
        receipt = tmp_path / 'receipts' / name; receipt.mkdir(parents=True)
        (receipt / 'worker.json').write_text(json.dumps({'pid': 123, 'launched_at': 900}))
        if elapsed:
            (receipt / 'guard.json').write_text(json.dumps({'at': 1000 - elapsed}))
    signals = []
    monkeypatch.setattr(watcher.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    assert watcher.watch(once=True) == 2
    assert signals == ([(123, signal.SIGKILL if elapsed else signal.SIGTERM)] * 3 if alive else [])


def test_terminal_hash_and_marker_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher.protocol, 'ROOT', tmp_path)
    name = 'e97softplus_graph_seed0'
    watcher.protocol.terminal(name, 143)
    assert watcher.protocol.rung_status(name) == 'censored'
    (tmp_path / name / 'result.json').write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        watcher.protocol.rung_status(name)
