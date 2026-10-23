"""X25 paper-recipe, fixed rung mapping, shared pools and guard regressions."""
import json
import signal
from types import SimpleNamespace

import pytest
import torch

from experiments.expressivity_tasks import run_recipe_controlled_separators as run
from scripts import watch_recipe_controlled_separators as watcher


def test_paper_recipe_is_constant_and_hot_recipe_defaults_are_unchanged(monkeypatch):
    assert run.protocol.WEIGHT_DECAY == .01
    assert run.protocol.learning_rate(5000) == .001
    for attr in ('ROOT', 'build_model', 'learning_rate', 'WEIGHT_DECAY'):
        monkeypatch.setattr(run.protocol, attr, getattr(run.protocol, attr))
    run.configure()
    assert run.protocol.WEIGHT_DECAY == 1e-4
    assert [run.protocol.learning_rate(i) for i in (0, 1, 5000, 60000)] == [1e-4] * 4
    for i in (-1, 60001):
        with pytest.raises(ValueError):
            run.learning_rate(i)
    assert (run.protocol.BATCH, run.protocol.CEILING, run.protocol.EVAL_EVERY) == (64, 60000, 500)


def test_five_rungs_only_and_initial_weights_match_hot_controls():
    assert run.CARDS == {'e97_graph_seed0': 0, 'e97softplus_graph_seed0': 7,
                         'gdn2_graph_seed0': 1, 'e97_integer_seed0': 2, 'gdn2_integer_seed0': 3}
    for arm in ('e97', 'e97softplus'):
        torch.manual_seed(0)
        model = run.build_model(arm, 'graph')
        torch.manual_seed(0)
        stock = run.softplus.STOCK_BUILD_MODEL('e97', 'graph')
        assert all(torch.equal(v, stock.state_dict()[k]) for k, v in model.state_dict().items())
        assert all(layer.state_activation == ('softplus' if arm == 'e97softplus' else 'tanh')
                   and layer.recurrent_state_precision == 'legacy' and layer.fused_inference
                   for layer in model.layers)
    for arm, task in (('e97', 'mod97'), ('e97softplus', 'integer'), ('gdn2', 'control')):
        with pytest.raises(ValueError):
            run.build_model(arm, task)
    with pytest.raises(ValueError, match='card binding'):
        watcher.spawn('e97_graph_seed0', 7)


def test_shared_dataset_link_is_actual_x23e_pool(tmp_path, monkeypatch):
    root, comparison = tmp_path / 'x25', tmp_path / 'x23e'
    root.mkdir(); (comparison / 'datasets').mkdir(parents=True)
    monkeypatch.setattr(run, 'ROOT', root)
    monkeypatch.setattr(run, 'COMPARISON_ROOT', comparison)
    run.link_datasets(); run.link_datasets()
    assert (root / 'datasets').resolve() == (comparison / 'datasets').resolve()
    (root / 'datasets').unlink(); (root / 'datasets').mkdir()
    with pytest.raises(ValueError, match='X23E dataset'):
        run.link_datasets()


def test_launch_disk_guard_precedes_push_gate_and_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    monkeypatch.setattr(watcher, 'configure_watcher', lambda: None)
    monkeypatch.setattr(watcher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(watcher.protocol, 'launch', lambda: pytest.fail('mutation before disk guard'))
    with pytest.raises(OSError, match='disk guard'):
        watcher.watch(initial=True)


@pytest.mark.parametrize('alive', [False, True])
@pytest.mark.parametrize('elapsed', [0, 31])
def test_guard_checks_identity_and_escalates_only_owned_groups(tmp_path, monkeypatch, alive, elapsed):
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
    expected = signal.SIGKILL if elapsed else signal.SIGTERM
    assert signals == ([(123, expected)] * 5 if alive else [])


def test_terminal_receipts_remain_hash_and_marker_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher.protocol, 'ROOT', tmp_path)
    name = 'e97softplus_graph_seed0'
    watcher.protocol.terminal(name, 143)
    assert watcher.protocol.rung_status(name) == 'censored'
    result = tmp_path / name / 'result.json'
    result.write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        watcher.protocol.rung_status(name)
