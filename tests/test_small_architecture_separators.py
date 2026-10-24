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
        watcher.spawn('e97_graph_seed0', 8)


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


@pytest.mark.parametrize('output,expected', [
    ('0', [0]), ('4,5', [4, 5]), ('0,2,3', [0, 2, 3]), ('', []),
    ('0,2,3\n', [0, 2, 3]), ('4, 5\n', [4, 5]), ('  \n', []),
])
def test_queue_parses_gpu_lease_comma_and_whitespace_output(monkeypatch, output, expected):
    states = {name: 'queued' for name in run.CARDS}
    monkeypatch.setattr(watcher.subprocess, 'check_output', lambda *a, **kw: output)
    checked, spawned = [], []
    def verify():
        checked.append(True)
        return {'amendments': [{'queue_rule': watcher.QUEUE_RULE}]}
    monkeypatch.setattr(watcher.protocol, 'verify_registration', verify)
    monkeypatch.setattr(watcher, 'spawn', lambda name, card: spawned.append((name, card)) or name)
    watcher.launch_queued(states)
    assert [card for _, card in spawned] == expected
    assert len(checked) == len(expected)
    assert all(states[name] == 'running' for name, _ in spawned)


def test_queue_prefers_registered_card_then_next_free_without_overlap(monkeypatch):
    states = {'e97_graph_seed0': 'running', 'e97softplus_graph_seed0': 'queued',
              'gdn2_graph_seed0': 'queued'}
    monkeypatch.setattr(watcher.subprocess, 'check_output', lambda *a, **kw: '0,2,3\n')
    monkeypatch.setattr(watcher.protocol, 'verify_registration',
                        lambda: {'amendments': [{'queue_rule': watcher.QUEUE_RULE}]})
    spawned = []
    monkeypatch.setattr(watcher, 'spawn', lambda name, card: spawned.append((name, card)) or name)
    watcher.launch_queued(states)
    assert spawned == [('e97softplus_graph_seed0', 0), ('gdn2_graph_seed0', 2)]
    assert watcher.launch_queued(states) == []
    assert len(spawned) == 2


def test_queue_fallback_requires_registered_amendment(monkeypatch):
    monkeypatch.setattr(watcher.subprocess, 'check_output', lambda *a, **kw: '0,2,3')
    monkeypatch.setattr(watcher.protocol, 'verify_registration',
                        lambda: {'amendments': [{'queue_rule': 'old fixed cards'}]})
    monkeypatch.setattr(watcher, 'spawn', lambda *a: pytest.fail('unregistered spawn'))
    with pytest.raises(ValueError, match='unregistered X27 queue fallback'):
        watcher.launch_queued({name: 'queued' for name in run.CARDS})


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


def mock_launch_gates(tmp_path, monkeypatch, task='X27'):
    reg = {'task': task, 'runs': list(run.CARDS), 'cards': run.CARDS,
           'amendments': [{'queue_rule': watcher.QUEUE_RULE}]}
    (tmp_path / 'registration.json').write_text(json.dumps(reg))
    receipt = tmp_path / 'receipts'
    receipt.mkdir()
    (receipt / 'launch.json').write_text('{"original": true}\n')
    for name in run.CARDS:
        arm, task_name, _ = name.split('_')
        (receipt / f'smoke_{arm}_{task_name}.json').write_text('{"status": "passed"}')
    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    monkeypatch.setattr(watcher.protocol, 'ROOT', tmp_path)
    monkeypatch.setattr(watcher.protocol, 'verify_registration', lambda: reg)
    def output(command, **kwargs):
        if command[:2] == ['git', 'rev-parse']:
            return 'origin/main' if '--abbrev-ref' in command else 'commit123'
        if command[:2] == ['git', 'ls-remote']:
            return 'commit123\trefs/heads/main'
        if command[:2] == ['bash', 'scripts/gpu_lease.sh']:
            return '0,2,3\n'
        pytest.fail(str(command))
    monkeypatch.setattr(watcher.subprocess, 'check_output', output)
    return reg


def test_restart_launches_queued_b_only_and_preserves_a_receipts(tmp_path, monkeypatch):
    mock_launch_gates(tmp_path, monkeypatch)
    name_a, name_b = 'e97_graph_seed0', 'e97softplus_graph_seed0'
    a = tmp_path / 'receipts' / name_a
    a.mkdir()
    (a / 'worker.json').write_text('{"original_arm_a": true}\n')
    before = {p: p.read_bytes() for p in (tmp_path / 'receipts').rglob('*') if p.is_file()}
    monkeypatch.setattr(watcher, 'configure_watcher', lambda: None)
    monkeypatch.setattr(watcher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=watcher.FREE_BYTES))
    monkeypatch.setattr(watcher.protocol, 'rung_status', lambda name:
                        'queued' if name == name_b else 'complete')
    monkeypatch.setattr(watcher.protocol, 'alive', lambda record: True)
    launches = []
    def popen(command, **kwargs):
        launches.append(command)
        return SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(watcher.subprocess, 'Popen', popen)
    monkeypatch.setattr(watcher.protocol, 'identity', lambda pid: {'pid': pid})
    monkeypatch.setattr(watcher.time, 'sleep', lambda _: (_ for _ in ()).throw(StopIteration()))
    with pytest.raises(StopIteration):
        watcher.watch(initial=True)
    assert launches == [['bash', 'scripts/launch_small_architecture_separators.sh',
                         '--worker', 'e97softplus', 'graph', '0']]
    assert all(p.read_bytes() == original for p, original in before.items())
    launch = json.loads((tmp_path / 'receipts' / name_b / 'launch.json').read_text())
    assert launch['card'] == launch['registered_card'] == 0
    assert launch['commit'] == 'commit123'
    with pytest.raises(FileExistsError):
        watcher.spawn(name_b, 0)
    assert len(launches) == 1


def test_other_watchers_keep_one_shot_guard(tmp_path, monkeypatch):
    mock_launch_gates(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='launch already exists'):
        watcher.protocol.launch()


def test_restartable_launch_rejects_other_families(tmp_path, monkeypatch):
    mock_launch_gates(tmp_path, monkeypatch, task='X25')
    with pytest.raises(ValueError, match='X27-only'):
        watcher.protocol.launch(restartable_x27=True)
