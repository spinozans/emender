"""CPU-only X12 recipe, T64 counters, post-W3 admission and native watcher tests."""
import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import run_invertible_stage12_separation_8m as r
from scripts import invertible_stage12_control as c
from scripts import watch_invertible_stage12_separation as w


def test_matched_four_layer_stock_models_and_all_parameter_adamw():
    for arm in r.ARMS:
        torch.manual_seed(0)
        model = r.build_model(arm)
        assert len(model.layers) == 4
        assert sum(p.numel() for p in model.parameters()) == r.PARAMETERS[arm]
        opt = r.make_optimizer(model)
        assert type(opt) is torch.optim.AdamW and not opt.state
        group = opt.param_groups[0]
        assert group['betas'] == (.9, .999) and group['eps'] == 1e-8
        assert group['lr'] == 0. and group['weight_decay'] == .01
        assert {id(p) for p in group['params']} == {id(p) for p in model.parameters()}
        if arm == 'gdn2':
            assert any(getattr(p, '_no_weight_decay', False) for p in group['params'])
            for layer in model.layers:
                assert layer.gdn.mode == 'chunk'
                assert layer.gdn.forward.__func__.__globals__['chunk_gated_delta_rule'] is not r.x7.fp32_gdn_recurrence
        else:
            assert all(layer.recurrent_state_precision == 'legacy' for layer in model.layers)
            assert all(layer.fused_inference for layer in model.layers)


def test_one_million_schedule_not_inherited_500k_schedule():
    for step, expected in [(0, 0.), (1, 2e-7), (5000, .001), (502500, .000505), (1000000, 1e-5)]:
        assert r.learning_rate(step) == pytest.approx(expected)
    assert r.learning_rate(500000) > .0005
    for step in (-1, 1000001):
        with pytest.raises(ValueError):
            r.learning_rate(step)


def test_uniform_update_lr_logged(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    inputs = torch.tensor([[[1., 2.], [3., 1.]], [[2., 1.], [1., 0.]]])
    y = torch.tensor([[0, 1], [1, 0]])
    mask = torch.tensor([[True, False], [True, True]])
    expected = torch.nn.functional.cross_entropy(model(inputs)[mask], y[mask]).item()
    monkeypatch.setattr(r.x7, 'forward_batch', lambda model, _: (model(inputs), y, mask))
    row = r.train_update(model, r.make_optimizer(model), None, 500001)
    assert row['loss'] == pytest.approx(expected)
    assert row['lr'] == r.learning_rate(500001) and math.isfinite(row['grad_norm'])


def pool():
    x = np.zeros((512, 66), dtype=np.int64)
    mask = np.ones_like(x, dtype=bool)
    mask[:, :2] = False
    return x, x.copy(), mask


def test_actual_all_64_position_evaluation_and_reconciliation(monkeypatch):
    def forward(model, batch):
        logits = torch.zeros((32, 66, 2))
        return logits, torch.from_numpy(batch[1]), torch.from_numpy(batch[2])
    monkeypatch.setattr(r.x7, 'forward_batch', forward)
    result = r.evaluate(torch.nn.Identity(), pool())
    assert result['perfect_prefix_depth'] == 64
    assert len(result['positions']) == 64
    assert [p['driver_position'] for p in result['positions']] == list(range(1, 65))
    assert all(p['correct'] == p['count'] == 512 for p in result['positions'])
    assert result['aggregate']['correct'] == 64 * 512
    assert result['aggregate']['loss'] == pytest.approx(math.log(2))


def test_runner_no_early_stop_t64_frozen_pool_lr_and_bounded_checkpoints(tmp_path, monkeypatch):
    name = r.NAMES[0]
    p = pool()
    registration = {'authorized_runs': [name], 'pool_array_sha256': {'0': [hashlib.sha256(a.tobytes()).hexdigest() for a in p]}}
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    r.x7.write_json(tmp_path / 'registration.json', registration)
    r.x7.write_json(tmp_path / 'receipts' / name / 'admission.json',
                    {'registration_sha256': r.x7.digest(tmp_path / 'registration.json'), 'gpu': 3})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: registration)
    monkeypatch.setattr(r.x7, 'frozen_pools', lambda *args: {'val': p})
    monkeypatch.setattr(r, 'CEILING', 3)
    monkeypatch.setattr(r, 'EVAL_EVERY', 1)
    model = torch.nn.Linear(1, 2, bias=False)
    model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    monkeypatch.setattr(r, 'build_model', lambda arm: model)
    monkeypatch.setattr(r, 'PARAMETERS', {'e97': 2})
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    lengths = []
    def forward(model, batch):
        lengths.append(batch[0].shape[1])
        return model.weight[:, 0].expand(32, 66, 2), torch.from_numpy(batch[1]) * 0, torch.from_numpy(batch[2])
    monkeypatch.setattr(r.x7, 'forward_batch', forward)
    monkeypatch.setattr(r, 'evaluate', lambda *_: {'aggregate': {'accuracy': 1., 'loss': 1.},
        'positions': [{'driver_position': j, 'correct': 512, 'count': 512} for j in range(1, 65)], 'perfect_prefix_depth': 64})
    r.run('e97')
    assert lengths == [66, 66, 66]  # Perfect cannot stop the run.
    out = tmp_path / name
    result = json.loads((out / 'result.json').read_text())
    assert result['total_steps'] == 3 and result['stop_reason'] == 'safety_ceiling'
    rows = json.loads((out / 'validation.json').read_text())
    assert [row['step'] for row in rows] == [0, 1, 2, 3]
    assert all(len(row['positions']) == 64 for row in rows)
    assert len(list(out.glob('*.pt'))) <= 3
    checkpoint = torch.load(rows[-1]['checkpoint']['path'], weights_only=False)
    assert checkpoint['optimizer_family'] == 'adamw'
    training = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    assert [row['lr'] for row in training] == [r.learning_rate(s) for s in (1, 2, 3)]


def mock_gate_git(monkeypatch, report, receipt):
    def git(*args):
        if args[0] == 'show':
            return report if args[1].endswith(':W3-report.md') else json.dumps(receipt)
        if args[0] == 'ls-tree':
            return 'receipts/w3-matcheddose/gate-verdict.json\n'
        return 'deadbeef\n'
    monkeypatch.setattr(c, 'git', git)


@pytest.mark.parametrize('report,receipt,complete', [
    ('## Gate verdict\nfinished', {'schema': 'emender-w3-verdict-v1', 'status': 'measurements-complete'}, True),
    ('## Pending\n', {'schema': 'emender-w3-verdict-v1', 'status': 'measurements-complete'}, False),
    ('## Gate verdict\n', {'status': 'launched'}, False),
    ('## Gate verdict\n', {'schema': 'emender-w3-verdict-v1', 'status': 'failed'}, False),
])
def test_remote_report_plus_terminal_receipt_required(monkeypatch, report, receipt, complete):
    mock_gate_git(monkeypatch, report, receipt)
    assert (c.completed_gate() is not None) == complete


def test_scheduler_polls_without_any_broker_call_before_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    (tmp_path / 'receipts/scheduler').mkdir(parents=True)
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: {})
    monkeypatch.setattr(c, 'git', lambda *args: '')
    monkeypatch.setattr(c, 'completed_gate', lambda: None)
    def no_broker(*args, **kwargs):
        raise AssertionError('no GPU broker calls before gate completion')
    monkeypatch.setattr(c.subprocess, 'check_output', no_broker)
    monkeypatch.setattr(c.time, 'sleep', lambda _: (_ for _ in ()).throw(SystemExit('end test wait')))
    assert c.schedule(1) == 1
    row = json.loads((tmp_path / 'receipts/scheduler/polls.jsonl').read_text())
    assert row['phase'] == 'WAIT_GATE' and row['gate'] is None


def test_worker_fails_closed_before_lease_if_gate_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    name = r.NAMES[0]
    r.x7.write_json(tmp_path / 'receipts' / name / 'admission.json', {'gate': {'origin_commit': 'abc'}})
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: {'authorized_runs': [name]})
    monkeypatch.setattr(c, 'completed_gate', lambda *_: None)
    monkeypatch.setattr(c.signal, 'signal', lambda *args: None)
    def no_lease(*args, **kwargs):
        raise AssertionError('must not acquire/release')
    monkeypatch.setattr(c.subprocess, 'run', no_lease)
    assert c.worker(name) == 1
    marker = json.loads((tmp_path / 'markers' / f'{name}.FAIL').read_text())
    assert 'W3 gate completion not verified' in marker['error']


def test_watcher_wait_done_tamper_and_full_growth_table(tmp_path):
    r.x7.write_json(tmp_path / 'receipts/scheduler/worker.json', c.identity(os.getpid()))
    assert w.status(tmp_path)[0] is None
    r.x7.write_json(tmp_path / 'receipts/scheduler/terminal.json', {'status': 'launched'})
    for name in r.NAMES:
        positions = [{'driver_position': j, 'count': 512, 'correct': 512 if j <= 4 else 1} for j in range(1, 65)]
        row = {'step': 1000000, 'positions': positions, 'perfect_prefix_depth': 4}
        result = {'name': name, 'status': 'complete', 'total_steps': 1000000,
                  'stop_reason': 'safety_ceiling', 'final': row}
        path = tmp_path / name / 'result.json'
        r.x7.write_json(path, result)
        r.x7.write_json(tmp_path / name / 'validation.json', [row])
        terminal = {'name': name, 'exit_code': 0, 'result_sha256': r.x7.digest(path)}
        r.x7.write_json(tmp_path / 'receipts' / name / 'terminal.json', terminal)
        r.x7.write_json(tmp_path / 'markers' / f'{name}.DONE', terminal)
    assert w.status(tmp_path)[0] == 0
    report = w.growth_report(tmp_path)
    assert '| P64 | 1 | 1 |' in report and '| P5 |' in report
    assert 'PARITY-AT-DEPTH' in report and 'ceiling remains binding' in report
    (tmp_path / r.NAMES[0] / 'result.json').write_text('{}')
    assert w.status(tmp_path)[0] == 1


def test_scheduler_admits_two_distinct_post_gate_cards_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    (tmp_path / 'receipts/scheduler').mkdir(parents=True)
    r.x7.write_json(tmp_path / 'registration.json', {})
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: {})
    monkeypatch.setattr(c, 'git', lambda *args: '')
    gate = {'origin_commit': 'abc', 'report_sha256': 'def'}
    monkeypatch.setattr(c, 'completed_gate', lambda: gate)
    commands = []
    def free_only(cmd, **kwargs):
        assert cmd == ['bash', 'scripts/gpu_lease.sh', 'list-free']
        return '4,5,7\n'
    def spawn(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(c.subprocess, 'check_output', free_only)
    monkeypatch.setattr(c.subprocess, 'Popen', spawn)
    assert c.schedule(1) == 0
    terminal = json.loads((tmp_path / 'receipts/scheduler/terminal.json').read_text())
    assert terminal['status'] == 'launched'
    assert [row['gpu'] for row in terminal['runs']] == [4, 5]
    assert [cmd[-1] for cmd in commands] == list(r.NAMES)
    for name, gpu in zip(r.NAMES, (4, 5)):
        admission = json.loads((tmp_path / 'receipts' / name / 'admission.json').read_text())
        assert admission['gpu'] == gpu and admission['gate'] == gate


def test_post_gate_worker_lease_heartbeat_done_and_release(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    name = r.NAMES[0]
    r.x7.write_json(tmp_path / 'registration.json', {})
    gate = {'origin_commit': 'abc', 'report_sha256': 'def'}
    r.x7.write_json(tmp_path / 'receipts' / name / 'admission.json',
        {'gpu': 5, 'gate': gate, 'registration_sha256': r.x7.digest(tmp_path / 'registration.json')})
    r.x7.write_json(tmp_path / name / 'result.json', {'total_steps': 1000000, 'status': 'complete'})
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: {'authorized_runs': [name]})
    monkeypatch.setattr(c, 'completed_gate', lambda *_: gate)
    monkeypatch.setattr(c.signal, 'signal', lambda *args: None)
    calls = []
    def broker(cmd, **kwargs):
        assert kwargs['env']['GPU_LEASE_PID'] == str(os.getpid())
        assert kwargs['env']['GPU_LEASE_VISIBLE'] == '5'
        calls.append(cmd[2:])
    class Child:
        pid = os.getpid()
        waits = 0
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise c.subprocess.TimeoutExpired('trainer', timeout)
            return 0
        def poll(self):
            return 0
    def spawn(cmd, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '5'
        assert cmd[-2:] == ['--arm', 'e97']
        return Child()
    monkeypatch.setattr(c.subprocess, 'run', broker)
    monkeypatch.setattr(c.subprocess, 'Popen', spawn)
    assert c.worker(name) == 0
    assert calls == [['acquire', '1', '--no-wait'], ['heartbeat', '5'], ['release', '5']]
    terminal = json.loads((tmp_path / 'markers' / f'{name}.DONE').read_text())
    assert terminal['exit_code'] == 0 and terminal['result_sha256'] == r.x7.digest(tmp_path / name / 'result.json')


def test_all_requested_predictions_and_explicit_competition():
    assert len(c.PREDICTIONS) == 4
    assert 'EXCEEDS E97' in c.PREDICTIONS[0]
    assert 'PARITY-AT-DEPTH' in c.PREDICTIONS[1]
    assert 'P5-first' in c.PREDICTIONS[2]
    assert 'GDN2 should not be able to do more nonlinear steps than it has layers' in c.PREDICTIONS[3]
