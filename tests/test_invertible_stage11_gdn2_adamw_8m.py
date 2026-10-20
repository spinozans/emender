"""X11 stock-kernel/init, exact X10-AdamW, ceiling/retention and native watcher tests."""
import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import run_invertible_stage11_gdn2_adamw_8m as r
from scripts import invertible_stage11_control as control
from scripts import watch_invertible_stage11_gdn2_adamw as watch


def test_stock_gdn2_matches_x7b_initialization_without_adapter():
    torch.manual_seed(0)
    model = r.build_model()
    torch.manual_seed(0)
    reference = r.x10.build_model('gdn2_uniform')
    assert sum(p.numel() for p in model.parameters()) == 9887056
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    for layer in model.layers:
        assert layer.gdn.mode == 'chunk'
        assert layer.gdn.forward.__func__.__globals__['chunk_gated_delta_rule'] is not r.x7.fp32_gdn_recurrence


def test_exact_x10_adamw_all_parameter_decay_and_schedule():
    model = r.build_model()
    opt = r.make_optimizer(model)
    reference = r.x10.make_optimizer(model, 'adamw')
    assert type(opt) is torch.optim.AdamW
    assert opt.defaults == reference.defaults
    assert not opt.state and len(opt.param_groups) == 1
    group = opt.param_groups[0]
    assert group['betas'] == (.9, .999) and group['eps'] == 1e-8
    assert group['weight_decay'] == .01 and group['lr'] == 0.
    assert {id(p) for p in group['params']} == {id(p) for p in model.parameters()}
    tagged = [p for p in model.parameters() if getattr(p, '_no_weight_decay', False)]
    assert tagged and all(any(p is q for q in group['params']) for p in tagged)
    for step, expected in [(1, 2e-7), (5000, .001), (252500, .000505), (500000, .00001)]:
        r.x10.rates(opt, 'adamw', step)
        assert group['lr'] == pytest.approx(expected)


def test_pool_exactly_matches_registered_x7b_and_x10():
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in r.x7.frozen_pools(0, 16, 61)['val']]
    for arm in ('gdn2_uniform', 'adamw'):
        reg = json.loads((r.x10.ROOTS[arm] / 'registration.json').read_text())
        assert hashes == reg['pool_array_sha256']['0']


def test_train_update_uses_actual_x10_uniform_ce_and_logs_lr(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 0.], [0., 3.]]))
    labels = torch.tensor([[[0, 1], [1, 0]]]).reshape(2, 2)
    mask = torch.tensor([[True, False], [True, True]])
    inputs = torch.tensor([[[1., 2.], [3., 1.]], [[2., 1.], [1., 0.]]])
    expected = torch.nn.functional.cross_entropy(model(inputs)[mask], labels[mask]).item()
    monkeypatch.setattr(r.x7, 'forward_batch', lambda model, _: (model(inputs), labels, mask))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    opt = r.make_optimizer(model)
    row = r.train_update(model, opt, None, 1)
    assert row['loss'] == pytest.approx(expected)
    assert row['lr'] == pytest.approx(2e-7) and math.isfinite(row['grad_norm'])
    assert len(opt.state) == len(list(model.parameters()))


def test_runner_500k_policy_no_metric_stop_and_bounded_retention(tmp_path, monkeypatch):
    pool = r.x7.frozen_pools(0, 16, 61)['val']
    registration = {'authorized_runs': [r.NAME],
                    'pool_array_sha256': {'0': [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]},
                    'resource_bounds': {'wall_seconds': 3600, 'peak_allocated_bytes': 8 * 1024**3}}
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    r.x7.write_json(tmp_path / 'registration.json', registration)
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: registration)
    monkeypatch.setattr(r, 'CEILING', 3)
    monkeypatch.setattr(r, 'EVAL_EVERY', 1)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    lengths = []
    def forward(model, batch):
        lengths.append(batch[0].shape[1])
        y, mask = torch.from_numpy(batch[1]) * 0, torch.from_numpy(batch[2])
        return model.embed.weight[0, :2].expand(*y.shape, 2), y, mask
    monkeypatch.setattr(r.x7, 'forward_batch', forward)
    monkeypatch.setattr(r.x7, 'evaluate', lambda model, pool: {
        'aggregate': {'accuracy': 1., 'loss': 4. - len(lengths)},
        'positions': [{'driver_position': j, 'correct': 512} for j in range(1, 17)],
        'perfect_prefix_depth': 16})
    r.run()
    assert lengths == [18, 18, 18]  # All-perfect cannot trigger an early stop.
    out = tmp_path / r.NAME
    result = json.loads((out / 'result.json').read_text())
    assert result['total_steps'] == 3 and result['stop_reason'] == 'safety_ceiling'
    rows = json.loads((out / 'validation.json').read_text())
    assert [row['step'] for row in rows] == [0, 1, 2, 3]
    assert all(len(row['positions']) == 16 for row in rows)
    saved = torch.load(rows[-1]['checkpoint']['path'], weights_only=False)
    assert saved['optimizer_family'] == 'adamw' and saved['weights'] == 'ordinary live parameters'
    assert all('train_mode' not in g for g in saved['optimizer']['param_groups'])
    assert len(list(out.glob('*.pt'))) <= 3
    training = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    assert [row['lr'] for row in training] == [r.x10.learning_rate(s) for s in (1, 2, 3)]


def test_watcher_pending_valid_done_tamper_and_fail(tmp_path):
    receipt = tmp_path / 'receipts' / r.NAME
    r.x7.write_json(receipt / 'worker.json', control.identity(os.getpid()))
    assert watch.status(tmp_path)[0] is None
    result = tmp_path / r.NAME / 'result.json'
    r.x7.write_json(result, {'name': r.NAME, 'status': 'complete', 'total_steps': 500000,
                             'stop_reason': 'safety_ceiling'})
    terminal = {'name': r.NAME, 'exit_code': 0, 'result_sha256': r.x7.digest(result)}
    r.x7.write_json(receipt / 'terminal.json', terminal)
    r.x7.write_json(tmp_path / 'markers' / f'{r.NAME}.DONE', terminal)
    assert watch.status(tmp_path)[0] == 0
    result.write_text('{}')
    assert watch.status(tmp_path)[0] == 1
    (tmp_path / 'markers' / f'{r.NAME}.FAIL').touch()
    assert watch.status(tmp_path)[0] == 1


def test_watcher_detects_dead_identity_and_missing_worker(tmp_path, monkeypatch):
    assert watch.status(tmp_path)[0] == 1
    record = control.identity(os.getpid())
    record['starttime'] = '-1'
    r.x7.write_json(tmp_path / 'receipts' / r.NAME / 'worker.json', record)
    monkeypatch.setattr(watch.time, 'sleep', lambda _: None)
    assert watch.status(tmp_path)[0] == 1


def test_worker_registration_failure_publishes_fail_without_release(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    def invalid(_):
        raise ValueError('changed registration')
    monkeypatch.setattr(r.x10, 'verify_registration', invalid)
    def no_command(*args, **kwargs):
        raise AssertionError('must not acquire or release an unowned lease')
    monkeypatch.setattr(control.subprocess, 'run', no_command)
    monkeypatch.setattr(control.signal, 'signal', lambda *args: None)
    assert control.worker() == 1
    terminal = json.loads((tmp_path / 'markers' / f'{r.NAME}.FAIL').read_text())
    assert 'changed registration' in terminal['error'] and terminal['exit_code'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='leased target GPU required')
def test_target_gpu_stock_kernel_adamw_and_all_16_positions():
    torch.manual_seed(99)
    model = r.build_model().cuda()
    task = r.x10.InvertibleQuadraticTask(61)
    pool = task.generate_batch(512, 18, np.random.default_rng(99))
    opt = r.make_optimizer(model)
    row = r.train_update(model, opt, tuple(a[:32] for a in pool), 1)
    assert math.isfinite(row['loss']) and row['lr'] == pytest.approx(2e-7)
    metrics = r.x7.evaluate(model, pool)
    assert len(metrics['positions']) == 16
    assert sum(p['correct'] for p in metrics['positions']) == metrics['aggregate']['correct']
    assert math.isfinite(metrics['aggregate']['loss'])
