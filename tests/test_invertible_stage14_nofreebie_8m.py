"""X14 stock-kernel/init, exact X10-AdamW, ceiling/retention and native watcher tests."""
import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import run_invertible_stage14_nofreebie_8m as r
from scripts import invertible_stage14_control as control
from scripts import watch_invertible_stage14_nofreebie as watch


def test_stock_gdn2_matches_x7b_initialization_without_adapter():
    torch.manual_seed(0)
    model = r.build_model('gdn2')
    torch.manual_seed(0)
    reference = r.x10.build_model('gdn2_uniform')
    assert sum(p.numel() for p in model.parameters()) == 9887056
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    for layer in model.layers:
        assert layer.gdn.mode == 'chunk'
        assert layer.gdn.forward.__func__.__globals__['chunk_gated_delta_rule'] is not r.x7.fp32_gdn_recurrence


def test_exact_x10_adamw_all_parameter_decay_and_schedule():
    model = r.build_model('gdn2')
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


def test_mask_excludes_only_p1_preserves_inputs_targets_and_frozen_pool():
    batch = r.x10.InvertibleQuadraticTask(61).generate_batch(32, 18, np.random.default_rng(0))
    original = batch[2].copy()
    x, y, mask = r.composition_batch(batch)
    assert x is batch[0] and y is batch[1]
    np.testing.assert_array_equal(batch[2], original)
    assert not mask[:, :3].any() and mask[:, 3:].all()
    assert mask.sum() == 32 * 15


def test_train_update_zero_p1_logit_gradient_uniform_p2_plus_and_exact_x10(monkeypatch):
    # Direct logit parameters expose loss gradients by position, including P1.
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(2, 18, 3))])
    batch = r.x10.InvertibleQuadraticTask(61).generate_batch(2, 18, np.random.default_rng(99))
    batch[1][:] %= 3
    labels = torch.from_numpy(batch[1])
    expected_mask = torch.from_numpy(r.composition_batch(batch)[2])
    expected = torch.nn.functional.cross_entropy(model[0][expected_mask], labels[expected_mask])
    expected_gradient, = torch.autograd.grad(expected, model[0])
    def forward(model, masked):
        return model[0], torch.from_numpy(masked[1]), torch.from_numpy(masked[2])
    monkeypatch.setattr(r.x7, 'forward_batch', forward)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    opt = r.make_optimizer(model)
    row = r.train_update(model, opt, batch, 1)
    assert row['loss'] == pytest.approx(expected.item())
    assert row['lr'] == pytest.approx(2e-7) and math.isfinite(row['grad_norm'])
    assert not model[0].grad[:, :3].any()
    torch.testing.assert_close(model[0].grad, expected_gradient)
    assert model[0].grad[:, 3:].abs().sum() > 0


def test_evaluation_p1_diagnostic_never_primary_or_best(monkeypatch):
    monkeypatch.setattr(r.x7, 'evaluate', lambda *_: {
        'aggregate': {'correct': 512, 'loss': 99}, 'perfect_prefix_depth': 1,
        'positions': [{'driver_position': j, 'correct': 512 if j == 1 else 0,
                       'loss': 0. if j == 1 else 7.} for j in range(1, 17)]})
    metrics = r.evaluate(None, None)
    assert metrics['aggregate']['correct'] == 0
    assert metrics['aggregate']['accuracy'] == 0 and metrics['aggregate']['loss'] == 7.
    assert metrics['positions'][0]['role'] == 'diagnostic-only non-composition'
    assert metrics['composition_perfect_prefix_length'] == 0
    assert metrics['all_positions_diagnostic_aggregate']['correct'] == 512


def test_runner_500k_policy_no_metric_stop_and_bounded_retention(tmp_path, monkeypatch):
    pool = r.x7.frozen_pools(0, 16, 61)['val']
    registration = {'authorized_runs': ['gdn2_seed0'],
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
    monkeypatch.setattr(r, 'evaluate', lambda model, pool: {
        'aggregate': {'accuracy': 1., 'loss': 4. - len(lengths)},
        'positions': [{'driver_position': j, 'correct': 512} for j in range(1, 17)],
        'composition_perfect_prefix_length': 15})
    from argparse import Namespace
    r.run(Namespace(arm='gdn2', seed=0))
    assert lengths == [18, 18, 18]  # All-perfect cannot trigger an early stop.
    out = tmp_path / 'gdn2_seed0'
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
    receipt = tmp_path / 'receipts' / 'gdn2_seed0'
    r.x7.write_json(receipt / 'worker.json', control.identity(os.getpid()))
    assert watch.arm_status(tmp_path, 'gdn2_seed0')[0] is None
    result = tmp_path / 'gdn2_seed0' / 'result.json'
    r.x7.write_json(result, {'name': 'gdn2_seed0', 'status': 'complete', 'total_steps': 500000,
                             'stop_reason': 'safety_ceiling'})
    terminal = {'name': 'gdn2_seed0', 'exit_code': 0, 'result_sha256': r.x7.digest(result)}
    r.x7.write_json(receipt / 'terminal.json', terminal)
    r.x7.write_json(tmp_path / 'markers' / 'gdn2_seed0.DONE', terminal)
    assert watch.arm_status(tmp_path, 'gdn2_seed0')[0] == 0
    result.write_text('{}')
    assert watch.arm_status(tmp_path, 'gdn2_seed0')[0] == 1
    (tmp_path / 'markers' / 'gdn2_seed0.FAIL').touch()
    assert watch.arm_status(tmp_path, 'gdn2_seed0')[0] == 1


def test_watcher_detects_dead_identity_and_missing_worker(tmp_path, monkeypatch):
    assert watch.arm_status(tmp_path, 'gdn2_seed0')[0] == 1
    record = control.identity(os.getpid())
    record['starttime'] = '-1'
    r.x7.write_json(tmp_path / 'receipts' / 'gdn2_seed0' / 'worker.json', record)
    monkeypatch.setattr(watch.time, 'sleep', lambda _: None)
    assert watch.arm_status(tmp_path, 'gdn2_seed0')[0] == 1


def test_worker_registration_failure_publishes_fail_without_release(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    def invalid(_):
        raise ValueError('changed registration')
    monkeypatch.setattr(r.x10, 'verify_registration', invalid)
    def no_command(*args, **kwargs):
        raise AssertionError('must not acquire or release an unowned lease')
    monkeypatch.setattr(control.subprocess, 'run', no_command)
    monkeypatch.setattr(control.signal, 'signal', lambda *args: None)
    assert control.worker('gdn2_seed0') == 1
    terminal = json.loads((tmp_path / 'markers' / 'gdn2_seed0.FAIL').read_text())
    assert 'changed registration' in terminal['error'] and terminal['exit_code'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='leased target GPU required')
@pytest.mark.parametrize('architecture', ['e97', 'gdn2'])
def test_target_gpu_stock_kernel_adamw_and_all_16_positions(architecture):
    torch.manual_seed(99)
    model = r.build_model(architecture).cuda()
    task = r.x10.InvertibleQuadraticTask(61)
    pool = task.generate_batch(512, 18, np.random.default_rng(99))
    opt = r.make_optimizer(model)
    row = r.train_update(model, opt, tuple(a[:32] for a in pool), 1)
    assert math.isfinite(row['loss']) and row['lr'] == pytest.approx(2e-7)
    metrics = r.evaluate(model, pool)
    assert len(metrics['positions']) == 16
    assert sum(p['correct'] for p in metrics['positions']) == metrics['all_positions_diagnostic_aggregate']['correct']
    assert math.isfinite(metrics['aggregate']['loss'])


def test_watcher_all_four_arms_and_tampered_completion(tmp_path):
    assert watch.status(tmp_path)[0] == 1
    for name in r.NAMES:
        r.x7.write_json(tmp_path / 'receipts' / name / 'worker.json', control.identity(os.getpid()))
    assert watch.status(tmp_path)[0] is None
    for name in r.NAMES:
        result = tmp_path / name / 'result.json'
        r.x7.write_json(result, {'name': name, 'status': 'complete', 'total_steps': 500000,
                               'stop_reason': 'safety_ceiling'})
        terminal = {'name': name, 'exit_code': 0, 'result_sha256': r.x7.digest(result)}
        r.x7.write_json(tmp_path / 'receipts' / name / 'terminal.json', terminal)
        r.x7.write_json(tmp_path / 'markers' / f'{name}.DONE', terminal)
    assert watch.status(tmp_path)[0] == 0
    (tmp_path / r.NAMES[-1] / 'result.json').write_text('{}')
    assert watch.status(tmp_path)[0] == 1


@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('architecture', ['e97', 'gdn2'])
def test_both_seeds_match_stage10_pattern_initialization(architecture, seed):
    torch.manual_seed(seed)
    model = r.build_model(architecture)
    torch.manual_seed(seed)
    reference = r.x10.build_model('adamw' if architecture == 'e97' else 'gdn2_uniform')
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)


def test_growth_report_labels_p1_and_censors_partial_run(tmp_path):
    for name in r.NAMES:
        r.x7.write_json(tmp_path / 'receipts' / name / 'worker.json', control.identity(os.getpid()))
        r.x7.write_json(tmp_path / name / 'validation.json', [{
            'step': 500, 'composition_perfect_prefix_length': 0,
            'positions': [{'correct': 0} for _ in range(16)]}])
    watch.growth_report(tmp_path)
    report = (tmp_path / 'GROWTH_REPORT.md').read_text()
    assert 'fixed-final valid: False' in report
    assert '|P1|diagnostic-only non-composition|' in report
    assert '|P2|composition|' in report and 'right-censored' in report
    assert all(f'## {name}' in report for name in r.NAMES)
