"""CPU-only X17 carry format, global wean, X12 recipe and fail-closed supervision."""
import hashlib
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import run_invertible_stage17_carrywean_8m as r
from experiments.expressivity_tasks import run_invertible_stage6_truecur_8m as x6
from experiments.expressivity_tasks import run_invertible_stage12_separation_8m as x12
from scripts import invertible_stage17_control as c
from scripts import watch_invertible_stage17_carrywean as w


def read(path):
    return json.loads(path.read_text())


def test_exact_x6_slot_format_previous_only_and_neutral_target_independence():
    batch = r.x10.InvertibleQuadraticTask(61).generate_batch(32, 66, np.random.default_rng(7))
    x, y, _ = batch
    for p in (0., .25, .5, 1.):
        got, positions = r.carry_inputs(x, y, p, np.random.default_rng(99))
        expected, original_positions = x6.carry_inputs(x, y, p, np.random.default_rng(99))
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(positions, original_positions)
        assert got.shape == (32, 129)
        np.testing.assert_array_equal(got[:, positions], x)
        np.testing.assert_array_equal(got[:, :3], x[:, :3])  # No slot before P1.
        if p == 1.:
            np.testing.assert_array_equal(got[:, 3::2], y[:, 2:-1])
    class NoTargets:
        def __getitem__(self, _):
            raise AssertionError('neutral evaluation must never inspect targets')
    neutral, _ = r.carry_inputs(x, NoTargets())  # No RNG necessary or consumed.
    assert np.all(neutral[:, 3::2] == 3721)
    for invalid in (-1., 1.01, float('nan')):
        with pytest.raises(ValueError):
            r.carry_inputs(x, y, invalid)


def test_one_global_wean_and_exact_x12_rate():
    assert r.CEILING == 1000000 and r.DEPTH == 64 and r.EVAL_EVERY == 500
    for step, p in ((0, .5), (1, .4999995), (500000, .25), (750000, .125), (1000000, 0.)):
        assert r.carry_probability(step) == pytest.approx(p)
        assert r.learning_rate(step) == x12.learning_rate(step)
    for step in (-1, 1000001):
        with pytest.raises(ValueError):
            r.carry_probability(step)


def test_stage10_model_plus_exact_x6_embedding_and_all_parameter_adamw():
    torch.manual_seed(0)
    base = x12.build_model('e97')
    torch.manual_seed(0)
    model = r.build_model()
    assert len(model.layers) == 4
    assert sum(p.numel() for p in model.parameters()) == 9901056
    for name, value in base.state_dict().items():
        assert torch.equal(model.state_dict()[name], value)
    assert model.embed.neutral.shape == (1, 256)
    assert model.embed(torch.tensor([[3721, 61, 0]])).shape == (1, 3, 256)
    assert all(layer.recurrent_state_precision == 'legacy' and layer.fused_inference for layer in model.layers)
    opt = r.make_optimizer(model)
    assert type(opt) is torch.optim.AdamW and not opt.state
    group = opt.param_groups[0]
    assert group['betas'] == (.9, .999) and group['eps'] == 1e-8
    assert group['weight_decay'] == .01 and group['lr'] == 0.
    assert {id(p) for p in group['params']} == {id(p) for p in model.parameters()}


def test_forward_uses_x3_padding_then_gathers_original_positions(monkeypatch):
    from contextlib import nullcontext
    batch = r.x10.InvertibleQuadraticTask(61).generate_batch(2, 66, np.random.default_rng(1))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.amp, 'autocast', lambda *a, **kw: nullcontext())
    seen = []
    def padded(model, x):
        seen.append(x)
        # Position-coded logits prove carry slots are gathered OUT before loss.
        return torch.arange(129).expand(2, 3, 129).transpose(1, 2)
    monkeypatch.setattr(r.x7, 'padded_forward', padded)
    logits, y, mask = r.forward_batch(None, batch)
    assert seen[0].shape == (2, 129) and torch.all(seen[0][:, 3::2] == r.NEUTRAL)
    assert logits.shape == (2, 66, 3)
    np.testing.assert_array_equal(logits[0, :, 0], np.r_[0, 1, 2 + 2 * np.arange(64)])
    np.testing.assert_array_equal(y, batch[1])
    np.testing.assert_array_equal(mask, batch[2])


def test_uniform_update_logs_actual_p_and_lr(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    inputs = torch.tensor([[[1., 2.], [3., 1.]], [[2., 1.], [1., 0.]]])
    y = torch.tensor([[0, 1], [1, 0]])
    mask = torch.tensor([[True, False], [True, True]])
    expected = torch.nn.functional.cross_entropy(model(inputs)[mask], y[mask]).item()
    seen = []
    rng = np.random.default_rng(0)
    def forward(model, batch, p, generator):
        seen.append((p, generator))
        return model(inputs), y, mask
    monkeypatch.setattr(r, 'forward_batch', forward)
    row = r.train_update(model, r.make_optimizer(model), None, 500001, rng)
    assert seen == [(r.carry_probability(500001), rng)]
    assert row['loss'] == pytest.approx(expected)
    assert row['carry_probability'] == r.carry_probability(500001)
    assert row['lr'] == x12.learning_rate(500001) and math.isfinite(row['grad_norm'])


def pool():
    x = np.zeros((512, 66), dtype=np.int64)
    mask = np.ones_like(x, dtype=bool)
    mask[:, :2] = False
    return x, x.copy(), mask


def test_all64_neutral_evaluation_reconciles_counts(monkeypatch):
    calls = []
    def forward(model, batch, probability=0., rng=None):
        calls.append((probability, rng))
        return torch.zeros((32, 66, 2)), torch.from_numpy(batch[1]), torch.from_numpy(batch[2])
    monkeypatch.setattr(r, 'forward_batch', forward)
    metrics = r.evaluate(torch.nn.Identity(), pool())
    assert calls == [(0., None)] * 16
    assert metrics['perfect_prefix_depth'] == 64 and len(metrics['positions']) == 64
    assert all(p['correct'] == p['count'] == 512 for p in metrics['positions'])
    assert metrics['aggregate']['correct'] == 64 * 512
    assert metrics['aggregate']['loss'] == pytest.approx(math.log(2))


def test_runner_full_horizon_not_early_stop_and_bounded_reloadable_checkpoints(tmp_path, monkeypatch):
    name, p = r.NAMES[0], pool()
    registration = {'authorized_runs': [name], 'pool_array_sha256': {'0': [hashlib.sha256(a.tobytes()).hexdigest() for a in p]}}
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    r.x7.write_json(tmp_path / 'registration.json', registration)
    r.x7.write_json(tmp_path / 'receipts' / name / 'admission.json',
                    {'registration_sha256': r.x7.digest(tmp_path / 'registration.json'), 'gpu': 4})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: registration)
    monkeypatch.setattr(r.x7, 'frozen_pools', lambda *args: {'val': p})
    monkeypatch.setattr(r, 'CEILING', 3)
    monkeypatch.setattr(r, 'EVAL_EVERY', 1)
    model = torch.nn.Linear(1, 2, bias=False)
    model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    monkeypatch.setattr(r, 'build_model', lambda: model)
    monkeypatch.setattr(r, 'PARAMETERS', 2)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    observed = []
    def forward(model, batch, probability=0., rng=None):
        observed.append((batch[0].shape, probability))
        return model.weight[:, 0].expand(32, 66, 2), torch.from_numpy(batch[1]) * 0, torch.from_numpy(batch[2])
    monkeypatch.setattr(r, 'forward_batch', forward)
    monkeypatch.setattr(r, 'evaluate', lambda *_: {'aggregate': {'accuracy': 1., 'loss': 1.},
        'positions': [{'driver_position': j, 'correct': 512, 'count': 512} for j in range(1, 65)], 'perfect_prefix_depth': 64})
    r.run()
    assert observed == [((32, 66), r.carry_probability(s)) for s in (1, 2, 3)]
    out = tmp_path / name
    result = read(out / 'result.json')
    assert result['total_steps'] == 3 and result['stop_reason'] == 'safety_ceiling'
    assert result['final']['carry_probability'] == result['final']['eval_carry_probability'] == 0.
    rows = read(out / 'validation.json')
    assert [row['step'] for row in rows] == [0, 1, 2, 3]
    assert len(list(out.glob('*.pt'))) <= 3
    saved = torch.load(rows[-1]['checkpoint']['path'], weights_only=False)
    assert saved['optimizer_family'] == 'adamw' and saved['numpy_rng']
    assert saved['optimizer']['state']
    training = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    assert [row['carry_probability'] for row in training] == [r.carry_probability(s) for s in (1, 2, 3)]


@pytest.mark.parametrize('reply', ['', 'oops', '8', '4,4', '4,,5', '4,'])
def test_scheduler_waits_without_lease_on_empty_or_invalid_broker(tmp_path, monkeypatch, reply):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    (tmp_path / 'receipts/scheduler').mkdir(parents=True)
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: {})
    def broker(cmd, **kwargs):
        assert cmd == ['bash', 'scripts/gpu_lease.sh', 'list-free']
        return reply
    monkeypatch.setattr(c.subprocess, 'check_output', broker)
    monkeypatch.setattr(c.subprocess, 'run', lambda *a, **kw: pytest.fail('no lease while waiting'))
    monkeypatch.setattr(c.subprocess, 'Popen', lambda *a, **kw: pytest.fail('no worker while waiting'))
    monkeypatch.setattr(c.time, 'sleep', lambda _: (_ for _ in ()).throw(SystemExit('end test')))
    assert c.schedule(1) == 1
    assert read(tmp_path / 'receipts/scheduler/polls.jsonl')['phase'] == 'WAIT_CARDS'
    assert not (tmp_path / 'receipts' / r.NAMES[0]).exists()


def test_scheduler_first_free_only_after_poll_and_no_gpu_acquire(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    (tmp_path / 'receipts/scheduler').mkdir(parents=True)
    r.x7.write_json(tmp_path / 'registration.json', {})
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: {})
    replies = iter(['', '4,5,7'])
    monkeypatch.setattr(c.subprocess, 'check_output', lambda *a, **kw: next(replies))
    monkeypatch.setattr(c.subprocess, 'run', lambda *a, **kw: pytest.fail('scheduler must not lease'))
    monkeypatch.setattr(c.time, 'sleep', lambda _: None)
    calls = []
    def spawn(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(c.subprocess, 'Popen', spawn)
    assert c.schedule(1) == 0
    assert len(calls) == 1 and calls[0][-1] == 'worker'
    assert read(tmp_path / 'receipts' / r.NAMES[0] / 'admission.json')['gpu'] == 4
    assert read(tmp_path / 'receipts/scheduler/terminal.json')['status'] == 'launched'


def setup_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    name = r.NAMES[0]
    r.x7.write_json(tmp_path / 'registration.json', {})
    r.x7.write_json(tmp_path / 'receipts' / name / 'admission.json',
        {'gpu': 4, 'registration_sha256': r.x7.digest(tmp_path / 'registration.json')})
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: {'authorized_runs': [name]})
    monkeypatch.setattr(c.signal, 'signal', lambda *args: None)
    return name


def test_worker_registration_tamper_fails_before_lease(tmp_path, monkeypatch):
    name = setup_worker(tmp_path, monkeypatch)
    path = tmp_path / 'receipts' / name / 'admission.json'
    row = read(path)
    row['registration_sha256'] = 'tampered'
    r.x7.write_json(path, row)
    monkeypatch.setattr(c.subprocess, 'run', lambda *a, **kw: pytest.fail('no broker before verified admission'))
    assert c.worker() == 1
    assert 'admission changed' in read(tmp_path / 'markers' / f'{name}.FAIL')['error']


def test_worker_lost_acquire_race_no_trainer_retry_or_release(tmp_path, monkeypatch):
    name = setup_worker(tmp_path, monkeypatch)
    calls = []
    def broker(cmd, **kwargs):
        calls.append(cmd[2:])
        raise c.subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(c.subprocess, 'run', broker)
    monkeypatch.setattr(c.subprocess, 'Popen', lambda *a, **kw: pytest.fail('no trainer without lease'))
    assert c.worker() == 1
    assert calls == [['acquire', '1', '--no-wait']]
    assert (tmp_path / 'markers' / f'{name}.FAIL').exists()


def test_worker_lease_heartbeat_and_release(tmp_path, monkeypatch):
    name = setup_worker(tmp_path, monkeypatch)
    r.x7.write_json(tmp_path / name / 'result.json', {'name': name, 'total_steps': 1000000,
        'status': 'complete', 'stop_reason': 'safety_ceiling', 'final': {'step': 1000000,
        'carry_probability': 0., 'eval_carry_probability': 0., 'positions': [{}] * 64}})
    calls = []
    def broker(cmd, **kwargs):
        assert kwargs['env']['GPU_LEASE_PID'] == str(os.getpid())
        assert kwargs['env']['GPU_LEASE_VISIBLE'] == '4'
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
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '4'
        assert cmd[-1] == 'experiments.expressivity_tasks.run_invertible_stage17_carrywean_8m'
        return Child()
    monkeypatch.setattr(c.subprocess, 'run', broker)
    monkeypatch.setattr(c.subprocess, 'Popen', spawn)
    assert c.worker() == 0
    assert calls == [['acquire', '1', '--no-wait'], ['heartbeat', '4'], ['release', '4']]
    assert read(tmp_path / 'markers' / f'{name}.DONE')['result_sha256'] == r.x7.digest(tmp_path / name / 'result.json')


def test_watcher_wait_complete_tamper_and_all_positions(tmp_path):
    name = r.NAMES[0]
    r.x7.write_json(tmp_path / 'receipts/scheduler/worker.json', c.identity(os.getpid()))
    assert w.status(tmp_path)[0] is None
    assert 'PENDING' in w.growth_report(tmp_path)
    row = {'step': 1000000, 'carry_probability': 0., 'eval_carry_probability': 0.,
           'positions': [{'driver_position': j, 'count': 512, 'correct': 50 if j == 2 else 1} for j in range(1, 65)],
           'perfect_prefix_depth': 0}
    r.x7.write_json(tmp_path / name / 'validation.json', [row])
    r.x7.write_json(tmp_path / name / 'result.json', {'name': name, 'status': 'complete', 'total_steps': 1000000,
        'stop_reason': 'safety_ceiling', 'final': row})
    marker = {'name': name, 'exit_code': 0, 'result_sha256': r.x7.digest(tmp_path / name / 'result.json')}
    r.x7.write_json(tmp_path / 'receipts/scheduler/terminal.json', {'status': 'launched'})
    r.x7.write_json(tmp_path / 'receipts' / name / 'terminal.json', marker)
    r.x7.write_json(tmp_path / 'markers' / f'{name}.DONE', marker)
    assert w.status(tmp_path)[0] == 0
    report = w.growth_report(tmp_path)
    assert '| P64 | 1 | 1 |' in report and '| P2 | 50 | 50 |' in report
    assert 'survives=True' in report and 'ALWAYS neutral' in report
    (tmp_path / name / 'result.json').write_text('{}')
    assert w.status(tmp_path)[0] == 1


def test_registered_predictions_and_no_unapproved_arm():
    assert r.NAMES == ('e97_carrywean_seed0',)
    assert len(c.PREDICTIONS) == 3
    assert 'WEAN-TRANSFER' in c.PREDICTIONS[0]
    assert 'CRUTCH-DEPENDENT-AT-DEPTH' in c.PREDICTIONS[1]
    assert 'P1 diagnostic' in c.PREDICTIONS[2]
