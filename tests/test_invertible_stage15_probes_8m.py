"""X15 source-named protocol, loader, schedule, retention and watcher checks."""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import run_invertible_stage15_probes_8m as r
from scripts import watch_invertible_stage15_probes as w


def test_muon3x_exact_x10_initialization_routing_decay_and_schedule():
    torch.manual_seed(0)
    model = r.build_model('muon3x')
    torch.manual_seed(0)
    reference = r.x10.build_model('muon')
    assert sum(p.numel() for p in model.parameters()) == 9900800
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    opt = r.make_optimizer(model, 'muon3x')
    assert type(opt) is r.x10.HybridMuon and not opt.state and not opt.aux.state
    assert opt.routes == r.parameter_routes(reference)
    assert all(g['weight_decay'] == .01 for g in opt.param_groups)
    assert sum(len(g['params']) for g in opt.param_groups) == len(list(model.parameters()))
    for step, lr in [(0, 0), (1, 6e-7), (5000, .003), (502500, .001515), (1000000, .00003)]:
        r.rates(opt, 'muon3x', step)
        assert all(g['lr'] == pytest.approx(lr) for g in opt.param_groups)
    with pytest.raises(ValueError):
        r.learning_rate(1000001)


def test_actual_x10_update_uniform_all64_and_rate_patch_restored(monkeypatch):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0)
    inputs = torch.ones(1, 66, 2)
    targets = torch.zeros(1, 66, dtype=torch.long)
    mask = torch.ones(1, 66, dtype=torch.bool)
    mask[:, :2] = False
    expected = torch.nn.functional.cross_entropy(model(inputs)[mask], targets[mask]).item()
    monkeypatch.setattr(r.x7, 'forward_batch', lambda m, _: (m(inputs), targets, mask))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    original = r.x10.rates
    row = r.train_update(model, opt, 'muon3x', None, 5000)
    assert row['loss'] == pytest.approx(expected) and row['lr'] == .003
    assert r.x10.rates is original


def test_evaluate_adjusts_only_expected_depth_and_restores(monkeypatch):
    def evaluate(model, pool):
        assert r.x7.DEPTH == 64
        return {'clean': True}
    monkeypatch.setattr(r.x7, 'evaluate', evaluate)
    assert r.evaluate(None, None) == {'clean': True}
    assert r.x7.DEPTH == 16
    assert r.CEILING == 1000000 and r.EVAL_EVERY == 500


def test_trusted_t64_forward_padding_targets_and_scalar_oracle(monkeypatch):
    pool = r.x7.frozen_pools(0, 64, 61)['val']
    batch = tuple(a[:2] for a in pool)
    captured = []
    class Capture(torch.nn.Module):
        def forward(self, tokens):
            captured.append(tokens.numpy())
            return torch.zeros((*tokens.shape, 1))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.amp, 'autocast', lambda *a, **k: nullcontext())
    logits, targets, mask = r.x7.forward_batch(Capture(), batch)
    assert captured[0].shape == (2, 80) and logits.shape == (2, 66, 1)
    np.testing.assert_array_equal(captured[0][:, :66], batch[0])
    assert (captured[0][:, 66:] == 0).all()
    np.testing.assert_array_equal(targets.numpy(), batch[1])
    np.testing.assert_array_equal(mask.numpy(), batch[2])
    for inputs, labels in zip(batch[0], batch[1]):
        x, y = int(inputs[0])-61, int(inputs[1])-122
        for j, driver in enumerate(inputs[2:]):
            x, y = y, (x + y*y + int(driver)) % 61
            assert int(labels[j+2]) == 61*x+y


def test_runner_scratch_t64_no_metric_stop_retention(tmp_path, monkeypatch):
    pool = r.x7.frozen_pools(0, 64, 61)['val']
    model = r.build_model('muon3x')
    reg = {'authorized_runs': ['muon3x_seed0'], 'pool_array_sha256': {'0': r.array_hashes(pool)},
           'parameter_routes': r.parameter_routes(model),
           'resource_bounds': {'muon3x': {'wall_seconds': 3600, 'peak_allocated_bytes': 8*1024**3}}}
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    r.x7.write_json(tmp_path / 'registration.json', reg)
    monkeypatch.setattr(r, 'verify_registration', lambda _: reg)
    monkeypatch.setattr(r, 'CEILING', 3)
    monkeypatch.setattr(r, 'EVAL_EVERY', 1)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    lengths = []
    def update(model, opt, arm, batch, step):
        lengths.append(batch[0].shape[1])
        assert not opt.state
        return {'loss': 1., 'optimizer_seconds': 0.}
    monkeypatch.setattr(r, 'train_update', update)
    monkeypatch.setattr(r, 'evaluate', lambda *_: {
        'aggregate': {'accuracy': 1., 'loss': 4. - len(lengths)},
        'positions': [{'driver_position': j, 'correct': 512} for j in range(1, 65)],
        'perfect_prefix_depth': 64})
    r.run(SimpleNamespace(arm='muon3x', seed=0))
    assert lengths == [66]*3
    out = tmp_path / 'muon3x_seed0'
    result = json.loads((out / 'result.json').read_text())
    assert result['total_steps'] == 3 and result['stop_reason'] == 'safety_ceiling'
    assert len(list(out.glob('step_*.pt'))) <= 3
    assert set(np.load(out / 'frozen_validation.npz').files) == {'T64_0', 'T64_1', 'T64_2'}


def test_zero_shot_rejects_hash_drift_before_deserialization(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'verify_registration', lambda _: {})
    cp = tmp_path / 'bad.pt'
    cp.write_bytes(b'not checkpoint')
    monkeypatch.setattr(r, 'CHAMPION', cp)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('untrusted deserialization'))
    with pytest.raises(AssertionError):
        r.zero_shot()


def terminal(root, name):
    steps = 0 if name == 'gdn2_zeroshot' else 1000000
    result = {'name': name, 'status': 'complete', 'total_steps': steps,
              'stop_reason': 'zero_shot_complete' if steps == 0 else 'safety_ceiling'}
    path = root / name / 'result.json'
    r.x7.write_json(path, result)
    marker = {'name': name, 'exit_code': 0, 'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    r.x7.write_json(root / 'receipts' / name / 'terminal.json', marker)
    r.x7.write_json(root / 'markers' / (name + '.DONE'), marker)


@pytest.mark.parametrize('name', r.NAMES)
def test_watcher_complete_tamper_fail_and_missing_identity(tmp_path, name):
    assert w.arm_status(tmp_path, name)[0] == 1
    terminal(tmp_path, name)
    assert w.arm_status(tmp_path, name)[0] == 0
    (tmp_path / name / 'result.json').write_text('{}')
    assert w.arm_status(tmp_path, name)[0] == 1
    r.x7.write_json(tmp_path / 'markers' / (name + '.FAIL'), {})
    assert w.arm_status(tmp_path, name)[0] == 1


def test_watcher_pending_dead_and_aggregation(tmp_path, monkeypatch):
    terminal(tmp_path, 'gdn2_zeroshot')
    r.x7.write_json(tmp_path / 'receipts/muon3x_seed0/worker.json', {'pid': 123})
    monkeypatch.setattr(w, 'process_alive', lambda _: True)
    assert w.status(tmp_path)[0] is None
    monkeypatch.setattr(w, 'process_alive', lambda _: False)
    monkeypatch.setattr(w.time, 'sleep', lambda _: None)
    assert w.status(tmp_path)[0] == 1
    terminal(tmp_path, 'muon3x_seed0')
    assert w.status(tmp_path)[0] == 0


def test_marginal_signature_and_all64_report(tmp_path):
    terminal(tmp_path, 'gdn2_zeroshot')
    path = tmp_path / 'gdn2_zeroshot/result.json'
    result = json.loads(path.read_text())
    pos = [{'driver_position': j+1, 'correct': 512 if j < 4 else 0,
            'loss': math.log(3721)} for j in range(64)]
    result.update(t16_control={'positions': pos[:16]}, final={'positions': pos})
    r.x7.write_json(path, result)
    # Rebind marker hash for valid completed zero-shot result.
    marker = {'name': 'gdn2_zeroshot', 'exit_code': 0, 'result_sha256': r.x7.digest(path)}
    r.x7.write_json(tmp_path / 'receipts/gdn2_zeroshot/terminal.json', marker)
    r.x7.write_json(tmp_path / 'markers/gdn2_zeroshot.DONE', marker)
    rows = [{'step': 500, 'aggregate': {'loss': math.log(3721)},
             'positions': [{**p, 'correct': int(j == 20)*50} for j,p in enumerate(pos)]}]
    r.x7.write_json(tmp_path / 'muon3x_seed0/validation.json', rows)
    sig = w.marginal_signature(rows)[0]
    assert sig['ce_minus_ln3721'] == 0 and sig['material_positions'] == [21]
    w.growth_report(tmp_path)
    text = (tmp_path / 'GROWTH_REPORT.md').read_text()
    assert '|P64|' in text and 'Prediction(c)' in text and 'censored@500' in text
    assert 'NOT proof' in text
    assert len(json.loads((tmp_path / 'marginal_signature.json').read_text())['rows'][0]['position_correct']) == 64


@pytest.mark.skipif(os.environ.get('EMENDER_X15_GPU_TEST') != '1', reason='explicit leased GPU opt-in')
def test_gpu_t64_actual_evaluation_all_positions():
    torch.manual_seed(99)
    model = r.build_model('muon3x').cuda()
    pool = r.InvertibleQuadraticTask(61).generate_batch(512, 66, np.random.default_rng(99))
    metrics = r.evaluate(model, pool)
    assert len(metrics['positions']) == 64
    assert sum(p['correct'] for p in metrics['positions']) == metrics['aggregate']['correct']
    assert math.isfinite(metrics['aggregate']['loss'])
