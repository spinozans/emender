"""X16 explicit loss, depth-local evaluation, curriculum state/retention and watchdog."""
import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import run_invertible_stage16_ignition_8m as r
from scripts import invertible_stage16_control as control
from scripts import watch_invertible_stage16_ignition as watch


def test_registered_cards_weights_and_exact_x10_optimizer():
    assert [a[2] for a in r.ARMS.values()] == [0, 1, 2, 3, 6]
    assert [a[1] for a in r.ARMS.values()] == [1/64, 1/32, 1/512, 1/2048, None]
    model = torch.nn.Linear(2, 2)
    opt = r.x14.make_optimizer(model)
    assert type(opt) is torch.optim.AdamW
    assert opt.defaults == r.x10.make_optimizer(model, 'adamw').defaults
    assert len(opt.param_groups) == 1 and not opt.state
    for step in (1, 5000, 100000, 500000):
        r.x10.rates(opt, 'adamw', step)
        assert opt.param_groups[0]['lr'] == r.x10.learning_rate(step)


@pytest.mark.parametrize('weight', [0., 1/64, 1/32, 1/512, 1/2048, 1/16, 1.])
def test_explicit_p1_share_loss_and_gradient(weight):
    torch.manual_seed(99)
    logits = torch.randn(2, 18, 3, requires_grad=True)
    y = torch.zeros(2, 18, dtype=torch.long)
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, 2:] = True
    original = mask.clone()
    loss = r.position_loss(logits, y, mask, weight)
    individual = torch.nn.functional.cross_entropy(logits[:, 2:].reshape(-1, 3), y[:, 2:].reshape(-1),
                                                   reduction='none').reshape(2, 16)
    weights = torch.tensor([weight] + [(1-weight)/15] * 15)
    expected = (individual * weights).sum(1).mean()
    actual_gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert not actual_gradient[:, :2].any()
    assert torch.equal(mask, original)
    if weight == 1/16:
        torch.testing.assert_close(loss, individual.mean())


def test_invalid_mask_and_extension_schedule_no_warmup_jump():
    with pytest.raises(ValueError):
        r.position_loss(torch.zeros(1, 18, 2), torch.zeros(1, 18, dtype=torch.long),
                        torch.ones(1, 18, dtype=torch.bool), 1/64)
    entry = r.x10.learning_rate(400000)
    assert r.extension_rate(0, entry) == pytest.approx(entry)
    assert r.extension_rate(1, entry) <= entry
    assert r.extension_rate(250000, entry) == pytest.approx((entry + 1e-5)/2)
    assert r.extension_rate(500000, entry) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        r.extension_rate(500001, entry)


def test_depth_local_evaluation_no_global_mutation(monkeypatch):
    model = torch.nn.Linear(1, 1)
    def forward(_, batch):
        y = torch.zeros(batch[1].shape, dtype=torch.long)
        mask = torch.from_numpy(batch[2])
        logits = torch.zeros(*y.shape, 2)
        logits[..., 0] = 10
        return logits, y, mask
    monkeypatch.setattr(r.x7, 'forward_batch', forward)
    for depth in (16, 64, 16):
        pool = r.x10.InvertibleQuadraticTask(61).generate_batch(512, depth+2, np.random.default_rng(99))
        pool[1][:] = 0
        metrics = r.evaluate(model, pool)
        assert len(metrics['positions']) == depth
        assert metrics['aggregate']['correct'] == 512 * depth
        assert metrics['perfect_prefix_depth'] == depth
    assert r.x7.DEPTH == 16


def metrics(depth, good=False):
    return {'aggregate': {'accuracy': .25, 'loss': 3.},
            'positions': [{'driver_position': j+1, 'correct': 508 if good and j < 4 else 0}
                          for j in range(depth)], 'perfect_prefix_depth': 0}


def setup_fake_run(tmp_path, monkeypatch, name, criterion_at=2):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'CEILING', 3)
    monkeypatch.setattr(r, 'EVAL_EVERY', 1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', str(r.ARMS[name][2]))
    registration = {'authorized_runs': [name], 'pool_array_sha256': {
        str(d): [hashlib.sha256(a.tobytes()).hexdigest() for a in r.x7.frozen_pools(0,d,61)['val']]
        for d in (16,64)}, 'resource_bounds': {'wall_seconds': 3600, 'peak_allocated_bytes': 2**34}}
    r.x7.write_json(tmp_path / 'registration.json', registration)
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: registration)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    calls = []
    def update(model, opt, batch, step, weight=None, lr=None):
        calls.append((id(model), id(opt), batch[0].shape[1]-2, step, weight, lr))
        # Real optimizer moments confirm no reset at transition.
        opt.zero_grad()
        next(model.parameters()).sum().backward()
        opt.step()
        return {'optimizer_seconds': 0., 'loss': 1.}
    monkeypatch.setattr(r, 'train_update', update)
    monkeypatch.setattr(r.x10, 'train_update', lambda model,opt,arm,batch,step: update(model,opt,batch,step))
    monkeypatch.setattr(r, 'evaluate', lambda model,pool: metrics(pool[0].shape[1]-2,
        good=sum(c[2] == 16 for c in calls) >= criterion_at))
    return calls


def test_curriculum_criterion_same_model_moments_full_500k_and_retention(tmp_path, monkeypatch):
    name = 'e97_curriculum_seed0'
    calls = setup_fake_run(tmp_path, monkeypatch, name)
    r.run(name)
    assert [c[2] for c in calls] == [16,16,64,64,64]
    assert len({c[0] for c in calls}) == len({c[1] for c in calls}) == 1
    assert [c[3] for c in calls] == [1,2,1,2,3]
    assert all(c[5] == r.extension_rate(c[3], r.x10.learning_rate(2)) for c in calls[2:])
    out = tmp_path / name
    rows = json.loads((out / 'validation.json').read_text())
    assert [(r['depth'],r['phase_step']) for r in rows] == [(16,0),(16,1),(16,2),(64,0),(64,1),(64,2),(64,3)]
    assert all(row['retention_t16'] is not None for row in rows if row['depth'] == 64)
    transition = json.loads((out / 'transition.json').read_text())
    assert transition['taught_positions'] == [1,2,3,4]
    assert transition['checkpoint']['path'] != rows[3]['checkpoint']['path']
    assert r.x7.digest(transition['checkpoint']['path']) == transition['checkpoint']['sha256']
    assert len(list(out.rglob('*.pt'))) <= 4
    saved = torch.load(rows[-1]['checkpoint']['path'], weights_only=False)
    assert next(iter(saved['optimizer']['state'].values()))['step'].item() == 5
    result = json.loads((out/'result.json').read_text())
    assert result['total_steps'] == 5 and result['t64_steps'] == 3
    assert control.valid_result(name, result)


def test_teaching_ceiling_does_not_silently_extend(tmp_path, monkeypatch):
    name = 'e97_curriculum_seed0'
    calls = setup_fake_run(tmp_path, monkeypatch, name, criterion_at=99)
    r.run(name)
    assert [c[2] for c in calls] == [16,16,16]
    result = json.loads((tmp_path/name/'result.json').read_text())
    assert result['status'] == 'criterion_not_reached' and result['t64_steps'] == 0
    assert control.valid_result(name, result)


def test_sweep_fixed_ceiling_not_ignition_stopped(tmp_path, monkeypatch):
    name = 'gdn2_w512_seed0'
    calls = setup_fake_run(tmp_path, monkeypatch, name, criterion_at=0)
    r.run(name)
    assert len(calls) == 3 and all(c[2] == 16 and c[4] == 1/512 for c in calls)
    assert len(list((tmp_path/name).rglob('*.pt'))) <= 3
    assert control.valid_result(name, json.loads((tmp_path/name/'result.json').read_text()))


def test_watcher_waits_for_all_and_detects_tamper(tmp_path):
    for name in r.ARMS:
        r.x7.write_json(tmp_path/'receipts'/name/'worker.json', control.identity(os.getpid()))
    assert watch.status(tmp_path)[0] is None
    name = next(iter(r.ARMS))
    r.x7.write_json(tmp_path/'markers'/f'{name}.FAIL', {})
    assert watch.status(tmp_path)[0] is None  # other arms still watched
    (tmp_path/'markers'/f'{name}.FAIL').unlink()
    for name in r.ARMS:
        result = {'name': name, 'status': 'complete', 'total_steps': 500000, 'stop_reason': 'safety_ceiling'}
        if name == 'e97_curriculum_seed0':
            result.update(total_steps=900000,t64_steps=500000,transition={'step':400000})
        path = tmp_path/name/'result.json'
        r.x7.write_json(path, result)
        terminal = {'name':name,'exit_code':0,'result_sha256':r.x7.digest(path)}
        r.x7.write_json(tmp_path/'receipts'/name/'terminal.json', terminal)
        r.x7.write_json(tmp_path/'markers'/f'{name}.DONE', terminal)
    assert watch.status(tmp_path)[0] == 0
    (tmp_path/name/'result.json').write_text('{}')
    assert watch.status(tmp_path)[0] == 1


def test_watcher_growth_retention_and_right_censoring(tmp_path):
    for name in r.ARMS:
        r.x7.write_json(tmp_path/'receipts'/name/'worker.json', control.identity(os.getpid()))
        rows = [{'depth':16,'phase_step':500,**metrics(16, True)}]
        if name == 'e97_curriculum_seed0':
            r.x7.write_json(tmp_path/name/'transition.json', {'step':500,'entry_lr':.0001,
                'taught_positions':[1,2,3,4],'t16':metrics(16,True)})
            rows.extend([{'depth':64,'phase_step':s,**metrics(64,True),'retention_t16':metrics(16,s==0)} for s in (0,500)])
        r.x7.write_json(tmp_path/name/'validation.json', rows)
    watch.growth_report(tmp_path)
    report = (tmp_path/'GROWTH_REPORT.md').read_text()
    assert 'right-censored' in report and 'fixed-final=False' in report
    assert 'new transient: []' in report and '|P1|True|508|0|0|508|500|' in report


def test_worker_invalid_registration_does_not_release_unowned_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: (_ for _ in ()).throw(ValueError('bad source')))
    monkeypatch.setattr(control.signal, 'signal', lambda *_: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('no lease operation on invalid source')
    monkeypatch.setattr(control.subprocess, 'run', forbidden)
    assert control.worker('gdn2_w512_seed0') == 1
    terminal = json.loads((tmp_path/'markers/gdn2_w512_seed0.FAIL').read_text())
    assert 'bad source' in terminal['error']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='leased target GPU required')
@pytest.mark.parametrize('architecture', ['e97','gdn2'])
def test_gpu_weighted_update_and_depth_safe_eval(architecture):
    torch.manual_seed(99)
    model = r.x14.build_model(architecture).cuda()
    opt = r.x14.make_optimizer(model)
    rng, task = np.random.default_rng(99), r.x10.InvertibleQuadraticTask(61)
    for depth in ((16,64,16) if architecture == 'e97' else (16,)):
        row = r.train_update(model,opt,task.generate_batch(32,depth+2,rng),1,
                             weight=1/64 if depth == 16 else None)
        assert math.isfinite(row['loss'])
        evaluated = r.evaluate(model, task.generate_batch(512,depth+2,rng))
        assert len(evaluated['positions']) == depth
        assert math.isfinite(evaluated['aggregate']['loss'])
