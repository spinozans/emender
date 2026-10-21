"""X16E fresh extension clock, preserved moments, retention, and fail-closed receipts."""
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from scripts import invertible_stage16_extension as r


def metrics(depth, count=512):
    return {'aggregate': {'accuracy': .1, 'loss': 1.},
            'positions': [{'driver_position': j+1, 'correct': count if j < 3 else 0} for j in range(depth)]}


def test_fresh_cosine_and_bounds():
    assert r.CEILING == 500000 and r.EVAL_EVERY == 500 and r.GPU == 6
    assert r.learning_rate(0) == pytest.approx(3e-4)
    assert r.learning_rate(250000) == pytest.approx(.0001515)
    assert r.learning_rate(500000) == pytest.approx(3e-6)
    with pytest.raises(ValueError):
        r.learning_rate(500001)
    with pytest.raises(ValueError):
        r.learning_rate(-1)


def test_extension_lifecycle_preserves_state_and_uses_only_t64(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'CEILING', 3)
    monkeypatch.setattr(r, 'EVAL_EVERY', 1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '6')
    monkeypatch.setattr(r, 'disk_guard', lambda: None)
    monkeypatch.setattr(r.prior, 'disk_guard', lambda: None)
    monkeypatch.setattr(r, 'write_json', r.x7.write_json)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda _: None)
    monkeypatch.setattr(r.x16.x14, 'build_model', lambda _: torch.nn.Linear(1,1))
    model = torch.nn.Linear(1,1)
    opt = r.x16.x14.make_optimizer(model)
    next(model.parameters()).sum().backward()
    opt.step()
    rng = np.random.default_rng(0)
    cp = tmp_path / 'teach.pt'
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(), 'cuda_rng': [],
        'step': 500000, 'config': {'depth':16}, 'optimizer_family':'adamw'}, cp)
    monkeypatch.setattr(r, 'CHECKPOINT', cp)
    (tmp_path / 'registration.json').write_text('{}')
    pools = {d: r.x7.frozen_pools(0,d,61)['val'] for d in (16,64)}
    reg = {'pool_array_sha256': {str(d):[hashlib.sha256(a.tobytes()).hexdigest() for a in pool] for d,pool in pools.items()},
           'source_checkpoint': {'path':str(cp)}, 'teach_counts': [p['correct'] for p in metrics(16)['positions']],
           'resource_bounds':{'wall_seconds':1000,'peak_allocated_bytes':100000}}
    monkeypatch.setattr(r, 'verify', lambda: reg)
    calls, evaluations = [], []
    def update(model,opt,batch,step,weight=None,lr=None):
        calls.append((id(model),id(opt),batch[0].shape[1]-2,step,weight,lr))
        opt.zero_grad()
        next(model.parameters()).sum().backward()
        opt.step()
        return {'optimizer_seconds':0., 'loss':1.}
    def evaluate(model,pool):
        depth = pool[0].shape[1]-2
        evaluations.append(depth)
        return metrics(depth)
    monkeypatch.setattr(r.x16,'train_update',update)
    monkeypatch.setattr(r.x16,'evaluate',evaluate)
    r.train()
    assert [c[2:5] for c in calls] == [(64,1,None),(64,2,None),(64,3,None)]
    assert len({c[0] for c in calls}) == len({c[1] for c in calls}) == 1
    assert [c[5] for c in calls] == [r.learning_rate(s) for s in (1,2,3)]
    assert evaluations == [64,16]*4
    out = tmp_path/r.NAME
    rows = json.loads((out/'validation.json').read_text())
    assert [row['phase_step'] for row in rows] == [0,1,2,3]
    assert all(len(row['retention_t16']['positions']) == 16 for row in rows)
    saved = torch.load(rows[-1]['checkpoint']['path'], weights_only=False)
    assert next(iter(saved['optimizer']['state'].values()))['step'].item() == 4
    assert len(list(out.rglob('*.pt'))) <= 3
    assert cp.exists()  # Never prune terminal teach evidence.
    assert r.valid_result(json.loads((out/'result.json').read_text()))


def test_watcher_tamper_and_retention_growth(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'disk_guard', lambda: None)
    monkeypatch.setattr(r.prior, 'write_text', lambda path,text: path.write_text(text))
    receipt = tmp_path/'receipts'/r.NAME
    r.x7.write_json(receipt/'worker.json', r.identity(os.getpid()))
    assert r.status()[0] is None
    rows = [{'phase_step':s,'depth':64, **metrics(64), 'retention_t16':metrics(16,c)} for s,c in ((0,512),(500,10))]
    r.x7.write_json(tmp_path/r.NAME/'validation.json',rows)
    r.growth_report()
    report = (tmp_path/'GROWTH_REPORT.md').read_text()
    assert 'right-censored' in report and 'new transient=[]' in report
    assert '|P1|512|10|10|502|500|' in report
    result = {'name':r.NAME,'status':'complete','total_steps':500000,'t64_steps':500000,
        'stop_reason':'safety_ceiling','final':{'phase_step':500000,'depth':64,**metrics(64),'retention_t16':metrics(16)}}
    path = tmp_path/r.NAME/'result.json'
    r.x7.write_json(path,result)
    terminal = {'name':r.NAME,'exit_code':0,'result_sha256':r.x7.digest(path)}
    r.x7.write_json(receipt/'terminal.json',terminal)
    r.x7.write_json(tmp_path/'markers'/f'{r.NAME}.DONE',terminal)
    assert r.status()[0] == 0
    path.write_text('{}')
    assert r.status()[0] == 1


def test_invalid_registration_never_acquires_or_releases_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    monkeypatch.setattr(r,'verify',lambda: (_ for _ in ()).throw(ValueError('changed checkpoint')))
    monkeypatch.setattr(r.signal,'signal',lambda *_:None)
    def forbidden(*args,**kwargs):
        raise AssertionError('lease invoked before verification')
    monkeypatch.setattr(r.subprocess,'run',forbidden)
    assert r.worker() == 1
    assert 'changed checkpoint' in json.loads((tmp_path/'markers'/f'{r.NAME}.FAIL').read_text())['error']


def test_verify_rejects_changed_checkpoint(tmp_path,monkeypatch):
    checkpoint = tmp_path/'teach.pt'
    checkpoint.write_bytes(b'tampered')
    monkeypatch.setattr(r,'CHECKPOINT',checkpoint)
    monkeypatch.setattr(r.x10,'verify_registration', lambda _: {'authorized_runs':[r.NAME],
        'gpu_binding':{r.NAME:6},'artifact_sha256':{},
        'source_checkpoint':{'path':str(checkpoint),'sha256':r.CHECKPOINT_SHA256}})
    with pytest.raises(ValueError,match='changed terminal checkpoint'):
        r.verify()
