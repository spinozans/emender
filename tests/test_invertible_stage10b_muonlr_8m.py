"""X10b source-named LR-only adapter and native-watcher regression tests."""
from argparse import Namespace
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import pytest
import torch
from torch import nn

from experiments.expressivity_tasks import run_invertible_stage10b_muonlr_8m as r
from experiments.expressivity_tasks import hybrid_muon as h
from scripts import watch_invertible_stage10b_muonlr as watch
from scripts import invertible_stage10b_control as control


@pytest.mark.parametrize('arm,multiplier', [('muonlr3x',3), ('muonlr10x',10)])
def test_lr_only_schedule_every_update_both_groups_and_auxiliary(arm, multiplier):
    model = nn.Module()
    model.embed = nn.Embedding(7, 4)
    layer = nn.Module()
    layer.qkv_proj = nn.Linear(4, 8, bias=False)
    model.layers = nn.ModuleList([layer])
    opt = r.make_optimizer(model, arm)
    assert isinstance(opt, h.HybridMuon)
    for step in range(500001):
        r.rates(opt, arm, step)
        assert all(g['lr'] == multiplier * r.x10.learning_rate(step) for g in opt.param_groups)
    for step in (1,5000,252500,500000):
        r.rates(opt, arm, step)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
        assert opt.aux.param_groups[0]['lr'] == r.learning_rate(arm,step)
    assert r.learning_rate(arm,5000) == pytest.approx(multiplier*1e-3)
    assert r.learning_rate(arm,500000) == pytest.approx(multiplier*1e-5)
    with pytest.raises(ValueError):
        r.learning_rate(arm,500001)


@pytest.mark.parametrize('arm', r.ARMS)
def test_exact_x10_initialization_routing_optimizer_and_pool(arm):
    torch.manual_seed(0)
    model = r.build_model(arm)
    torch.manual_seed(0)
    ref = r.x10.build_model('muon')
    assert all(torch.equal(v,ref.state_dict()[k]) for k,v in model.state_dict().items())
    reg = json.loads((control.BASELINE_ROOT/'registration.json').read_text())
    assert h.parameter_routes(model) == reg['parameter_routes']
    assert [hashlib.sha256(a.tobytes()).hexdigest() for a in r.x7.frozen_pools(0,16,61)['val']] == reg['pool_array_sha256']['0']
    assert r.make_optimizer(model,arm).state_dict()['family'] == h.VERSION
    assert r.point(arm) == r.x10.point('muon')
    assert r.CARDS[arm] not in (0,1,2)


@pytest.mark.parametrize('arm', r.ARMS)
def test_delegated_x10_loop_restores_globals_safety_ceiling_and_lr_logging(tmp_path,monkeypatch,arm):
    original = r.x10.build_model
    pool = r.x7.frozen_pools(0,16,61)['val']
    reg = {'authorized_runs':[f'{arm}_seed0'],
           'pool_array_sha256':{'0':[hashlib.sha256(a.tobytes()).hexdigest() for a in pool]},
           'parameter_routes':h.parameter_routes(r.build_model(arm)),
           'resource_bounds':{arm:{'wall_seconds':3600,'peak_allocated_bytes':8*1024**3}}}
    r.x7.write_json(tmp_path/'registration.json',reg)
    monkeypatch.setitem(r.ROOTS,arm,tmp_path)
    monkeypatch.setattr(r.x10,'verify_registration',lambda _:reg)
    monkeypatch.setattr(r.x10,'CEILING',3)
    monkeypatch.setattr(r.x10,'EVAL_EVERY',1)
    monkeypatch.setattr(torch.nn.Module,'cuda',lambda self:self)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    lengths=[]
    def forward(model,batch):
        lengths.append(batch[0].shape[1])
        y,mask=torch.from_numpy(batch[1])*0,torch.from_numpy(batch[2])
        return model.embed.weight[0,:2].expand(*y.shape,2),y,mask
    monkeypatch.setattr(r.x7,'forward_batch',forward)
    monkeypatch.setattr(r.x7,'evaluate',lambda *_:{'aggregate':{'accuracy':1.,'loss':1.},
        'positions':[{'correct':512} for _ in range(16)],'perfect_prefix_depth':16})
    r.run(Namespace(arm=arm,seed=0))
    assert r.x10.build_model is original
    assert lengths == [18,18,18]
    out=tmp_path/f'{arm}_seed0'
    result=json.loads((out/'result.json').read_text())
    assert result['total_steps']==3 and result['stop_reason']=='safety_ceiling'
    rows=[json.loads(s) for s in (out/'training.jsonl').read_text().splitlines()]
    assert [row['lr'] for row in rows] == [r.learning_rate(arm,s) for s in (1,2,3)]
    assert len(list(out.glob('*.pt'))) <= 3
    assert len(json.loads((out/'validation.json').read_text())[-1]['positions'])==16


def test_watcher_pending_failure_terminal_hash_binding(tmp_path):
    for name in watch.NAMES:
        r.x7.write_json(tmp_path/'receipts'/name/'worker.json',control.identity(os.getpid()))
    assert watch.status(tmp_path)[0] is None
    for name in watch.NAMES:
        path=tmp_path/name/'result.json'
        r.x7.write_json(path,{'status':'complete','total_steps':500000})
        terminal={'name':name,'exit_code':0,'result_sha256':r.x7.digest(path)}
        r.x7.write_json(tmp_path/'receipts'/name/'terminal.json',terminal)
        r.x7.write_json(tmp_path/'markers'/f'{name}.DONE',terminal)
    assert watch.status(tmp_path)[0]==0
    (tmp_path/watch.NAMES[0]/'result.json').write_text('{}')
    assert watch.status(tmp_path)[0]==1
    (tmp_path/'markers'/f'{watch.NAMES[0]}.FAIL').touch()
    assert watch.status(tmp_path)[0]==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='exclusive target GPU required')
@pytest.mark.parametrize('arm',r.ARMS)
def test_gpu_higher_lr_full_model_update_and_per_position_validation(arm):
    torch.manual_seed(99)
    model=r.build_model(arm).cuda()
    opt=r.make_optimizer(model,arm)
    pool=r.InvertibleQuadraticTask(61).generate_batch(512,18,r.np.random.default_rng(99))
    for step in (1,5000):
        metrics=r.train_update(model,opt,arm,tuple(a[:32] for a in pool),step)
        assert math.isfinite(metrics['loss'])
        assert metrics['lr']==r.learning_rate(arm,step)
    positions=r.x7.evaluate(model,pool)
    assert len(positions['positions'])==16
    assert sum(p['correct'] for p in positions['positions'])==positions['aggregate']['correct']
    assert math.isfinite(positions['aggregate']['loss'])
