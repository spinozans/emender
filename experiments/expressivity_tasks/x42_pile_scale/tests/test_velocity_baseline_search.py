import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x42_pile_scale import velocity_baseline_search as space
from experiments.expressivity_tasks.x42_pile_scale import velocity_baseline_parent as parent
from experiments.expressivity_tasks.x42_pile_scale import velocity_baseline_train as train


@pytest.mark.parametrize('arm', ['transformer','gdn2'])
def test_space_and_capacity(arm):
    assert len(space.dimensions(arm)) == (6 if arm == 'transformer' else 9)
    for vector in (space.center(arm), np.zeros(len(space.dimensions(arm))), np.ones(len(space.dimensions(arm)))):
        policy = space.decode(arm,vector)
        space.validate_policy(policy,arm)
        with torch.device('meta'): model=space.SearchBaseline(arm,policy)
        assert abs(space.anchors.counts(model)['trainable']/space.anchors.CAPACITY-1)<.001
        if arm=='gdn2':
            assert model.model.blocks[0].layer.gdn2.gdn2.expand_v == policy.expansion
    with pytest.raises(ValueError): space.decode(arm,[float('nan')]*len(space.dimensions(arm)))


@pytest.mark.parametrize('arm', ['transformer','gdn2'])
def test_resume_failure_and_cap(tmp_path,monkeypatch,arm):
    reg=tmp_path/'reg.json';reg.write_text('{}');out=tmp_path/'run'
    launched=[];calls=0
    class Child:
        pid=-1
        def poll(self):return 0
    def launch(arm,policy,path,registration):
        path.mkdir(parents=True,exist_ok=True);launched.append(path.name)
        return Child(),path
    def result(*args):
        nonlocal calls
        calls+=1
        if calls==5:raise RuntimeError('mock child failed')
        return dict(score=2.)
    monkeypatch.setattr(parent,'verify_search',lambda *a:dict(population=4,cma_seed=4202))
    monkeypatch.setattr(parent,'anchor_reference',lambda *a:None)
    monkeypatch.setattr(parent,'launch',launch)
    monkeypatch.setattr(parent,'wait_result',result)
    monkeypatch.setattr(parent,'process_receipt',lambda c,p:dict(pid=-1,output=str(p),start_ticks='0'))
    with pytest.raises(RuntimeError,match='mock child failed'):parent.run(arm,reg,out,2)
    saved=json.loads((out/'generation.json').read_text())
    assert saved['cma']['generation']==1 and saved['cma']['phase']=='asked'
    pop=saved['cma']['x'];before=len(launched)
    parent.run(arm,reg,out,2)
    assert launched[before:]==[f'g2_c{i:02}' for i in range(4)]
    assert json.loads((out/'generation.json').read_text())['cma']['x']==pop
    before=len(launched);parent.run(arm,reg,out,2)
    assert len(launched)==before
    assert json.loads((out/'parent.json').read_text())['status']=='generation_cap_reached'
    with pytest.raises(ValueError):parent.run(arm,reg,out,0)


def test_partial_generation_preserves_completed(tmp_path,monkeypatch):
    reg=tmp_path/'reg.json';reg.write_text('{}');out=tmp_path/'run';names=[]
    class Child:
        pid=-1
        def poll(self):return 0
    def launch(arm,policy,path,registration):
        path.mkdir(parents=True,exist_ok=True);names.append(path.name);return Child(),path
    def result(*args):
        if len(names)==3:raise RuntimeError('failed')
        return dict(score=1.)
    monkeypatch.setattr(parent,'verify_search',lambda *a:dict(population=4,cma_seed=4202))
    monkeypatch.setattr(parent,'anchor_reference',lambda *a:None)
    monkeypatch.setattr(parent,'launch',launch);monkeypatch.setattr(parent,'wait_result',result)
    monkeypatch.setattr(parent,'process_receipt',lambda c,p:dict(pid=-1,output=str(p),start_ticks='0'))
    with pytest.raises(RuntimeError):parent.run('transformer',reg,out,1)
    saved=json.loads((out/'generation.json').read_text());assert set(saved['completed'])=={'0','1'}
    parent.run('transformer',reg,out,1)
    assert names==['g1_c00','g1_c01','g1_c02','g1_c02','g1_c03']


def test_endpoint_fail_closed(tmp_path):
    class Child:
        pid=1
        def wait(self):return 0
    policy=space.SearchPolicy()
    good=dict(status='completed',score_eligible=True,train_seconds=2701,
        train_budget_seconds=2700,batch=64,context=2048,eval_windows=24,arm='gdn2',
        registration_sha256='digest',policy=asdict(policy),physical_gpu='5',score=3.,nll_heldout=3.)
    for change in ({},dict(physical_gpu='7'),dict(score=4.),dict(train_seconds=2699),
                   dict(registration_sha256='wrong'),dict(batch=32),dict(arm='transformer')):
        (tmp_path/'result.json').write_text(json.dumps({**good,**change}))
        if change:
            with pytest.raises(RuntimeError):parent.wait_result(Child(),tmp_path,'gdn2','digest',policy)
        else:assert parent.wait_result(Child(),tmp_path,'gdn2','digest',policy)['score']==3.


def test_microbatch_preserves_update(monkeypatch,tmp_path):
    # A real optimizer compares B64 versus accumulated B8, on the same window.
    instances=[]
    class Fake(torch.nn.Linear):
        def __init__(self,*args):super().__init__(1,1);instances.append(self)
        def cuda(self):return self
        def loss(self,x,y):return (self(x)-y).square().mean()
    monkeypatch.setenv('GPU_LEASE_HELD','5');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','5')
    for flag in train.PRODUCTION_FLAGS:monkeypatch.setenv(flag,'1')
    monkeypatch.setattr(train,'SearchBaseline',Fake)
    monkeypatch.setattr(train,'configure_production',lambda:None)
    monkeypatch.setattr(train.torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(train.torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(train.np,'memmap',lambda *a,**kw:np.zeros(100))
    monkeypatch.setattr(train,'window',lambda *a:(torch.arange(64.).view(-1,1)/64,torch.ones(64,1)))
    monkeypatch.setattr(train,'evaluate_baseline',lambda *a:space.anchors.baseline_metrics(5.,24))
    adam=train.torch.optim.AdamW
    monkeypatch.setattr(train.torch.optim,'AdamW',lambda params,**kw:adam(params,**{**kw,'fused':False}))
    reg=tmp_path/'reg.json';reg.write_text('{}')
    for micro in (64,8):
        clock=iter([0.,2701.]);monkeypatch.setattr(train.time,'monotonic',lambda:next(clock,2701.))
        result=train.train('transformer',replace(space.SearchPolicy(),microbatch=micro),reg,tmp_path/str(micro))
        assert result['steps']==1 and result['tokens_consumed']==64*2048
    for a,b in zip(instances[0].parameters(),instances[1].parameters()):torch.testing.assert_close(a,b)
