import ast
import inspect
import json
import pytest
import torch
from experiments.expressivity_tasks.x42_pile_scale import velocity_baselines as b
from experiments.expressivity_tasks.x42_pile_scale import velocity_train as original
from experiments.expressivity_tasks.x42_pile_scale.velocity_anchor_report import report

@pytest.mark.parametrize('arm',['transformer','gdn2'])
def test_capacity_and_tokenizer(arm):
    with torch.device('meta'):
        model=b.Baseline(arm)
    assert model.model.embed.num_embeddings==50257
    assert model.model.head.out_features==50257
    assert abs(b.counts(model)['trainable']/b.CAPACITY-1)<.001


def test_update_loop_is_identical():
    def loop(fn):
        tree=ast.parse(inspect.getsource(fn))
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.While))
        # Only emender's detached gate offset is inapplicable to baselines.
        node.body=[n for n in node.body if not (isinstance(n,ast.Assign) and
            any(isinstance(x,ast.Attribute) and x.attr=='gate_cost_lambda' for x in ast.walk(n)))]
        return ast.dump(node,include_attributes=False)
    assert loop(b.train)==loop(original.train)
    assert b.window is original.window
    assert b.TRAIN_SECONDS==2700 and b.EVAL_CONTEXT==2048


def test_score_asymmetry_and_nonfinite():
    r=b.baseline_metrics(5.3,24)
    assert r['score']==r['nll_heldout']==5.3
    assert r['occ_frac'] is None and r['degeneracy_flags'] is None
    with pytest.raises(ValueError):b.baseline_metrics(float('nan'),24)


def test_pending_and_failed_report(tmp_path):
    rows=report(tmp_path)
    assert rows[1]['status']=='queued'
    (tmp_path/'transformer_failure.json').write_text('{}')
    assert report(tmp_path)[1]['status']=='failed'
    assert 'pending' in (tmp_path/'ANCHOR.md').read_text()


def test_harness_one_update_cpu(monkeypatch,tmp_path):
    import numpy as np
    class Fake(torch.nn.Linear):
        def __init__(self,arm):super().__init__(1,1)
        def cuda(self):return self
        def loss(self,x,y):return self(x).square().mean()
    monkeypatch.setenv('GPU_LEASE_HELD','0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    for flag in b.PRODUCTION_FLAGS:monkeypatch.setenv(flag,'1')
    monkeypatch.setattr(b,'Baseline',Fake)
    monkeypatch.setattr(b,'configure_production',lambda:None)
    monkeypatch.setattr(b.torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(b.torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(b.np,'memmap',lambda *args,**kw:np.zeros(100))
    monkeypatch.setattr(b,'window',lambda *args:(torch.ones(1,1),torch.ones(1,1)))
    monkeypatch.setattr(b,'evaluate_baseline',lambda *args:b.baseline_metrics(5.,24))
    adam=b.torch.optim.AdamW
    monkeypatch.setattr(b.torch.optim,'AdamW',lambda params,**kw:adam(params,**{**kw,'fused':False}))
    registration=tmp_path/'reg.json';registration.write_text('{}')
    r=b.train('transformer',b.Policy(),registration,tmp_path/'run',seconds=.000001)
    assert r['steps']==1 and r['tokens_consumed']==64*2048
    assert r['status']=='engineering_only' and not r['score_eligible']
