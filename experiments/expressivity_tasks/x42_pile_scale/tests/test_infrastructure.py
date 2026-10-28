import json
import numpy as np
import pytest
from experiments.expressivity_tasks.x42_pile_scale.cmaes import CMAES
from experiments.expressivity_tasks.x42_pile_scale.data import Stream, build
from experiments.expressivity_tasks.x42_pile_scale.model import memory
from experiments.expressivity_tasks.x42_pile_scale.search import decode, encode, CENTER
from experiments.expressivity_tasks.x42_pile_scale.train import gate

@pytest.mark.parametrize('rotated',[False,True])
def test_reference_quadratics(rotated):
    rng=np.random.default_rng(42)
    q=np.linalg.qr(rng.normal(size=(5,5)))[0]
    es=CMAES(np.full(5,3.),sigma=.8,population=12)
    best=float('inf')
    for _ in range(220):
        x=es.ask()
        z=x@q if rotated else x
        f=np.sum(z*z*(np.geomspace(1,100,5) if rotated else 1),axis=1)
        best=min(best,float(f.min()))
        es.tell(f)
    assert best < 1e-10


def test_sampler_exact_order(tmp_path):
    path=tmp_path/'tokens'
    np.arange(201,dtype='<u2').tofile(path)
    a,b=Stream(path,10),Stream(path,10)
    assert len({tuple(a.window(i)) for i in range(a.n)})==a.n
    assert all(np.array_equal(a.window(i),b.window(i)) for i in range(a.n*2))


def test_shape_memory():
    from pathlib import Path
    root=Path(__file__).parents[1]
    for name in ('pilot','final'):
        c=json.loads((root/'configs'/f'{name}.json').read_text())
        m=memory(c)
        assert m['headroom_48GiB'] > 4*2**30
    assert 1.2e9 < m['parameters'] < 1.5e9


def test_vector_roundtrip():
    v=decode(encode(CENTER))
    assert v['budget3']==1472
    assert v['match_coef']==pytest.approx(-1)


def test_gate_fails_closed(tmp_path):
    p=tmp_path/'verdict.json'
    p.write_text('{}')
    with pytest.raises(RuntimeError):
        gate(p,dict(architecture='pure'))


def test_pipeline_dedup_and_budget(tmp_path):
    class Encoded:
        ids=[1,2,3,4]*32
    class Tokenizer:
        def encode(self,text):
            return Encoded()
        def save(self,path):
            __import__('pathlib').Path(path).write_text('fixture tokenizer')
    rows=[dict(text=f'document {i}') for i in range(1000)]
    rows.insert(1,rows[0])
    receipt=build(rows,Tokenizer(),tmp_path/'corpus',4096,{'fixture':True},.5)
    assert receipt['duplicates']==1
    assert receipt['complete']
    assert sum(receipt['tokens'].values())*2 <=4096
    assert receipt['split_sha256']['train']==receipt['split_sha256']['valid'] # equal fixture tokens, distinct documents


def test_pure_cannot_silently_use_attention():
    from experiments.expressivity_tasks.x42_pile_scale.model import make_model
    with pytest.raises(ValueError):
        make_model({'architecture':'pure'})


def test_checkpointed_model_forward_backward():
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.model import LM
    model=LM(dict(width=16,heads=2,layers=1,activation_checkpointing=True))
    ids=torch.tensor([[1,2,3]])
    model(ids).float().square().mean().backward()
    assert torch.isfinite(model.embed.weight.grad).all()


def test_pair_rejects_exposure_mismatch(tmp_path):
    from experiments.expressivity_tasks.x42_pile_scale.paired import plan
    cfg=dict(seed=1,context=16,microbatch=1,accumulation=2,tokens=64,eval_every=1,eval_batches=1)
    (tmp_path/'manifest.json').write_text('{}')
    result=plan([cfg,cfg.copy()],tmp_path)
    assert result['steps']==2
    with pytest.raises(ValueError):
        plan([cfg,dict(cfg,seed=2)],tmp_path)


def test_existing_emender_adapter_pure_has_no_attention():
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.emender import factory, PureBlock
    cfg=dict(width=32,layers=1,heads=2,activation_checkpointing=True,architecture='pure',lane_positions=[0],microbatch=1,context=16)
    with torch.device('meta'):
        model=factory(cfg)
    assert all(isinstance(block,PureBlock) for block in model.blocks)
    assert model.blocks[0].lane.cell.match_coef.shape==(4,)
    assert model.extra_workspace_bytes>0
