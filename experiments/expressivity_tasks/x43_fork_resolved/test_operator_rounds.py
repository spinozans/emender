import pytest
import torch
from experiments.expressivity_tasks.x43_fork_resolved.test_c14_b_enabled import fixture
from experiments.expressivity_tasks.x43_fork_resolved.operator_rounds import IterativeForward, fingerprint


def test_k1_byte_exact_and_no_checkpoint_mutation():
    model,x,a,c=fixture();model.eval()
    state={n:v.clone() for n,v in model.state_dict().items()}
    with torch.inference_mode():
        expected,_=model(x,a,c,checkpointed=False)
    runner=IterativeForward(model,store_arm=True)
    actual,_=runner(x,a,c,windows=[(0,0),(1,0)],valid_lengths=[4,4],forks=torch.ones_like(x,dtype=torch.bool),k=1)
    assert fingerprint(actual)==fingerprint(expected)
    assert fingerprint(-actual[...,99])==fingerprint(-expected[...,99])
    assert all(torch.equal(state[n],v) for n,v in model.state_dict().items())


def test_rounds_repeat_deterministically_and_reset_at_window():
    model,x,a,c=fixture();model.eval()
    run=IterativeForward(model,store_arm=True)
    kwargs=dict(windows=[(0,0),(1,0)],valid_lengths=[3,4],forks=torch.ones_like(x,dtype=torch.bool),k=2)
    first,_=run(x,a,c,**kwargs); telemetry=run.telemetry.copy()
    second,_=run(x,a,c,**kwargs)
    assert torch.equal(first,second)
    assert telemetry==run.telemetry
    assert all(r['position']<kwargs['valid_lengths'][r['batch']] for r in telemetry)
    assert any(r['round']==2 for r in telemetry)
    # Causality: later byte/answer cannot alter an earlier emitted prediction.
    xx=x.clone();xx[:,3]=255
    changed,_=run(xx,a,c,**kwargs)
    assert torch.equal(first[:,:3],changed[:,:3])
    assert all(p.grad is None for p in model.parameters())


def test_invalid_round_budget_fails_closed():
    model,x,a,c=fixture();model.eval()
    with pytest.raises(ValueError):
        IterativeForward(model,store_arm=True)(x,a,c,windows=[(0,0),(1,0)],valid_lengths=[4,4],forks=x.bool(),k=3)


def test_actual_store_objects_survive_round_and_position(monkeypatch):
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_store import SharedStore
    model,x,a,c=fixture();model.eval()
    observed={};original=SharedStore.stream
    def record(self,*args,**kwargs):
        # Last positional argument is the input-position clock. Persistent
        # store identity must see 0,0,1,1: rounds do not advance input time.
        observed.setdefault(self,[]).append(args[-1])
        return original(self,*args,**kwargs)
    monkeypatch.setattr(SharedStore,'stream',record)
    run=IterativeForward(model,store_arm=True,epsilon=0.)
    run(x[:1,:2],a[:1,:2],c[:1,:2],windows=[(0,0)],valid_lengths=[2],forks=torch.ones(1,2,dtype=torch.bool),k=2)
    assert len(observed)==sum(block.cell.policy.N for block in model.head.base.blocks)
    assert all(clocks==[0,0,1,1] for clocks in observed.values())
