"""Enabled integration gates; native journal certification remains separate."""
from dataclasses import replace
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_model import StoreIntegrated
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_store import SharedStore
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_suffix import queried_suffix
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Policy
from experiments.expressivity_tasks.x43_fork_resolved.c6 import count


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    model = StoreIntegrated().train()
    ids = torch.tensor([[97,98,97,0],[99,99,99,99]])
    answers = torch.zeros(2,4,260)
    answers[...,97] = 1
    answers[...,256:] = torch.tensor([1.,0.,0.,0.6931472])
    lengths = torch.ones(2,4,dtype=torch.long)
    return model,ids,answers,lengths


def test_inventory_enabled_finite_backward_and_checkpoint_parity():
    model,ids,answers,lengths = fixture()
    from experiments.expressivity_tasks.x43_fork_resolved.c11 import Boosted
    dense = Boosted('emender',2000000)
    assert count(model) == 2133700
    assert {k:tuple(v.shape) for k,v in model.named_parameters()} == {k:tuple(v.shape) for k,v in dense.named_parameters()}
    lp,_ = model(ids,answers,lengths,checkpointed=False)
    loss = -lp[...,99].mean(); loss.backward()
    gradients = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    assert torch.isfinite(lp).all()
    assert gradients['device_input.weight'].abs().sum() > 0
    assert all(torch.isfinite(g).all() for g in gradients.values())
    model.zero_grad(set_to_none=True)
    replay,_ = model(ids,answers,lengths,checkpointed=True)
    assert torch.equal(lp,replay)
    (-replay[...,99].mean()).backward()
    for name,p in model.named_parameters():
        if name in gradients:
            assert torch.equal(p.grad,gradients[name]), name
    assert all(r['peak_occupancy'] <= r['budget'] for layer in model.telemetry for r in layer)
    assert sum(r['array_slots_read'] for layer in model.telemetry for r in layer) > 0


def test_causal_prefix_reset_and_masked_padding():
    model,ids,answers,lengths = fixture()
    model.eval()
    base,_ = model(ids,answers,lengths,checkpointed=False)
    changed_ids = ids.clone(); changed_ids[:,3] = 255
    changed_answers = answers.clone(); changed_answers[:,3] = 0
    changed,_ = model(changed_ids,changed_answers,lengths,checkpointed=False)
    assert torch.equal(base[:,:3],changed[:,:3])
    short,_ = model(ids[:1,:2],answers[:1,:2],lengths[:1,:2],checkpointed=False)
    padded,_ = model(ids[:1],answers[:1],lengths[:1],valid_lengths=[2],checkpointed=False)
    assert torch.allclose(short,padded[:,:2],atol=1e-6,rtol=1e-6)
    repeat,_ = model(ids,answers,lengths,checkpointed=False)
    assert torch.equal(base,repeat)


def store():
    cfg = replace(Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.).configs()[0],budget=1)
    return SharedStore(cfg)


def test_pressure_duplicate_merge_stale_and_delayed_correction():
    s = store(); key = torch.ones(64); v = torch.ones(128,requires_grad=True)
    owner = queried_suffix((1,0),b'a',1)
    s.insert_array(key,v,owner,ord('b'),0,torch.tensor(.5))
    saved_key = s.slots[0].key
    # Wrong actual byte arrives as an observed INPUT at the next time only.
    s.observe((1,0),b'aa',ord('a'),v*3,1)
    assert s.ops['CORRECT'] == 1
    assert torch.equal(s.slots[0].value,v*3)
    s.insert_array(torch.zeros(64),v*2,owner,ord('b'),1,torch.tensor(.5))
    assert s.ops['duplicate_merges'] == 1
    assert s.slots[0].key is saved_key
    different = queried_suffix((1,0),b'ba',2)
    s.insert_array(key,v,different,ord('a'),1,torch.tensor(.5))
    assert s.ops['capacity_merges'] == 1
    assert len(s.live()) == 1
    s.observe((1,0),b'bac',ord('c'),v,2)
    assert not s.live()
    assert s.ops['stale_retirements'] == 1
    assert s.ops['peak_occupancy'] == 1


def test_pop_and_array_only_decay_empty_preserves_stream():
    s = store(); v = torch.ones(128); key = torch.ones(64)
    owner = queried_suffix((0,0),b'a',1)
    s.insert_array(key,v,owner,1,0,torch.tensor(1.))
    ticks = 0
    while s.live():
        s.decay(array_only=True); ticks += 1
        assert ticks < 1000
    assert s.ops['natural_retirements'] == 1
    assert ticks <= 129
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_store import Slot
    s.slots[0] = Slot(key,v,torch.tensor(1.),0)
    s.decay(array_only=True)
    assert float(s.slots[0].strength) == 1.


def test_zero_consultation_creates_no_array_and_bad_context_fails():
    import pytest
    model,ids,answers,lengths = fixture()
    model(ids,torch.zeros_like(answers),lengths,checkpointed=False)
    assert sum(r['array_requests'] for layer in model.telemetry for r in layer) == 0
    lengths[0,0] = 2
    with pytest.raises(ValueError,match='reset'):
        model(ids,answers,lengths)
