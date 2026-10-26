import pytest
import torch
from .test_parity import device
from experiments.expressivity_tasks.x30_tape_cell import naive_reference as oracle
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import tape_scan


def tensors(state):
    return (state.native.keys,state.native.values,state.native.strength,
            state.native.live,state.native.age,state.history,*state.metadata)


@pytest.mark.parametrize('slots',[1,7,37,736,11776])
def test_all_field_exact_rollback(slots):
    dev=device()
    torch.manual_seed(72)
    cfg=oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=2,budget=slots,
                              decay_rate=.8,free_threshold=.2)
    initial=oracle.init_state(2,cfg,dev)
    initial.keys.normal_();initial.values.normal_();initial.strength.uniform_()
    initial.live.copy_(torch.rand_like(initial.strength)>.3)
    initial.age.copy_(torch.randint(0,4,initial.age.shape,device=dev))
    k,q=[torch.randn(2,9,2,4,device=dev) for _ in range(2)]
    v=torch.randn(2,9,2,5,device=dev)
    beta=torch.ones(2,9,2,device=dev);beta[:,2]=0
    pop=torch.rand_like(beta);logit=torch.randn_like(beta)
    inputs=(k,q,v,beta,pop,logit)
    _,fresh,_,_=tape_scan(*inputs,initial,cfg,record_all=True)
    assert fresh.all_field_journal.counts.gt(0).all()
    restored=fresh.all_field_journal.undo_(fresh)
    assert restored.metadata is None and restored.tokens==0
    for name in ('keys','values','strength','live','age'):
        assert torch.equal(getattr(restored.native,name),getattr(initial,name))
    assert restored.history.eq(0).all()
    # Persistent metadata, including bitmap words, inverse heap positions,
    # terminal strengths, lazy refs, timestamps, birth links and history.
    _,mid,_,_=tape_scan(*inputs,initial,cfg)
    expected=[x.view(torch.uint8).clone() for x in tensors(mid)]
    pointers=[x.data_ptr() for x in tensors(mid)]
    _,end,_,_=tape_scan(*inputs,mid,cfg,record_all=True)
    journal=end.all_field_journal
    assert pointers==[x.data_ptr() for x in tensors(end)]
    restored=journal.undo_(end)
    assert restored.tokens==9
    for actual,old in zip(tensors(restored),expected):
        assert torch.equal(actual.view(torch.uint8),old)
    with pytest.raises(ValueError,match='unconsumed'):
        journal.undo_(end)
    # Restored metadata must remain usable, not merely look like old tensors.
    expected_state=restored.snapshot()
    out,actual,trace,_=tape_scan(*inputs,restored,cfg)
    expected_out,expected_final,expected_trace,_=tape_scan(*inputs,expected_state,cfg)
    assert torch.equal(out,expected_out)
    assert torch.equal(trace.case,expected_trace.case)
    for a,b in zip(tensors(actual),tensors(expected_final)):
        assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))


def test_all_field_snapshot_parent_and_pressure_expiry():
    dev=device()
    cfg=oracle.TapeHeadConfig(key_dim=2,value_dim=2,n_heads=1,budget=1,decay_rate=1.)
    initial=oracle.init_state(1,cfg,dev)
    initial.live.fill_(True);initial.strength.fill_(1);initial.age.fill_(3)
    initial.values.fill_(2)
    k=torch.ones(1,130,1,2,device=dev);v=torch.full_like(k,10)
    beta=torch.full((1,130,1),.75,device=dev);off=torch.zeros_like(beta)
    _,first,_,_=tape_scan(k,k,v,beta,off,off,initial,cfg,record_all=True)
    branch=first.snapshot()
    before=first.native.values.clone()
    restored=branch.all_field_journal.undo_(branch)
    assert torch.equal(restored.native.values,initial.values)
    assert torch.equal(first.native.values,before)
    assert restored.history.eq(0).all()
    parent=first.all_field_journal
    _,second,_,_=tape_scan(k,k,v,beta,off,off,first,cfg,record_all=True)
    restored=second.all_field_journal.undo_(second)
    assert restored.all_field_journal is parent and restored.tokens==130
    restored=parent.undo_(restored)
    assert restored.tokens==0 and restored.history.eq(0).all()
    assert torch.equal(restored.native.values,initial.values)


def test_oversized_journal_fails_before_allocation():
    from experiments.expressivity_tasks.x31_tape_cell_fused.all_field_journal import AllFieldJournal
    with pytest.raises(ValueError,match='int32 index range'):
        AllFieldJournal.allocate(1024,16,11776,64,128,(torch.empty(0),),0,True)


@pytest.mark.parametrize("write", [False, True])
def test_compact_reservation_and_unchanged_capture(write):
    from experiments.expressivity_tasks.x31_tape_cell_fused.all_field_journal import AllFieldJournal
    dev=device()
    reservations=[]
    entries=[]
    for slots in (736,11776):
        cfg=oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=slots,decay_rate=1.)
        # Equal event streams with different physical backing capacity.
        k=torch.zeros(1,4,1,4,device=dev)
        v=torch.zeros(1,4,1,5,device=dev)
        off=torch.zeros(1,4,1,device=dev)
        beta=torch.ones_like(off) if write else off
        _,state,_,_=tape_scan(k,k,v,beta,off,off,None,cfg,record_all=True)
        journal=state.all_field_journal
        for buffer in (journal.addresses,journal.before,journal.widths):
            assert buffer.untyped_storage().nbytes()==buffer.numel()*buffer.element_size()
        reservations.append(journal.nbytes)
        entries.append(int(state.all_field_journal.counts.sum()))
        state.all_field_journal.undo_(state)
    assert max(reservations)/min(reservations)<1.5
    # Newly initialized heap size can differ from its allocator before-image;
    # native/metadata dense spills have no changed bits in this fixture.
    if write:
        assert entries[0]==entries[1] and entries[0]>1
    else:
        assert max(entries)<=1
