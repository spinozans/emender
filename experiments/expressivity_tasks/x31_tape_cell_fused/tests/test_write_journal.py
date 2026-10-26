import pytest
import torch

from .test_parity import device
from experiments.expressivity_tasks.x30_tape_cell import naive_reference as oracle
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import tape_scan
from experiments.expressivity_tasks.x31_tape_cell_fused.write_journal import WriteJournal


@pytest.mark.parametrize('slots',[1,7,736,11776])
def test_live_payload_before_images_and_exact_undo(slots):
    dev = device()
    torch.manual_seed(72)
    b,h,t,dk,dv = 2,2,9,4,5
    cfg = oracle.TapeHeadConfig(key_dim=dk,value_dim=dv,n_heads=h,budget=slots,
                               decay_rate=.8,free_threshold=.2)
    initial = oracle.init_state(b,cfg,dev)
    initial.keys.normal_(); initial.values.normal_(); initial.strength.uniform_()
    initial.live.copy_(torch.rand_like(initial.strength)>.3)
    initial.age.copy_(torch.randint(0,4,initial.age.shape,device=dev))
    k,q = [torch.randn(b,t,h,dk,device=dev) for _ in range(2)]
    v = torch.randn(b,t,h,dv,device=dev)
    beta = torch.ones(b,t,h,device=dev); beta[:,2] = 0
    pop = torch.rand_like(beta)
    logit = torch.randn_like(beta); coef = torch.randn(h,device=dev)
    inputs = (k,q,v,beta,pop,logit)
    actual,final,trace,_ = tape_scan(*inputs,initial,cfg,coef,record_payload=True)
    journal = final.write_journal
    ids = torch.where(trace.alloc_target>=0,trace.alloc_target,
                      torch.where((trace.case==4)|(trace.case==5),trace.matched,-1))
    assert torch.equal(journal.slots,ids.transpose(1,2).reshape(b*h,t))
    # Independent one-token oracle boundaries give the true old payload at
    # each write, including multiple writes to the same slot within a chunk.
    native = initial
    expected = []
    for step in range(t):
        for row in range(b*h):
            slot = journal.slots[row,step].item()
            if slot >= 0:
                assert torch.equal(journal.keys[row,step],native.keys.reshape(b*h,slots,dk)[row,slot])
                torch.testing.assert_close(journal.values[row,step],
                    native.values.reshape(b*h,slots,dv)[row,slot],atol=1e-5,rtol=1e-5)
            else:
                assert journal.keys[row,step].eq(0).all()
                assert journal.values[row,step].eq(0).all()
        out,native,_ = oracle.tape_scan(*(x[:,step:step+1] for x in inputs),
                                         native,cfg,append_err_coef=coef)
        expected.append(out)
    torch.testing.assert_close(actual,torch.cat(expected,1),atol=1e-5,rtol=1e-5)
    # Preserve the owned continuation and rollback an explicit diagnostic copy.
    keys,values = final.native.keys.clone(),final.native.values.clone()
    journal.undo_payload_(keys,values)
    assert torch.equal(keys,initial.keys)
    assert torch.equal(values,initial.values)
    # Reserved memory is O(tokens * payload width), independent of S. This
    # is not a claim of compact O(actual mutations) event allocation.
    assert journal.nbytes == b*h*t*(8+4*(dk+dv))


def test_repeated_cap_corrections_capture_intermediate_values():
    dev = device()
    cfg = oracle.TapeHeadConfig(key_dim=2,value_dim=2,n_heads=1,budget=1,decay_rate=1.)
    native = oracle.init_state(1,cfg,dev)
    native.live.fill_(True); native.strength.fill_(1); native.age.fill_(3)
    native.values.fill_(2)
    k = torch.ones(1,4,1,2,device=dev)
    v = torch.full_like(k,10)
    beta = torch.full((1,4,1),.75,device=dev)
    off = torch.zeros_like(beta)
    _,final,trace,_ = tape_scan(k,k,v,beta,off,off,native,cfg,record_payload=True)
    assert trace.case.eq(5).all()
    assert final.write_journal.slots.eq(0).all()
    expected = torch.tensor([2.,8.,9.5,9.875],device=dev)
    assert torch.equal(final.write_journal.values[0,:,0],expected)
    assert torch.equal(final.write_journal.values[0,:,1],expected)
    branch = final.snapshot()
    assert branch.write_journal.slots.data_ptr() != final.write_journal.slots.data_ptr()
    assert branch.write_journal.keys.data_ptr() != final.write_journal.keys.data_ptr()
    assert branch.write_journal.values.data_ptr() != final.write_journal.values.data_ptr()
    branch.write_journal.values.zero_()
    assert torch.equal(final.write_journal.values[0,:,0],expected)


def test_payload_rollback_rejects_invalid_slot():
    journal = WriteJournal.allocate(1,1,2,3,'cpu')
    journal.slots.fill_(7)
    with pytest.raises(ValueError,match='outside store'):
        journal.undo_payload_(torch.zeros(1,7,2),torch.zeros(1,7,3))


def test_payload_continuation_reuses_storage_and_undoes_both_chunks():
    dev = device()
    cfg = oracle.TapeHeadConfig(key_dim=2,value_dim=2,n_heads=1,budget=1,decay_rate=1.)
    native = oracle.init_state(1,cfg,dev)
    native.live.fill_(True); native.strength.fill_(1); native.age.fill_(3)
    native.values.fill_(2)
    k = torch.ones(1,2,1,2,device=dev)
    v = torch.full_like(k,10)
    beta = torch.full((1,2,1),.75,device=dev); off = torch.zeros_like(beta)
    _,mid,_,_ = tape_scan(k,k,v,beta,off,off,native,cfg,record_payload=True)
    first = mid.write_journal
    pointers = [mid.native.keys.data_ptr(),mid.native.values.data_ptr()]
    _,end,_,_ = tape_scan(k,k,v,beta,off,off,mid,cfg,record_payload=True)
    assert pointers == [end.native.keys.data_ptr(),end.native.values.data_ptr()]
    assert torch.equal(first.values[0,:,0],torch.tensor([2.,8.],device=dev))
    assert torch.equal(end.write_journal.values[0,:,0],torch.tensor([9.5,9.875],device=dev))
    keys,values = end.native.keys.clone(),end.native.values.clone()
    end.write_journal.undo_payload_(keys,values)
    first.undo_payload_(keys,values)
    assert torch.equal(keys,native.keys) and torch.equal(values,native.values)
