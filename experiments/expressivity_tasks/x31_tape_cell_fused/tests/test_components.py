import torch
import pytest
from .test_parity import device
from experiments.expressivity_tasks.x31_tape_cell_fused.append_phase import allocate_free_ranks,append_phase_
from experiments.expressivity_tasks.x31_tape_cell_fused.sparse_checkpoint import DeltaJournal
from experiments.expressivity_tasks.x30_tape_cell.float_decay import f32


def test_rank_scan_and_parallel_mutation():
    dev = device()
    n,s,t,dk,dv = 2,7,3,4,5
    live = torch.tensor([[1,0,1,0,0,0,0],[0,1,0,1,0,1,0]],device=dev,dtype=torch.bool)
    slots,overflow = allocate_free_ranks(live,t)
    assert not overflow.any()
    assert slots.tolist() == [[1,3,4],[0,2,4]]
    keys = torch.zeros(n,s,dk,device=dev); values = torch.zeros(n,s,dv,device=dev)
    strength = torch.zeros(n,s,device=dev); age = torch.zeros(n,s,device=dev,dtype=torch.int64)
    torch.manual_seed(3)
    kin = torch.randn(n,t,dk,device=dev); v = torch.randn(n,t,dv,device=dev)
    gates = torch.rand(n,t,device=dev)
    curve = [1.]
    for _ in range(t): curve.append(f32(curve[-1]*f32(.8)))
    append_phase_(keys,values,strength,live,age,kin,v,gates,slots,torch.tensor(curve,device=dev))
    for i in range(n):
        for j in range(t):
            slot = slots[i,j]
            torch.testing.assert_close(keys[i,slot],kin[i,j])
            torch.testing.assert_close(values[i,slot],gates[i,j]*v[i,j])
            assert strength[i,slot] == curve[t-j] and age[i,slot] == t-j
    _,overflow = allocate_free_ranks(live,3)
    assert overflow.all()


def test_reverse_journal_repeated_slot_and_padding():
    dev = device()
    store = torch.zeros(1,7,3,device=dev)
    store[0,2] = 9
    # slot 2 changed from 1 to 4, then from 4 to 9; padding does nothing.
    journal = DeltaJournal(torch.tensor([[2,2,-1]],device=dev),
                           torch.tensor([[[1.,1.,1.],[4.,4.,4.],[0.,0.,0.]]],device=dev))
    journal.undo_(store)
    assert store[0,2].eq(1).all() and store[0,1].eq(0).all()
    # Bytes depend on events and fields, never on the backing slot budget.
    assert journal.nbytes == 3*8+3*3*4
    larger = torch.zeros(1,11776,3,device=dev)
    journal.undo_(larger)
    assert larger[0,2].eq(1).all() and journal.nbytes == 60


def test_rollback_invalid_slot_fails_before_launch():
    journal = DeltaJournal(torch.tensor([[7]]),torch.ones(1,1,3))
    with pytest.raises(ValueError,match='outside store'):
        journal.undo_(torch.zeros(1,7,3))
