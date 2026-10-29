"""CPU regressions for checkpoint lifetime and auxiliary gradient wiring."""
from types import SimpleNamespace
import copy
import torch
from experiments.expressivity_tasks.x40_emender_lm.part6_checkpoint import checkpoint_block
from experiments.expressivity_tasks.x40_emender_lm.part6_memory import terms


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(3));self.cell=SimpleNamespace(gate_mean=None)
        self.recorded=[]
    def forward(self,x,ids):
        self.recorded.append(torch.is_grad_enabled())
        y=x*self.weight
        self.cell.gate_mean=y.mean()
        return y


def test_recorded_checkpoint_returns_gate_and_releases_attribute():
    block=Block();other=copy.deepcopy(block)
    x=torch.arange(6,dtype=torch.float32).reshape(2,3).requires_grad_()
    y=x.detach().clone().requires_grad_()
    direct=block(x,None);(direct.sum()+.5*block.cell.gate_mean).backward()
    value,gate=checkpoint_block(other,y,None)
    assert other.cell.gate_mean is None
    (value.sum()+.5*gate).backward()
    assert other.cell.gate_mean is None
    assert other.recorded==[True,True]
    assert torch.equal(value,direct)
    assert torch.equal(x.grad,y.grad)
    assert torch.equal(block.weight.grad,other.weight.grad)


def test_native_accounting_uses_the_heterogeneous_pilot_budgets():
    # Independent byte ledger for B32/T512, one block, four distinct policies.
    # The superclass budget 2944 is not the capacity of every pilot head.
    ledger=terms(1,32,512)
    assert ledger['native_payload_bytes']==62_128_128
    assert ledger['journal_reserved_bytes']==669_516_800
    assert ledger['trace_freed_bytes']==41_418_752


def test_completed_journal_packing_owns_compact_storage(monkeypatch):
    from experiments.expressivity_tasks.x40_emender_lm import model
    from experiments.expressivity_tasks.x40_emender_lm.part6_journal import packed_journals
    for stores in (1,3):
        journal=SimpleNamespace(addresses=torch.arange(stores*100).reshape(stores,100),
                                before=torch.zeros(stores,100,dtype=torch.int64),
                                widths=torch.ones(stores,100,dtype=torch.int32),
                                counts=torch.full((stores,),7),capacity=100)
        original=journal.addresses
        monkeypatch.setattr(model,'wired_forward',lambda *args,**kwargs:(None,SimpleNamespace(all_field_journal=journal),None,None))
        rows=[]
        with packed_journals(rows):model.wired_forward()
        assert journal.addresses.shape==(stores,7)
        assert torch.equal(journal.addresses,original[:,:7])
        assert journal.addresses.untyped_storage().nbytes()==stores*7*8
        assert journal.addresses.data_ptr()!=original.data_ptr()
        assert rows[0]['measured_max_events_per_lane']==7


def test_allocator_peak_is_reconstructed_before_terminal_cleanup():
    from experiments.expressivity_tasks.x40_emender_lm.part6_allocator import summarize
    events=[dict(action='alloc',addr=1,size=100,frames=[dict(filename='all_field_journal.py',name='allocate')]),
            dict(action='alloc',addr=2,size=50,frames=[]),
            dict(action='free_requested',addr=1),dict(action='oom',size=200),dict(action='free_requested',addr=2)]
    result=summarize(dict(device_traces=[events],segments=[]))['devices'][0]
    assert result['peak_allocated_bytes_from_history']==150
    assert result['categories']==dict(journals=100,unattributed=50)
    assert result['oom_events'][0]['size']==200
    assert result['oom_events'][0]['pending_free_bytes']==100
    assert result['oom_events'][0]['logical_categories']==dict(unattributed=50)


def test_microbatch_weighting_preserves_mean_loss_and_gradients():
    torch.manual_seed(6)
    model=torch.nn.Linear(3,5);other=copy.deepcopy(model)
    x=torch.randn(8,3);target=torch.randn(8,5)
    dense=(model(x)-target).square().mean();dense.backward()
    total=0.
    for start in range(0,8,2):
        loss=(other(x[start:start+2])-target[start:start+2]).square().mean()/4
        total+=float(loss.detach());loss.backward()
    torch.testing.assert_close(torch.tensor(total),dense.detach())
    for p,q in zip(model.parameters(),other.parameters()):torch.testing.assert_close(p.grad,q.grad)


def test_bitwise_gate_distinguishes_signed_zero_and_rejects_nan():
    from experiments.expressivity_tasks.x40_emender_lm.part6_gate import bitwise_equal
    assert bitwise_equal(torch.tensor(1.),torch.tensor(1.))
    assert not bitwise_equal(torch.tensor(0.),torch.tensor(-0.))
    assert not bitwise_equal(torch.tensor(float('nan')),torch.tensor(float('nan')))
