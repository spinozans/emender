"""The isolated diagnostic executes the production fresh core faithfully."""
import torch
from experiments.expressivity_tasks.x35_tape_cell_v3 import cell_fused as cell
from experiments.expressivity_tasks.x35_tape_cell_v3.structural_cost_part5 import preparation, launch
from experiments.expressivity_tasks.x35_tape_cell_v3.tests.test_backing import inputs


def test_prepared_core_matches_complete_scan_routes():
    cfg = cell.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=2,budget=8)
    xs = inputs(cfg,8)
    prepare = preparation()
    with torch.no_grad():
        captured,state = prepare(*xs,None,cfg)
        assert state.native.live.shape == (1,2,8)
        assert not state.native.live.any()
        launch(captured)
        _,final,trace,_ = cell._tape_scan(*xs,None,cfg)
        # Kernel signature positions are asserted by comparison with public
        # decisions, catching drift in capture boundaries or argument layout.
        args,_ = captured
        for index,name in ((13,'case'),(14,'matched'),(15,'alloc_target'),(16,'pop_target')):
            assert torch.equal(args[index].transpose(1,2),getattr(trace,name))
        assert torch.equal(state.native.keys,final.native.keys)
        assert torch.equal(state.native.values,final.native.values)


def test_diagnostic_samples_support_registered_accounting(monkeypatch):
    from experiments.expressivity_tasks.x35_tape_cell_v3 import structural_cost_part5 as diagnostic
    from experiments.expressivity_tasks.x35_tape_cell_v3.gates_part5 import account
    # Exercise the diagnostic accounting path without collecting performance
    # samples before full parity. Preserve physical_slots returned by account.
    def accounting_only(fn,tokens):
        sample = fn()
        return account(sample)
    monkeypatch.setattr(diagnostic,'measure_core',lambda *args: {})
    rows = diagnostic.collect(accounting_only)
    assert len(rows) == 6
    assert all(row['physical_slots'] == 16 for row in rows)
