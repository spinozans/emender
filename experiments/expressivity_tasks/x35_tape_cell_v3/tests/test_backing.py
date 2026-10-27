"""Allocation-horizon backing, promotion, replay and architectural cap checks.

These checks do not certify live-row pooling; historical touched rows remain
resident, a documented structural residual.
"""
import pytest
import torch
from experiments.expressivity_tasks.x35_tape_cell_v3 import cell_fused as candidate
from experiments.expressivity_tasks.x35_tape_cell_v3.backing import PrefixNative, FIELDS
from experiments.expressivity_tasks.x31_tape_cell_fused.tests.test_parity import device
from experiments.expressivity_tasks.x30_tape_cell import naive_reference as oracle


def inputs(cfg, tokens, seed=354):
    torch.manual_seed(seed)
    dev = device()
    k, q = [torch.randn(1,tokens,cfg.n_heads,cfg.key_dim,device=dev) for _ in range(2)]
    v = torch.randn(1,tokens,cfg.n_heads,cfg.value_dim,device=dev)
    beta = torch.ones(1,tokens,cfg.n_heads,device=dev)
    return k,q,v,beta,torch.zeros_like(beta),torch.zeros_like(beta)


@pytest.mark.parametrize('budget', [3, 33, 11776])
def test_lazy_continuation_preserves_cap_pressure_and_dead_rows(budget):
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=2,budget=budget,
                               decay_rate=.8,free_threshold=.2)
    xs = list(inputs(cfg, 18))
    # Include no-write, POP death and reuse; no materialization between chunks.
    xs[3][:,4:7] = 0
    xs[4][:,9:12] = 1
    expected = oracle.init_state(1,cfg,device())
    state = None
    dense = None
    for start,stop in ((0,2),(2,5),(5,9),(9,13),(13,18)):
        chunk = [x[:,start:stop] for x in xs]
        out,state,trace,pressure = candidate.tape_scan(*chunk,state,cfg)
        ref,expected,rt = oracle.tape_scan(*chunk,expected,cfg)
        dense_out,dense,dense_trace,dense_pressure = candidate._tape_scan(*chunk,dense,cfg)
        assert torch.equal(out,dense_out)
        assert torch.equal(pressure,dense_pressure)
        torch.testing.assert_close(out,ref,atol=1e-5,rtol=1e-5)
        for name in ('case','matched','alloc_target','pop_target','freed'):
            assert torch.equal(getattr(trace,name),getattr(rt,name)),name
    for name in FIELDS:
        assert torch.equal(getattr(state.native,name),getattr(dense.native,name)),name
        if name != "usage":  # Usage EMA is outside the X31 functional scope.
            torch.testing.assert_close(getattr(state.native,name),getattr(expected,name),atol=1e-5,rtol=1e-5)
    # Public materialization is stable and subsequent continuation consumes it.
    pointers = [getattr(state.native,n).data_ptr() for n in FIELDS]
    _,end,_,_ = candidate.tape_scan(*(x[:,:1] for x in xs),state,cfg)
    assert pointers == [getattr(end.native,n).data_ptr() for n in FIELDS]
    with pytest.raises(ValueError,match='consumed'):
        candidate.tape_scan(*(x[:,:1] for x in xs),state,cfg)


def test_sparse_fresh_training_does_not_materialize_budget_until_public_access():
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=2,budget=11776)
    xs = [x.requires_grad_() for x in inputs(cfg, 6)]
    out,state,trace,_ = candidate.tape_scan(*xs,None,cfg)
    assert isinstance(state.native,PrefixNative)
    assert state.native.compact.keys.shape[2] == 6
    assert trace._trace.freed.shape[-1] == 6
    grads = torch.autograd.grad(out.sum(),xs,retain_graph=True)
    assert all(torch.isfinite(g).all() for g in grads)
    # Mutating materialized public state must remain visible to replay checks.
    state.native.live.fill_(False)
    with pytest.raises(RuntimeError,match='modified in place'):
        torch.autograd.grad(out.sum(),xs)


def test_lazy_training_continuation_and_public_final_state_adjoints():
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=33,
                               decay_rate=.5,free_threshold=.2)
    base = inputs(cfg,8)
    a = [x.clone().requires_grad_() for x in base]
    b = [x.clone().requires_grad_() for x in base]
    first,state,_,_ = candidate.tape_scan(*(x[:,:3] for x in a),None,cfg)
    last,state,_,_ = candidate.tape_scan(*(x[:,3:] for x in a),state,cfg)
    expected,ref,_ = oracle.tape_scan(*b,oracle.init_state(1,cfg,device()),cfg)
    actual = torch.cat((first,last),dim=1)
    loss = actual.sum()+state.native.values.sum()+state.native.strength.sum()
    other = expected.sum()+ref.values.sum()+ref.strength.sum()
    ga = torch.autograd.grad(loss,a)
    gb = torch.autograd.grad(other,b,allow_unused=True)
    for x,y,z in zip(ga,gb,b):
        torch.testing.assert_close(x,torch.zeros_like(z) if y is None else y,atol=1e-5,rtol=1e-5)


def test_recording_after_lazy_prefix_promotion_and_snapshot_are_exact():
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=33)
    xs = inputs(cfg,4)
    _,state,_,_ = candidate.tape_scan(*xs,None,cfg)
    branch = state.snapshot()
    original = [getattr(branch.native,n).clone() for n in FIELDS]
    _,final,_,_ = candidate.tape_scan(*xs,state,cfg,record_all=True)
    restored = final.all_field_journal.undo_(final)
    for name,x in zip(FIELDS,original):
        assert torch.equal(getattr(restored.native,name),x)


def test_materialized_metadata_mutation_is_visible_to_compact_replay():
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=33)
    xs = [x.requires_grad_() for x in inputs(cfg,3)]
    out,state,_,_ = candidate.tape_scan(*xs,None,cfg)
    _ = state.native.keys
    state.metadata[0].fill_(-1)
    with pytest.raises(RuntimeError,match='modified in place'):
        torch.autograd.grad(out.sum(),xs)


def test_orphan_public_native_and_routing_trace_api():
    import weakref
    from experiments.expressivity_tasks.x30_tape_cell.routing_trace import RoutingTrace
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=33)
    _,state,trace,_ = candidate.tape_scan(*inputs(cfg,3),None,cfg)
    native = state.native
    ref = weakref.ref(state)
    del state
    assert ref() is None  # Timed calls must release ownership without cyclic GC.
    assert native.keys.shape == (1,1,33,4)
    assert isinstance(trace,RoutingTrace)
    combined = RoutingTrace.cat([trace.slice_tokens(0,1),trace.slice_tokens(1,3)])
    assert torch.equal(combined.freed,trace.freed)
    assert torch.equal(combined.case,trace.case)


@pytest.mark.parametrize('decay',[.125,.5,.8,.9995,1.])
def test_incremental_read_strengths_match_timestamp_jumps_bitwise(monkeypatch,decay):
    # The diagnostic specialization retains the previous timestamp-jump
    # emission arithmetic. Routing, death, mutation and replay stay identical.
    original = candidate.scan_kernel
    class TimestampViews:
        def __getitem__(self, grid):
            launch = original[grid]
            def invoke(*args,**kwargs):
                kwargs["IncrementalViews"] = False
                return launch(*args,**kwargs)
            return invoke
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=2,budget=33,
                               decay_rate=decay,free_threshold=.2)
    xs = list(inputs(cfg,19,seed=355))
    xs[3][:,3:6]=0
    xs[4][:,9:11]=1
    native = oracle.init_state(1,cfg,device())
    native.live[:,:,:7]=True;native.strength[:,:,:7]=.9
    native.keys.normal_();native.values.normal_();native.age.random_(0,5)
    a=b=None
    for start,stop in ((0,3),(3,9),(9,12),(12,19)):
        chunk = [x[:,start:stop] for x in xs]
        monkeypatch.setattr(candidate,'scan_kernel',original)
        ao,a,at,_ = candidate.tape_scan(*chunk,native if a is None else a,cfg)
        monkeypatch.setattr(candidate,'scan_kernel',TimestampViews())
        bo,b,bt,_ = candidate.tape_scan(*chunk,native if b is None else b,cfg)
        assert torch.equal(ao,bo)
        for name in FIELDS:
            assert torch.equal(getattr(a.native,name),getattr(b.native,name)),name
        for name in ('case','matched','alloc_target','pop_target','freed'):
            assert torch.equal(getattr(at,name),getattr(bt,name)),name


def test_orphan_materialized_native_still_detects_replay_mutation():
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=33)
    xs = [x.requires_grad_() for x in inputs(cfg,3)]
    out,state,_,_ = candidate.tape_scan(*xs,None,cfg)
    native = state.native
    del state
    native.live.fill_(False)
    with pytest.raises(RuntimeError,match='modified in place'):
        torch.autograd.grad(out.sum(),xs)


def test_frozen_replay_ignores_mutation_of_consumed_promoted_state():
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=33)
    base = inputs(cfg,8)
    a = [x.clone().requires_grad_() for x in base]
    b = [x.clone().requires_grad_() for x in base]
    first,old,_,_ = candidate.tape_scan(*(x[:,:3] for x in a),None,cfg)
    last,state,_,_ = candidate.tape_scan(*(x[:,3:] for x in a),old,cfg)
    # Consuming continuation has already frozen the earlier replay. Promoting
    # and then modifying the consumed public handle must not poison that copy.
    old.native.live.fill_(False)
    actual = torch.cat((first,last),dim=1)
    expected,_,_ = oracle.tape_scan(*b,oracle.init_state(1,cfg,device()),cfg)
    ga = torch.autograd.grad(actual.sum(),a)
    gb = torch.autograd.grad(expected.sum(),b,allow_unused=True)
    for x,y,z in zip(ga,gb,b):
        torch.testing.assert_close(x,torch.zeros_like(z) if y is None else y,atol=1e-5,rtol=1e-5)


def test_physical_horizon_cannot_change_architectural_exhaustion_or_pressure():
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=3,decay_rate=1.)
    xs = list(inputs(cfg,8))
    xs[2].zero_()  # Every enabled write requests append, including at capacity.
    _,state,_,_ = candidate.tape_scan(*(x[:,:2] for x in xs),None,cfg)
    assert isinstance(state.native,PrefixNative)
    assert state.native.compact.live.shape[-1] == 2
    _,state,trace,pressure = candidate.tape_scan(*(x[:,2:] for x in xs),state,cfg)
    assert trace.case[0,:,0].tolist() == [3,5,5,5,5,5]
    assert trace.alloc_target[0,:,0].tolist() == [2,-1,-1,-1,-1,-1]
    assert pressure[0,-1,0].item() == 5/8
    assert state.native.live.sum().item() == cfg.slots


def test_public_native_field_assignment_updates_materialized_continuation():
    cfg = oracle.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=1,budget=33)
    xs = inputs(cfg,3)
    _,state,_,_ = candidate.tape_scan(*xs,None,cfg)
    native = state.native
    replacement = torch.full((1,1,33,5),2.,device=device())
    native.values = replacement
    assert native.values is replacement
    assert state.native.values is replacement
    expected = state.snapshot()
    a,_,_,_ = candidate.tape_scan(*xs,state,cfg)
    b,_,_,_ = candidate.tape_scan(*xs,expected,cfg)
    assert torch.equal(a,b)
