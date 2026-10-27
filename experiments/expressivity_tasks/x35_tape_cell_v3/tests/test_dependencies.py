import os
import pytest
import torch
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig, tape_scan as naive, init_state
from experiments.expressivity_tasks.x30_tape_cell.routing_trace import RoutingTrace, R_EMPTY, R_CONFLICT, R_ZERO_RESIDUAL, assert_traces_equal
from experiments.expressivity_tasks.x35_tape_cell_v3.classifier import classify, segments


def fixture(values, device='cpu', coef=0.):
    cfg = TapeHeadConfig(n_heads=1, key_dim=4, value_dim=4, budget=7, decay_rate=1.)
    t = len(values)
    k = torch.ones(1, t, 1, 4, device=device)
    v = torch.zeros_like(k)
    v[0, :, 0, 0] = torch.tensor(values, device=device)
    on = torch.ones(1, t, 1, device=device)
    off = torch.zeros_like(on)
    return cfg, (k, k.clone(), v, on, off, off), torch.tensor([coef], device=device)


def run(values, device='cpu', coef=0., fused=False):
    cfg, args, coefficient = fixture(values, device, coef)
    if fused:
        from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import tape_scan
        return tape_scan(*args, None, cfg, coefficient)
    return naive(*args, init_state(1, cfg, device), cfg, append_err_coef=coefficient)


def test_constant_a_run_is_not_append_run():
    _, _, tr = run([1.] * 8)
    assert tr.case.flatten().tolist() == [R_EMPTY, R_CONFLICT, 2, R_CONFLICT, 2, R_CONFLICT, 2, R_CONFLICT]
    cert = classify(tr)
    assert not cert.mutation_candidate and not cert.entry_pure
    assert cert.intra_match == 4


def test_true_append_run_still_depends_on_prior_append():
    _, _, tr = run([1., .5])
    assert tr.case.flatten().tolist() == [R_EMPTY, R_ZERO_RESIDUAL]
    assert tr.alloc_target.flatten().tolist() == [0, 1]
    cert = classify(tr)
    assert cert.mutation_candidate and not cert.entry_pure
    assert cert.intra_match == 1
    assert segments(tr) == [(0, 1, 'entry_pure'), (1, 2, 'entry_pure')]


def test_chunk_entry_routing_counterexample():
    _, _, actual = run([1., 1.])
    _, _, independent = run([1.])
    assert int(actual.case[0, 1, 0]) == R_CONFLICT
    assert int(independent.case[0, 0, 0]) == R_EMPTY


def test_append_gate_dependency_with_nonzero_coefficient():
    cfg, args, coef = fixture([1., .5005], coef=1.)
    _, full, tr = naive(*args, init_state(1, cfg, 'cpu'), cfg, append_err_coef=coef)
    assert int(tr.case[0, 1, 0]) == R_ZERO_RESIDUAL
    # Entry-empty evaluation incorrectly uses has-live=false and hence gate=0.5.
    assert full.values[0, 0, 1, 0] > args[2][0, 1, 0, 0] * .5


def test_new_append_changes_causal_read_and_gradient():
    cfg, args, coef = fixture([1., 1.])
    args[2].requires_grad_()
    out, _, _ = naive(*args, init_state(1, cfg, 'cpu'), cfg, append_err_coef=coef)
    assert out[0, 1, 0, 0] == .5
    out.sum().backward()
    assert args[2].grad[0, 0, 0, 0] == .5
    independent, _, _ = run([1.])
    assert torch.count_nonzero(independent) == 0


def test_missing_death_evidence_fails_closed():
    _, _, tr = run([0., 0.])
    tr.freed = None
    with pytest.raises(ValueError, match='death trace'):
        classify(tr)


def test_segmentation_covers_events_once():
    _, _, tr = run([1.] * 8)
    parts = segments(tr)
    assert [t for a, b, _ in parts for t in range(a, b)] == list(range(8))
    for a, b, label in parts:
        assert classify(tr, start=a, stop=b).entry_pure == (label == 'entry_pure')


@pytest.mark.parametrize('values', [[1., 1.], [1., .5], [1.] * 8])
def test_x31_confirms_counterexamples_bitwise(values):
    assert os.environ.get('GPU_LEASE_HELD') in tuple(map(str, range(7)))
    expected, native, trace = run(values, 'cuda')
    actual, state, recorded, _ = run(values, 'cuda', fused=True)
    assert_traces_equal(trace, recorded)
    assert torch.equal(actual, expected)
    for field in ('keys', 'values', 'strength', 'live', 'age'):
        assert torch.equal(getattr(state.native, field), getattr(native, field)), field


def test_integer_allocation_scan_preserves_holes_and_no_write():
    from experiments.expressivity_tasks.x35_tape_cell_v3.allocation import allocate
    free = torch.tensor([False, True, False, True, True, False, True])
    append = torch.tensor([True, False, True, True, False, True])
    assert allocate(free, append).tolist() == [1, -1, 3, 4, -1, 6]
    with pytest.raises(ValueError, match='insufficient'):
        allocate(free, torch.ones(5, dtype=torch.bool))
    assert allocate(torch.zeros(3, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)).tolist() == [-1] * 4


def test_x31_counterexample_gradient_bitwise():
    assert os.environ.get('GPU_LEASE_HELD') in tuple(map(str, range(7)))
    from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import tape_scan
    cfg, args, coef = fixture([1., 1.], 'cuda')
    args[2].requires_grad_()
    reference, _, _ = naive(*args, init_state(1, cfg, 'cuda'), cfg, append_err_coef=coef)
    expected, = torch.autograd.grad(reference.sum(), args[2])
    actual, _, _, _ = tape_scan(*args, None, cfg, coef)
    observed, = torch.autograd.grad(actual.sum(), args[2])
    assert torch.equal(expected, observed)
    assert observed[0, 0, 0, 0] == .5
