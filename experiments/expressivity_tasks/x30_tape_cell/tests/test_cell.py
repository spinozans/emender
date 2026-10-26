import copy
from dataclasses import replace

import pytest
import torch

from experiments.expressivity_tasks.x30_tape_cell import naive_reference as naive
from experiments.expressivity_tasks.x30_tape_cell.cell import (
    FreeList, TapeCell, empty_state, from_native, tape_scan, replay_trace)
from experiments.expressivity_tasks.x30_tape_cell.routing_trace import assert_traces_equal


def inputs(cfg, seed, b=2, t=19):
    torch.manual_seed(seed)
    shape = (b, t, cfg.n_heads)
    k = torch.randn(*shape, cfg.key_dim)
    q = torch.randn_like(k, requires_grad=True)
    # Repeats exercise zero residual explicitly in the op-level test below.
    v = torch.randn(*shape, cfg.value_dim, requires_grad=True)
    beta = (torch.rand(*shape) > .25).float()
    pop = torch.rand(*shape, requires_grad=True)
    logit = torch.randn(*shape, requires_grad=True)
    return k, q, v, beta, pop, logit


def initial(cfg, b, seed):
    torch.manual_seed(seed)
    state = naive.init_state(b, cfg, 'cpu')
    state.keys.normal_()
    state.values.normal_()
    state.strength.uniform_(.3, 1.)
    state.live = torch.rand_like(state.strength) > .4
    state.age.random_(1, 6)  # deliberately tied births
    return state


@pytest.mark.parametrize('budget', [1, 3, 11])
@pytest.mark.parametrize('seed', [0, 1, 7])
@pytest.mark.parametrize('decay', [.8, .999995])
def test_naive_parity(budget, seed, decay):
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=2,
                                budget=budget, decay_rate=decay, free_threshold=.15)
    args = inputs(cfg, seed)
    init = initial(cfg, 2, seed+100)
    coef = torch.tensor([.1, -.2], requires_grad=True)
    out_n, state_n, trace_n = naive.tape_scan(*args, init, cfg, append_err_coef=coef)
    loss_n = out_n.square().sum() + state_n.values.square().sum()*.07 + state_n.strength.sum()*.13
    grad_n = torch.autograd.grad(loss_n, (args[1], args[2], args[4], args[5], coef))
    state_e = from_native(init, cfg)
    out_e, final, trace_e, pressure = tape_scan(*args, state_e, cfg, append_err_coef=coef, chunk_size=4)
    mat = final.materialize()
    assert_traces_equal(trace_n, trace_e.dense())
    torch.testing.assert_close(out_n, out_e, atol=1e-5, rtol=1e-5)
    for name, value in mat.items():
        torch.testing.assert_close(getattr(state_n, name), value, atol=1e-5, rtol=1e-5)
    loss_e = out_e.square().sum() + mat['values'].square().sum()*.07 + mat['strength'].sum()*.13
    grad_e = torch.autograd.grad(loss_e, (args[1], args[2], args[4], args[5], coef))
    for a, b in zip(grad_n, grad_e):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert pressure.min() >= 0 and pressure.max() <= 1 and not pressure.requires_grad
    trace_e.dense().assert_integral()


def test_all_five_cases_and_reuse():
    cfg = naive.TapeHeadConfig(key_dim=1, value_dim=1, n_heads=1,
                                budget=2, decay_rate=1.)
    k = q = torch.ones(1, 8, 1, 1)
    v = torch.tensor([1., .5, 3., 9., 9., 9., 9., 9.]).reshape(1, 8, 1, 1)
    beta = torch.tensor([1., 1., 1., 0., 1., 1., 1., 1.]).reshape(1, 8, 1)
    pop = torch.zeros_like(beta)
    pop[:, 7] = 1.
    logit = torch.zeros_like(beta)
    args = k, q, v, beta, pop, logit
    a, sa, tr = naive.tape_scan(*args, naive.init_state(1, cfg, 'cpu'), cfg)
    b, sb, te, p = tape_scan(*args, None, cfg, chunk_size=2)
    assert_traces_equal(tr, te.dense())
    # empty -> identical appended value -> zero residual append -> overwrite,
    # gap -> no-match cap, followed by cap; pop frees the first free slot.
    assert tr.case.flatten().tolist() == [1, 3, 4, 0, 5, 5, 5, 2]
    torch.testing.assert_close(a, b)
    assert p[0, 4, 0] == pytest.approx(1/3)


def test_replay_consumes_trace_and_not_thresholds():
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=2,
                                budget=5, decay_rate=.8, free_threshold=.15)
    args = inputs(cfg, 42)
    state = empty_state(2, cfg, 'cpu')
    out, final, trace, _ = tape_scan(*args, state, cfg)
    changed = copy.deepcopy(state)
    changed.management.cfg = replace(cfg, eps_residual=1e6,
                                     write_enable_threshold=2., pop_threshold=2.,
                                     free_threshold=.99)
    with torch.enable_grad():
        replay, last = replay_trace(*args, changed, trace)
    torch.testing.assert_close(out, replay, atol=1e-5, rtol=1e-5)
    for name, value in final.materialize().items():
        torch.testing.assert_close(value, last.materialize()[name])


def test_freelist_lowest_free_and_word_boundaries():
    for size in (1, 65, 736, 11776):
        f = FreeList(size)
        assert [f.take() for _ in range(size)] == list(range(size))
        assert f.take() == -1
        for j in (size-1, 0, size//2):
            f.release(j)
        assert [f.take() for _ in range(len(set((size-1, 0, size//2))))] == sorted(set((size-1, 0, size//2)))


def test_module_freezes_geometry_and_masks_pad():
    torch.manual_seed(0)
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=2, budget=4)
    module = TapeCell(7, cfg, chunk_size=2)
    x = torch.randn(2, 5, 7, requires_grad=True)
    ids = torch.tensor([[1, 2, 0, 1, 0], [1, 0, 3, 0, 1]])
    out, state, tr, p = module(x, ids)
    out.square().sum().backward()
    assert not tr.dense().w_on[ids == 0].any()
    assert module.W_k.weight.grad is None
    assert module.W_write.weight.grad is None
    assert module.W_q.weight.grad is not None
    assert module.W_v.weight.grad is not None
    assert p.shape == (2, 5, 2)


def test_streaming_chunk_equivalence():
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=2, budget=7,
                                decay_rate=.8, free_threshold=.15)
    args = inputs(cfg, 22)
    with torch.no_grad():
        out, final, trace, p = tape_scan(*args, None, cfg, chunk_size=19)
        first, middle, _, p1 = tape_scan(*(a[:, :8] for a in args), None, cfg, chunk_size=3)
        second, last, _, p2 = tape_scan(*(a[:, 8:] for a in args), middle, cfg, chunk_size=4)
    torch.testing.assert_close(out, torch.cat((first, second), 1))
    torch.testing.assert_close(p, torch.cat((p1, p2), 1))
    for name, value in final.materialize().items():
        torch.testing.assert_close(value, last.materialize()[name])


@pytest.mark.parametrize('heads', [4, 16])
def test_head_independence_outputs_traces_and_gradients(heads):
    cfg1 = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=1, budget=7)
    args1 = inputs(cfg1, 18, b=1, t=9)
    out1, state1, trace1, p1 = tape_scan(*args1, None, cfg1, chunk_size=3)
    grads1 = torch.autograd.grad(out1.square().sum(), (args1[1], args1[2], args1[4], args1[5]))
    cfg = replace(cfg1, n_heads=heads)
    args = tuple(a.detach().repeat_interleave(heads, dim=2).requires_grad_(a.requires_grad)
                 for a in args1)
    out, state, trace, pressure = tape_scan(*args, None, cfg, chunk_size=3)
    torch.testing.assert_close(out, out1.repeat_interleave(heads, 2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pressure, p1.repeat_interleave(heads, 2))
    grads = torch.autograd.grad(out.square().sum(), (args[1], args[2], args[4], args[5]))
    for a, expected in zip(grads, grads1):
        torch.testing.assert_close(a, expected.repeat_interleave(heads, 2), atol=1e-5, rtol=1e-5)
    for name in ('case', 'matched', 'alloc_target', 'pop_target', 'w_on', 'freed'):
        assert torch.equal(getattr(trace.dense(), name), getattr(trace1.dense(), name).repeat_interleave(heads, 2))


def test_imported_age_zero_does_not_hide_age_one():
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=2, budget=5)
    init = initial(cfg, 2, 0)
    init.live.fill_(True)
    init.age[:, :, 0] = 0
    init.age[:, :, 1] = 1
    args = inputs(cfg, 9, t=2)
    out, last, trace = naive.tape_scan(*args, init, cfg)
    actual, final, tr, _ = tape_scan(*args, from_native(init, cfg), cfg)
    assert_traces_equal(trace, tr.dense())
    torch.testing.assert_close(out, actual)


def test_window_pressure_expires_requests_and_caps():
    cfg = naive.TapeHeadConfig(n_heads=1, budget=1)
    from experiments.expressivity_tasks.x30_tape_cell.cell import Management
    m = Management(1, cfg, window=3)
    assert [m.pressure(0, c) for c in (1, 5, 0, 0, 1)] == [0., .5, .5, 1., 0.]


def test_imported_already_decayed_rows_die_postpop_without_scan():
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=2, budget=5)
    init = initial(cfg, 2, 0)
    init.live.fill_(True)
    init.strength[:, :, 0] = 0.
    init.strength[:, :, 1] = cfg.free_threshold
    args = inputs(cfg, 4, t=3)
    expected, _, trace = naive.tape_scan(*args, init, cfg)
    actual, _, tr, _ = tape_scan(*args, from_native(init, cfg), cfg)
    assert_traces_equal(trace, tr.dense())
    torch.testing.assert_close(actual, expected)


def test_detached_write_gate_has_no_backward_path():
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=1, budget=2)
    args = tuple(a.detach() for a in inputs(cfg, 1, b=1, t=3))
    args = (*args[:3], args[3].requires_grad_(), *args[4:])
    read, state, _, _ = tape_scan(*args, None, cfg)
    read.sum().backward()
    assert args[3].grad is None


def test_initial_state_gradients():
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=2, budget=5,
                                decay_rate=.8, free_threshold=.15)
    state = initial(cfg, 2, 10)
    leaves = (state.keys.requires_grad_(), state.values.requires_grad_(), state.strength.requires_grad_())
    args = tuple(x.detach() for x in inputs(cfg, 99, t=9))
    expected, last, _ = naive.tape_scan(*args, state, cfg)
    loss = expected.square().sum() + last.values.sum()*.1 + last.strength.sum()*.2 + last.keys.sum()*.03
    grad_n = torch.autograd.grad(loss, leaves)
    out, final, _, _ = tape_scan(*args, from_native(state, cfg), cfg, chunk_size=3)
    mat = final.materialize()
    loss_e = out.square().sum() + mat['values'].sum()*.1 + mat['strength'].sum()*.2 + mat['keys'].sum()*.03
    grad_e = torch.autograd.grad(loss_e, leaves)
    for a, b in zip(grad_n, grad_e):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('decay, threshold', [(.8, .20971521735191345), (.9, .3874203860759735)])
def test_exact_scheduled_death_at_adversarial_rounding_boundary(decay, threshold):
    cfg = naive.TapeHeadConfig(key_dim=1, value_dim=1, n_heads=1, budget=1,
                                decay_rate=decay, free_threshold=threshold)
    init = naive.init_state(1, cfg, 'cpu')
    init.live.fill_(True)
    init.strength.fill_(1.)
    init.age.fill_(1)
    k = q = v = torch.ones(1, 12, 1, 1)
    beta = pop = logit = torch.zeros(1, 12, 1)
    expected, last, trace = naive.tape_scan(k, q, v, beta, pop, logit, init, cfg)
    actual, final, tr, _ = tape_scan(k, q, v, beta, pop, logit, from_native(init, cfg), cfg)
    assert_traces_equal(trace, tr.dense())
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
    for name, value in final.materialize().items():
        torch.testing.assert_close(getattr(last, name), value, atol=1e-5, rtol=1e-5)


def test_long_default_decay_parity():
    cfg = naive.TapeHeadConfig(key_dim=2, value_dim=2, n_heads=1, budget=3)
    torch.manual_seed(3)
    init = naive.init_state(1, cfg, 'cpu')
    init.keys.normal_()
    init.values.normal_()
    init.live.fill_(True)
    init.strength[:] = torch.tensor([.31, .7, 1.])
    init.age.fill_(1)
    k = torch.randn(1, 2304, 1, 2)
    q = torch.randn_like(k, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    beta = pop = torch.zeros(1, 2304, 1)
    logit = torch.zeros_like(beta, requires_grad=True)
    expected, last, trace = naive.tape_scan(k, q, v, beta, pop, logit, init, cfg)
    actual, final, tr, _ = tape_scan(k, q, v, beta, pop, logit, from_native(init, cfg), cfg)
    assert_traces_equal(trace, tr.dense())
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
    mat = final.materialize()
    for name, value in mat.items():
        torch.testing.assert_close(getattr(last, name), value, atol=1e-5, rtol=1e-5)
    grad_n = torch.autograd.grad(expected.square().mean(), q)[0]
    grad_e = torch.autograd.grad(actual.square().mean(), q)[0]
    torch.testing.assert_close(grad_n, grad_e, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('seed', range(5))
def test_imported_birth_ties_survive_new_allocations_and_pops(seed):
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=2, budget=9,
                                decay_rate=.9, free_threshold=.1)
    init = initial(cfg, 2, seed)
    init.age.random_(0, 3)
    args = inputs(cfg, seed+100, t=13)
    expected, last, tr = naive.tape_scan(*args, init, cfg)
    actual, final, trace, _ = tape_scan(*args, from_native(init, cfg), cfg, chunk_size=3)
    assert_traces_equal(tr, trace.dense())
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for name, value in final.materialize().items():
        torch.testing.assert_close(getattr(last, name), value, atol=1e-5, rtol=1e-5)


def test_trace_override_uses_checkpoint_replay_without_mutating_entry_store():
    cfg = naive.TapeHeadConfig(key_dim=3, value_dim=5, n_heads=2, budget=9,
                                decay_rate=.8, free_threshold=.15)
    init = initial(cfg, 2, 31)
    init.age.zero_()
    state = from_native(init, cfg)
    before = state.materialize()
    args = inputs(cfg, 41, t=15)
    expected, _, trace, p = tape_scan(*args, state, cfg, chunk_size=3)
    # Alter only architectural thresholds. Every routing choice must still
    # come from the trace, including the recorded deaths and allocations.
    changed = copy.deepcopy(state)
    changed_cfg = replace(cfg, write_enable_threshold=2., pop_threshold=2.,
                          eps_residual=1e5, free_threshold=.99)
    changed.management.cfg = changed_cfg
    actual, _, tr, pressure = tape_scan(*args, changed, changed_cfg,
                                        chunk_size=4, trace_override=trace)
    assert_traces_equal(trace.dense(), tr.dense())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pressure, p)
    grads = torch.autograd.grad(actual.square().sum(), (args[1], args[2], args[4], args[5]))
    assert all(torch.isfinite(g).all() for g in grads)
    for name, value in before.items():
        torch.testing.assert_close(value, state.materialize()[name])


def test_trailing_pad_trimming_preserves_supervised_loss_and_gradients():
    import numpy as np
    from experiments.expressivity_tasks.x30_tape_cell.train_stage1 import WordModel, trim_mqar
    from experiments.expressivity_tasks.x28e_memory_head.generators.stage1_mqar import generate_pool
    torch.manual_seed(7)
    model = WordModel(heads=1, budget=736)
    xs, ys, mask, _ = generate_pool(128, 2, 91)
    ids, labels, m = torch.from_numpy(xs), torch.from_numpy(ys-1), torch.from_numpy(mask)
    full = model(ids)
    loss_full = torch.nn.functional.cross_entropy(full[m], labels[m])
    params = [p for p in model.parameters() if p.requires_grad]
    grad_full = torch.autograd.grad(loss_full, params, allow_unused=True)
    t = trim_mqar(xs, mask)
    assert t == 31
    prefix = model(ids[:, :t])
    loss_prefix = torch.nn.functional.cross_entropy(prefix[m[:, :t]], labels[:, :t][m[:, :t]])
    grad_prefix = torch.autograd.grad(loss_prefix, params, allow_unused=True)
    torch.testing.assert_close(loss_prefix, loss_full, atol=1e-5, rtol=1e-5)
    for a, b in zip(grad_full, grad_prefix):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_pop_free_reallocate_and_decay_free_same_slot_same_token():
    cfg = naive.TapeHeadConfig(key_dim=1, value_dim=1, n_heads=1, budget=1,
                                decay_rate=.1, free_threshold=.2)
    init = naive.init_state(1, cfg, 'cpu')
    init.live.fill_(True)
    init.strength.fill_(.3)
    init.values.fill_(1.)
    init.age.fill_(1)
    k = q = v = torch.ones(1, 1, 1, 1)
    beta = pop = torch.ones(1, 1, 1)
    logit = torch.zeros_like(beta, requires_grad=True)
    expected, last, tr = naive.tape_scan(k, q, v, beta, pop, logit, init, cfg)
    actual, final, trace, _ = tape_scan(k, q, v, beta, pop, logit, from_native(init, cfg), cfg)
    assert int(tr.freed.item()) == 3
    assert_traces_equal(tr, trace.dense())
    torch.testing.assert_close(actual, expected)
    for name, value in final.materialize().items():
        torch.testing.assert_close(getattr(last, name), value)
    grad_n = torch.autograd.grad(last.values.sum(), logit)[0]
    grad_e = torch.autograd.grad(final.values.sum(), logit)[0]
    torch.testing.assert_close(grad_n, grad_e)
