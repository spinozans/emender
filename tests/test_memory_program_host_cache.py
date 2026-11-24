import pytest
import torch
from experiments.expressivity_tasks.x43_fork_resolved.run import Control
from experiments.expressivity_tasks.x47_xsa_attention.cached_host_check import check_cache, make_cache


@pytest.fixture
def model():
    torch.manual_seed(7601)
    return Control(2000324).eval()


def test_real_block_architecture_reordered_uneven_query_and_capacity(model):
    report = check_cache(model)
    assert report["stats"] == dict(
        prefill_batches=4, prefill_tokens=135, step_batches=3, step_tokens=6, released=4
    )
    assert report["checks"]["hidden"] == 13 and report["all_released_zero"]


def test_byte_budget_refuses_before_tensor_allocation_and_limits_slots(model):
    n = (128 * 2 * 2 * 256 + 512) * 4
    with pytest.raises(ValueError, match="budget"):
        make_cache(model, 2, budget=2 * n - 1)
    with pytest.raises(ValueError, match="context"):
        make_cache(model, 1, 129)
    a = make_cache(model, 2, budget=2 * n)
    assert a.tensor_bytes == sum(t.numel() * t.element_size() for t in a.slots.values()) == 2 * n
    a.admit("x")
    a.admit("y")
    with pytest.raises(ValueError, match="full"):
        a.admit("z")
    with pytest.raises(ValueError, match="duplicate"):
        a.admit("x")


@torch.no_grad()
def test_rejected_handles_or_frontiers_cannot_overwrite_other_session(model):
    a, b = [make_cache(model, 2) for _ in range(2)]
    x = a.admit("x")
    y = a.admit("y")
    foreign = b.admit("x")
    a.prefill([x, y], [b"\xff", b"\x00"])
    snap = {k: v.clone() for k, v in a.slots.items()}
    for bad in [
        lambda: a.read([x], [2]),
        lambda: a.advance([x], [0], [1]),
        lambda: a.advance([x, x], [1, 1], [2, 3]),
        lambda: a.advance([y], [1], [256]),
        lambda: a.release([foreign]),
        lambda: a.prefill([y], [b"x"]),
    ]:
        with pytest.raises(ValueError):
            bad()
    assert all(torch.equal(v, a.slots[k]) for k, v in snap.items())
    h, p = a.read([y], [1])
    h.add_(1)
    p.fill_(0)
    assert all(torch.equal(v, a.slots[k]) for k, v in snap.items())
    a.release([x])
    x2 = a.admit("x")
    assert x.slot == x2.slot and x.generation != x2.generation
    with pytest.raises(ValueError, match="stale"):
        a.release([x])
    with pytest.raises(ValueError, match="stale"):
        a.read([x2], [1])
    with pytest.raises(ValueError, match="raw bytes"):
        a.prefill([x2], [[256]])


@torch.no_grad()
def test_driver_mixed_prefill_step_and_read_preserves_order_without_extra_work(model):
    from types import SimpleNamespace
    from experiments.expressivity_tasks.x47_xsa_attention.cached_host_bench import HostDriver
    from experiments.expressivity_tasks.x47_xsa_attention.serving_bench import causal_last

    a = make_cache(model, 3)
    ss = [SimpleNamespace(id=str(i), visible=1, pending=False) for i in range(3)]
    lives = [SimpleNamespace(data=bytes([i])) for i in range(3)]
    d = HostDriver(a, ss, lives)
    d.inputs([2, 0])
    ss[0].visible += 1
    lives[0].data += b"Y"
    ids = [2, 0, 1]
    h, p = d.inputs(ids)
    expected = causal_last(model, [lives[i].data for i in ids], "cpu")
    torch.testing.assert_close(h, expected[0], atol=5e-5, rtol=2e-5)
    torch.testing.assert_close(p, expected[1], atol=5e-5, rtol=2e-5)
    ss[0].visible += 2
    lives[0].data += b"xx"
    with pytest.raises(ValueError, match="frontier"):
        d.inputs([0])
    report = d.report()
    assert report["encoded_tokens"] == 4 and report["all_released_zero"]


def test_inference_only_and_changed_model_rejected(model):
    a = make_cache(model, 1)
    h = a.admit("x")
    with pytest.raises(ValueError, match="inference"):
        a.prefill([h], [b"x"])
    with torch.no_grad():
        model.embed.weight.add_(1)
        with pytest.raises(ValueError, match="weights"):
            a.prefill([h], [b"x"])


@torch.no_grad()
def test_failed_partial_backend_write_invalidates_only_its_admissions(model):
    a = make_cache(model, 3)
    x, y, z = [a.admit(k) for k in "xyz"]
    encode = a.backend.encode
    a.prefill([z], [b"good"])
    snap = [t[z.slot].clone() for t in a.slots.values()]

    def fail(slots, ix, old, ids, prompt):
        if ids.shape[1] == 2:
            raise RuntimeError("injected backend failure")
        return encode(slots, ix, old, ids, prompt)

    a.backend.encode = fail
    with pytest.raises(RuntimeError):
        a.prefill([x, y], [b"x", b"yy"])
    with pytest.raises(ValueError, match="stale"):
        a.read([x], [1])
    assert all(
        not t[[x.slot, y.slot]].any() and torch.equal(t[z.slot], v) for t, v in zip(a.slots.values(), snap)
    )
    a.read([z], [4])
