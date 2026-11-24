import torch
import pytest

from ndm.memory_program import (
    Field,
    Hit,
    Instruction as I,
    Limits,
    MemoryPolicy,
    Reader,
    Session,
    Span,
    teacher_force,
    episode_objective,
)
from ndm.memory_program.packed import PackedMemoryPolicy, LabelIds, encode_pages
from ndm.memory_program.runtime import Page
from experiments.expressivity_tasks.x47_xsa_attention.packed_program_check import exercise


def test_full_conditional_distributions_and_delayed_credit():
    check = exercise("cpu")
    assert check["checks"]["factor_logp"] > 70 and check["delayed_grad_norm"] > 0


def test_flat_mixed_pages_reference_and_mass_gradients():
    torch.manual_seed(755)
    a = MemoryPolicy(2, dim=8, heads=2, buckets=64)
    b = PackedMemoryPolicy(2, dim=8, heads=2, buckets=64)
    b.load_state_dict(a.state_dict())
    ps = []
    qs = []
    ns = [0, 1, 37, 256]
    widths = [4, 1, 5, 2]
    for n, f in zip(ns, widths):
        fields = tuple(Field(f"b{i}", "byte") for i in range(f))
        keys = tuple(tuple(bytes([(j % 7 + i) % 256]) for i in range(f)) for j in range(n))
        mass = torch.randn(n).mul(12).requires_grad_()
        v = torch.randn(n, 8, requires_grad=True)
        ps.append(Page(fields, keys, mass, v))
        qs.append(
            Page(fields, keys, mass.detach().clone().requires_grad_(), v.detach().clone().requires_grad_())
        )
    vs = [a.view(p) for p in ps]
    ref = torch.cat(
        [
            torch.stack([v.summary for v in vs]),
            torch.cat([v.rows for v in vs]),
            torch.cat([v.fields for v in vs]),
        ]
    )
    pack = encode_pages(b, qs, LabelIds(b.buckets))
    torch.testing.assert_close(pack, ref, rtol=2e-5, atol=5e-5)
    delta = torch.randn_like(ref)
    ref.backward(delta)
    pack.backward(delta)
    for p, q in zip(ps[1:], qs[1:]):
        torch.testing.assert_close(p.log_mass.grad, q.log_mass.grad, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(p.value.grad, q.value.grad, rtol=2e-4, atol=2e-4)


def test_rollout_cache_lifetime_replay_not_on_policy():
    torch.manual_seed(7)
    m = PackedMemoryPolicy(2, dim=8, heads=2)
    run = m.batch_runner()
    with torch.no_grad():
        m.host.weight.add_(1)
    with pytest.raises(ValueError, match="update"):
        run([Session()], torch.zeros(1, 2), torch.zeros(1, 256))

    def encode(x):
        return torch.zeros(*x.shape, 2), torch.zeros(*x.shape, 256)

    raw = [b"abcd", b"\x00\xff"]
    first = teacher_force(
        m, encode, [Session(), Session()], raw, mode="sample", generator=torch.Generator().manual_seed(341)
    )
    traces = [[[e.inst for e in d.trace] for d in ds] for ds in first.decisions]
    assert first.targets == 4
    replay = teacher_force(m, encode, [Session(), Session()], raw, mode="replay", replay=traces)
    assert replay.objective is None
    torch.testing.assert_close(first.mean_nll, replay.mean_nll)
    with pytest.raises(ValueError, match="on-policy"):
        episode_objective(replay.decisions[0], list(raw[0][1:]))


def test_budgets_no_partial_page_or_speculative_query():
    calls = []

    def read(keys, cap):
        calls.append((keys, cap))
        return [(Hit((b"\x00",)), Hit((b"\xff",))) for k in keys]

    r = Reader("bank", "v1", "foo", "text", (Field("symbol", "byte"),), read)
    torch.manual_seed(27)
    a = MemoryPolicy(2)
    b = PackedMemoryPolicy(2)
    b.load_state_dict(a.state_dict())
    h = torch.randn(2, 2)
    log = torch.randn(2, 256)
    plan = [
        I("SLICE", (0, 1)),
        I("LOOKUP", (0, 0, 0)),
        I("LOOKUP", (1, 2, 0)),
        I("JOIN", (1, 0, 1, 0)),
        I("RELATE", (1, 1)),
        I("EMIT", (1, 2)),
    ]

    def execute(m):
        ss = [Session([r], session_id=str(i), limits=Limits(rows=2, requests=1)) for i in range(2)]
        for s in ss:
            s.observe(b"\x1e")
        return m(ss, h, log, replay=[plan, plan[:2] + [I("EMIT", (-1, -1))]])

    dr, dp = execute(a), execute(b)
    assert calls == [((b"\x1e",), 2)] * 2
    assert [e.status for e in dp[0].trace[2:5]] == [
        "refused: key budget",
        "refused: join exceeds row budget",
        "refused: pair relation exceeds row budget",
    ]
    for x, y in zip(dr, dp):
        assert x.trace == y.trace and x.requests == y.requests == 1
        torch.testing.assert_close(x.log_probs, y.log_probs)


def test_natural_witness_version_renaming_does_not_collapse_addresses():
    from experiments.expressivity_tasks.x47_xsa_attention.packed_program_profile import BindingNames

    n = BindingNames(
        [Session(session_id="s", document=0, start=11), Session(session_id="t", document=5, start=22)]
    )
    assert n.key(Span("tape", "s", 0, 11, 12)) != n.key(Span("tape", "t", 0, 11, 12))
    assert n.key(Span("P100", "s", 0, 11, 12))[2] == "s"
    assert n.origin(("P100", "s", "spans", None)) == ("P100", "s", "spans", None)
    assert n.origin(("tape", "s", "spans", 3)) != n.origin(("tape", "s", "spans", 4))


def test_packed_fresh_causal_prefix_ignores_future_bytes():
    torch.manual_seed(745)
    m = PackedMemoryPolicy(4, dim=12, heads=2)

    def encode(x):
        h = x.float().cumsum(1)[:, :, None].expand(-1, -1, 4) / 256
        return h, torch.zeros(*x.shape, 256)

    def run(raw):
        return teacher_force(
            m, encode, [Session()], [raw], mode="sample", generator=torch.Generator().manual_seed(745)
        )

    x = run(b"abcd\xff")
    y = run(b"abcXY")
    for dx, dy in zip(x.decisions[0][:3], y.decisions[0][:3]):
        assert [(e.inst, e.status) for e in dx.trace] == [(e.inst, e.status) for e in dy.trace]
        torch.testing.assert_close(dx.log_probs, dy.log_probs)
        torch.testing.assert_close(dx.log_policy, dy.log_policy)
