import asyncio
from threading import Event

import pytest
import torch

from ndm.memory_program import Field, Hit, Instruction as I, PackedMemoryPolicy, Reader, Session
from ndm.memory_program.serving import _resolve, serve_ready, StreamUniforms
from experiments.expressivity_tasks.x47_xsa_attention.serving_check import check_ready


@pytest.mark.parametrize("cohort", [False, True])
def test_ready_full_events_pages_scores_and_delayed_gradients(cohort):
    report = check_ready("cpu", cohort=cohort)
    assert report["pages"] >= 10 and report["delayed_grad_norm"] > 0


@pytest.mark.parametrize("cohort", [False, True])
def test_fast_client_emits_again_while_slow_client_waits(cohort):
    entered = Event()
    advanced = Event()
    calls = []

    def slow(keys, cap):
        calls.append(keys)
        entered.set()
        assert advanced.wait(5), "batch-wide head-of-line stall"
        return [(Hit((b"z",)),)]

    mount = Reader("s", "v", "find", "text", (Field("v", "byte"),), slow)
    m = PackedMemoryPolicy(2, dim=8, heads=2)
    ss = [Session([mount], session_id=f"s{i}") for i in range(2)]
    [s.observe(b"a") for s in ss]
    cnt = [0, 0]
    emit = I("EMIT", (-1, -1))
    sl = I("SLICE", (0, 1))
    read = I("LOOKUP", (0, 0, 0))
    plan = [[[sl, read, emit], [emit], [emit]], [[sl, I("PROJECT", (0, 0)), emit], [emit], [emit]]]

    def deliver(ids, ds):
        for i, d in zip(ids, ds):
            cnt[i] += 1
            if i == 1 and cnt[i] == 2:
                assert entered.wait(3) and ss[0].visible == 1 and not ss[0].pending
                advanced.set()
            ss[i].observe(b"Y")

    with torch.no_grad():
        stats = asyncio.run(
            serve_ready(
                m,
                ss,
                3,
                lambda ids: (torch.zeros(len(ids), 2), torch.zeros(len(ids), 256)),
                deliver,
                offload=[mount],
                replay=plan,
                cohort=cohort,
            )
        )
    assert cnt == [3, 3] and stats["emits"] == 6 and calls == [(b"a",)]


@pytest.mark.parametrize("cohort", [False, True])
def test_provider_error_does_not_commit_staged_write_or_emit_empty(cohort):
    def broken(keys, cap):
        raise RuntimeError("native IO failed")

    r = Reader("s", "v", "find", "text", (Field("v", "byte"),), broken)
    m = PackedMemoryPolicy(2, dim=8, heads=2)
    s = Session([r])
    s.observe(b"abc")
    seen = []
    plan = [[[I("SLICE", (0, 1)), I("SAVE", (0, 1)), I("LOOKUP", (1, 0, 0)), I("EMIT", (-1, -1))]]]
    with pytest.raises(ExceptionGroup) as error:
        asyncio.run(
            serve_ready(
                m,
                [s],
                1,
                lambda ids: (torch.zeros(1, 2), torch.zeros(1, 256)),
                lambda *x: seen.append(x),
                replay=plan,
                offload=[r],
                cohort=cohort,
            )
        )
    assert (
        "native IO failed" in repr(error.value)
        and not seen
        and not any(s.cells)
        and s.visible == 3
        and not s.pending
    )


def test_cohort_unequal_program_lengths_do_not_fragment_backbone_when_no_io_is_pending():
    m = PackedMemoryPolicy(2, dim=8, heads=2)
    n, ticks = 3, 5
    ss = [Session(session_id=str(i)) for i in range(n)]
    for s in ss:
        s.observe(b"z")
    seen = []
    sl, em = I("SLICE", (0, 1)), I("EMIT", (-1, -1))
    plan = [[[sl] * ((i + t) % 3) + [em] for t in range(ticks)] for i in range(n)]

    def inputs(ids):
        seen.append(sorted((i, ss[i].visible) for i in ids))
        return torch.ones(len(ids), 2), torch.zeros(len(ids), 256)

    def deliver(ids, decisions):
        for i in ids:
            ss[i].observe(b"x")

    with torch.no_grad():
        data = asyncio.run(serve_ready(m, ss, ticks, inputs, deliver, replay=plan, cohort=True))
    assert data["emits"] == n * ticks
    assert seen == [[(i, t + 1) for i in range(n)] for t in range(ticks)]


def test_only_bound_immutable_cpu_reader_offloaded_and_batches_preserve_keys():
    calls = []

    def get(keys, cap):
        calls.append((keys, cap))
        return [(Hit((k,)),) for k in keys]

    r = Reader("s", "v", "one", "text", (Field("v", "text"),), get, batch_limit=2)
    table, *_ = _resolve(r, 8, [(0, (b"a", b"b")), (1, (b"a", b"c"))], True)
    assert list(table) == [b"a", b"b", b"c"] and calls == [((b"a", b"b"), 8), ((b"c",), 8)]
    live = Reader(
        "s", "one", "one", "text", (Field("v", "text"),), get, session_id="one", watermark=lambda: 0
    )
    s = Session([live], session_id="one")
    m = PackedMemoryPolicy(2, dim=8, heads=2)
    for reader in (r, live):
        with pytest.raises(ValueError, match="offload"):
            asyncio.run(serve_ready(m, [s], 1, None, None, offload=[reader]))
    bad = Reader(
        "s",
        "v",
        "values",
        "text",
        (Field("v", "text"),),
        lambda keys, cap: [(Hit((k,), value=torch.ones(8)),) for k in keys],
    )
    with pytest.raises(ValueError, match="payload"):
        _resolve(bad, 8, [(0, (b"x",))], True)


def test_fresh_categorical_draws_independent_of_serving_order_and_legal_padding():
    ss = [Session(session_id=f"d{i}") for i in range(4)]
    [s.observe(b"z") for s in ss]
    a = StreamUniforms(ss, [12, 31, 40, 41], 8, 345, "cpu")
    b = StreamUniforms([ss[2]], [40], 8, 345, "cpu")
    v = torch.tensor(
        [[0.1, 2.0, -torch.inf], [3.0, -torch.inf, -torch.inf], [1.0, 0.0, 5.0], [9.0, 2.0, 1.0]]
    )
    for t in range(8):
        coords = [(s.id, 1 + t, 2, 1) for s in ss]
        js, lp = a.sample_factor(v, [2, 1, 3, 3], coords)
        ks, q = b.sample_factor(v[2:3], [3], [coords[2]])
        assert js[2] == ks[0] and lp[2] == q[0] and js[1] == 0 and js[0] < 2


def test_c14_decode_uses_last_real_byte_and_not_ragged_padding():
    from experiments.expressivity_tasks.x43_fork_resolved.run import Control
    from experiments.expressivity_tasks.x47_xsa_attention.serving_bench import causal_last

    torch.manual_seed(81)
    net = Control(2000324).eval()
    windows = [b"\x00", b"xy\x1e", b"z\xff"]
    with torch.no_grad():
        h, p = causal_last(net, windows, "cpu", 0)
        h2, q = causal_last(net, windows, "cpu", 255)
        oracle = torch.cat([net(torch.tensor([list(w)]))[:, -1].log_softmax(1) for w in windows])
        torch.testing.assert_close(h, h2)
        torch.testing.assert_close(p, q)
        torch.testing.assert_close(p, oracle, atol=1e-5, rtol=2e-5)


def test_direct_backoff_binds_positive_byte_mass_not_unavailable_or_end():
    from collections import Counter
    from ndm.memory_program import Unavailable
    from experiments.expressivity_tasks.x47_xsa_attention.serving_bench import direct

    def h(byte):
        out = [0] * 257
        out[byte] = 7
        return tuple(out)

    class Bank:
        def histograms(self, keys):
            return [h(255) if k == b"5678" else Unavailable("unmaterialized") for k in keys]

    class Live:
        data = b"12345678"

        def histogram(self, k):
            return h(30 if k == b"78" else 256)

    work = Counter()
    base = torch.full((2, 256), 1 / 256.0)
    output = direct(Bank(), [Live(), Live()], base.log(), work)
    expect = base.clone()
    expect[:, [255, 30]] += 2
    torch.testing.assert_close(output.exp(), expect / 5)
    assert work == {"archive_batches": 3, "archive_keys": 3, "live_keys": 6}


def test_triton_policy_does_not_silently_use_cpu():
    from ndm.memory_program.triton_operands import TritonMemoryPolicy

    m = TritonMemoryPolicy(2, dim=8, heads=2)
    s = Session()
    s.observe(b"X")
    with pytest.raises(ValueError, match="CUDA operand"):
        m([s], torch.zeros(1, 2), torch.zeros(1, 256))
