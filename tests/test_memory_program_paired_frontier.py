from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from ndm.memory_program import PackedMemoryPolicy, Session, Reader, Field, Hit, Unavailable
from ndm.memory_program.frontier_reads import FrontierReads, StageLimit
from ndm.memory_program.paired_frontier import FixedFrontierPair
from ndm.memory_program.runtime import OPS, Span


def record(*, answer=None):
    calls = []
    wm = [0]

    def read(keys, cap):
        calls.append((keys, cap))
        return (
            answer
            if answer is not None
            else [
                (Hit((b"\x00", b"k\x00"), 2), Hit((b"\x00", b"k\x00"), 4), Hit((b"\xff", b"k\xff"))),
                (),
                Unavailable("cap"),
            ]
        )

    r = Reader(
        "tape",
        "local",
        "byte_edges",
        "text",
        (Field("symbol", "byte"), Field("extension", "text")),
        read,
        session_id="local",
        watermark=lambda: wm[0],
    )
    return r, calls, wm


def test_only_issued_structural_answers_transferred_once_and_not_past_frontier():
    r, calls, _ = record()
    tr = FrontierReads(lambda x: None, StageLimit())
    w = torch.zeros(256, 4)
    issue, consume = [tr.view(r, p, w) for p in ("issue", "consume")]
    keys = (b"k", b"\x00", b"\x1e")
    tr.begin()
    a = issue.read_many(keys, 8)
    tr.handoff()
    b = consume.read_many(keys, 8)
    tr.finish()
    assert a == b and [h.mass for h in b[0]] == [2, 4, 1] and b[1:] == ((), Unavailable("cap"))
    assert calls == [(keys, 8)] and not tr._calls and tr.peak["rows"] == 3
    with pytest.raises(ValueError, match="phase"):
        consume.read_many(keys, 8)
    tr.begin()
    issue.read_many(keys, 8)
    assert len(calls) == 2
    tr.close()


def test_span_vectors_are_built_separately_including_gradient_into_only_second_embedding():
    loc = Span("archive", "v", 1, 0, 2)
    calls = []

    def read(keys, cap):
        calls.append(keys)
        return [(Hit((b"\x00\xff",)),)]

    r = Reader("archive", "v", "contents", "span", (Field("text", "text"),), read)
    a = torch.arange(512, dtype=torch.float32).reshape(256, 2).requires_grad_()
    b = (a.detach().clone() * 3).requires_grad_()
    tr = FrontierReads(lambda _: None, StageLimit())
    tr.begin()
    one = tr.view(r, "issue", a).read_many((loc,), 1)[0][0]
    tr.handoff()
    two = tr.view(r, "consume", b).read_many((loc,), 1)[0][0]
    tr.finish()
    tr.close()
    assert torch.equal(3 * one.value, two.value) and one.keys == two.keys and len(calls) == 1
    two.value.sum().backward()
    assert (
        a.grad is None
        and torch.equal(b.grad[[0, 255]], torch.full((2, 2), 0.5))
        and b.grad[1:255].count_nonzero() == 0
    )


@pytest.mark.parametrize(
    "fault", ["cap", "keys", "same_schema_reader", "watermark", "authority", "unconsumed"]
)
def test_wrong_consumer_fails_closed_without_second_native_call(fault):
    r, calls, wm = record(answer=[()])
    ready = [True]
    w = torch.zeros(256, 2)

    def guard(_):
        if not ready[0]:
            raise ValueError("expired")

    tr = FrontierReads(guard, StageLimit())
    tr.begin()
    tr.view(r, "issue", w).read_many((b"A",), 2)
    tr.handoff()
    if fault == "watermark":
        wm[0] += 1
    if fault == "authority":
        ready[0] = False
    with pytest.raises(ValueError):
        if fault == "unconsumed":
            tr.finish()
        else:
            tr.view(replace(r) if fault == "same_schema_reader" else r, "consume", w).read_many(
                (b"B" if fault == "keys" else b"A",), 1 if fault == "cap" else 2
            )
    assert len(calls) == 1 and tr.phase == "closed" and not tr._calls


@pytest.mark.parametrize("fault", ["partial", "tensor", "mass", "rows", "bytes", "schema"])
def test_ambiguous_or_unbudgeted_staging_is_not_truncation(fault):
    h = Hit((b"x", b"kx"))
    reply = [(h, h)]
    if fault == "partial":
        reply = []
    if fault == "tensor":
        reply = [(replace(h, value=torch.ones(3)),)]
    if fault == "mass":
        reply = [(replace(h, mass=-1),)]
    r, calls, _ = record(answer=reply)
    w = torch.zeros(256, 3)
    tr = FrontierReads(
        lambda _: None,
        StageLimit(rows=1 if fault == "rows" else 16, text_bytes=1 if fault == "bytes" else 2000),
    )
    tr.begin()
    with pytest.raises(ValueError):
        tr.view(replace(r, operation="other") if fault == "schema" else r, "issue", w).read_many((b"k",), 8)
    assert len(calls) == int(fault != "schema") and not tr._calls
    tr.close()


def models():
    torch.manual_seed(933)
    p = PackedMemoryPolicy(2, dim=8, heads=2, buckets=256)
    with torch.no_grad():
        p.transitions.weight.fill_(-80)
        p.transitions.weight[-1, OPS.index("SLICE")] = 60
        p.transitions.weight[-1, OPS.index("RECALL")] = 100
        for a, b in [("SLICE", "LOOKUP"), ("LOOKUP", "SAVE"), ("SAVE", "EMIT"), ("RECALL", "EMIT")]:
            p.transitions.weight[OPS.index(a), OPS.index(b)] = 60
        p.prior[OPS.index("SAVE"), 4] = 50
        p.prior[OPS.index("EMIT"), 4:] = 50
    return p, deepcopy(p)


def source():
    data = bytearray()
    calls = []

    def reply(keys, cap):
        calls.append((len(data), keys))
        return [(Hit((b"a", k + b"a"), 2), Hit((b"b", k + b"b"), 1)) for k in keys]

    bound = Reader(
        "tape",
        "s",
        "byte_edges",
        "text",
        (Field("symbol", "byte"), Field("extension", "text")),
        reply,
        session_id="s",
        watermark=lambda: len(data),
    )
    return Session([bound], advance=data.extend, session_id="s", document=4, start=12), data, calls


def test_causal_fixed_choices_with_two_value_states_one_read_and_delayed_adjoint():
    p, q = models()
    s, data, calls = source()
    rng = torch.Generator().manual_seed(5)
    with torch.no_grad():
        q.writer.bias.add_(0.4)
        q.transitions.weight[:, OPS.index("EMIT")] = 1000
    with FixedFrontierPair(p, q, [s], guard=lambda _: None, byte_limit=8) as fx:
        fx.observe([b"\x00"])
        a = fx.predict(torch.zeros(1, 2), torch.zeros(1, 256), generator=rng)
        assert [e.inst.op for e in a.selectors[0].trace] == ["SLICE", "LOOKUP", "SAVE", "EMIT"]
        fx.observe([b"\xff"])
        b = fx.predict(torch.zeros(1, 2), torch.zeros(1, 256), generator=rng)
        assert b.values[0].trace[0].inst.op == "RECALL" and b.values[0].trace == b.selectors[0].trace
        assert (
            not torch.equal(b.values[0].log_probs, b.selectors[0].log_probs)
            and not b.selectors[0].log_probs.requires_grad
        )
        assert data == b"\x00\xff" and len(calls) == 1 and fx.work()["frames_still_staged"] == 0
        fx.observe([b"q"])
    (-b.values[0].log_probs[31]).backward()
    assert (
        q.writer.weight.grad.norm() > 0
        and all(x.grad is None for x in p.parameters())
        and q.pointer.weight.grad is None
    )


def test_drift_and_cross_thread_or_duplicate_owner_do_not_issue():
    p, q = models()
    s, data, calls = source()
    with pytest.raises(ValueError, match="alias"):
        FixedFrontierPair(p, p, [s], guard=lambda _: None, byte_limit=8)
    fx = FixedFrontierPair(p, q, [s], guard=lambda _: None, byte_limit=8)
    with pytest.raises(ValueError):
        FixedFrontierPair(p, q, [s], guard=lambda _: None, byte_limit=8)
    with ThreadPoolExecutor(1) as th, pytest.raises(ValueError, match="foreign"):
        th.submit(fx.observe, [b"x"]).result()
    assert not data and not fx.closed
    fx.observe([b"\x00"])
    with torch.no_grad():
        p.writer.bias.add_(1)
    with pytest.raises(ValueError, match="changed"):
        fx.predict(torch.zeros(1, 2), torch.zeros(1, 256), generator=torch.Generator())
    assert fx.closed and data == b"\x00" and not calls and s.pending and not s.readers
