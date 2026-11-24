from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from ndm.memory_program import PackedMemoryPolicy, Session, Reader, Field, Hit
from ndm.memory_program.frozen_routes import capture_routes, record_training_routes, replay_routes
from ndm.memory_program.training import teacher_force
from ndm.memory_program.runtime import OPS

A = "e" * 64
WORDS = [b"\x00abb\xff\x1e"]
LIMIT = 30


def model():
    torch.manual_seed(933)
    p = PackedMemoryPolicy(2, dim=8, heads=2, buckets=64)
    with torch.no_grad():
        p.transitions.weight.fill_(-80)
        p.transitions.weight[-1, OPS.index("SLICE")] = 60
        p.transitions.weight[-1, OPS.index("RECALL")] = 100
        for a, b in [
            ("SLICE", "LOOKUP"),
            ("LOOKUP", "SAVE"),
            ("SAVE", "EMIT"),
            ("RECALL", "EMIT"),
        ]:
            p.transitions.weight[OPS.index(a), OPS.index(b)] = 60
        p.prior[OPS.index("SAVE"), 4] = 50
        p.prior[OPS.index("EMIT"), 4:] = 50
    return p


def encode(x):
    return torch.zeros(*x.shape, 2), torch.zeros(*x.shape, 256)


def fresh(sid, *, bad=None, doc=4, op="byte_edges"):
    data = bytearray()
    calls = []

    def read(keys, cap):
        calls.append((len(data), keys))
        out = [Hit((b"a",), 2), Hit((b"a",), 4), Hit((b"b",), 1)]
        if bad == "mass":
            out[0] = Hit((b"a",), 1)
        if bad == "order":
            out.reverse()
        return [tuple(out) for _ in keys]

    r = Reader(
        "tape",
        sid,
        op,
        "text",
        (Field("symbol", "byte"),),
        read,
        session_id=sid,
        watermark=lambda: len(data),
    )
    return [Session([r], document=doc, start=12, advance=data.extend, session_id=sid)], calls


def captured():
    p = model()
    s, _ = fresh("original")
    tape, b = capture_routes(
        deepcopy(p).requires_grad_(False),
        encode,
        s,
        WORDS,
        authority=A,
        max_steps=LIMIT,
        generator=torch.Generator().manual_seed(5),
    )
    return p, tape, b


def test_frozen_executed_choices_still_allow_credit_from_later_recall():
    p, t, b = captured()
    s, _ = fresh("new")
    with torch.no_grad():
        p.writer.bias.add_(0.4)
        p.transitions.weight[:, OPS.index("EMIT")] = 1000
    out = replay_routes(p, encode, s, WORDS, t, authority=A, max_steps=LIMIT)
    assert out.objective is None
    last = out.decisions[0][-1]
    assert (
        last.trace[0].inst.op == "RECALL"
        and last.mode == "replay"
        and not torch.equal(last.log_probs, b.decisions[0][-1].log_probs)
    )
    (-last.log_probs[31]).backward()
    assert (
        p.writer.weight.grad.abs().max() > 0
        and p.op.weight.grad is None
        and p.transitions.weight.grad is None
        and p.pointer.weight.grad is None
    )


@pytest.mark.parametrize("bad", ["mass", "order"])
def test_same_shape_wrong_native_result_stops_before_any_next_observation(bad):
    p, t, _ = captured()
    s, calls = fresh("replacement", bad=bad)
    with pytest.raises(ValueError, match="frozen result"):
        replay_routes(p, encode, s, WORDS, t, authority=A, max_steps=LIMIT)
    assert len(calls) == 1 and s[0].visible == 1 and not any(s[0].cells)


def test_mismatched_request_refused_before_reader_is_called():
    p, t, _ = captured()
    doc = t.programs[0]
    first = doc[0]
    index = next(i for i, c in enumerate(first) if c.request is not None)
    bad = replace(first[index], request=bytes(32))
    t = replace(t, programs=((first[:index] + (bad,) + first[index + 1 :], *doc[1:]),))
    s, calls = fresh("none")
    with pytest.raises(ValueError, match="before dispatch"):
        replay_routes(p, encode, s, WORDS, t, authority=A, max_steps=LIMIT)
    assert s[0].visible == 1 and not calls


def test_tape_is_bound_to_observations_schema_and_budget_not_last_unseen_label():
    p, t, _ = captured()
    for ss, ws, auth, cap in [
        (fresh("a", doc=6), WORDS, A, 30),
        (fresh("b", op="other"), WORDS, A, 30),
        (fresh("c"), [b"Z" + WORDS[0][1:]], A, 30),
        (fresh("d"), WORDS, "d" * 64, 30),
        (fresh("e"), WORDS, A, 4),
    ]:
        s, calls = ss
        with pytest.raises(ValueError):
            replay_routes(p, encode, s, ws, t, authority=auth, max_steps=cap)
        assert not calls and s[0].visible == 0
    s, _ = fresh("f")
    out = replay_routes(p, encode, s, [WORDS[0][:-1] + b"q"], t, authority=A, max_steps=30)
    assert out.targets == 5


def test_inband_returns_the_live_cost_graph_with_unchanged_actions_and_adjoints():
    p = model()
    q = deepcopy(p)
    ss, calls = fresh("annotated")
    tt, other = fresh("ordinary")
    cost = dict(key_price=0.091, instruction_price=0.07, baseline=1.0)
    tape, b = record_training_routes(
        p, encode, ss, WORDS, authority=A, max_steps=LIMIT, generator=torch.Generator().manual_seed(5), **cost
    )
    a = teacher_force(q, encode, tt, WORDS, mode="sample", generator=torch.Generator().manual_seed(5), **cost)
    assert tape.instructions() == [[[e.inst for e in d.trace] for d in a.decisions[0]]]
    assert calls == other and torch.equal(a.objective, b.objective) and b.objective.requires_grad
    x = replay_routes(q, encode, fresh("checked")[0], WORDS, tape, authority=A, max_steps=LIMIT)
    assert all(
        torch.equal(v.log_probs, w.log_probs) and w.mode == "replay"
        for v, w in zip(b.decisions[0], x.decisions[0], strict=True)
    )
    a.objective.backward()
    b.objective.backward()
    for v, w in zip(p.parameters(), q.parameters(), strict=True):
        assert (v.grad is None) == (w.grad is None)
        if v.grad is not None:
            torch.testing.assert_close(v.grad, w.grad)
    assert p.writer.weight.grad.norm() > 0 and p.pointer.weight.grad is not None


def test_inband_refuses_detachment_and_parameter_drift_before_seal():
    p = model()

    def run(enc):
        return record_training_routes(
            p,
            enc,
            fresh("once")[0],
            WORDS,
            authority=A,
            max_steps=LIMIT,
            generator=torch.Generator().manual_seed(5),
        )

    with torch.no_grad(), pytest.raises(ValueError, match="live train-mode"):
        run(encode)

    # Mutate a real parameter while constructing the episode; never publish that tape.
    def changed(x):
        with torch.no_grad():
            p.writer.bias.add_(1.0)
        return encode(x)

    with pytest.raises(ValueError, match="changed|crossed a model update"):
        run(changed)
