"""Platform contracts, not a benchmark dataset or a hand-authored policy used for training."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from ndm.memory_program import (
    Decision,
    Field,
    Hit,
    Instruction as I,
    Limits,
    MaterializedBranches,
    MemoryPolicy,
    Reader,
    Session,
    Span,
    Unavailable,
    branch_reader,
    episode_objective,
    teacher_force,
)
from ndm.memory_program.runtime import Frame, Page
from ndm.memory_program.machine import Views


def model():
    torch.manual_seed(7301)
    return MemoryPolicy(6, dim=12, heads=3, buckets=256)


def session(readers=(), prefix=b"x", **kw):
    s = Session(readers, **kw)
    s.observe(prefix)
    return s


def predict(m, sessions, *, plans=None, **kwargs):
    return m(sessions, torch.ones(len(sessions), 6), torch.zeros(len(sessions), 256), replay=plans, **kwargs)


def histogram(pairs, end=0):
    h = [0] * 256 + [end]
    for b, n in pairs:
        h[b] = n
    return tuple(h)


def test_compiled_direct_normalization_and_chosen_only_coalescing():
    m = model()
    calls = []
    h = histogram([(0, 11), (30, 7), (255, 3), (19, 2), (1, 1)], end=99)

    def counts(keys):
        calls.append(keys)
        return [h] * len(keys)

    archive = branch_reader("collection", "pinned", counts)
    never = branch_reader("other", "v", lambda _: pytest.fail("speculative read"))
    ss = [session([archive, never], key) for key in (b"az", b"bz", b"\xff")]
    ds = predict(m, ss)
    assert calls == [(b"z", b"\xff")]
    for d in ds:
        assert [e.inst.op for e in d.trace] == ["SLICE", "LOOKUP", "EMIT"] and d.requests == 1
        assert d.trace[1].rows == 5 and d.trace[1].reader == ("collection", "pinned", "byte_edges", None)
        torch.testing.assert_close(d.log_probs.exp(), 0.5 / 256 + 0.5 * torch.tensor(h[:256]) / 24)
    with pytest.raises(ValueError, match="observe"):
        predict(m, ss)
    direct = ds[0].log_probs
    ss[0].observe(b"q")
    alone = predict(m, [ss[0]])[0]
    assert calls[-1] == (b"q",)
    torch.testing.assert_close(alone.log_probs, direct)
    # Same name/version strings don't conflate distinct reader instances/authorized views.
    other = branch_reader("collection", "pinned", lambda keys: [histogram([(42, 1)]) for k in keys])
    pair = predict(m, [session([archive]), session([other])])
    assert pair[1].log_probs.argmax() == 42
    assert pair[0].log_probs.argmax() == 0


def test_unavailable_overflow_empty_and_end_are_different():
    m = model()
    limit = Limits(rows=2)
    xs = []

    def counts(keys):
        return [
            None if k == b"u" else histogram([(1, 1), (2, 1), (3, 1)]) if k == b"o" else histogram([], end=2)
            for k in keys
        ]

    normal = branch_reader("s", "1", counts)
    end = branch_reader("s", "1", counts, structural_end=True)
    for k in (b"u", b"o", b"e"):
        xs.append(session([normal, end], k, limits=limit))
    us, over, empty = predict(m, xs)
    assert "not materialized" in us.trace[1].status and "budget" in over.trace[1].status
    assert us.trace[1].register is over.trace[1].register is None
    assert empty.trace[1].status == "empty" and empty.trace[1].register == 1
    for d in (us, over, empty):
        torch.testing.assert_close(d.log_probs.exp(), torch.ones(256) / 256)
    assert end.read_many((b"e",), 2) == [(Hit((b"e",), 2),)]


def test_live_watermark_and_no_implicit_observation():
    tape = bytearray()
    observed = []

    def counts(keys):
        observed.append(bytes(tape))
        return [histogram([(tape[-1], 1)]) for k in keys]

    reader = branch_reader("live", "one", counts, session_id="one", watermark=lambda: len(tape))
    with pytest.raises(ValueError, match="wrong live"):
        Session([reader], session_id="other")
    s = session([reader], b"\x00\x1e\xff", advance=tape.extend, session_id="one")
    d = predict(model(), [s])[0]
    assert observed == [b"\x00\x1e\xff"] and d.trace[1].reader[-1] == 3
    assert s.visible == len(tape) == 3
    s.observe(b"Y")
    tape.append(99)  # externally advanced native stream must fail closed
    with pytest.raises(ValueError, match="watermark"):
        predict(model(), [s])


def joins():
    ref = Span("book", "v1", 7, 17, 19)
    wrong = replace(ref, version="v0")
    calls = []

    def mount(name, hits):
        def get(keys, cap):
            calls.append((name, keys))
            return [hits for k in keys]

        return Reader(name, "v1", "attributes", "text", (Field("ref", "span"), Field("letter", "byte")), get)

    left = mount("mentions", (Hit((ref, b"x"), 2), Hit((ref, b"y"), 3), Hit((wrong, b"z"), 99)))
    right = mount(
        "catalog", (Hit((ref, b"B"), 7), Hit((ref, b"C"), 11), Hit((replace(ref, document=0), b"D"), 13))
    )
    # Neither the controller nor the join opcode knows what mentions/catalog mean.
    return left, right, calls


WRITE = [I("SLICE", (0, 1)), I("LOOKUP", (0, 0, 0)), I("SAVE", (1, 2)), I("EMIT", (-1, -1))]
READ = [
    I("RECALL", (2,)),
    I("SLICE", (0, 1)),
    I("LOOKUP", (1, 0, 1)),
    I("JOIN", (0, 2, 2, 2)),
    I("SAVE", (3, 0)),
    I("EMIT", (4, 7)),
]


def test_delayed_exact_many_to_many_join_and_real_writer_gradient():
    m = model()
    l, r, calls = joins()
    s = session([l, r], session_id="doc")
    old_hidden = torch.ones(1, 6, requires_grad=True)
    m([s], old_hidden, torch.zeros(1, 256), replay=[WRITE])
    saved = s.cells[2]
    saved.value.retain_grad()
    s.observe(b" next question?")
    d = predict(m, [s], plans=[READ])[0]
    assert calls == [("mentions", (b"x",)), ("catalog", (b"?",))]
    assert d.trace[3].rows == 4
    result = s.cells[0]
    assert [row[3::4] for row in result.keys] == [(b"x", b"B"), (b"x", b"C"), (b"y", b"B"), (b"y", b"C")]
    torch.testing.assert_close(result.log_mass.exp(), torch.tensor([14.0, 22.0, 21.0, 33.0]))
    assert (
        result.fields[2].origin[:3] == ("mentions", "v1", "attributes")
        and result.fields[6].origin[0] == "catalog"
    )
    (-d.log_probs[66]).backward()
    for grad in (saved.value.grad, old_hidden.grad, m.writer.weight.grad):
        assert grad is not None and grad.abs().sum() > 0
    # Identical raw tape, historical keys/masses, operations, requests and host. Erase only the saved vector.
    alt = session([l, r], b"x next question?", session_id="doc")
    alt.cells[2] = replace(saved, value=torch.zeros_like(saved.value))
    erased = predict(m, [alt], plans=[READ])[0]
    assert erased.trace == d.trace
    assert (d.log_probs - erased.log_probs).abs().max() > 1e-5
    assert not any(Session([l, r]).cells)


def test_learned_compatibility_and_join_output_as_next_address():
    m = model()
    l, r, _ = joins()
    called = []

    def translate(keys, cap):
        called.append(keys)
        return [(Hit((k.lower(),)),) for k in keys]

    follow = Reader("translation", "v1", "lower", "byte", (Field("translated", "byte"),), translate)
    s = session([l, r, follow], limits=Limits(steps=8))
    predict(m, [s], plans=[WRITE])
    s.observe(b"?")
    # Same interpreter as the equijoin. All 3x3 scored pairs, not a hidden top two.
    plan = READ[:3] + [I("RELATE", (0, 2)), I("LOOKUP", (3, 7, 2)), I("EMIT", (4, 8))]
    d = predict(m, [s], plans=[plan])[0]
    assert d.trace[3].rows == 9 and d.trace[4].rows == 9 and called == [(b"B", b"C", b"D")]
    (-d.log_probs[ord("d")]).backward()
    assert m.pair_query.weight.grad.norm() > 0
    assert torch.isfinite(m.pair_query.weight.grad).all()
    # JOIN can feed LOOKUP which can feed a second JOIN, without another implementation.
    s = session([l, r, follow], limits=Limits(steps=8, columns=16))
    predict(m, [s], plans=[WRITE])
    s.observe(b"!")
    plan = READ[:4] + [I("LOOKUP", (3, 7, 2)), I("JOIN", (4, 6, 0, 2)), I("EMIT", (5, 8))]
    d = predict(m, [s], plans=[plan])[0]
    assert d.trace[5].rows == 8 and called[-1] == (b"B", b"C")


@pytest.mark.parametrize(
    ("op", "args", "reason"), [("JOIN", (0, 2, 2, 2), "join"), ("RELATE", (0, 2), "pair")]
)
def test_no_partial_join_or_partial_commit(op, args, reason):
    m = model()
    l, r, _ = joins()
    s = session([l, r], limits=Limits(rows=3))
    predict(m, [s], plans=[WRITE])
    s.observe(b"?")
    d = predict(m, [s], plans=[READ[:3] + [I(op, args), I("EMIT", (-1, -1))]])[0]
    assert reason in d.trace[3].status and d.trace[3].register is None
    assert not s.cells[0] and s.cells[2] is not None
    s2 = session([l, r])
    bad = WRITE[:3] + [I("JOIN", (0, 1, 0, 0)), I("EMIT", (-1, -1))]  # span != text
    with pytest.raises(ValueError):
        predict(m, [s2], plans=[bad])
    assert not any(s2.cells) and not s2.pending


def test_query_can_be_constructed_from_nonadjacent_bytes_without_appending():
    m = model()
    keys = []

    def read(ks, cap):
        keys.extend(ks)
        return [(Hit((b"!",)),) for k in ks]

    reader = Reader("s", "1", "answers", "text", (Field("out", "byte"),), read)
    s = session([reader], b"A---\xff", limits=Limits(steps=8))
    plan = [
        I("SLICE", (4, 1)),
        I("SLICE", (0, 1)),
        I("RELATE", (0, 1)),
        I("CONCAT", (2, 0, 2)),
        I("PROJECT", (3, 4)),
        I("LOOKUP", (4, 0, 0)),
        I("EMIT", (5, 1)),
    ]
    d = predict(m, [s], plans=[plan])[0]
    assert keys == [b"A\xff"] and d.log_probs.argmax() == ord("!")
    assert s.recent == b"A---\xff" and not any(s.cells)
    # Reducing a schema never changes multiplicity; an excessive key isn't truncated.
    s = session([reader], b"A---\xff", limits=Limits(pattern=1, steps=8))
    d = predict(m, [s], plans=[plan[:4] + [I("EMIT", (-1, -1))]])[0]
    assert "byte budget" in d.trace[3].status
    assert keys == [b"A\xff"]


def test_bind_query_to_selected_row_and_enforce_key_budget():
    m = model()
    l, r, calls = joins()
    s = session([l, r], limits=Limits(requests=1))
    plan = [
        I("SLICE", (0, 1)),
        I("LOOKUP", (0, 0, 0)),
        I("FOCUS", (1, 1)),
        I("LOOKUP", (2, 3, 1)),
        I("EMIT", (2, 3)),
    ]
    d = predict(m, [s], plans=[plan])[0]
    assert d.trace[3].status == "refused: key budget" and d.requests == 1
    assert calls == [("mentions", (b"x",))]
    s = session([l, r])
    d = predict(m, [s], plans=[plan])[0]
    assert d.trace[3].keys == (b"y",) and calls[-1] == ("catalog", (b"y",))


def test_learned_opcode_can_switch_between_two_legal_programs():
    # Exactly marginalize over the two legal initial instructions on a nonempty tape
    # with no saved cells: SLICE (then direct read) vs EMIT. No opcode labels.
    m = model()
    r = branch_reader("s", "v", lambda keys: [histogram([(0, 1)]) for _ in keys])

    def alternatives():
        ds = predict(
            m,
            [session([r]), session([r])],
            plans=[[I("EMIT", (-1, -1))], [I("SLICE", (0, 1)), I("LOOKUP", (0, 0, 0)), I("EMIT", (1, 2))]],
        )
        # Recompute only the FIRST instruction probabilities; the continuations here
        # are held fixed, not claimed to be the marginalized policy for a general episode.
        f = Frame(session([r]))
        get = Views(m)
        h = m.rebuild([f], torch.ones(1, 6), [get], [None], ["start"], 0)[0]
        logits = [
            m.choose(f, h, get, None, 0, "greedy", inst, None)[1]
            for inst in (I("EMIT", (-1, -1)), I("SLICE", (0, 1)))
        ]
        return torch.stack(logits).exp(), torch.stack([-d.log_probs[100] for d in ds])

    assert predict(m, [session([r])])[0].trace[0].inst.op == "SLICE"
    # Zero the learnable transition margin, keeping the compiled tie break (SLICE).
    # The check is of the sign and executed action, not a convergence benchmark.
    with torch.no_grad():
        m.transitions.weight[-1].zero_()
    probs, costs = alternatives()
    assert probs.sum().item() == pytest.approx(1.0) and costs[0] < costs[1]
    opt = torch.optim.SGD(m.op.parameters(), lr=0.03)
    opt.zero_grad()
    (probs * costs).sum().backward()
    opt.step()
    after = predict(m, [session([r])])[0]
    assert [e.inst.op for e in after.trace] == ["EMIT"]


def test_on_policy_byte_loss_changes_the_executed_operand_not_just_the_vote():
    m = model()
    target = ord("B")
    readers = [
        branch_reader(chr(c), "v", lambda keys, c=c: [histogram([(c, 1)]) for k in keys]) for c in (65, 66)
    ]
    # Isolate the mount operand for this gradient test; no replay/forced mount/op labels or answer in the input.
    with torch.no_grad():
        m.transitions.weight.mul_(25)
        m.prior.mul_(25)
    before = predict(m, [session(readers)])[0]
    assert before.trace[1].reader[0] == "A"
    sampled = predict(m, [session(readers)], mode="sample", generator=torch.Generator().manual_seed(21))[0]
    assert sampled.log_policy.item() == pytest.approx(-math.log(2), abs=1e-5)
    loss = episode_objective([sampled], [target], baseline=2.0)
    opt = torch.optim.SGD(m.pointer.parameters(), lr=0.01)
    opt.zero_grad()
    loss.loss.backward()
    opt.step()
    after = predict(m, [session(readers)])[0]
    assert after.trace[1].reader[0] == "B" and after.log_probs[target] > before.log_probs[target]
    with pytest.raises(ValueError, match="on-policy"):
        episode_objective([after], [target])


def test_a_next_byte_can_train_which_columns_are_joined():
    # Conditional on two available relations and a JOIN, enumerate only its last operand.
    # The two right-hand columns have identical marginals. No key-choice labels.
    m = model()
    a = Span("catalog", "v", 1, 2, 4)
    b = replace(a, start=3)
    col = (Field("entity", "span"),)
    left = Page(col, ((a,),), torch.zeros(1), torch.zeros(1, m.dim))
    right = Page(
        (Field("c0", "span"), Field("c1", "span"), Field("out", "byte")),
        ((a, b, b"x"), (b, a, b"y")),
        torch.zeros(2),
        torch.zeros(2, m.dim),
    )

    def alternatives():
        lps = []
        costs = []
        for key in (0, 1):
            frame = Frame(Session())
            frame.reg.extend([left, right])
            get = Views(m)
            h = torch.ones(m.dim)
            inst, lp = m.choose(frame, h, get, "LOOKUP", 2, "greedy", I("JOIN", (0, 0, 1, key)), None)
            frame.reg.append(frame.execute(inst, h, m))
            out = m.emit(I("EMIT", (2, 3)), frame, h, torch.zeros(256), get)
            lps.append(lp)
            costs.append(-out[ord("y")])
        return torch.stack(lps).softmax(0), torch.stack(costs)

    p, cost = alternatives()
    torch.testing.assert_close(p, torch.tensor([0.5, 0.5]))
    assert cost[0] > cost[1]
    opt = torch.optim.SGD(m.pointer.parameters(), lr=0.02)
    opt.zero_grad()
    (p * cost).sum().backward()
    opt.step()
    new, cost2 = alternatives()
    assert new[1] > new[0] and (new * cost2).sum() < (p * cost).sum()


def test_ragged_teacher_forcing_all_real_targets_and_no_future_tape():
    m = model()
    calls = []
    causal = torch.nn.Embedding(256, 6)
    head = torch.nn.Linear(6, 256)

    def encode(x):
        calls.append(tuple(x.shape))
        h = causal(x)
        return h, head(h)

    def run(raw, mode):
        tapes = [bytearray() for _ in raw]
        before = [[] for _ in raw]
        ss = []
        for i in range(len(raw)):

            def branch(keys, i=i):
                before[i].append(bytes(tapes[i]))
                return [histogram([(ord("x"), 1)]) for k in keys]

            ss.append(
                Session(
                    [
                        branch_reader(
                            "live", str(i), branch, session_id=str(i), watermark=lambda i=i: len(tapes[i])
                        )
                    ],
                    session_id=str(i),
                    advance=tapes[i].extend,
                )
            )
        result = teacher_force(m, encode, ss, raw, mode=mode, generator=torch.Generator().manual_seed(117))
        assert list(map(bytes, tapes)) == raw
        return result, before

    raw = [b"abx", b"ab\xff", b"abcde"]
    out, before = run(raw, "greedy")
    assert calls == [(2, 2), (1, 4)] and out.targets == 8 and out.objective is None
    assert before == [[b"a", b"ab"], [b"a", b"ab"], [b"a", b"ab", b"abc", b"abcd"]]
    for i in (0, 1):
        torch.testing.assert_close(out.decisions[0][i].log_probs, out.decisions[1][i].log_probs)
    expected = -torch.stack(
        [d.log_probs[y] for ds, w in zip(out.decisions, raw) for d, y in zip(ds, w[1:])]
    ).mean()
    torch.testing.assert_close(expected, out.mean_nll)
    trained, _ = run(raw, "sample")
    trained.objective.backward()
    for g in (m.op.weight.grad, m.pointer.weight.grad, causal.weight.grad):
        assert g is not None and g.norm() > 0


def test_return_to_go_prices_and_later_loss_reach_earlier_actions():
    logp = torch.ones(256).log_softmax(0)
    a, b = torch.tensor(-1.0, requires_grad=True), torch.tensor(-2.0, requires_grad=True)

    def d(lp, i):
        return Decision(logp, lp, (None, None), 1, "one", i, "sample")

    ds = [d(a, 3), d(b, 4)]
    obj = episode_objective(ds, [0, 255], key_price=0.5, instruction_price=0.25)
    c = math.log(256) + 0.75
    torch.testing.assert_close(obj.cost_to_go, torch.tensor([2 * c, c]))
    obj.loss.backward()
    assert a.grad.item() == pytest.approx(c) and b.grad.item() == pytest.approx(c / 2)
    with pytest.raises(ValueError, match="document"):
        episode_objective([ds[0], replace(ds[1], session_id="two")], [0, 255])


def test_m20_directed_batching_no_fetch_of_siblings_and_no_invented_miss():
    class Pages:
        root, nodes = 1, 3

        def __init__(self):
            self.calls = []

        def rows(self, ids):
            self.calls.append(tuple(ids))
            records = []
            for i in ids:
                counts = np.zeros(257, np.uint64)
                child = np.zeros(256, np.uint32)
                if i == 1:
                    counts[[97, 98, 99]] = [9, 5, 3]
                    child[[97, 98]] = [2, 3]
                elif i == 2:
                    counts[[0, 30, 255, 256]] = [2, 1, 3, 3]
                records.append({"h": counts, "child": child})
            return records

    p = Pages()
    with pytest.raises(PermissionError):
        MaterializedBranches("bank", "v", p, scope="partial")
    assert not p.calls
    mount = MaterializedBranches("bank", "v", p, scope="whole", cache_pages=2)
    hit, miss, unknown = mount.histograms((b"a", b"never", b"c?"))
    assert hit[255] == 3 and hit[256] == 3 and sum(miss) == 0 and isinstance(unknown, Unavailable)
    assert p.calls == [(1,), (2,)]  # even the materialized sibling b was not asked for
    assert mount.reader(structural_end=True).read_many((b"a",), 2) == [(Hit((b"a",), 3),)]
    assert len(p.calls) == 2
