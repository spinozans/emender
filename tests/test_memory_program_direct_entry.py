from copy import deepcopy

import pytest
import torch
from torch import nn
from ndm.memory_program import PackedMemoryPolicy, Session, branch_reader
from ndm.memory_program.direct import DirectTrust, DocumentCounts, direct_branches, direct_features
from ndm.memory_program.direct_entry import DirectEntryProgram, EntryStream, direct_entry_teacher_force
from ndm.memory_program.runtime import OPS, Refused, Unavailable
from ndm.memory_program.serving import StreamUniforms
from ndm.memory_program.sparse_clock import _uniform

WORDS = (b"first\x00\xff=abcde, abcde.", b"\x1e second doc??!", b"XYZ\xff:abcdefghijklm nop")


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(256, 6)
        self.out = nn.Linear(6, 256)

    def forward(self, x):
        h = self.e(x).cumsum(1).tanh()
        return h, self.out(h)


class Bank:
    source, version = "fixture", "immutable"

    def __init__(self):
        self.calls = []

    def histograms(self, keys):
        self.calls.append(tuple(keys))
        out = []
        for k in keys:
            if len(k) > 2:
                out.append(Unavailable("unmaterialized branch"))
            else:
                row = [0] * 257
                row[0 if not k else k[-1]] = 7
                row[255] += 2
                row[256] = 1
                out.append(tuple(row))
        return out


class Program(PackedMemoryPolicy):
    after_emit = None

    def forward(self, sessions, *args, **kw):
        ds = super().forward(sessions, *args, **kw)
        if self.after_emit is not None:
            for s in sessions:
                self.after_emit(s)
        return ds


def models(bias):
    torch.manual_seed(402)
    net = Host()
    p = Program(6, dim=8, heads=2, buckets=128)
    m = DirectEntryProgram(p, DirectTrust(6, 8))
    with torch.no_grad():
        m.entry.linear.bias.fill_(bias)
        p.transitions.weight.fill_(-60)
        p.transitions.weight[-1, OPS.index("SLICE")] = 60
        p.transitions.weight[-1, OPS.index("RECALL")] = 100
        for a, b in [("SLICE", "LOOKUP"), ("LOOKUP", "SAVE"), ("SAVE", "EMIT"), ("RECALL", "EMIT")]:
            p.transitions.weight[OPS.index(a), OPS.index(b)] = 60
        p.prior[OPS.index("SAVE"), 4] = 60
        p.prior[OPS.index("SLICE"), :2] = 60
        p.prior[OPS.index("EMIT"), 4:] = 60
    return net, m


def bindings(words, scored=None):
    tapes = []
    reads = []

    def make(i, w):
        tape = bytearray()
        tapes.append(tape)
        sid = f"entry-fixture-{i}"

        def append(data):
            pos = len(tape)
            assert w[pos : pos + len(data)] == data
            if scored is not None and pos:
                assert len(data) == 1 and (sid, pos) in scored
            tape.extend(data)

        def hist(keys):
            reads.append((sid, len(tape), tuple(keys)))
            assert tuple(keys) == (tape[-1:],)
            h = [0] * 257
            h[tape[-1]] = 2
            h[(tape[-1] + 1) % 256] = 3
            return [h]

        r = branch_reader("tape", sid, hist, session_id=sid, watermark=lambda: len(tape))
        # Deliberately identical public origin, different private bindings.
        return Session([r], advance=append, session_id=sid, document=4, start=19)

    return [make(i, w) for i, w in enumerate(words)], tapes, reads


@pytest.mark.parametrize("bias", [-100.0, 100.0, -0.8])
def test_exact_choices_all_byte_credit_and_writer_across_a_direct_gap(bias):
    host, m = models(bias)
    pairs = [(host, m), (deepcopy(host), deepcopy(m))]
    outs = []
    sigs = []
    for sched, (net, p) in zip(("index", "queue"), pairs, strict=True):
        scored = set()
        ss, tapes, reads = bindings(WORDS, scored)
        bank = Bank()
        direct = {}

        def mark(s):
            k = (s.id, s.visible)
            assert k not in scored and s.pending
            scored.add(k)

        def observe(s, v, a):
            mark(s)
            assert v.tail == bytes(tapes[ss.index(s)][-8:])
            direct[s.id, s.visible] = (v.tail, v.counts, a)

        p.program.after_emit = mark
        v = direct_entry_teacher_force(
            p,
            net,
            ss,
            WORDS,
            bank=bank,
            row_ids=[0, 1, 2],
            gate_seed=7823,
            program_seed=7809,
            clock=sched,
            key_price=0.001,
            instruction_price=0.0001,
            baseline_per_remaining=2.0,
            on_direct=observe,
        )
        assert tuple(bytes(t) for t in tapes) == WORDS and all(
            s.pending and not s.readers and not any(s.cells) for s in ss
        )
        assert (
            v.targets == len(scored) == sum(map(len, WORDS)) - 3
            and v.work["append_calls"] == v.work["append_bytes"] == v.targets + 3
        )
        expected = v.mean_nll.new_zeros(())
        for i, (ds, losses, future) in enumerate(zip(v.decisions, v.losses, v.future, strict=True)):
            for t, d in enumerate(ds):
                pick = d.program
                if pick is None:
                    assert (
                        d.keys == len(direct[ss[i].id, t + 1][-1].probes)
                        and d.nonemit == 0
                        and len(d.trust) == 4
                    )
                else:
                    assert (
                        (ss[i].id, t + 1) not in direct
                        and d.keys == pick.requests
                        and d.nonemit == len(pick.trace) - 1
                        and d.trust is None
                    )
                cost = sum(
                    -later.log_probs[WORDS[i][j + 1]].detach() + 0.001 * later.keys + 0.0001 * later.nonemit
                    for j, later in enumerate(ds[t:], t)
                )
                lp = d.gate_logp if pick is None else d.gate_logp + pick.log_policy
                torch.testing.assert_close(future[t], cost)
                torch.testing.assert_close(losses[t], -d.log_probs[WORDS[i][t + 1]])
                expected = expected + losses[t] + (cost - 2 * (len(ds) - t)) * lp
        torch.testing.assert_close(expected / v.targets, v.objective, atol=1e-5, rtol=1e-5)
        if bias < -90:
            assert len(direct) == v.targets and not reads and not v.widths["program"]
        elif bias > 90:
            assert (
                v.work["program"] == v.targets and not bank.calls and not direct and not v.widths["direct"]
            )
        else:
            assert direct and reads and v.work["direct_keys"] > v.work["direct"]
            delayed = next(
                b.program
                for ds in v.decisions
                for j, b in enumerate(ds)
                if j
                and ds[j - 1].program is None
                and b.program is not None
                and b.program.trace[0].inst.op == "RECALL"
            )
            (gw,) = torch.autograd.grad(-delayed.log_probs[13], p.program.writer.weight, retain_graph=True)
            assert gw.abs().max() > 0
        v.objective.backward()
        if bias == -0.8:
            assert all(
                z.grad.abs().max() > 0
                for z in (p.entry.linear.weight, p.direct.vote.weight, p.program.writer.weight)
            )
        outs.append(v)
        sigs.append((direct, sorted(reads)))
    assert sigs[0] == sigs[1]
    if bias == -0.8:
        assert outs[1].work["held_entry_ticks"] > 0
    for x, y in zip(outs[0].decisions, outs[1].decisions, strict=True):
        for a, b in zip(x, y, strict=True):
            assert (a.program is None) == (b.program is None)
            if a.program is not None:
                assert a.program.trace == b.program.trace
            else:
                torch.testing.assert_close(a.trust, b.trust)
            for k in ("gate_logp", "log_policy", "log_probs"):
                torch.testing.assert_close(getattr(a, k), getattr(b, k), atol=5e-5, rtol=2e-5)
    for a, b in zip(
        [*host.parameters(), *m.parameters()],
        [*pairs[1][0].parameters(), *pairs[1][1].parameters()],
        strict=True,
    ):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("bias", [-100.0, 100.0, -0.8])
def test_live_conditional_predictor_matches_the_standalone_chosen_head_only(bias):
    net, m = models(bias)
    word = WORDS[0]
    ss, tapes, reads = bindings([word])
    oracle, _, oracle_reads = bindings([word])
    state = EntryStream(ss[0], byte_limit=len(word))
    rng1, rng2 = [StreamUniforms(s, [0], len(word), 7809, "cpu") for s in (ss, oracle)]
    count = DocumentCounts(4, 19, byte_limit=len(word))
    runner = m.program.batch_runner()
    b1, b2 = Bank(), Bank()
    state.observe(word[:1])
    oracle[0].observe(word[:1])
    count.observe(word[0], at=0)
    with torch.no_grad():
        # Match the runner's gradient context stamp.
        runner = m.program.batch_runner()
        for t, y in enumerate(word[1:]):
            h, b = net(torch.tensor([list(word[: t + 1])]))
            h, b = h[:, -1], b[:, -1]
            enter = _uniform(7823, 0, t + 1) < torch.sigmoid(m.entry.linear(h)).item()
            calls0, reads0 = len(b1.calls), len(reads)
            d = m([state], h, b, bank=b1, row_ids=[0], gate_seed=7823, program_rng=rng1)[0]
            assert enter == (d.program is not None)
            if enter:
                ref = runner(oracle, h, b, mode="sample", generator=rng2)[0]
                assert ref.trace == d.program.trace and len(b1.calls) == calls0
                lp = ref.log_probs
            else:
                v = count.view(at=t + 1)
                lp, mix = m.direct(h, b, direct_features([v], direct_branches(b2, [v], grouped=False)))
                lp = lp[0]
                torch.testing.assert_close(d.trust, mix[0])
                assert len(reads) == reads0
            torch.testing.assert_close(d.log_probs, lp)
            state.observe(bytes([y]))
            oracle[0].observe(bytes([y]))
            count.observe(y, at=t + 1)
    assert reads == oracle_reads and b1.calls == b2.calls and bytes(tapes[0]) == word
    state.close()
    count.close()


def test_private_counts_refuse_double_ownership_overbudget_and_unscored_observations():
    ss, tapes, _ = bindings([b"abc"])
    stream = EntryStream(ss[0], byte_limit=1)
    with pytest.raises(ValueError, match="exclusive"):
        EntryStream(ss[0], byte_limit=3)
    with pytest.raises(Refused):
        stream.observe(b"ab")
    assert not tapes[0]
    stream.observe(b"a")
    with pytest.raises(ValueError, match="after emit"):
        stream.observe(b"b")
    ss[0].observe(b"b")  # foreign advance of the shared program session
    with pytest.raises(ValueError, match="changed"):
        stream.check()
    stream.close()
    with pytest.raises(ValueError, match="closed"):
        stream.observe(b"c")


def test_ambiguous_chosen_direct_read_closes_all_private_states():
    _, model = models(-100.0)
    bank = Bank()
    bank.histograms = lambda keys: [(1,)] * len(keys)
    ss, tapes, reads = bindings([b"a", b"b"])
    qs = [EntryStream(s, byte_limit=2) for s in ss]
    for q, w in zip(qs, (b"a", b"b"), strict=True):
        q.observe(w)
    with pytest.raises(ValueError, match="histogram"):
        model(
            qs,
            torch.zeros(2, 6),
            torch.zeros(2, 256),
            bank=bank,
            row_ids=[0, 1],
            gate_seed=7823,
            mode="greedy",
        )
    assert all(q.closed and not q._counts.rows for q in qs) and not reads and tapes == [b"a", b"b"]
