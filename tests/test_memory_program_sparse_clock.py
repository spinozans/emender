from copy import deepcopy
import pytest
import torch
from torch import nn
from ndm.memory_program import PackedMemoryPolicy, Session, branch_reader
from ndm.memory_program.runtime import OPS
from ndm.memory_program.sparse_clock import SparseProgram, sparse_teacher_force
from ndm.memory_program.serving import StreamUniforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 6)
        self.head = nn.Linear(6, 256)

    def forward(self, x):
        h = self.embed(x).cumsum(1).tanh()
        return h, self.head(h)


WORDS = (b"first\x00\xff=abcde, abcde.", b"\x1e second doc??!", b"XYZ\xff:abcdefghijklm nop", b"a")


def sessions(words):
    tapes = []
    seen = []
    ss = []

    def bind(i, w):
        tape = bytearray()
        tapes.append(tape)
        sid = f"sparse-{i}"
        calls = []
        seen.append(calls)

        def append(b):
            assert w[len(tape) : len(tape) + len(b)] == b
            tape.extend(b)

        def hist(keys):
            calls.append((len(tape), tuple(keys)))
            out = []
            for k in keys:
                assert k == tape[-1:]
                v = [0] * 257
                v[tape[-1]] = 2
                v[(tape[-1] + 1) % 256] = 3
                out.append(v)
            return out

        r = branch_reader("tape", sid, hist, session_id=sid, watermark=lambda: len(tape))
        return Session([r], advance=append, session_id=sid, document=i)

    for i, w in enumerate(words):
        ss.append(bind(i, w))
    return ss, tapes, seen


def models(bias):
    torch.manual_seed(402)
    net = Net()
    p = PackedMemoryPolicy(6, dim=8, heads=2, buckets=128)
    m = SparseProgram(p)
    with torch.no_grad():
        m.entry.linear.bias.fill_(bias)
        p.transitions.weight.fill_(-60)
        p.transitions.weight[-1, OPS.index("SLICE")] = 60
        p.transitions.weight[-1, OPS.index("RECALL")] = 100
        for a, b in [("SLICE", "LOOKUP"), ("LOOKUP", "SAVE"), ("RECALL", "EMIT"), ("SAVE", "EMIT")]:
            p.transitions.weight[OPS.index(a), OPS.index(b)] = 60
        p.prior[OPS.index("SAVE"), 4] = 60
        p.prior[OPS.index("SLICE"), :2] = 60
        p.prior[OPS.index("EMIT"), 4:] = 60
    return net, m


@pytest.mark.parametrize("bias", [-100.0, 100.0, -0.8])
def test_sparse_clock_all_bytes_route_writer_and_host_adjoints(bias):
    a, m = models(bias)
    b = deepcopy(a)
    p = deepcopy(m)
    ws = WORDS[:3]
    outs = []
    seen = []
    for net, model, schedule in [(a, m, "index"), (b, p, "compact")]:
        ss, tapes, reads = sessions(ws)
        v = sparse_teacher_force(
            model,
            net,
            ss,
            ws,
            row_ids=[0, 1, 2],
            gate_seed=7823,
            program_seed=7809,
            clock=schedule,
            key_price=0.001,
            instruction_price=0.0001,
            baseline_per_remaining=2,
        )
        assert [bytes(x) for x in tapes] == list(ws)
        assert v.work["append_bytes"] == sum(map(len, ws)) and v.work["enter"] + v.work["base"] == v.targets
        if bias == -0.8:
            # This loss is exclusively at an actual later read of the old written page, across a bypass gap.
            old = next(
                d
                for ds in v.decisions
                for prev, d in zip(ds, ds[1:])
                if d.visible > prev.visible + 1 and d.trace[0].inst.op == "RECALL"
            )
            (g,) = torch.autograd.grad(-old.log_probs[13], model.program.writer.weight, retain_graph=True)
            assert g.abs().max() > 0
        v.objective.backward()
        outs.append(v)
        seen.append(reads)
    x, y = outs
    assert x.entries == y.entries and seen[0] == seen[1]
    assert [[d.trace for d in ds] for ds in x.decisions] == [[d.trace for d in ds] for ds in y.decisions]
    for i, ds in enumerate(x.decisions):
        assert [c[0] for c in seen[0][i]] == [
            d.visible for d in ds if any(e.inst.op == "LOOKUP" for e in d.trace)
        ]
    for k in ["byte_logp", "gate_logp", "joint_logp"]:
        for t, u in zip(getattr(x, k), getattr(y, k)):
            torch.testing.assert_close(t, u, atol=5e-5, rtol=2e-5)
    torch.testing.assert_close(x.objective, y.objective)
    torch.testing.assert_close(x.nll, y.nll)
    for v, w in zip([*m.parameters(), *a.parameters()], [*p.parameters(), *b.parameters()]):
        assert (v.grad is None) == (w.grad is None)
        if v.grad is not None:
            torch.testing.assert_close(v.grad, w.grad, atol=2e-4, rtol=2e-4)
    if bias > 90:
        assert x.work["base"] == 0
    if bias < -90:
        assert x.work["enter"] == y.work["program_batches"] == 0 and not any(seen[0])
    if bias == -0.8:
        assert x.work["append_calls"] > y.work["append_calls"] and m.entry.linear.weight.grad.abs().max() > 0


def test_live_gate_has_no_future_schedule_and_matches_teacher_forcing_one_observation_at_a_time():
    net, m = models(-0.8)
    ws = WORDS[:1]
    ss, tapes, _ = sessions(ws)
    v = sparse_teacher_force(m, net, ss, ws, row_ids=[0], gate_seed=7823, program_seed=7809)
    ss, tapes, _ = sessions(ws)
    rng = StreamUniforms(ss, [0], len(ws[0]), 7809, "cpu")
    w = ws[0]
    ss[0].observe(w[:1])
    for t in range(len(w) - 1):
        h, b = net(torch.tensor([list(w[: t + 1])]))
        d = m(ss, h[:, -1], b[:, -1], row_ids=[0], gate_seed=7823, program_rng=rng)[0]
        assert (d.program is not None) == (t in v.entries[0])
        torch.testing.assert_close(d.log_probs, v.byte_logp[0][t])
        torch.testing.assert_close(d.log_policy, v.joint_logp[0][t])
        ss[0].observe(w[t + 1 : t + 2])
    assert bytes(tapes[0]) == w
