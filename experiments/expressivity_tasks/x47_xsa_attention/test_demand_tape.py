"""S17 gates use small synthetic words and separate actual local native session processes."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager

import pytest
import torch
from torch import nn
from ndm.memory_program import Limits, PackedMemoryPolicy, Session, Span, Unavailable
from ndm.memory_program.byte_service import ByteService, CapacityError
from ndm.memory_program.direct import DirectTrust
from ndm.memory_program.direct_entry import DirectEntryProgram, EntryStream
from ndm.memory_program.entry_service import EntryReadout
from ndm.memory_program.host_cache import CachedByteHost
from ndm.memory_program.runtime import OPS, Refused
from .demand_tape_check import EntryTrace, compare
from .private_tape import PrivateTape
from .workbench_native import EXE, Live

native = pytest.mark.skipif(not EXE.exists(), reason="registered local native session binary")


@native
def test_debt_is_private_observed_prefix_and_never_a_structural_end():
    with ExitStack() as stack:
        a, b = [
            stack.enter_context(closing(PrivateTape(7, 100, byte_limit=64, demand=d))) for d in (False, True)
        ]
        for x in (a, b):
            x.advance(b"\xff\x00\x1e\x00")
            _, loc, content, follow = x.readers()
            s = Span("tape", x.id, 7, 101, 102)
            assert content.read_many((s,), 1)[0][0].keys == (b"\x00",)
            assert follow.read_many((s,), 1)[0][0].keys[1] == b"\x1e\x00"
            assert follow.read_many((Span("tape", x.id, 7, 103, 104),), 1) == [()]
            assert isinstance(loc.read_many((b"\xff" * 17,), 8)[0], Unavailable)
        assert a.id != b.id and a.native is not None and b.native is None and b.watermark() == 4
        assert isinstance(b.readers()[2].read_many((Span("tape", a.id, 7, 100, 101),), 1)[0], Unavailable)
        for x in (a, b):
            h = x.histogram(b"\x00")
            assert sum(h) == h[0x1E] == 1 and h[256] == 0
            assert x.stats["native_bytes"] == 4 and x.stats["native_appends"] == 1
            x.advance(b"\xff")
            x.advance(b"\x00")
        assert b.native.visible == 4 < b.watermark() == a.native.visible == 6
        for x in (a, b):
            _, loc, _, _ = x.readers()
            # All three complete matches including the open suffix; no first-cap truncation.
            pos = loc.read_many((b"\x00",), 3)[0]
            assert [p.keys[0].start for p in pos] == [101, 103, 105]
            assert isinstance(loc.read_many((b"\x00",), 2)[0], Unavailable)
            assert x.stats["locate_rpcs"] == 1 and x.native.visible == 6
            x.advance(b"Q")
        assert a.stats["native_appends"] == 4 and b.stats["native_appends"] == 2
    assert a.stats["unsynced_at_close"] == 0 and b.stats["unsynced_at_close"] == 1
    assert all(x.exit["rc"] == 0 and x.exit["peak_kib"] and not x.data and x.native is None for x in (a, b))
    with pytest.raises(ValueError, match="closed"):
        loc.read_many((b"x",), 1)


@native
def test_cold_tape_revoke_before_first_native_demand_and_owner_refusal():
    t = PrivateTape(3, 4, byte_limit=2, demand=True)
    readers = t.readers()
    try:
        with ThreadPoolExecutor(1) as worker, pytest.raises(ValueError, match="foreign"):
            worker.submit(readers[0].read_many, (b"x",), 8).result()
        t.advance(b"xy")
        with pytest.raises(Refused):
            t.advance(b"z")
        assert t.watermark() == 2 and t.native is None
    finally:
        t.close()
    for r, k in zip(readers, (b"x", b"x", Span("tape", t.id, 3, 4, 5), b"foreign"), strict=True):
        with pytest.raises(ValueError, match="closed"):
            r.read_many((k,), 1)
    assert t.exit is None and not t.stats["native_starts"] and t.stats["unsynced_at_close"] == 2


@native
@pytest.mark.parametrize("fault", ["append_ack", "branch_reply", "dead_worker"])
def test_ambiguous_native_tape_never_replays_or_accepts_more_observations(fault, monkeypatch):
    t = PrivateTape(0, 0, byte_limit=8, demand=True)
    r = t.readers()[0]
    try:
        t.advance(b"x")
        if fault == "append_ack":
            monkeypatch.setattr(Live, "append", lambda *a: None)
        elif fault == "branch_reply":
            monkeypatch.setattr(
                Live, "branch", lambda *a: (_ for _ in ()).throw(ValueError("malformed branch reply"))
            )
        else:
            r.read_many((b"x",), 8)
            t.native.proc.kill()
            t.native.proc.wait()
        with pytest.raises((ValueError, RuntimeError)):
            r.read_many((b"x",), 8)
        assert t.closed and t.exit["rc"] is not None and t.stats["native_starts"] == 1
        with pytest.raises(ValueError, match="closed"):
            t.advance(b"z")
        with pytest.raises(ValueError, match="closed"):
            r.read_many((b"x",), 8)
    finally:
        t.close()


class TinyHost:
    device, width, max_context = torch.device("cpu"), 4, 64

    def __init__(self):
        self.net = nn.Module().eval()

    def spec(self, c):
        return {"seen": ((1,), torch.long)}

    def encode(self, arena, slots, old, ids, prompt):
        x = (ids.sum(1) + (0 if prompt else arena["seen"][slots, 0])) % 256
        arena["seen"][slots, 0] = x
        return x[:, None].float().expand(-1, 4) / 255, torch.zeros(len(x), 256).scatter(
            1, x[:, None], 1.0
        ).log_softmax(1)


class DirectBank:
    source, version = "fixture", "v1"

    def histograms(self, keys):
        h = (3,) + (0,) * 254 + (2, 1)
        return [h for k in keys]


def init(bias):
    torch.manual_seed(45)
    p = PackedMemoryPolicy(4, dim=8, heads=2, buckets=64)
    m = DirectEntryProgram(p, DirectTrust(4, 8)).eval().requires_grad_(False)
    m.entry.linear.bias.fill_(bias)
    # Synthetic witness only: on first entry, index a slice and SAVE, later RECALL.
    p.transitions.weight.fill_(-100)
    p.transitions.weight[-1, OPS.index("SLICE")] = 90
    p.transitions.weight[-1, OPS.index("RECALL")] = 150
    for a, b in [("SLICE", "LOOKUP"), ("LOOKUP", "SAVE"), ("SAVE", "EMIT"), ("RECALL", "EMIT")]:
        p.transitions.weight[OPS.index(a), OPS.index(b)] = 90
    p.prior[OPS.index("SLICE"), :2] = 90
    p.prior[OPS.index("SAVE"), 4] = 90
    return m


@native
@pytest.mark.parametrize("bias", [-100.0, 100.0, 0.0])
@torch.no_grad()
def test_real_conditional_service_scores_before_observe_through_gaps_and_reorders(bias):
    rows = [dict(doc=5, start=0), dict(doc=5, start=1)]
    prompts = [b"\xff\x00aba", b"\x1e\x00aba"]
    m = init(bias)
    traces = []
    for demand in (False, True):
        tapes = []
        tr = EntryTrace(rows, prompts)
        m.program.record_page = tr.page
        fn = EntryReadout(m, DirectBank(), gate_seed=7823, on_direct=tr.direct)
        pool = CachedByteHost(TinyHost(), 2, 64, 1 << 20)

        @contextmanager
        def binding(i):
            with closing(PrivateTape(5, i, byte_limit=64, demand=demand)) as live:
                tapes.append(live)
                s = Session(
                    live.readers()[:1],
                    limits=Limits(),
                    advance=live.advance,
                    session_id=live.id,
                    document=5,
                    start=i,
                )
                tr.names.live[s.id] = (5, i)
                with closing(EntryStream(s, byte_limit=64)) as q:
                    yield s, q
                assert live.data == prompts[i] + tr.raw[i]

        with ByteService(pool, entry=fn, mode="sample", seed=7530) as sv:
            ts = []
            for i in range(2):
                ts.append(sv.admit(prompts[i], lambda i=i: binding(i), row_id=i, limit=8))
            with pytest.raises(CapacityError):
                sv.admit(b"x", lambda: pytest.fail("opened before capacity"), row_id=3, limit=2)
            for j in range(8):
                ix = ([1, 0] if demand else [0, 1]) if j < 3 else [0]
                for i, r in zip(ix, sv.step([ts[i] for i in ix], trace=True), strict=True):
                    tr.commit(i, r)
                if j == 2:
                    sv.release([ts[1]])
                    with pytest.raises(ValueError, match="stale"):
                        sv.step([ts[1]])
        assert (
            sv.work["acknowledged"] == 11 and sv.work["prefill_tokens"] == 10 and sv.work["step_tokens"] == 9
        )
        assert pool.empty and not sv.noise.tables and all(t.closed for t in tapes)
        if bias == -100.0:
            assert not fn.work["program_batches"] and fn.work["logical_probes"] == 11
            if demand:
                assert all(t.exit is None for t in tapes)
        elif bias == 100.0:
            assert (
                not fn.work["direct_batches"]
                and fn.work["program_keys"]
                and all(t.exit["rc"] == 0 for t in tapes)
            )
        else:
            assert fn.work["program_keys"] and 0 < fn.work["logical_probes"] < 11
        traces.append(tr)
    c = compare(traces[1], traces[0])
    assert c["passed"] and c["positions"] == 11 and c["numeric"]["byte_logp"]["elements"] == 11 * 256
    # The checker must notice a non-argmax score change too, not just hashes of responses.
    traces[1].records[0][0]["scores"]["byte_logp"][3] += 0.02
    assert not compare(traces[1], traces[0])["passed"]


@torch.no_grad()
def test_swapped_direct_responses_at_the_same_public_frontier_never_acknowledge():
    class Swapped(EntryReadout):
        def __call__(self, *a, **kw):
            return super().__call__(*a, **kw)[::-1]

    svc = ByteService(
        CachedByteHost(TinyHost(), 2, 64, 1 << 20), entry=Swapped(init(-100.0), DirectBank(), gate_seed=7823)
    )
    streams = [EntryStream(Session(document=9, start=8), byte_limit=6) for _ in range(2)]

    @contextmanager
    def binding(q):
        with closing(q):
            yield q.session, q

    ts = [svc.admit(b"xy", lambda q=q: binding(q), limit=1, row_id=i) for i, q in enumerate(streams)]
    with pytest.raises(ValueError, match="unbound direct-entry"):
        svc.step(ts)
    assert svc.closed and not svc.work["acknowledged"] and svc.arena.empty
    assert all(q.closed and q.session.visible == 2 for q in streams)


@pytest.mark.parametrize("change", ["gate", "direct", "program", "bank", "foreign_attachment"])
@torch.no_grad()
def test_changed_conditional_weights_or_foreign_binding_poison_service(change):
    m = init(-100.0)
    bank = DirectBank()
    fn = EntryReadout(m, bank, gate_seed=7823)
    pool = CachedByteHost(TinyHost(), 2, 64, 1 << 20)
    s = Session()
    q = EntryStream(Session() if change == "foreign_attachment" else s, byte_limit=6)

    @contextmanager
    def binding():
        with closing(q):
            yield s, q

    with ByteService(pool, entry=fn) as svc:
        with pytest.raises(ValueError):
            t = svc.admit(b"123", binding, limit=2, row_id=1)
            if change == "foreign_attachment":
                pytest.fail("bound another private stream")
            if change == "bank":
                bank.version = "changed"
            else:
                next({"gate": m.entry, "direct": m.direct, "program": m.program}[change].parameters()).add_(1)
            svc.step([t])
    assert q.closed and pool.empty and not fn.work["predictions"] and not svc.work["acknowledged"]
