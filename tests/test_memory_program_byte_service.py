from contextlib import contextmanager
import pytest
import torch
from torch import nn
from ndm.memory_program import PackedMemoryPolicy, Session
from ndm.memory_program.host_cache import CachedByteHost
from ndm.memory_program.byte_service import ArrivalUniforms, ByteService, CapacityError, Ticket
from ndm.memory_program.serving import StreamUniforms
from ndm.memory_program.direct import DirectTrust
from ndm.memory_program.direct_service import DirectReadout, DirectStream
from ndm.memory_program.runtime import Limits, Refused


class Host:
    device = torch.device("cpu")
    width = 4
    max_context = 64

    def __init__(self):
        self.net = nn.Module().eval()

    def spec(self, c):
        return {"sum": ((1,), torch.long)}

    def encode(self, arena, slots, old, ids, prompt):
        z = (ids.sum(1) + (0 if prompt else arena["sum"][slots, 0])) % 256
        arena["sum"][slots, 0] = z
        return z[:, None].expand(-1, 4).float(), torch.zeros(len(z), 256).scatter(
            1, z[:, None], 5.0
        ).log_softmax(1)


def arena():
    return CachedByteHost(Host(), 2, 64, 1 << 20)


def binding(tag, closed, bad=False):
    @contextmanager
    def open_():
        count = 0

        def observe(data):
            nonlocal count
            if bad and count:
                raise RuntimeError("native append failed")
            count += len(data)

        s = Session(session_id=tag, advance=observe)
        try:
            yield s, tag
        finally:
            closed.append(tag)

    return open_


@torch.no_grad()
def test_new_and_continuing_reordered_without_duplicate_encodes_or_capacity_side_effects():
    a = arena()
    closed = []
    service = ByteService(a, direct=lambda resources, lp: lp)
    x = service.admit(b"\x00\x1e", binding("a", closed), limit=3, row_id=1)
    assert service.step([x])[0].byte == 30
    y = service.admit(b"\xff", binding("b", closed), limit=2, row_id=2)
    with pytest.raises(CapacityError):
        service.admit(b"x", lambda: pytest.fail("opened before capacity"), limit=1, row_id=3)
    for bad in ([x, x], [Ticket(x.generation)], []):
        with pytest.raises(ValueError, match="ticket"):
            service.step(bad)
    assert [r.byte for r in service.step([y, x])] == [255, 60]
    service.release([x])
    assert closed == ["a"]
    z = service.admit(b"\x01", binding("c", closed), limit=1, row_id=3)
    with pytest.raises(ValueError, match="stale"):
        service.step([x])
    assert [r.byte for r in service.step([y, z])] == [254, 1]
    assert not service.ready
    with pytest.raises(ValueError, match="exhausted"):
        service.step([z])
    service.close()
    assert (
        set(closed) == {"a", "b", "c"}
        and a.empty
        and service.available == 0
        and service.work["prefill_tokens"] == 4
        and service.work["step_tokens"] == 2
    )
    assert all(not v.count_nonzero() for v in a.slots.values())


@torch.no_grad()
def test_partial_append_failure_poison_all_bindings_and_no_continuation():
    a = arena()
    closed = []
    service = ByteService(a, direct=lambda r, p: p)
    ts = [service.admit(b"abc", binding(str(i), closed, bad=i == 1), limit=2, row_id=i) for i in range(2)]
    with pytest.raises(RuntimeError, match="native append failed"):
        service.step(ts)
    assert (
        service.closed
        and a.empty
        and set(closed) == {"0", "1"}
        and not service.work["acknowledged"]
        and not service.ready
    )
    with pytest.raises(ValueError, match="closed"):
        service.step([ts[0]])


class DirectBank:
    source, version = "fixture", "v1"

    def __init__(self):
        self.calls = []

    def histograms(self, keys):
        self.calls.append(keys)
        return [(0,) * 255 + (5, 2) for k in keys]


def direct_open(s, attachment=None):
    @contextmanager
    def open_():
        try:
            yield s.session, s if attachment is None else attachment
        finally:
            s.close()

    return open_


@torch.no_grad()
def test_trained_bypass_private_duplicate_document_origins_and_score_before_append():
    h = DirectTrust(4).eval().requires_grad_(False)
    h.vote.weight.zero_()
    h.vote.bias.copy_(torch.tensor([-80.0, -80.0, -80.0, 80.0]))
    bank = DirectBank()
    obs = []
    fn = DirectReadout(
        h, bank, observe=lambda s, v, a, lp, w: obs.append((s.id, s.visible, v.latest, lp.argmax().item()))
    )
    pool = arena()
    ss = [DirectStream(2, 8, limits=Limits(), byte_limit=64) for _ in range(2)]
    with ByteService(pool, feature_direct=fn) as svc:
        ts = [
            svc.admit(w, direct_open(s), limit=2, row_id=0)
            for s, w in zip(ss, [b"ab\x00ab", b"ab\xffab"], strict=True)
        ]
        replies = svc.step(ts[::-1])
        assert [r.byte for r in replies] == [255, 0]
        assert obs == [(ss[1].session.id, 5, 255, 255), (ss[0].session.id, 5, 0, 0)]
        assert [s.counts.visible for s in ss] == [6, 6] and bank.calls == [(b"b\xffab", b"b\x00ab")]
        svc.release(ts[:1])
        assert ss[0].counts.closed and not ss[0].counts.rows and ss[1].counts.rows
        svc.step(ts[1:])
        assert fn.work["predictions"] == 3 and fn.work["trust_calls"] == 2
    assert pool.empty and all(s.counts.closed and not s.counts.tail for s in ss)


@pytest.mark.parametrize("fault", ["attachment", "frontier", "features", "weights", "version"])
@torch.no_grad()
def test_feature_direct_fault_cannot_read_or_continue(fault):
    head = DirectTrust(4).eval().requires_grad_(False)
    bank = DirectBank()
    fn = DirectReadout(head, bank)
    a = arena()
    s = DirectStream(2, 0, limits=Limits(), byte_limit=64)
    other = DirectStream(2, 0, limits=Limits(), byte_limit=64)
    sv = ByteService(
        a, feature_direct=fn, features=(lambda h: h[:, :2]) if fault == "features" else (lambda h: h)
    )
    if fault == "frontier":
        s.counts.observe(3, at=0)
    with pytest.raises(ValueError):
        t = sv.admit(b"a", direct_open(s, other if fault == "attachment" else None), limit=2, row_id=0)
        if fault in ("attachment", "frontier"):
            pytest.fail("attached before validation")
        if fault == "weights":
            head.vote.bias.add_(1)
        if fault == "version":
            bank.version = "v2"
        sv.step([t])
    assert not bank.calls and sv.closed and a.empty and s.counts.closed
    other.close()


def test_private_direct_prompt_cap_is_atomic():
    s = DirectStream(42, 3, limits=Limits(), byte_limit=2)
    s.session.observe(b"a")
    with pytest.raises(Refused):
        s.session.observe(b"bc")
    assert s.counts.visible == s.session.visible == 1 and s.counts.tail == b"a"
    s.close()


def test_admission_uniforms_agree_with_original_after_reorder_drop_and_rerequest():
    ss = [Session(session_id=f"row{i}") for i in range(3)]
    for i, s in enumerate(ss):
        s.observe(b"z" * (i + 1))
    ids = [7, 16, 2]
    a = ArrivalUniforms(7530)
    ref = StreamUniforms(ss, ids, 12, 7530, "cpu")
    for s, i in zip(ss, ids):
        a.admit(s, i, 12)
    torch.manual_seed(40)
    scores = torch.randn(3, 13)
    for t in range(12):
        coords = [(ss[i].id, ss[i].visible + t, t % 6, i % 5) for i in (2, 0, 1)]
        y = a.sample_factor(scores, [13] * 3, coords)
        z = ref.sample_factor(scores, [13] * 3, coords)
        assert y[0] == z[0] and torch.equal(y[1], z[1])
    a.drop(ss[1].id)
    assert ss[1].id not in a.tables


@torch.no_grad()
def test_real_cpu_policy_call_drops_private_state_and_rejects_changed_controller():
    closed = []
    p = PackedMemoryPolicy(4, dim=8, heads=2).eval()
    a = arena()
    e = ByteService(a, policy=p, mode="sample")
    t = e.admit(b"a", binding("priv", closed), limit=4, row_id=0)
    r = e.step([t], trace=True)[0]
    assert r.visible_before == 1 and r.decision.session_id == "priv" and r.decision.mode == "sample"
    e.release([t])
    assert not e.noise.tables and not e.ready and closed == ["priv"]
    y = e.admit(b"other", binding("second", closed), limit=3, row_id=1)
    p.host.weight.add_(0.1)
    with pytest.raises(ValueError, match="changed service policy"):
        e.step([y])
    assert e.closed and a.empty and not e.noise.tables and closed == ["priv", "second"]


@pytest.mark.parametrize("fail", [False, True])
@torch.no_grad()
def test_open_intake_while_chosen_native_read_is_blocked_and_drain_on_error(fail):
    import asyncio
    from threading import Event
    from ndm.memory_program import branch_reader

    async def witness():
        a = arena()
        p = PackedMemoryPolicy(4, dim=8, heads=2).eval()
        e = ByteService(a, policy=p, mode="greedy")
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        available = asyncio.Event()
        unblock = Event()
        done = Event()
        closed = []
        returned = []
        names = {}

        def hist(keys):
            assert keys == (b"s",)
            loop.call_soon_threadsafe(started.set)
            assert unblock.wait(5), "intake or replacement stalled"
            done.set()
            return [[0] * 257]

        r = branch_reader("shared", "stable", hist)

        @contextmanager
        def private():
            try:
                yield Session([r], session_id="slow"), None
            finally:
                assert done.is_set(), "discarded before draining a native read"
                closed.append("slow")

        async def intake():
            t = e.admit(b"s", private, limit=1, row_id=0)
            names[t] = "slow"
            yield t
            await asyncio.wait_for(started.wait(), 3)
            for i, tag in enumerate(("peer", "new")):
                if i:
                    await asyncio.wait_for(available.wait(), 3)
                t = e.admit(b"pq", binding(tag, closed), limit=5, row_id=1 + i)
                names[t] = tag
                yield t

        def ack(replies):
            stop = []
            for x in replies:
                tag = names[x.ticket]
                returned.append(tag)
                if tag != "slow":
                    assert started.is_set() and not done.is_set()
                    if returned.count(tag) == 2:
                        if fail:
                            unblock.set()
                            raise RuntimeError("owner delivery failed")
                        stop.append(x.ticket)
            return stop

        def reap(ts):
            for t in ts:
                tag = names[t]
                assert tag in closed
                if tag == "peer":
                    available.set()
                if tag == "new":
                    unblock.set()

        if fail:
            with pytest.raises(ExceptionGroup, match="TaskGroup") as err:
                await e.drive(intake(), ack, reap, offload=[r])
            assert (
                "owner delivery failed" in repr(err.value)
                and e.closed
                and returned == ["peer"] * 2
                and set(closed) == {"peer", "slow"}
            )
        else:
            stats = await e.drive(intake(), ack, reap, offload=[r])
            assert (
                stats["emits"] == 5
                and returned == ["peer"] * 2 + ["new"] * 2 + ["slow"]
                and closed == ["peer", "new", "slow"]
            )
        assert done.is_set() and a.empty and not e.ready and not e.noise.tables
        e.close()

    asyncio.run(witness())
