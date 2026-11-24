"""S20 off-card qualification: real F6 native positions, real values before birth, and fault cases."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
from weakref import ref

import numpy as np
import pytest
import torch
from ndm.memory_program import Limits, Session, Span, Unavailable
from ndm.memory_program.byte_service import ByteService
from ndm.memory_program.host_cache import CachedByteHost
from . import scoped_records as lib
from .demand_position_trace import NativeLog, ProgramTrace, compare_program
from .demand_position_verify import activation
from .entry_admissions import EntryAdmissions
from .offered_waves import DescriptorOffers, serve_group
from .test_demand_tape import TinyHost, init, native
from .test_scoped_records import Encoder, mount

pytestmark = native


def raw(desc, n):
    with desc.text.open("rb") as f:
        return f.read(n)


def test_real_contents_and_follower_before_first_locate_no_native_exit_to_invent():
    def use():
        desc, fr, _, domain = mount()
        enc = Encoder()
        weak = list(map(ref, (desc, desc.starts, enc)))
        with domain, domain.borrow(desc.name, desc.version, scope="whole") as g:
            with lib.ScopedRecords(desc, fr, scope="whole", branches=g, encode=enc, defer_position=True) as s:
                loc, get, nxt = s.readers()
                span = Span(desc.name, desc.version, 0, 0, 8)
                bs = raw(desc, 24)
                assert loc.read_many((), 8) == [] and isinstance(
                    loc.read_many((b"x" * 257,), 8)[0], Unavailable
                )
                assert isinstance(get.read_many((replace(span, version="else"),), 8)[0], Unavailable)
                a = get.read_many((span,), 8)[0][0]
                b = nxt.read_many((span,), 8)[0][0]
                assert a.keys == (bs[:8],) and a.value.item() == sum(bs[:8]) and b.keys[1] == bs[8:24]
                s.check()
                assert s.pid is s.start_ns is s.birth is None
            assert s.final["pid"] is s.final["rc"] is s.final["start_ns"] is None and s.final["fd_closed"]
            assert s.final["work"]["preads"] == 2
        return weak, get

    weak, r = use()
    assert not any(v() is not None for v in weak)
    with pytest.raises(ValueError, match="retired"):
        r.read_many((), 8)


def test_certified_overcap_skips_native_unknown_parent_requires_live_worker_ordered_subset():
    desc, fr, p, domain = mount()
    key = raw(desc, 8)
    calls = []

    def rows(ids):
        assert tuple(ids) == (1,)
        h = np.ones(257, np.uint64)
        h[ord("z")] = 15
        return [dict(h=h, child=np.zeros(256, np.uint32))]

    p.rows = rows
    with domain, domain.borrow(desc.name, desc.version, scope="whole") as g:
        with lib.ScopedRecords(
            desc,
            fr,
            scope="whole",
            branches=g,
            defer_position=True,
            on_native=lambda a, b, c, d, e, f, h: calls.append((a, b, tuple(c), e, f, h)),
        ) as s:
            loc, _, _ = s.readers()
            assert loc.read_many((b"z",), 8) == [Unavailable("raw interval exceeds 8")] and s.pid is None
            over, hits, invalid = loc.read_many((b"z", key, b""), 8)
            assert isinstance(over, Unavailable) and hits and isinstance(invalid, Unavailable)
            pid = s.pid
            assert pid > 0 and calls[0][:3] == ((b"z", key, b""), 8, (key,))
            assert calls[0][3] <= s.birth["start"] < s.birth["ready"] <= calls[0][4] < calls[0][5]
            assert loc.read_many((key,), 8) == [hits] and s.pid == pid and len(calls) == 2
        assert s.final["rc"] == 0 and s.final["work"]["certified_overcap"] == 2


def test_unused_companion_expires_no_birth_even_during_next_same_descriptor_grant():
    d, r, _, domain = mount()
    with domain:
        with domain.borrow(d.name, d.version, scope="whole") as g:
            s = lib.ScopedRecords(d, r, scope="whole", branches=g, defer_position=True)
            old = s.readers()
        with (
            domain.borrow(d.name, d.version, scope="whole") as g,
            lib.ScopedRecords(d, r, scope="whole", branches=g, defer_position=True) as successor,
        ):
            for op in old:
                with pytest.raises(ValueError, match="retired"):
                    op.read_many((), 8)
            assert s.closed and s.final["birth"] is successor.birth is None


def test_foreign_thread_never_activates_parked_locator():
    d, r, _, domain = mount()
    with (
        domain,
        domain.borrow(d.name, d.version, scope="whole") as g,
        lib.ScopedRecords(d, r, scope="whole", branches=g, defer_position=True) as s,
    ):
        loc = s.readers()[0]
        with ThreadPoolExecutor(1) as ex, pytest.raises(ValueError, match="foreign"):
            ex.submit(loc.read_many, (b"query",), 8).result()
        assert s.pid is None and loc.read_many((), 8) == []


def test_both_native_launches_bound_to_pinned_binary_index_and_corpus_and_same_answers():
    d, r, _, domain = mount()
    result = []
    pids = []
    key = raw(d, 8)
    with domain:
        for defer in (False, True):
            with (
                domain.borrow(d.name, d.version, scope="whole") as g,
                lib.ScopedRecords(d, r, scope="whole", branches=g, defer_position=defer) as s,
            ):
                assert (s.pid is None) == defer
                loc, get, follow = s.readers()
                hits = loc.read_many((key,), 8)[0]
                spans = tuple(v.keys[0] for v in hits)
                result.append((spans, get.read_many(spans, 8), follow.read_many(spans, 8)))
                pids.append(s.pid)
                cmd = Path(f"/proc/{s.pid}/cmdline").read_bytes().split(b"\0")[:4]
                assert cmd[1] == b"positional-worker"
                for arg, path in zip((cmd[0], cmd[2], cmd[3]), (lib.EXE, d.index, d.text), strict=True):
                    assert arg.startswith(b"/proc/self/fd/")
                    assert os.path.samestat(
                        os.stat(path), os.stat(f"/proc/{s.pid}/fd/" + arg.decode().split("/")[-1])
                    )
            assert s.final["rc"] == 0
    assert pids[0] != pids[1] and result[0] == result[1]


@pytest.mark.parametrize("fault", ["pin", "birth", "request"])
def test_failed_late_use_is_terminal_never_restarts(monkeypatch, fault):
    d, r, _, domain = mount()
    key = raw(d, 8)
    starts = []
    orig = lib.PositionalWorker
    with (
        domain,
        domain.borrow(d.name, d.version, scope="whole") as g,
        lib.ScopedRecords(d, r, scope="whole", branches=g, defer_position=True) as s,
    ):

        def bad(*a, **kw):
            raise ValueError("fault")

        def start(*a, **kw):
            starts.append(1)
            if fault == "birth":
                return bad()
            w = orig(*a, **kw)
            w.request = bad
            return w

        loc = s.readers()[0]
        monkeypatch.setattr(lib, "PositionalWorker", start)
        if fault == "pin":
            monkeypatch.setattr(s._pins, "check", bad)
        with pytest.raises(ValueError):
            loc.read_many((key,), 8)
        assert s.closed and s.final["fd_closed"] and len(starts) == int(fault != "pin")
        assert s.final["rc"] == (0 if fault == "request" else None)
        if fault == "birth":
            assert s.birth["ready"] is None and s.pid is None
        with pytest.raises(ValueError, match="retired"):
            loc.read_many((key,), 8)
        assert len(starts) == int(fault != "pin")


def test_late_worker_death_refuses_empty_value_read_without_rebirth():
    d, r, _, domain = mount()
    with (
        domain,
        domain.borrow(d.name, d.version, scope="whole") as g,
        lib.ScopedRecords(d, r, scope="whole", branches=g, defer_position=True) as s,
    ):
        loc, get, _ = s.readers()
        loc.read_many((raw(d, 8),), 8)
        pid = s.pid
        os.kill(pid, 9)
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        with pytest.raises(ValueError):
            get.read_many((), 8)
        assert s.closed and s.final["rc"] == -9 and s.final["fd_closed"] and s.pid == pid


def test_batch_limit_or_unselected_internal_issue_never_starts_workspace():
    d, r, _, domain = mount()
    with (
        domain,
        domain.borrow(d.name, d.version, scope="whole") as g,
        lib.ScopedRecords(d, r, scope="whole", branches=g, defer_position=True) as s,
    ):
        loc = s.readers()[0]
        with pytest.raises(ValueError, match="batch"):
            loc.read_many((b"x",) * 17, 8)
        with pytest.raises(ValueError, match="no chosen"):
            s._issue((b"x",))
        assert not s.closed and s.pid is None


@torch.no_grad()
def test_unconditional_generated_trace_has_only_actual_program_scores_and_preserves_cancellations():
    model = init(0.0).program
    words = [b"\xff\x00\x1e" + bytes([i]) for i in range(8)]
    rows = [dict(doc=i, start=2) for i in range(8)]

    def session(row, t):
        return Session(
            t.readers()[:1], Limits(), advance=t.advance, session_id=t.id, document=row["doc"], start=2
        )

    traces = []
    for _ in range(2):
        tr = ProgramTrace(rows, words)
        model.record_page = tr.page
        c = EntryAdmissions(
            rows,
            tr,
            admitted_rows=range(8),
            byte_limit=64,
            demand=True,
            encode=None,
            session=session,
            check=lambda: None,
            program_only=True,
        )
        with (
            DescriptorOffers(width=8, waves=1) as offers,
            ByteService(
                CachedByteHost(TinyHost(), 8, 64, 1 << 20), policy=model, mode="sample", seed=7530
            ) as svc,
        ):
            ticks, stale = serve_group(svc, offers.take(0), words, c)
            assert stale == 1 and len(ticks) == 12
        c.finish()
        assert all(set(r["scores"]) == {"byte_logp", "joint_logp"} for rs in tr.records for r in rs)
        assert tr.prefix(0, 3)["length"] == 3 and tr.prefix(7, 12)["length"] == 12
        traces.append(tr)
    assert compare_program(*traces)["positions"] == 87 and compare_program(*traces)["passed"]
    traces[0].records[3][1]["scores"]["byte_logp"][11] += 0.1
    assert not compare_program(*traces)["passed"]


def test_positive_native_charge_must_be_inside_selected_issuer_tick():
    c = dict(
        tick=0,
        issuers=[1],
        chosen=2,
        keys=1,
        cap=8,
        sha="5" * 64,
        lookup_begin=101,
        before=103,
        begin=220,
        end=401,
        lookup_end=408,
        ranks=12,
        lf_steps=5,
        lf_rows=2,
        verified_spans=2,
        exposed_spans=2,
        overcap=0,
    )
    w = dict(
        start=3,
        ready=85,
        ticks=[dict(start=100, ack=420, rows=[0, 1])],
        native_reads=[c],
        position=dict(
            pid=4,
            rc=0,
            peak_kib=123,
            start_ns=90,
            birth=dict(
                reason="unresolved occurrences", start=107, ready=202, handshake_bytes=23, arena_bytes=40
            ),
            work=dict(worker_keys=1, worker_batches=1, ranks=12, lf_steps=5, lf_rows=2, verified_spans=2),
        ),
    )
    assert activation(w, True)["first_native_ms"] == (401 - 103) / 1e6
    w["position"]["birth"]["start"] = 93
    with pytest.raises(ValueError):
        activation(w, True)
    w["position"]["birth"]["start"] = 107
    c["issuers"] = [2]
    with pytest.raises(ValueError):
        activation(w, True)


def test_raw_subset_attribution_excludes_certified_peer_and_includes_each_actual_issuer():
    trace = SimpleNamespace(
        origins={("d", i): i for i in range(3)},
        names=SimpleNamespace(live={str(i): ("d", i) for i in range(3)}),
    )
    log = NativeLog(trace)
    log.raw(
        (b"certified", b"needed"),
        8,
        [b"needed"],
        ([[]], dict(rows=[dict(raw=0, status="complete", hits=[], lf=[], ranks=1)])),
        3,
        4,
        6,
    )
    op = SimpleNamespace(source="P100", operation="occurrences")
    log.lookup(op, 2, 7, [("0", (b"certified",)), ("1", (b"needed",)), ("2", (b"needed",))])
    (c,) = log.finish([dict(start=1, ack=10, rows=[0, 1, 2])])
    assert (
        c["tick"] == 0 and c["issuers"] == [1, 2] and c["chosen"] == 2 and c["keys"] == 1 and not log.pending
    )


def test_absent_locator_not_a_clean_exit_or_zero_peak():
    w = dict(
        position=dict(pid=None, birth=None, work={}, rc=None, start_ns=None, peak_kib=None), native_reads=[]
    )
    assert activation(w, True)["birth_ms"] is None
    w["position"]["rc"] = 0
    with pytest.raises(ValueError, match="imagined"):
        activation(w, True)
