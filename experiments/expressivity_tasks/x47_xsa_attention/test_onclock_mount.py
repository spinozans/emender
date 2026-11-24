"""S19: synthetic intake, grant, cancellation and clock regressions. No natural S19 trial here."""

from concurrent.futures import ThreadPoolExecutor
from threading import get_native_id
from time import perf_counter_ns as clock
from types import SimpleNamespace

import pytest
import torch
from ndm.memory_program import Limits, Session
from ndm.memory_program.byte_service import ByteService
from ndm.memory_program.entry_service import EntryReadout
from ndm.memory_program.host_cache import CachedByteHost
from .demand_tape_check import EntryTrace
from .entry_admissions import EntryAdmissions
from .offered_waves import DescriptorOffers, serve_group
from .onclock_mount_bench import refuse
from .onclock_mount_verify import OPS, latencies, ratio_panel
from .test_demand_tape import DirectBank, TinyHost, init, native


def test_offers_real_time_under_lock_sender_has_no_client_or_source():
    with DescriptorOffers(width=2, waves=3, spacing_ns=2_000_000) as q:
        batches = [q.take(i) for i in range(3)]
        assert q.sender != get_native_id() and q.q.high <= 6 and q.q.empty()
        for j, rs in enumerate(batches):
            assert [r["row"] for r in rs] == [2 * j, 2 * j + 1]
            assert all(
                set(r) == {"row", "scheduled", "offered", "taken", "acks"}
                and not r["acks"]
                and q.origin + j * 2_000_000 == r["scheduled"] <= r["offered"] <= r["taken"] <= clock()
                for r in rs
            )
    assert not q.thread.is_alive()


@pytest.mark.parametrize("args", [dict(width=16, waves=3), dict(width=True), dict(spacing_ns=-1)])
def test_no_unbounded_or_malformed_intake(args):
    with pytest.raises(ValueError, match="bounded"):
        DescriptorOffers(**args)


def test_no_early_or_foreign_grant_and_exception_joins_sender():
    q = DescriptorOffers(width=2, waves=4, spacing_ns=400_000_000)
    with pytest.raises(ValueError, match="unavailable"):
        q.take(0)
    with pytest.raises(ValueError, match="unfinished"):
        with q, ThreadPoolExecutor(1) as ex:
            with pytest.raises(ValueError, match="unavailable"):
                ex.submit(q.take, 0).result()
            q.take(0)
            with pytest.raises(ValueError, match="unavailable"):
                q.take(2)
    assert not q.thread.is_alive()
    with pytest.raises(ValueError, match="spent"):
        with q:
            pytest.fail("spent trial restarted")


@native
@torch.no_grad()
def test_generated_group_uses_only_spent_offers_and_closes_real_private_bindings():
    m = init(0.0)
    rows = [dict(doc=i, start=7) for i in range(8)]
    words = [b"\xff\x00\x1e" + bytes([i]) for i in range(8)]
    tr = EntryTrace(rows, words)
    m.program.record_page = tr.page

    def session(row, t):
        return Session(
            t.readers()[:1],
            Limits(),
            advance=t.advance,
            session_id=t.id,
            document=row["doc"],
            start=7,
        )

    guards = []
    clients = EntryAdmissions(
        rows,
        tr,
        admitted_rows=range(8),
        byte_limit=64,
        demand=True,
        encode=None,
        session=session,
        check=lambda: guards.append(1),
    )
    with pytest.raises(ValueError, match="unoffered"), clients.binding(8):
        pytest.fail("bound a future group")
    assert not guards
    read = EntryReadout(m, DirectBank(), gate_seed=7823, on_direct=tr.direct)
    pool = CachedByteHost(TinyHost(), 8, 64, 1 << 20)
    ns = []
    with (
        DescriptorOffers(width=8, waves=1) as q,
        ByteService(pool, entry=read, mode="sample", seed=7530) as service,
    ):
        rows = q.take(0)
        ticks, stale = serve_group(service, rows, words, clients, progress=ns.append)
        assert stale == 1 and ns == [8, 16, 24, *range(31, 88, 7)] and len(guards) == 8 and len(ticks) == 12
    closed = clients.finish()
    assert len(closed) == 8 and all(v["matched"] and v["scrub"] for v in closed) and pool.empty
    assert sum(v["work"]["observations"] for v in closed) == 95 and 0 < read.work["enter"] < 87
    assert all(t["rows"] == list(range(int(j >= 3), 8)) for j, t in enumerate(ticks))
    with pytest.raises(ValueError, match="repeated"), clients.binding(0):
        pytest.fail("reused prior observations")


def test_probe_requires_all_stale_handles_to_reject_for_retirement_not_some_other_fault():
    def fail(msg):
        def fetch(*_):
            raise ValueError(msg)

        return fetch

    rs = [
        SimpleNamespace(
            operation=k,
            read_many=fail("expired branch grant" if j == 0 else "retired record reader"),
        )
        for j, k in enumerate(OPS)
    ]
    assert refuse(rs) == OPS
    rs[-1].read_many = fail("file corruption")
    with pytest.raises(ValueError, match="different stale"):
        refuse(rs)
    rs[1].read_many = lambda *_: []
    with pytest.raises(ValueError, match="remained usable"):
        refuse(rs)


def fixture():
    origin = 1_000_000_000
    setup = origin + 2_000_000
    done = setup + 300_000_000
    x = dict(origin=origin, schedule_start=setup, schedule_ready=done, peak_waiting=24)
    ws = []
    for j in range(4):
        scheduled = origin + 60_000_000 * j
        take = origin + 1_000_000 if not j else done + 1000
        start = max(take + 1000, done) + 1000
        ready = start + 100_000_000
        rs = [
            dict(
                row=i + 8 * j,
                scheduled=scheduled,
                offered=scheduled + 100 + i,
                taken=take + i,
                opening=ready + 100 * (i + 1),
                admitted=ready + 100 * (i + 1) + 50,
                acks=[],
            )
            for i in range(8)
        ]
        ts = []
        t = ready + 1000
        for step in range(12):
            ack = t + 500
            current = rs[int(step >= 3) :]
            ts.append(dict(start=t, ack=ack, rows=[r["row"] for r in current]))
            for r in current:
                r["acks"].append(ack)
                if step + 1 == (3 if r["row"] % 8 == 0 else 12):
                    r.update(
                        released=ack + 200,
                        status="finished" if step + 1 == 12 else "canceled",
                    )
            t = ack + 1000
        done = t + 2000
        ws.append(
            dict(
                start=start,
                ready=ready,
                stale_checked=ready + 30,
                retired_at=t + 1000,
                requests=rs,
                ticks=ts,
                after_release=dict(clock_ns=done),
                mount=dict(open_ns=50_000_000),
            )
        )
    x.update(close_start=done + 100, schedule_closed=done + 25_000_100, end=done + 26_000_000)
    return x, ws


def test_primary_clock_charges_actual_first_offer_final_close_and_wait_not_scheduled_time():
    x, ws = fixture()
    z = latencies(x, ws)
    assert (
        z["charged_setup_s"] == 0.7 and z["lifecycle_s"] == (x["end"] - ws[0]["requests"][0]["offered"]) / 1e9
    )
    assert z["first_grant_wait_ms"]["p95"] > 400 and z["offered_during_mount"] == 24
    assert z["sojourn_ms"]["p95"] > z["first_ms"]["p95"] and z["peak_waiting"] == 24


@pytest.mark.parametrize("fault", ["mount_before_offers", "cross_group"])
def test_backdated_setup_or_inactive_row_rejected(fault):
    x, ws = fixture()
    if fault == "mount_before_offers":
        x["schedule_start"] = x["origin"] + 102
    else:
        ws[1]["ticks"][8]["rows"][0] = 0
    with pytest.raises(ValueError):
        latencies(x, ws)


def test_recommendation_can_fail_cold_wait_or_current_rss_even_with_good_mean_lifecycle():
    def m(image):
        return dict(
            lifecycle_s=5 if image else 10,
            charged_setup_s=3 if image else 7,
            first_ms=dict(p95=400 if image else 800),
            first_grant_wait_ms=dict(p95=100),
            inter_ms=dict(p95=1),
            m20_peak_mib=91,
            position_peak_mib=202,
            peak_allocated_mib=1200,
            rss_drift_mib=12,
        )

    cells = {f"{k}-{r}": m(k == "image") for k in ("fresh", "image") for r in "AB"}
    assert ratio_panel(cells)["passed"]
    cells["image-A"]["first_grant_wait_ms"]["p95"] = 165
    cells["image-B"]["rss_drift_mib"] = 65
    d = ratio_panel(cells)
    assert not d["passed"] and set(d["failed"]) == {
        "cold_first_p95",
        "max_image_rss_drift_mib",
    }
