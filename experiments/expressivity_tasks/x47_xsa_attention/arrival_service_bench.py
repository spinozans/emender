"""S12: bounded real admissions and tight per-client oracle on exogenously offered local work."""

from __future__ import annotations
import argparse, copy, hashlib, json, os, resource
from collections import Counter
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
from time import perf_counter_ns as clock
import torch
from ndm.memory_program import Limits
from ndm.memory_program.byte_service import ByteService
from ndm.memory_program.resident_host import ResidentArena
from .arrival_domain import ServingDomain
from .wide_host_bench import Bound
from .large_host_bench import prepare_large
from .packed_program_profile import BindingNames
from .serving_bench import direct, measure, process_usage, ROUTE_SEED
from .workbench_run import dump
from .workbench_witness import require


def target(i):
    return 3 if i % 8 == 0 else 12


class Mailbox(Queue):
    high = 0

    def _put(self, item):
        super()._put(item)
        self.high = max(self.high, self._qsize())  # lock held by Queue.put


def event_update(digest, d, row):
    names = BindingNames([])
    names.live[d.session_id] = (row["doc"], row["start"])
    parts = tuple(
        (v.inst, v.status, v.register, v.rows, names.origin(v.reader), tuple(map(names.key, v.keys)))
        for v in d.trace
    )
    buf = repr((d.visible, d.requests, parts)).encode()
    digest.update(len(buf).to_bytes(8, "little"))
    digest.update(buf)


def children_cpu():
    x = resource.getrusage(resource.RUSAGE_CHILDREN)
    return x.ru_utime + x.ru_stime


def trial(run, n, pool, cpu, selected, ws, man, mode, offer, prior):
    name = f"{offer}-{mode}"
    q = Mailbox(32)
    stop = Event()
    fault = []
    limits = Limits(pattern=16)
    beg = clock()
    sent = 0

    def progress(phase):
        dump(
            run / "progress.json",
            dict(cell=name, phase=phase, total_emitted=prior + sent, server_pid=os.getpid()),
        )

    progress("setup")
    trace = [dict(row=i, acks=[]) for i in range(32)]
    out = [hashlib.sha256() for _ in trace]
    ev = [hashlib.sha256() for _ in trace]
    active = {}
    ticks = []
    dw = Counter()
    host0 = pool.stats.copy()
    bucket0 = {w: g.work.copy() for w, g in pool.backend.buckets.items()}
    dispatch0 = pool.backend.work.copy()
    maxnoise = 0
    tested = 0
    with ServingDomain(man, cpu, limits) as domain:
        pids = [domain.pages.proc.pid, domain.record.worker.proc.pid]
        cpu0, native0, _ = process_usage(pids)
        child0 = children_cpu()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        origin = clock()

        def now():
            return clock() - origin

        def produce():
            try:
                for i in range(32):
                    schedule = (i // 4 * 60 if offer == "spaced" else i // 16 * 50) * 1000000
                    if stop.wait(max(0, (schedule - now()) / 1e9)):
                        return
                    q.put_nowait((i, schedule, now()))
            except BaseException as e:
                fault.append(e)

        def bypass(lives, lp):
            return direct(domain.bank, lives, lp, dw)

        kwargs = (
            dict(policy=cpu, features=lambda h: h[:, :256].contiguous(), mode="sample", seed=ROUTE_SEED)
            if mode == "sample"
            else dict(direct=bypass)
        )
        with ByteService(pool, limits=limits, **kwargs) as service:
            producer = Thread(target=produce, name="bounded-descriptor-offer")
            producer.start()
            try:
                while sent < 348:
                    while service.available:
                        try:
                            if active:
                                i, sched, offered = q.get_nowait()
                            else:
                                i, sched, offered = q.get(timeout=1)
                        except Empty:
                            if fault or (not active and not producer.is_alive()):
                                raise RuntimeError("lost offered work") from (fault[0] if fault else None)
                            break
                        opening = now()
                        t = service.admit(
                            ws[i][:32], lambda i=i: domain.binding(selected[i]), limit=12, row_id=i
                        )
                        active[t] = i
                        trace[i].update(scheduled=sched, offered=offered, opening=opening, admitted=now())
                    if not active:
                        continue
                    ts = list(active)
                    start = now()
                    replies = service.step(ts, trace=mode == "sample")
                    ack = now()
                    retire = []
                    ticks.append(dict(start=start, ack=ack, rows=[active[t] for t in ts]))
                    maxnoise = max(maxnoise, len(service.noise.tables))
                    for t, r in zip(ts, replies, strict=True):
                        i = active[t]
                        x = trace[i]
                        require(r.ticket is t and r.visible_before == 32 + len(x["acks"]), "stale response")
                        out[i].update(bytes([r.byte]))
                        if mode == "sample":
                            event_update(ev[i], r.decision, selected[i])
                        x["acks"].append(ack)
                        sent += 1
                        if len(x["acks"]) == target(i):
                            retire.append(t)
                    if retire:
                        service.release(retire)
                        end = now()
                        for t in retire:
                            i = active.pop(t)
                            trace[i].update(status="canceled" if target(i) == 3 else "finished", released=end)
                            if target(i) == 3:
                                try:
                                    service.step([t])
                                    raise AssertionError("retired ticket ran")
                                except ValueError as e:
                                    require("stale" in str(e), "wrong refusal")
                                    tested += 1
                    progress("serving")
                producer.join()
                require(
                    not fault and q.empty() and not active and not service.ready and not service.noise.tables,
                    "unfinished requests",
                )
                torch.cuda.synchronize()
                end = now()
                work = service.work.copy()
            finally:
                stop.set()
                producer.join()
        require(
            pool.empty and all(not t.count_nonzero().item() for t in pool.slots.values()),
            "retained slot data",
        )
        cpu1, native1, rss = process_usage(pids)
        child1 = children_cpu()
        shared_processes = (domain.pages.proc, domain.record.worker.proc)
        private = list(domain.reaped)
        stats = dict(
            provider=domain.provider,
            archive=domain.record.stats,
            live=domain.live_work,
            private_opened=domain.opened,
            private_peak=domain.peak,
            private_closed=private,
            shared_pids=pids,
            m20_pages=domain.bank.pages_fetched,
        )
    shared_after = [p.poll() for p in shared_processes]
    require(
        all(rc == 0 for _, rc in private) and shared_after == [0, 0], "native process did not exit cleanly"
    )
    for i, x in enumerate(trace):
        x.update(byte_sha=out[i].hexdigest(), event_sha=ev[i].hexdigest() if mode == "sample" else None)
    first = min(x["offered"] for x in trace)
    rep = dict(
        name=name,
        mode=mode,
        offer=offer,
        requests=trace,
        ticks=ticks,
        emitted=sent,
        wall_ns=end - first,
        lifecycle_ns=clock() - beg,
        domain_setup_ns=origin - beg,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        server_rss_kib=rss[0],
        shared_rss_kib=sum(rss[1:]),
        peak_queue=q.high,
        peak_uniforms=maxnoise,
        stale_cancels=tested,
        work=work,
        direct_work=dw,
        host=pool.stats - host0,
        buckets={w: g.work - bucket0[w] for w, g in pool.backend.buckets.items()},
        dispatch=pool.backend.work - dispatch0,
        end_zero=True,
        domain=stats,
        shared_exits=shared_after,
        cpu_core_s=cpu1 - cpu0 + native1 - native0 + child1 - child0,
    )
    dump(run / (name + ".json"), rep)
    print(
        f"{name}: {sent} returned bytes, {sent * 1e9 / rep['wall_ns']:.1f}/s, peak {domain.peak} active, queue {q.high}",
        flush=True,
    )
    return sent


@torch.no_grad()
def main(run):
    n, eager, fused, rows, ws, man = prepare_large(run)
    cpu = copy.deepcopy(eager).cpu()
    del fused
    idn = json.loads((run / "identity.json").read_text())
    idn.update(generated_length=12, active_cap=16, waiting_cap=32, step_widths=[1, 4, 8, 16])
    dump(run / "identity.json", idn)
    start = clock()
    total = 0
    with ResidentArena(n, 16, 64, (1, 4, 8, 16), byte_budget=256 << 20, private_budget=64 << 20) as pool:
        pool.set_cover(True)
        torch.cuda.synchronize()
        dump(
            run / "setup.json",
            dict(
                wall_ns=clock() - start, tensor_bytes=pool.tensor_bytes, private_bytes=pool.backend.pool_bytes
            ),
        )

        def host(n, s, l):
            return Bound(pool, n, s, l)

        for block in (0, 1):
            ix = list(range(block * 16, block * 16 + 16))
            for m in ("direct", "fused-sample"):
                total += measure(
                    run,
                    n,
                    eager if m == "direct" else cpu,
                    m,
                    block,
                    [rows[i] for i in ix],
                    [ws[i] for i in ix],
                    ix,
                    man,
                    total,
                    label=f"oracle-{m}",
                    host_factory=host,
                    prefix=32,
                    length=12,
                    oracle_lengths=[target(i) for i in ix],
                )
        for offer, mode in [
            ("spaced", "direct"),
            ("burst", "sample"),
            ("spaced", "sample"),
            ("burst", "direct"),
        ]:
            total += trial(run, n, pool, cpu, rows, ws, man, mode, offer, total)
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
