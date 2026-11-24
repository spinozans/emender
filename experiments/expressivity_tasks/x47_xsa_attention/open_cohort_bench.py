"""S13. Same independently offered CPU-sampled predictor; selected immutable reads inline or parked."""

from __future__ import annotations
import argparse, asyncio, copy, hashlib, json, os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from queue import Empty
from threading import Thread, Event
from time import perf_counter_ns as clock
import torch
from ndm.memory_program import Limits
from ndm.memory_program.byte_service import ByteService
from ndm.memory_program.resident_host import ResidentArena
from .arrival_domain import ServingDomain
from .arrival_service_bench import Mailbox, children_cpu, event_update, target
from .large_host_bench import prepare_large
from .serving_bench import LockedBranches, process_usage, ROUTE_SEED
from .workbench_run import dump
from .workbench_witness import require


class PositionTrace:
    """In-process, transient attribution; retain only wall intervals, row ids and counts."""

    def __init__(self, domain, origin):
        self.origin = origin
        self.raw = []
        self.groups = []
        self.reader = next(r for r in domain.shared if r.operation == "occurrences")
        call = domain.record.worker.request

        def timed(keys):
            a = clock()
            out = call(keys)
            self.raw.append((a, clock(), tuple(keys)))
            return out

        domain.record.worker.request = timed

    def group(self, r, a, b, items, rows):
        if r is not self.reader:
            return
        chosen = defaultdict(set)
        for who, ks in items:
            for k in ks:
                chosen[k].add(rows[who])
        self.groups.append((a, b, chosen))

    def receipt(self):
        out = []
        for a, b, ks in self.raw:
            (chosen,) = [u for lo, hi, u in self.groups if lo <= a < b <= hi]
            issuers = sorted(set.union(*(chosen[k] for k in ks)))
            require(issuers and all(chosen[k] for k in ks), "unattributed native request")
            out.append(dict(start=a - self.origin, end=b - self.origin, issuers=issuers, keys=len(ks)))
        self.groups.clear()
        self.raw.clear()
        return out


async def trial(run, pool, cpu, selected, ws, man, schedule, offer, prior):
    name = f"{offer}-{schedule}"
    q = Mailbox(32)
    stop = Event()
    fault = []
    loop = asyncio.get_running_loop()
    wake = asyncio.Event()
    limits = Limits(pattern=16)
    beg = clock()
    t0 = pool.stats.copy()
    p0 = cpu.calls.copy()
    b0 = {w: g.work.copy() for w, g in pool.backend.buckets.items()}
    d0 = pool.backend.work.copy()
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
    sids = {}
    batches = []
    stale = []
    maxnoise = 0
    with ServingDomain(man, cpu, limits, bank_type=LockedBranches) as domain:
        pids = [domain.pages.proc.pid, domain.record.worker.proc.pid]
        cpu0, native0, _ = process_usage(pids)
        child0 = children_cpu()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        origin = clock()
        position = PositionTrace(domain, origin)

        def now():
            return clock() - origin

        def produce():
            try:
                for i in range(32):
                    due = (i // 4 * 60 if offer == "spaced" else i // 16 * 50) * 1000000
                    if stop.wait(max(0, (due - now()) / 1e9)):
                        return
                    q.put_nowait((i, due, now()))
                    loop.call_soon_threadsafe(wake.set)
            except BaseException as e:
                fault.append(e)
            finally:
                loop.call_soon_threadsafe(wake.set)

        @contextmanager
        def binding(i):
            with domain.binding(selected[i]) as (s, live):
                sids[s.id] = i
                try:
                    yield s, live
                finally:
                    del sids[s.id]

        with ByteService(
            pool,
            limits=limits,
            policy=cpu,
            features=lambda h: h[:, :256].contiguous(),
            mode="sample",
            seed=ROUTE_SEED,
        ) as service:

            def pop():
                nonlocal maxnoise
                if not service.available:
                    return None
                try:
                    i, due, offered = q.get_nowait()
                except Empty:
                    return None
                opening = now()
                t = service.admit(ws[i][:32], lambda: binding(i), limit=12, row_id=i)
                active[t] = i
                trace[i].update(scheduled=due, offered=offered, opening=opening, admitted=now())
                maxnoise = max(maxnoise, len(service.noise.tables))
                return t

            def ack(replies):
                nonlocal sent
                at = now()
                before = sent
                ret = []
                batch = []
                for r in replies:
                    t = r.ticket
                    i = active[t]
                    x = trace[i]
                    require(r.visible_before == 32 + len(x["acks"]), "stale delivery")
                    out[i].update(bytes([r.byte]))
                    event_update(ev[i], r.decision, selected[i])
                    x["acks"].append(at)
                    sent += 1
                    batch.append(i)
                    if len(x["acks"]) == target(i):
                        ret.append(t)
                batches.append(dict(ack=at, rows=batch))
                if before // 16 != sent // 16:
                    progress("serving")
                return ret

            def reap(ts):
                at = now()
                for t in ts:
                    i = active.pop(t)
                    trace[i].update(released=at, status="canceled" if target(i) == 3 else "finished")
                    if target(i) == 3:
                        stale.append(t)
                wake.set()

            async def wait():
                # Only the main lane clears. Recheck both mailbox/capacity under this level-triggered scheme.
                wake.clear()
                if fault:
                    raise RuntimeError("offer failed") from fault[0]
                if service.available and not q.empty():
                    return
                await asyncio.wait_for(wake.wait(), 10)

            async def arrivals():
                n = 0
                while n < 32:
                    t = pop()
                    if t is None:
                        await wait()
                        continue
                    n += 1
                    yield t

            producer = Thread(target=produce, name="bounded-descriptor-offer")
            producer.start()
            stats = {}
            try:
                if schedule == "open":
                    stats = await service.drive(
                        arrivals(),
                        ack,
                        reap,
                        trace=True,
                        offload=[position.reader],
                        on_lookup=lambda r, a, b, items: position.group(r, a, b, items, active),
                    )
                else:
                    assert schedule == "barrier"
                    while sent < 348:
                        while pop() is not None:
                            pass
                        if not active:
                            await wait()
                            continue
                        ts = list(active)
                        ret = ack(
                            service.step(
                                ts,
                                trace=True,
                                on_lookup=lambda r, a, b, items: position.group(r, a, b, items, sids),
                            )
                        )
                        if ret:
                            service.release(ret)
                            reap(ret)
                producer.join()
                require(
                    not fault
                    and q.empty()
                    and not sids
                    and not active
                    and not service.ready
                    and not service.noise.tables,
                    "unfinished intake",
                )
                torch.cuda.synchronize()
                end = now()
                work = service.work.copy()
                for t in stale:
                    try:
                        service.step([t])
                        raise AssertionError("retired ticket accepted")
                    except ValueError as e:
                        require("stale" in str(e), "wrong refusal")
            finally:
                stop.set()
                producer.join()
        require(
            pool.empty and all(not v.count_nonzero().item() for v in pool.slots.values()),
            "retained slot data",
        )
        cpu1, native1, rss = process_usage(pids)
        child1 = children_cpu()
        shared_processes = [domain.pages.proc, domain.record.worker.proc]
        ds = dict(
            private_opened=domain.opened,
            private_peak=domain.peak,
            private_closed=domain.reaped,
            shared_pids=pids,
            provider=domain.provider,
            archive=domain.record.stats,
            live=domain.live_work,
            m20_pages=domain.bank.pages_fetched,
        )
        pos = position.receipt()
    exits = [p.poll() for p in shared_processes]
    require(exits == [0, 0] and all(rc == 0 for _, rc in ds["private_closed"]), "native leak")
    for i, x in enumerate(trace):
        x.update(byte_sha=out[i].hexdigest(), event_sha=ev[i].hexdigest())
    rep = dict(
        name=name,
        offer=offer,
        schedule=schedule,
        requests=trace,
        batches=batches,
        emitted=sent,
        wall_ns=end - min(x["offered"] for x in trace),
        lifecycle_ns=clock() - beg,
        domain_setup_ns=origin - beg,
        domain=ds,
        work=work,
        engine=stats,
        controller=cpu.calls - p0,
        host=pool.stats - t0,
        buckets={w: g.work - b0[w] for w, g in pool.backend.buckets.items()},
        dispatch=pool.backend.work - d0,
        peak_queue=q.high,
        peak_uniforms=maxnoise,
        stale_cancels=len(stale),
        positions=pos,
        end_zero=True,
        shared_exits=exits,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        server_rss_kib=rss[0],
        shared_rss_kib=sum(rss[1:]),
        cpu_core_s=cpu1 - cpu0 + native1 - native0 + child1 - child0,
    )
    dump(run / (name + ".json"), rep)
    print(
        f"{name}: {sent} returns, {sent * 1e9 / rep['wall_ns']:.1f}/s, {len(pos)} native position calls",
        flush=True,
    )
    return sent


@torch.no_grad()
def main(run):
    net, eager, fused, rows, ws, man = prepare_large(run)
    cpu = copy.deepcopy(eager).cpu()
    del fused
    idn = json.loads((run / "identity.json").read_text())
    idn.update(generated_length=12, active_cap=16, waiting_cap=32, step_widths=[1, 4, 8, 16])
    dump(run / "identity.json", idn)
    start = clock()
    total = 0
    with ResidentArena(net, 16, 64, (1, 4, 8, 16), byte_budget=256 << 20, private_budget=64 << 20) as pool:
        pool.set_cover(True)
        torch.cuda.synchronize()
        dump(
            run / "setup.json",
            dict(
                wall_ns=clock() - start, tensor_bytes=pool.tensor_bytes, private_bytes=pool.backend.pool_bytes
            ),
        )
        for offer, sched in [
            ("spaced", "barrier"),
            ("spaced", "open"),
            ("burst", "open"),
            ("burst", "barrier"),
        ]:
            total += asyncio.run(trial(run, pool, cpu, rows, ws, man, sched, offer, total))
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
