"""S14-style closed and independently offered real-client clocks. Predictor agnostic."""

from queue import Empty
from threading import Event, Thread
from time import perf_counter_ns as clock

from .arrival_service_bench import Mailbox, target
from .serving_verify import require


def serve_offers(service, prompts, binding, offer, on_reply, progress, *, trace, synchronize):
    """Only the intake thread offers descriptors; only the owner admits, steps and retires."""
    require(len(prompts) == 32 and offer in ("closed", "spaced", "burst"), "unknown offer schedule")
    rows = [dict(row=i, acks=[]) for i in range(32)]
    q, stop = Mailbox(32), Event()
    active, ticks, faults = {}, [], []
    peak = retired = next_closed = sent = 0
    closed = offer == "closed"
    expected = 384 if closed else 348
    origin = clock()

    def now():
        return clock() - origin

    def produce():
        try:
            for i in range(32):
                sched = (i // 4 * 60 if offer == "spaced" else i // 16 * 50) * 1_000_000
                if stop.wait(max(0, (sched - now()) / 1e9)):
                    return
                q.put_nowait((i, sched, now()))
        except BaseException as e:
            faults.append(e)

    def admit(i, sched, offered):
        opening = now()
        t = service.admit(prompts[i], lambda: binding(i), limit=12, row_id=i)
        active[t] = i
        rows[i].update(scheduled=sched, offered=offered, opening=opening, admitted=now())

    producer = None if closed else Thread(target=produce, name="bounded-byte-offers")
    if producer is not None:
        producer.start()
    try:
        while sent < expected:
            if closed:
                if not active:
                    require(next_closed in (0, 16), "closed oracle did not finish")
                    for i in range(next_closed, next_closed + 16):
                        admit(i, 0, now())
                    next_closed += 16
            else:
                while service.available:
                    try:
                        offered = q.get_nowait() if active else q.get(timeout=1)
                    except Empty:
                        if faults or (not active and not producer.is_alive()):
                            raise RuntimeError("lost offered descriptor") from (faults[0] if faults else None)
                        break
                    admit(*offered)
            if not active:
                continue
            ts = list(active)
            start = now()
            replies = service.step(ts, trace=trace)
            ack = now()
            peak = max(peak, len(ts))
            ticks.append(dict(start=start, ack=ack, rows=[active[t] for t in ts]))
            retire = []
            for t, r in zip(ts, replies, strict=True):
                i = active[t]
                acks = rows[i]["acks"]
                require(
                    r.ticket is t and r.visible_before == len(prompts[i]) + len(acks),
                    "wrong delivered frontier",
                )
                on_reply(i, r)
                acks.append(ack)
                sent += 1
                if len(acks) == (12 if closed else target(i)):
                    retire.append(t)
            if retire:
                service.release(retire)
                tm = now()
                for t in retire:
                    i = active.pop(t)
                    canceled = not closed and target(i) == 3
                    rows[i].update(status="canceled" if canceled else "finished", released=tm)
                    if canceled:
                        try:
                            service.step([t])
                        except ValueError as e:
                            require("stale" in str(e), "wrong cancellation refusal")
                            retired += 1
                        else:
                            raise ValueError("advanced retired ticket")
            progress(sent)
        require(
            not active and not service.ready and not service.noise.tables and q.empty() and not faults,
            "unfinished requests",
        )
        synchronize()
        end = now()
    finally:
        if producer is not None:
            stop.set()
            producer.join()
    return origin, dict(
        requests=rows,
        ticks=ticks,
        emitted=sent,
        wall_ns=end - min(r["offered"] for r in rows),
        peak_active=peak,
        peak_queue=q.high,
        stale_cancels=retired,
    )
