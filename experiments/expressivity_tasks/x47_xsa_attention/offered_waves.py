"""Finite FIFO of descriptors, never of mounted readers, sessions or host slots.

Only the owner drains a complete group. Offers are stamped under the mailbox lock at insertion,
not inferred afterwards from a scheduled arrival or from the owner's first poll.
"""

from queue import Empty, Queue
from threading import Event, Thread, get_ident, get_native_id
from time import perf_counter_ns as clock

from .arrival_service_bench import target
from .serving_verify import require

WIDTH = 8
WAVES = 4
SPACING = 60_000_000


class _Mailbox(Queue):
    high = 0

    def _put(self, item):
        row, scheduled = item
        super()._put((row, scheduled, clock()))
        self.high = max(self.high, self._qsize())

    def _get(self):
        return (*super()._get(), clock())


class DescriptorOffers:
    """One bounded owner. The sole sender knows only row numbers and deadlines."""

    def __init__(self, *, width=WIDTH, waves=WAVES, spacing_ns=SPACING):
        if (
            any(type(v) is not int for v in (width, waves, spacing_ns))
            or not 1 <= width <= 16
            or not 1 <= waves <= 32
            or width * waves > 32
            or not 0 <= spacing_ns <= 1_000_000_000
        ):
            raise ValueError("invalid bounded offer schedule")
        self.width, self.waves, self.spacing = width, waves, spacing_ns
        self.q, self.stop = _Mailbox(width * waves), Event()
        self.owner, self.origin, self.thread, self.sender = (
            get_ident(),
            None,
            None,
            None,
        )
        self.error, self.next = None, 0

    def _produce(self):
        self.sender = get_native_id()
        try:
            for g in range(self.waves):
                at = self.origin + g * self.spacing
                if self.stop.wait(max(0, at - clock()) / 1e9):
                    return
                for i in range(g * self.width, (g + 1) * self.width):
                    self.q.put_nowait((i, at))
        except BaseException as e:
            self.error = e

    def __enter__(self):
        require(
            self.owner == get_ident() and self.origin is None,
            "spent or foreign offer clock",
        )
        self.thread = Thread(target=self._produce, name="bounded-row-descriptors")
        self.origin = clock()
        self.thread.start()
        return self

    def take(self, group):
        require(
            self.owner == get_ident()
            and self.thread is not None
            and not self.stop.is_set()
            and type(group) is int
            and group == self.next < self.waves,
            "unavailable offer group",
        )
        offers = []
        for i in range(group * self.width, (group + 1) * self.width):
            try:
                j, scheduled, offered, taken = self.q.get(timeout=3)
            except Empty as e:
                raise RuntimeError("missing offered row") from (self.error or e)
            require(
                j == i and self.origin + group * self.spacing == scheduled <= offered <= taken <= clock(),
                "early, reordered or missing actual offer",
            )
            offers.append(dict(row=j, scheduled=scheduled, offered=offered, taken=taken, acks=[]))
        self.next += 1
        return offers

    def __exit__(self, typ, exc, tb):
        self.stop.set()
        self.thread.join()
        if typ is None:
            require(
                self.error is None and self.q.empty() and self.next == self.waves,
                "unfinished offered workload",
            )


def serve_group(service, offered, prompts, clients, *, progress=lambda n: None, on_lookup=None):
    """Synchronous waves: no advance after a cancel, no unoffered binding, no prebuilt next group."""
    ix = [r["row"] for r in offered]
    require(
        len(ix) == WIDTH
        and ix == list(range(ix[0], ix[0] + WIDTH))
        and ix[0] in range(0, 32, WIDTH)
        and all(r["scheduled"] <= r["offered"] <= clock() and not r["acks"] for r in offered),
        "expected eight complete unconsumed offers",
    )
    active = {}
    for r in offered:
        i = r["row"]
        r["opening"] = clock()
        t = service.admit(prompts[i], lambda i=i: clients.binding(i), limit=12, row_id=i)
        r["admitted"] = clock()
        active[t] = r
    ticks, stale = [], 0
    while active:
        ts = list(active)
        start = clock()
        replies = service.step(ts, trace=True, on_lookup=on_lookup)
        ack = clock()
        ticks.append(dict(start=start, ack=ack, rows=[r["row"] for r in active.values()]))
        retire = []
        for t, reply in zip(ts, replies, strict=True):
            r = active[t]
            i = r["row"]
            require(
                reply.ticket is t and reply.visible_before == len(prompts[i]) + len(r["acks"]),
                "wrong acknowledged owner/frontier",
            )
            clients.reply(i, reply)
            r["acks"].append(ack)
            if len(r["acks"]) == target(i):
                retire.append(t)
        if retire:
            service.release(retire)
            released = clock()
            for t in retire:
                r = active.pop(t)
                canceled = target(r["row"]) == 3
                r.update(released=released, status="canceled" if canceled else "finished")
                if canceled:
                    try:
                        service.step([t])
                    except ValueError as e:
                        require("stale" in str(e), "different ticket refusal")
                        stale += 1
                    else:
                        raise ValueError("advanced canceled client")
        progress(sum(len(r["acks"]) for r in offered))
    require(
        not service.ready and not service.noise.tables and stale == 1,
        "remaining tickets/streams",
    )
    return ticks, stale
