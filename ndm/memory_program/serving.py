"""Bounded ready scheduler for fixed cohorts or an owned live intake. One continuation per session, one main policy/device lane.

Only explicitly registered immutable, CPU-only readers can leave the main thread. An emitted
response is committed independently of other clients; an un-emitted frame never commits on error.
This is an in-process engine, not a network/tenant service. The caller owns exclusive sessions.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns

import numpy as np
import torch

from .machine import Decision, Event, Views
from .runtime import Frame, Refused, Unavailable


def place_inputs(hidden, base, *, width, device):
    """Bind one causal host packet to a controller. No candidate/row payload crosses this seam."""
    dev = torch.device(device)
    if (
        hidden.ndim != 2
        or len(hidden) == 0
        or hidden.shape[1] != width
        or base.shape != (len(hidden), 256)
        or hidden.device != base.device
        or hidden.dtype != torch.float32
        or base.dtype != torch.float32
    ):
        raise ValueError("invalid FP32 host packet")
    if hidden.device == dev:
        return hidden, base
    if hidden.device.type != "cuda" or dev.type != "cpu":
        raise ValueError("unsupported controller placement")
    return torch.cat((hidden, base), 1).to(dev).split((width, 256), 1)


class StreamUniforms:
    """Action-independent random numbers, not instructions. Stable under regrouping and parking."""

    def __init__(self, sessions, row_ids, horizon, seed, device):
        if len(sessions) != len(row_ids) or len(set(row_ids)) != len(row_ids):
            raise ValueError("duplicate stream seed")
        self.rows = {s.id: (i, s.visible) for i, s in enumerate(sessions)}
        steps = max(s.limits.steps for s in sessions)
        noise = np.stack(
            [
                np.random.default_rng(np.random.SeedSequence([seed, i])).random(
                    (horizon, steps, 5), dtype=np.float32
                )
                for i in row_ids
            ]
        )
        self.uniform = torch.from_numpy(noise).to(device)

    def sample_factor(self, scores, lengths, coords):
        if scores.shape[1] == 1:
            return [0] * len(coords), scores[:, 0] * 0
        loc = [
            (self.rows[sid][0], vis - self.rows[sid][1], step, factor) for sid, vis, step, factor in coords
        ]
        if any(not 0 <= t < self.uniform.shape[1] for _, t, _, _ in loc):
            raise ValueError("outside registered serving horizon")
        idx = torch.tensor(loc, device=scores.device).T
        u = self.uniform[idx[0], idx[1], idx[2], idx[3]]
        lp = scores.float().log_softmax(1)  # candidate layout is already masked with -inf
        j = (
            (lp.detach().exp().cumsum(1) <= u[:, None])
            .sum(1)
            .minimum(torch.as_tensor(lengths, device=u.device) - 1)
        )
        return j.tolist(), lp.gather(1, j[:, None])[:, 0]


def _resolve(reader, cap, items, background):
    # Coalesce only *chosen* keys on the same reader object and cap. Frame.execute restores duplicates.
    begin = perf_counter_ns()
    keys = tuple(dict.fromkeys(k for _, ks in items for k in ks))
    table = {}
    for pos in range(0, len(keys), reader.batch_limit):
        chunk = keys[pos : pos + reader.batch_limit]
        rows = reader.read_many(chunk, cap)
        for k, hits in zip(chunk, rows, strict=True):
            if background and not isinstance(hits, Unavailable) and any(h.value is not None for h in hits):
                raise ValueError("offloaded reader returned a numeric payload")
            table[k] = hits
    return table, begin, perf_counter_ns()


class _Queue:
    """Each actor can enqueue at most one item; a callback batches compatible ready actors."""

    def __init__(self, dispatch, offload, pool, on_io, on_lookup=None):
        self.loop = asyncio.get_running_loop()
        self.dispatch = dispatch
        self.offload = offload
        self.pool = pool
        self.on_io = on_io
        self.on_lookup = on_lookup
        self.groups = {}
        self.failed = None

    async def call(self, key, item):
        if self.failed is not None:
            raise self.failed
        fut = self.loop.create_future()
        if key not in self.groups:
            self.groups[key] = []
            self.schedule(key)
        self.groups[key].append((item, fut))
        return await fut

    def schedule(self, key):
        self.loop.call_soon(self.flush, key)

    def flush(self, key):
        jobs = [(x, f) for x, f in self.groups.pop(key) if not f.cancelled()]
        if not jobs:
            return
        items, fs = zip(*jobs)

        def fail(exc):
            self.failed = exc
            for f in fs:
                if not f.done():
                    f.set_exception(exc)

        def finish(out):
            for f, x in zip(fs, out, strict=True):
                if not f.done():
                    f.set_result(x)

        def read_done(result):
            table, start, end = result
            if self.on_lookup is not None:
                self.on_lookup(key[1], start, end, items)
            if self.on_io is not None:
                self.on_io(key[1], start, end, [i for i, _ in items])
            finish([table] * len(fs))

        def callback(task):
            try:
                read_done(task.result())
            except BaseException as e:
                fail(e)

        try:
            if self.failed is not None:
                raise self.failed
            if key[0] != "lookup":
                finish(self.dispatch(key, items))
                return
            _, reader, cap = key
            if reader in self.offload:
                self.loop.run_in_executor(self.pool, _resolve, reader, cap, items, True).add_done_callback(
                    callback
                )
            else:
                read_done(_resolve(reader, cap, items, False))
        except BaseException as e:
            fail(e)


class _CohortQueue(_Queue):
    """Pump one group, then let its resumed clients reach their next dependency.

    Finish this byte's known work before starting a new host batch. There is no
    cohort of fixed membership and no timer: fully parked clients are not in
    groups at all. Depth orders only ready compute, never instructions within
    a session. Submit chosen I/O early, and acknowledge an emit immediately.
    """

    scheduled = False

    def schedule(self, key=None):
        if not self.scheduled:
            self.scheduled = True
            self.loop.call_soon(self.pump)

    def pump(self):
        self.scheduled = False
        if not self.groups:
            return

        def priority(key):
            kind = key[0]
            if kind == "lookup" and key[1] in self.offload:
                return (0, 0)
            if kind in ("lookup", "deliver", "retire"):
                return (1, 0)
            if kind == "choose":
                return (2, key[1])
            return (3, 0)  # inputs for the next observed byte

        self.flush(min(self.groups, key=priority))
        # flush has already queued continuations on successful synchronous work.
        if self.groups:
            self.schedule()


async def serve_ready(
    policy,
    sessions,
    horizon,
    inputs,
    deliver,
    *,
    offload=(),
    mode="greedy",
    generator=None,
    replay=None,
    on_io=None,
    on_lookup=None,
    cohort=False,
    arrivals=None,
    capacity=None,
    retire=None,
    finished=None,
):
    """inputs(ids) supplies current causal [B,H],[B,256]; deliver(ids, decisions) must observe one byte.

    Both callbacks and all tensor operations run on the calling thread. Choose a single model/grad
    context for the whole call. No concurrent mutation of this model or these sessions. Small fixed
    replays with grad enabled qualify the engine; a serving application normally uses no_grad().
    Alternatively an owner supplies arrivals (async iterable of key, Session, response limit),
    capacity and retire(keys), and no initial sessions or replay. The owner controls admission;
    once delivered, finished(key) may request byte-boundary retirement. The worker is joined
    before returning, even on an exception; arrivals do not give it any private live capabilities.
    """
    n = len(sessions)
    slow = frozenset(offload)
    dynamic = arrivals is not None
    if dynamic and (
        n
        or replay is not None
        or type(capacity) is not int
        or capacity < 1
        or retire is None
        or torch.is_grad_enabled()
    ):
        raise ValueError("invalid dynamic intake/grad context")
    if not dynamic and any(x is not None for x in (capacity, retire, finished)):
        raise ValueError("retirement requires dynamic intake")
    active = {}
    if (
        (not dynamic and n < 1)
        or len({s.id for s in sessions}) != n
        or type(horizon) is not int
        or horizon < 1
        or mode not in ("greedy", "sample")
    ):
        raise ValueError("invalid service batch")
    if replay is not None and (
        len(replay) != n or any(len(plans) != horizon or any(not x for x in plans) for plans in replay)
    ):
        raise ValueError("bad service replay")
    if (
        any(s.pending for s in sessions)
        or (not dynamic and not slow.issubset({r for s in sessions for r in s.readers}))
        or any(r.watermark is not None for r in slow)
    ):
        raise ValueError("unavailable session or unsafe offload binding")
    if policy.embed.weight.dtype != torch.float32 or torch.is_autocast_enabled(
        policy.embed.weight.device.type
    ):
        raise ValueError("ready controller requires FP32")
    memo = policy._memo()
    stats = Counter()

    def dispatch(key, items):
        tag = key[0]
        stats[tag + "_batches"] += 1
        if tag == "retire":
            # The issuing byte coroutine has returned. No private scratch or in-flight
            # lookup may use the retired admission, even if a different client is parked.
            for key in items:
                del active[key]
            memo["labels"].ids.clear()
            retire(list(items))
            return [None] * len(items)
        if tag == "inputs":
            h, b = inputs(list(items))
            if (
                h.shape != (len(items), policy.host.in_features)
                or b.shape != (len(items), 256)
                or h.dtype != torch.float32
            ):
                raise ValueError("invalid service host state")
            return list(zip(h, b, strict=True))
        if tag == "deliver":
            ids, ds = zip(*items, strict=True)
            deliver(list(ids), list(ds))
            stats["emits"] += len(ids)
            return [None] * len(ids)
        _, step = key
        if policy._stamp() != memo["stamp"]:
            raise ValueError("weights or grad context changed during service")
        frames, hosts, base, gets, last, status, spec = map(list, zip(*items, strict=True))
        planner = policy.planner_type(policy, memo)
        h = planner.rebuild(frames, torch.stack(hosts), gets, last, status, step)
        choices = planner.choose_many(
            frames, h, gets, last, step, mode, spec if replay is not None else None, generator
        )
        ix = [i for i, (inst, _) in enumerate(choices) if inst.op == "EMIT"]
        emitted = {}
        if ix:
            emitted = dict(
                zip(
                    ix,
                    planner.emit_many(
                        [frames[i] for i in ix],
                        h[ix],
                        torch.stack([base[i] for i in ix]),
                        [choices[i][0] for i in ix],
                    ),
                    strict=True,
                )
            )
        return [
            (inst, lp, x, emitted.get(i)) for i, ((inst, lp), x) in enumerate(zip(choices, h, strict=True))
        ]

    async def stream(i, s, limit):
        saved = {}
        for t in range(limit):
            before = s.visible
            f = Frame(s)
            get = Views(policy)
            trace = []
            scores = []
            for p in f.cells:
                if p is not None and id(p) in saved and saved[id(p)][0] is p:
                    get.cache[id(p)] = saved[id(p)][1]
            host, base = await queue.call(("inputs",), i)
            for step in range(f.l.steps):
                spec = None
                if replay is not None:
                    if step >= len(replay[i][t]):
                        raise ValueError("service replay did not emit")
                    spec = replay[i][t][step]
                inst, lp, h, out = await queue.call(
                    ("choose", step),
                    (
                        f,
                        host,
                        base,
                        get,
                        trace[-1].inst.op if trace else None,
                        trace[-1].status if trace else "start",
                        spec,
                    ),
                )
                scores.append(lp)
                binding = None
                keys = ()
                table = None
                reg = None
                rows = 0
                status = "ok"
                if inst.op == "EMIT":
                    if s.visible != before or (replay is not None and step + 1 != len(replay[i][t])):
                        raise ValueError("stale observation or replay after emit")
                    trace.append(Event(inst, "ok", 0, None, None, ()))
                    s.cells = f.cells
                    s.pending = True
                    saved = {id(p): (p, get(p)) for p in f.cells if p is not None}
                    d = Decision(
                        out,
                        torch.stack(scores).sum(),
                        tuple(trace),
                        f.requests,
                        s.id,
                        before,
                        "replay" if replay is not None else mode,
                    )
                    await queue.call(("deliver",), (i, d))
                    if s.visible != before + 1 or s.pending:
                        raise ValueError("deliver must observe exactly one byte")
                    if finished is not None and finished(i):
                        return
                    break
                try:
                    if inst.op == "LOOKUP":
                        r = s.readers[inst.args[-1]]
                        binding = (r.source, r.version, r.operation, r.watermark() if r.watermark else None)
                        r, keys = f.request(inst.args)
                        table = await queue.call(("lookup", r, f.l.rows), (i, keys)) if keys else {}
                    if s.visible != before:
                        raise ValueError("observation changed during service")
                    p = f.execute(inst, h, policy, table, get=get)
                    reg = len(f.reg)
                    f.reg.append(p)
                    rows = len(p.keys)
                    if not rows:
                        status = "empty"
                    if getattr(policy, "record_page", None) is not None:
                        policy.record_page(s, step, reg, p)
                except Refused as e:
                    status = "refused: " + str(e)
                trace.append(Event(inst, status, rows, reg, binding, keys))
            else:
                raise ValueError("no service emit")

    async def lifetime(key, s, limit):
        await stream(key, s, limit)
        await queue.call(("retire",), key)

    # The worker owns at most one active native position call. All submitted work is drained
    # before returning to the owner of the descriptors, including on provider/task failure.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="immutable-memory-read") as pool:
        queue = (_CohortQueue if cohort else _Queue)(dispatch, slow, pool, on_io, on_lookup)
        async with asyncio.TaskGroup() as group:
            if dynamic:
                async for key, s, limit in arrivals:
                    if (
                        len(active) >= capacity
                        or key in active
                        or any(v.id == s.id for v in active.values())
                        or s.pending
                        or type(limit) is not int
                        or not 1 <= limit <= horizon
                    ):
                        raise ValueError("invalid dynamic client binding or capacity")
                    active[key] = s
                    group.create_task(lifetime(key, s, limit))
            else:
                for i, s in enumerate(sessions):
                    group.create_task(stream(i, s, horizon))
    return dict(stats)
