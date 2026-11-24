"""Bounded synchronous byte service in one trusted owner thread. No transport or tenant auth.

The caller owns the empty arena, fixed weights and admitted immutable domain. Each open_binding()
returns a context manager for a *fresh* (Session, private attachment); the service takes exclusive
ownership until release. No background descriptor work or outstanding device callbacks may use it.
A synchronous return is the delivery boundary, not an exactly-once network protocol.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from threading import get_ident
from time import perf_counter_ns

import numpy as np
import torch
from .runtime import Limits, Session
from .machine import Decision
from .direct_entry import EntryDecision
from .serving import place_inputs, serve_ready


class CapacityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, eq=False)
class Ticket:
    generation: int  # identity, not a guessable slot index or an external credential


@dataclass(frozen=True, slots=True)
class Reply:
    ticket: Ticket
    byte: int
    visible_before: int
    decision: Decision | EntryDecision | None = None  # opt-in private trace, belongs only to this caller


@dataclass(slots=True)
class _Client:
    session: Session
    attachment: object
    host: object
    resources: ExitStack
    prompt: bytes | None
    origin: int
    limit: int
    row_id: int
    sent: int = 0
    last: int | None = None


class ArrivalUniforms:
    """CPU workbench RNG. Only live session IDs, action-independent coordinates and bounded tables."""

    def __init__(self, seed):
        self.seed = seed
        self.tables = {}

    def admit(self, s, row, limit):
        if s.id in self.tables:
            raise ValueError("duplicate stream")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, row]))
        self.tables[s.id] = (s.visible, rng.random((limit, s.limits.steps, 5), dtype=np.float32))

    def drop(self, sid):
        self.tables.pop(sid, None)

    def sample_factor(self, scores, lengths, coords):
        if scores.device.type != "cpu":
            raise ValueError("CPU controller required")
        if scores.shape[1] == 1:
            return [0] * len(coords), scores[:, 0] * 0
        us = []
        for sid, vis, step, factor in coords:
            origin, table = self.tables[sid]
            i = vis - origin
            if not 0 <= i < len(table) or not 0 <= step < table.shape[1] or not 0 <= factor < 5:
                raise ValueError("outside admitted stream horizon")
            us.append(float(table[i, step, factor]))
        u = scores.new_tensor(us, dtype=torch.float32)
        lp = scores.float().log_softmax(1)
        j = (lp.detach().exp().cumsum(1) <= u[:, None]).sum(1).minimum(torch.as_tensor(lengths) - 1)
        return j.tolist(), lp.gather(1, j[:, None])[:, 0]


class ByteService:
    """Advance selected live clients once. No admission queue, socket, cross-client value cache or updates.

    In program mode provide an eval-mode CPU PackedMemoryPolicy and a causal feature selection.
    For an explicit bypass instead provide direct(attachments, base_logp) -> [B,256] log probabilities.
    A trained bypass can instead provide feature_direct(sessions, attachments, features(h), base_logp).
    Its validate_binding(session, attachment) runs before host admission; it owns placement and head checks. All deliver argmax bytes; 'sample' refers only to
    workbench instructions and operands (and the gate for entry=EntryReadout). For that predictor
    mount a fresh (session, EntryStream) and close the stream on exit; use distinct active row IDs.
    """

    def __init__(
        self,
        arena,
        *,
        policy=None,
        features=lambda h: h,
        mode="greedy",
        seed=0,
        direct=None,
        feature_direct=None,
        entry=None,
        limits=Limits(),
    ):
        if (
            not arena.empty
            or torch.is_grad_enabled()
            or mode not in ("greedy", "sample")
            or sum(x is not None for x in (policy, direct, feature_direct, entry)) != 1
            or (feature_direct is not None and mode != "greedy")
        ):
            raise ValueError("need an exclusive empty inference store and one predictor")
        if policy is not None and (
            policy.training
            or policy.embed.weight.device.type != "cpu"
            or policy.embed.weight.dtype != torch.float32
        ):
            raise ValueError("service requires fixed FP32 CPU controller")
        if type(seed) is not int or seed < 0 or not isinstance(limits, Limits):
            raise ValueError("invalid stream seed/limits")
        self.limits = limits
        self.arena = arena
        self.policy = policy
        self.direct = direct
        self.feature_direct = feature_direct
        self.entry = entry
        if entry is not None:
            entry.check()
        self.features = features
        self.mode = mode
        self.noise = ArrivalUniforms(seed)
        self._stamp = policy._stamp() if policy is not None else None
        self._thread = get_ident()
        self._clients = {}
        self._generation = 0
        self.closed = False
        self._driving = self._faulted = False
        self.work = Counter()

    def _guard(self):
        if self.closed or self._faulted or get_ident() != self._thread or torch.is_grad_enabled():
            raise ValueError("closed or foreign service owner/inference context")

    @property
    def available(self):
        return 0 if self.closed else self.arena.capacity - len(self._clients)

    @property
    def ready(self):
        return tuple(t for t, c in self._clients.items() if c.sent < c.limit)

    def admit(self, prompt, open_binding, *, limit, row_id):
        self._guard()
        if not self.available:
            raise CapacityError("active capacity exhausted")
        if (
            not isinstance(prompt, bytes)
            or not prompt
            or type(limit) is not int
            or limit < 1
            or len(prompt) + limit > self.arena.context
            or type(row_id) is not int
            or row_id < 0
            or (
                self.entry is not None
                and (row_id >= 2**64 or any(c.row_id == row_id for c in self._clients.values()))
            )
        ):
            raise ValueError("unbounded or invalid byte request")
        try:
            with ExitStack() as stack:
                s, attachment = stack.enter_context(open_binding())
                if (
                    not isinstance(s, Session)
                    or s.limits != self.limits
                    or s.visible
                    or s.pending
                    or any(x is not None for x in s.cells)
                ):
                    raise ValueError("fresh private session required")
                if self.feature_direct is not None:
                    self.feature_direct.validate_binding(s, attachment)
                if self.entry is not None:
                    self.entry.validate_binding(s, attachment)
                host = self.arena.admit(s.id)
                (s if self.entry is None else attachment).observe(prompt)
                self._generation += 1
                t = Ticket(self._generation)
                self._clients[t] = _Client(
                    s, attachment, host, stack.pop_all(), prompt, len(prompt), limit, row_id
                )
                if (self.policy is not None or self.entry is not None) and self.mode == "sample":
                    self.noise.admit(s, row_id, limit)
                self.work["admissions"] += 1
                return t
        except BaseException:
            if self._driving:
                self._faulted = True  # the driving task joins its immutable worker before shutdown
            else:
                self._shutdown()
            raise

    def _select(self, tickets):
        self._guard()
        ts = tuple(tickets)
        if not ts or len(set(ts)) != len(ts) or any(t not in self._clients for t in ts):
            raise ValueError("empty, stale, duplicate or foreign service ticket")
        return ts, [self._clients[t] for t in ts]

    def _inputs(self, cs):
        order = [None] * len(cs)
        # Snapshot membership before clearing any consumed prompt; a just-prefilled
        # admission must not also enter the continuation group in this same tick.
        groups = {
            fresh: [i for i, c in enumerate(cs) if (c.prompt is not None) == fresh] for fresh in (True, False)
        }
        for fresh, ix in groups.items():
            if not ix:
                continue
            group = [cs[i] for i in ix]
            hs = [c.host for c in group]
            if fresh:
                out = self.arena.prefill(hs, [c.prompt for c in group])
                self.work["prefill_tokens"] += sum(len(c.prompt) for c in group)
                for c in group:
                    c.prompt = None
            else:
                out = self.arena.advance(hs, [c.session.visible - 1 for c in group], [c.last for c in group])
                self.work["step_tokens"] += len(hs)
            if len(ix) == len(cs):
                return out
            for i, h, p in zip(ix, *out, strict=True):
                order[i] = (h, p)
        return tuple(torch.stack(xs) for xs in zip(*order, strict=True))

    def _policy_packet(self, h, base):
        h = self.features(h)
        dev = self.policy.embed.weight.device
        other = h.device != dev
        start = perf_counter_ns()
        h, base = place_inputs(h, base, width=self.policy.host.in_features, device=dev)
        if other:
            self.work["transfer_batches"] += 1
            self.work["transfer_bytes"] += (h.numel() + base.numel()) * 4
            self.work["blocking_transfer_ns"] += perf_counter_ns() - start
        return h, base

    def _ack(self, ts, cs, ds, lp, trace):
        if self.policy is not None and any(
            d.session_id != c.session.id
            or d.visible != c.session.visible
            or d.mode != self.mode
            or not c.session.pending
            for c, d in zip(cs, ds, strict=True)
        ):
            raise ValueError("unbound decision")
        if self.entry is not None:
            for c, d in zip(cs, ds, strict=True):
                c.attachment.check()
                if (
                    not isinstance(d, EntryDecision)
                    or (d.session_id, d.visible) != (c.session.id, c.session.visible)
                    or (d.program is None) != (d.trust is not None)
                    or not c.session.pending
                    or (
                        d.program is not None
                        and (d.program.session_id, d.program.visible, d.program.mode)
                        != (c.session.id, c.session.visible, self.mode)
                    )
                ):
                    raise ValueError("unbound direct-entry decision")
        if lp.shape != (len(cs), 256) or lp.dtype != torch.float32:
            raise ValueError("bad output packet")
        ok = lp.isfinite().all(1) & (lp.logsumexp(1).abs() < 5e-5)
        raw = torch.stack([lp.argmax(1), ok.long()], 1).tolist()
        if not all(v for _, v in raw):
            raise ValueError("invalid byte distribution")
        replies = []
        for t, c, (b, _), d in zip(ts, cs, raw, ds, strict=True):
            visible = c.session.visible
            (c.session if self.entry is None else c.attachment).observe(bytes([b]))
            c.last = b
            c.sent += 1
            replies.append(Reply(t, b, visible, d if trace else None))
        self.work["acknowledged"] += len(replies)
        self.work["batches"] += 1
        return replies

    def step(self, tickets, *, trace=False, on_lookup=None):
        if self._driving:
            raise ValueError("open driver owns the active tickets")
        ts, cs = self._select(tickets)
        if any(c.sent >= c.limit for c in cs):
            raise ValueError("exhausted response limit")
        try:
            if any(c.session.pending or c.session.visible != c.origin + c.sent for c in cs):
                raise ValueError("invalid live frontier")
            if self.policy is not None and self.policy._stamp() != self._stamp:
                raise ValueError("changed service policy")
            if self.entry is not None:
                self.entry.check()
                if on_lookup is not None:
                    raise ValueError("lookup callbacks not supported by direct-entry service")
            h, base = self._inputs(cs)
            ds = [None] * len(cs)
            if self.entry is not None:
                ds = self.entry(
                    [c.session for c in cs],
                    [c.attachment for c in cs],
                    self.features(h),
                    base,
                    row_ids=[c.row_id for c in cs],
                    mode=self.mode,
                    generator=self.noise if self.mode == "sample" else None,
                )
                lp = torch.stack([d.log_probs for d in ds])
            elif self.feature_direct is not None:
                lp = self.feature_direct(
                    [c.session for c in cs], [c.attachment for c in cs], self.features(h), base
                )
                if any(c.session.pending or c.session.visible != c.origin + c.sent for c in cs):
                    raise ValueError("direct read advanced a client")
            elif self.direct is not None:
                lp = self.direct([c.attachment for c in cs], base)
            else:
                h, base = self._policy_packet(h, base)
                # Fresh memo each call. No retired client's page/label can survive in a server-global runner.
                ds = self.policy.batch_runner()(
                    [c.session for c in cs],
                    h,
                    base,
                    mode=self.mode,
                    generator=self.noise if self.mode == "sample" else None,
                    on_lookup=on_lookup,
                )
                lp = torch.stack([d.log_probs for d in ds])
            return self._ack(ts, cs, ds, lp, trace)
        except BaseException:
            # Some host writes / native appends may already have happened; never continue an uncertain tick.
            self._shutdown()
            raise

    async def drive(self, tickets, on_batch, on_retire, *, offload=(), on_lookup=None, trace=False):
        """Drive an async iterable of admitted tickets. on_batch(replies) may return tickets to stop.

        on_retire(tickets) is notified after release. Only the owner may admit during this loop;
        close/step/release are exclusive. Normal early stop is at a delivery boundary. Canceling
        the whole drive is a fatal drain, not a promise to cancel an already issued native walk.
        """
        self._guard()
        if self._driving or self.policy is None or self.policy._stamp() != self._stamp:
            raise ValueError("open drive requires one fixed workbench owner")
        stopping = set()
        self._driving = True

        async def arrivals():
            async for t in tickets:
                _, (c,) = self._select((t,))
                yield t, c.session, c.limit - c.sent

        def inputs(ts):
            _, cs = self._select(ts)
            if self.policy._stamp() != self._stamp or any(
                c.sent >= c.limit or c.session.pending or c.session.visible != c.origin + c.sent for c in cs
            ):
                raise ValueError("invalid open host frontier")
            return self._policy_packet(*self._inputs(cs))

        def deliver(ts, ds):
            _, cs = self._select(ts)
            replies = self._ack(ts, cs, ds, torch.stack([d.log_probs for d in ds]), trace)
            done = set(on_batch(replies) or ())
            if not done.issubset(ts):
                raise ValueError("cannot retire another in-flight client")
            stopping.update(done)

        def retire(ts):
            self._release(ts)
            stopping.difference_update(ts)
            on_retire(ts)

        try:
            stats = await serve_ready(
                self.policy,
                (),
                self.arena.context,
                inputs,
                deliver,
                mode=self.mode,
                generator=self.noise if self.mode == "sample" else None,
                offload=offload,
                cohort=True,
                on_lookup=on_lookup,
                arrivals=arrivals(),
                capacity=self.arena.capacity,
                retire=retire,
                finished=stopping.__contains__,
            )
            if self._clients or self._faulted:
                raise ValueError("unserved or faulted intake")
            return stats
        except BaseException:
            self._shutdown()
            raise
        finally:
            self._driving = False

    def _discard(self, t):
        c = self._clients.pop(t)
        self.noise.drop(c.session.id)
        s = c.session
        s.cells = [None] * s.limits.cells
        s.recent = b""
        s.pending = True
        s.readers = ()
        s.advance = None
        c.prompt = c.last = c.attachment = None
        c.resources.close()
        self.work["closed_bindings"] += 1

    def release(self, tickets):
        if self._driving:
            raise ValueError("open driver owns the active tickets")
        self._release(tickets)

    def _release(self, tickets):
        ts, cs = self._select(tickets)
        try:
            self.arena.release([c.host for c in cs])
            for t in ts:
                self._discard(t)
        except BaseException:
            if self._driving:
                self._faulted = True
            else:
                self._shutdown()
            raise

    def _shutdown(self):
        self.closed = True
        with ExitStack() as cleanup:
            for t in tuple(self._clients):
                cleanup.callback(self._discard, t)
            self.arena.clear()

    def close(self):
        if self._driving:
            raise ValueError("drain the open driver before closing")
        if self.closed:
            return
        self._guard()
        self._shutdown()

    def __enter__(self):
        self._guard()
        return self

    def __exit__(self, *exc):
        self.close()
