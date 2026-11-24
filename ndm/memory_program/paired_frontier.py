"""Causal fixed-execution control. One native observation owner, two independent neural machines.

The owner supplies fresh co-licensed raw sessions, an authority check (None means whole owner), a
fixed selector and a separate value model. Only the selector chooses. Between observe() calls the
value machine executes this frontier's verified choices against just-issued structural results.
On error discard the episode; the caller owns and must reap its native resources. Not general RPC.
"""

from dataclasses import dataclass, replace
import os
from threading import get_ident

import torch
from .frozen_routes import FrontierChoices, digest
from .frontier_reads import FrontierReads, StageLimit
from .runtime import Session


def _state(m):
    return tuple((id(p), p._version) for p in (*m.parameters(), *m.buffers()))


@dataclass(frozen=True)
class FixedPredictions:
    # Sampled scores are detached diagnostics. Only values.log_probs go into the learner objective.
    selectors: tuple
    values: tuple
    choice_digests: tuple


class FixedFrontierPair:
    def __init__(self, selector, values, sources, *, guard, byte_limit, stage_limit=StageLimit()):
        self._who = os.getpid(), get_ident()
        self.closed = False
        self._busy = False
        if (
            not 0 < len(sources) <= 32
            or type(byte_limit) is not int
            or not 1 <= byte_limit <= 4096
            or len({s.id for s in sources}) != len(sources)
            or len({(s.document, s.start) for s in sources}) != len(sources)
            or any(
                s.visible
                or s.pending
                or s.recent
                or any(c is not None for c in s.cells)
                or getattr(s, "_pair_owner", None) is not None
                or getattr(s, "_entry_owner", None) is not None
                or s.limits.steps > 16
                or s.limits.rows > 1024
                or s.limits.columns > 32
                or s.limits.requests > 128
                or s.limits.cells > 16
                or s.limits.recent > 256
                or s.limits.pattern > 256
                for s in sources
            )
        ):
            raise ValueError("unbound or excessive paired episode")
        self._models = selector, values
        if (
            selector.host.in_features != values.host.in_features
            or type(selector) is not type(values)
            or [(k, p.shape) for k, p in selector.named_parameters()]
            != [(k, p.shape) for k, p in values.named_parameters()]
            or any(
                not m.training
                or not all(
                    p.requires_grad and p.grad is None and p.dtype == torch.float32 and p.device.type == "cpu"
                    for p in m.parameters()
                )
                for m in self._models
            )
        ):
            raise ValueError("fresh separate train-mode FP32 models required")
        storages = [
            {t.untyped_storage().data_ptr() for t in (*m.parameters(), *m.buffers()) if t.numel()}
            for m in self._models
        ]
        if storages[0] & storages[1]:
            raise ValueError("selector/value state alias")
        self._reads = FrontierReads(guard, stage_limit)
        self._source = tuple(sources)
        self._limit = byte_limit
        self._nonce = object()
        self._bindings = tuple((s.id, s.document, s.start, s.readers, s.advance, s.limits) for s in sources)
        self._views = []
        self._runs = []
        self._stamp = tuple(map(_state, self._models))
        try:
            guard(None)
            for s in sources:
                s._pair_owner = self._nonce
            for m, phase in zip(self._models, ("issue", "consume"), strict=True):
                shared = {
                    r: self._reads.view(r, phase, m.embed.weight)
                    for r in dict.fromkeys(r for s in sources for r in s.readers)
                }
                self._views.append(
                    tuple(
                        Session(
                            [shared[r] for r in s.readers],
                            s.limits,
                            session_id=s.id,
                            document=s.document,
                            start=s.start,
                        )
                        for s in sources
                    )
                )
                self._runs.append(m.batch_runner())
            self._check()
        except BaseException:
            self.close()
            raise

    def _own(self):
        if self._who != (os.getpid(), get_ident()):
            raise ValueError("foreign paired owner")

    def _check(self):
        self._own()
        if self.closed or self._busy:
            raise ValueError("closed or busy paired executor")
        if (
            not torch.is_grad_enabled()
            or torch.is_autocast_enabled("cpu")
            or any(not m.training or not all(p.requires_grad for p in m.parameters()) for m in self._models)
            or tuple(map(_state, self._models)) != self._stamp
        ):
            raise ValueError("changed paired model/precision")
        self._reads.guard(None)
        for i, s in enumerate(self._source):
            sid, d, start, rs, adv, lim = self._bindings[i]
            if (
                getattr(s, "_pair_owner", None) is not self._nonce
                or (s.id, s.document, s.start) != (sid, d, start)
                or s.readers is not rs
                or s.advance is not adv
                or s.limits is not lim
                or any(v is not None for v in s.cells)
                or any(
                    (v.visible, v.recent, v.pending) != (s.visible, s.recent, s.pending)
                    for v in (self._views[0][i], self._views[1][i])
                )
                or any(r.watermark and r.watermark() != s.visible for r in rs)
            ):
                raise ValueError("changed source or paired frontier")

    def _fault(self):
        self._busy = False
        self.close()

    def observe(self, observed):
        """One real prompt on first use, then one delivered/teacher-forced byte per completed prediction."""
        self._own()
        if self.closed or self._busy:
            raise ValueError("closed or busy paired executor")
        try:
            self._check()
            if len(observed) != len(self._source) or any(
                type(x) is not bytes
                or not x
                or s.visible + len(x) > self._limit
                or (s.visible and (not s.pending or len(x) != 1))
                for s, x in zip(self._source, observed)
            ):
                raise ValueError("invalid paired observation")
            self._busy = True
            for s, a, b, x in zip(self._source, *self._views, observed, strict=True):
                s.observe(x)
                a.observe(x)
                b.observe(x)  # only the source has an append callback
            self._busy = False
            self._check()
        except BaseException:
            self._fault()
            raise

    def predict(self, hidden, base, *, generator):
        self._own()
        if self.closed or self._busy:
            raise ValueError("closed or busy paired executor")
        try:
            self._check()
            self._busy = True
            n = len(self._source)
            if (
                generator is None
                or any(not s.visible or s.pending for s in self._source)
                or hidden.shape != (n, self._models[0].host.in_features)
                or base.shape != (n, 256)
                or any(
                    t.requires_grad or t.dtype != torch.float32 or t.device.type != "cpu"
                    for t in (hidden, base)
                )
            ):
                raise ValueError("expected one frozen causal host packet per current frontier")
            a, b = self._views
            capture = FrontierChoices(a)
            self._reads.begin()
            src = self._runs[0](a, hidden, base, mode="sample", generator=generator, observer=capture)
            choices = capture.finish()
            self._reads.handoff()
            verify = FrontierChoices(b, choices)
            vals = self._runs[1](
                b,
                hidden,
                base,
                mode="greedy",
                replay=[[c.instruction for c in row] for row in choices],
                observer=verify,
            )
            verify.finish()
            self._reads.finish()
            for raw, left, right in zip(self._source, src, vals, strict=True):
                if (
                    (left.session_id, left.visible) != (raw.id, raw.visible)
                    or (right.session_id, right.visible) != (raw.id, raw.visible)
                    or left.mode != "sample"
                    or right.mode != "replay"
                    or left.trace != right.trace
                    or left.requests != right.requests
                ):
                    raise ValueError("changed executed path")
                raw.pending = True
            self._busy = False
            self._check()
            return FixedPredictions(
                tuple(
                    replace(s, log_probs=s.log_probs.detach(), log_policy=s.log_policy.detach()) for s in src
                ),
                tuple(vals),
                tuple(digest(row).hex() for row in choices),
            )
        except BaseException:
            self._fault()
            raise

    def work(self):
        return dict(
            total=dict(self._reads.total),
            peak_per_frontier=dict(self._reads.peak),
            frames_still_staged=len(self._reads._calls),
            encoders={k: dict(v) for k, v in self._reads.encoders.items()},
        )

    def close(self):
        self._own()
        if self._busy:
            raise ValueError("cannot close a running paired frontier")
        if self.closed:
            return
        self.closed = True
        self._reads.close()
        self._runs.clear()
        for group in [*self._views, self._source]:
            for s in group:
                if s in self._source:
                    if getattr(s, "_pair_owner", None) is not self._nonce:
                        continue
                    del s._pair_owner
                s.cells = [None] * s.limits.cells
                s.recent = b""
                s.pending = True
                s.readers = ()
                s.advance = None
        self._views.clear()
        self._bindings = self._source = self._models = ()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()
