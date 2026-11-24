"""Trained direct readout for a trusted single-owner ByteService; no workbench or future targets."""

from collections import Counter
from threading import get_ident
from time import perf_counter_ns

import torch

from .direct import DirectTrust, DocumentCounts, direct_branches, direct_features
from .runtime import Refused, Session
from .serving import place_inputs


class DirectStream:
    """Private follower counts for one admission, even for two requests at the same document origin."""

    __slots__ = ("_counts", "_session")

    def __init__(self, document, start, *, limits, byte_limit):
        self._counts = DocumentCounts(document, start, byte_limit=byte_limit)
        self._session = Session(limits=limits, document=document, start=start, advance=self._append)

    @property
    def session(self):
        return self._session

    @property
    def counts(self):
        return self._counts

    def check(self, session):
        d = self._counts
        if (
            session is not self._session
            or d.closed
            or session.pending
            or session.advance != self._append
            or (session.document, session.start, session.visible) != (d.document, d.start, d.visible)
        ):
            raise ValueError("unbound or stale direct attachment")

    def _append(self, raw):
        self.check(self._session)
        d = self._counts
        if d.visible + len(raw) > d.byte_limit:
            raise Refused("direct document byte budget")
        for b in raw:
            d.observe(b, at=d.visible)

    def view(self, session):
        self.check(session)
        return self._counts.view(at=session.visible)

    def close(self):
        self._counts.close()
        self._session.recent = b""
        self._session.pending = True
        self._session.advance = None


class DirectReadout:
    """ByteService feature_direct adapter. Caller owns the verified immutable whole-bank capability.

    An optional trusted diagnostic may observe evidence *after* scoring. The readout itself
    retains neither the client stream nor its query history and never consults reference answers.
    """

    def __init__(self, head, bank, *, observe=None):
        if (
            not isinstance(head, DirectTrust)
            or head.training
            or any(
                p.requires_grad or p.device.type != "cpu" or p.dtype != torch.float32
                for p in head.parameters()
            )
        ):
            raise ValueError("frozen CPU FP32 direct head required")
        self.head = head
        self.bank = bank
        self.observe = observe
        self._owner = get_ident()
        self._version = self._stamp()
        self.work = Counter()

    def _stamp(self):
        return (
            id(self.bank),
            self.bank.source,
            self.bank.version,
            id(getattr(self.bank, "pages", self.bank)),
            self.head.host_dim,
            self.head.training,
            tuple((id(p), p._version, p.requires_grad, p.device, p.dtype) for p in self.head.parameters()),
        )

    def _guard(self):
        if get_ident() != self._owner or torch.is_grad_enabled() or self._stamp() != self._version:
            raise ValueError("changed direct owner/head/bank or not in inference")

    def validate_binding(self, session, attachment):
        self._guard()
        if not isinstance(attachment, DirectStream):
            raise ValueError("private direct stream required")
        attachment.check(session)

    def __call__(self, sessions, attachments, h, base):
        self._guard()
        if (
            not sessions
            or len(attachments) != len(sessions)
            or len(sessions) != len(h)
            or len({s.id for s in sessions}) != len(sessions)
        ):
            raise ValueError("invalid direct feature batch")
        views = [a.view(s) for a, s in zip(attachments, sessions, strict=True)]
        other = h.device.type != "cpu"
        start = perf_counter_ns()
        h, base = place_inputs(h, base, width=self.head.host_dim, device="cpu")
        if other:
            self.work.update(
                transfer_batches=1,
                transfer_bytes=(h.numel() + base.numel()) * 4,
                blocking_transfer_ns=perf_counter_ns() - start,
            )
        branches = direct_branches(self.bank, views, grouped=True, work=self.work)
        lp, weights = self.head(h, base, direct_features(views, branches))
        self.work.update(trust_calls=1, predictions=len(views))
        if self.observe is not None:
            for s, v, a, l, w in zip(sessions, views, branches, lp, weights, strict=True):
                self.observe(s, v, a, l, w)
        return lp
