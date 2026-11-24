"""Revocable, synchronous positional/value capabilities subordinate to a whole-archive branch grant.

The owner must exit this scope before returning its branch grant. Reader metadata and a small
expired token may escape, not the callback environment, document starts or value encoder. This
is a trusted local lifetime discipline, not an ACL against code introspecting private objects.
"""

from dataclasses import replace
from functools import partial
import os
from pathlib import Path
from threading import get_ident
from time import perf_counter_ns as clock

from ndm.memory_program.branch_domain import BranchGrant
from ndm.memory_program.runtime import Hit, Unavailable
from .b2_worker import PositionalWorker
from .memory_program_mounts import ArchiveRecords
from .pinned_branch import FilePins
from .workbench_native import EXE, EXE_SHA
from .workbench_run import sha


class _GrantedArchive(ArchiveRecords):
    """The scope, not a bare descriptor, may start the native locator on this adapter."""

    def _prepare_position(self):
        self.worker = None

    def _positions(self, unresolved):
        return self.issue(unresolved)


class ScopedRecords:
    """Eager by default. defer_position delays only the native process, never mount verification.

    pid/start_ns stay None until a real workspace handshakes. birth timestamps distinguish
    admission from a chosen unresolved occurrence read. on_native, if provided, is a trusted
    synchronous meter of that real call (chosen keys, cap, unresolved keys, result and three
    monotonic timestamps); it must not modify or retain the raw data. Do not hand it to clients.
    """

    def __init__(
        self,
        desc,
        file_receipt,
        *,
        scope,
        branches,
        encode=None,
        excluded_docs=(),
        defer_position=False,
        on_native=None,
    ):
        if type(defer_position) is not bool or (on_native is not None and not callable(on_native)):
            raise ValueError("invalid synchronous locator option")
        if scope != "whole" or excluded_docs:
            raise PermissionError("only admitted whole-record scope")
        if not isinstance(branches, BranchGrant):
            raise ValueError("active branch grant required")
        self._owner = os.getpid(), get_ident()
        self._parent = branches
        self._record = self._pins = None
        self._ops = ()
        self._closed = False
        self.final = self.birth = self.pid = self.start_ns = None
        self._active_call = None
        self._on_native = on_native  # trusted synchronous post-call diagnostic; no stored key list
        branches._claim_companion(desc, self._revoke)  # one use; no file or process before admission
        try:
            self._pins = FilePins([EXE, desc.index])
            if sha(Path(f"/proc/self/fd/{self._pins.fd(0)}")) != EXE_SHA:
                raise ValueError("wrong positional executable")
            self._record = _GrantedArchive(desc, file_receipt, scope=scope, branches=branches, encode=encode)
            self._record.issue = self._issue
            self._ops = self._record.readers()
            self._validate()
            if not defer_position:
                self._start("admission")
        except BaseException:
            self._revoke()
            raise

    @property
    def closed(self):
        return self._closed

    def _own(self):
        if self._owner != (os.getpid(), get_ident()):
            raise ValueError("foreign scoped-record owner")

    def _validate(self):
        # Only reached under the owner. Validate even empty, out-of-scope or EOF-only
        # key lists: they cannot be used to observe a retired capability.
        if self.closed:
            raise ValueError("retired record reader")
        rec = self._record
        self._parent.check_descriptor(rec.desc)
        rec.verify()
        self._pins.check()
        if rec.worker is None:
            if self.birth is not None:
                raise ValueError("ambiguous positional worker birth")
        elif rec.worker.proc.poll() is not None or not os.path.samestat(
            os.fstat(self._pins.fd(0)), os.stat(f"/proc/{self.pid}/exe")
        ):
            raise ValueError("dead or changed positional worker")

    def _start(self, reason):
        self._validate()  # including index/exec pins and exact parent descriptor, even on a late start
        if self.birth is not None:
            raise ValueError("spent positional launch")
        rec = self._record
        self.birth = dict(reason=reason, start=clock(), ready=None)
        w = PositionalWorker(rec.desc, EXE, pinned_fds=(self._pins.fd(0), self._pins.fd(1), rec.f.fileno()))
        rec.worker = w
        rec.stack.callback(w.close)
        self.pid, self.start_ns = w.proc.pid, w.start_ns
        if (
            set(w.ready) != {"ready", "index_ns", "verifier_ns", "ctx_ns", "arena_bytes"}
            or w.ready["ready"] != 1
            or w.ready["arena_bytes"] <= 0
            or any(type(v) is not int or v < 0 for v in w.ready.values())
        ):
            raise ValueError("incomplete positional workspace handshake")
        self._validate()
        self.birth.update(ready=clock(), handshake_bytes=w.ready_bytes, arena_bytes=w.ready["arena_bytes"])

    def _issue(self, unresolved):
        self._own()
        if self._active_call is None or self._active_call[0] != 0 or not 1 <= len(unresolved) <= 16:
            raise ValueError("no chosen positional call")
        _, keys, cap = self._active_call
        # The raw frame must be an ordered subset of the actual selected keys, after certification.
        remaining = iter(keys)
        if not all(type(k) is bytes and 1 <= len(k) <= 256 and k in remaining for k in unresolved):
            raise ValueError("different unresolved keys")
        before = clock()
        if self._record.worker is None:
            self._start("unresolved occurrences")
        begin = clock()
        result = self._record.worker.request(unresolved)
        end = clock()
        if self._on_native is not None:
            self._on_native(keys, cap, unresolved, result, before, begin, end)
        return result

    def _check(self):
        self._own()
        if self.closed:
            raise ValueError("retired record reader")
        try:
            self._validate()
        except BaseException:
            self._revoke()
            raise

    def _dispatch(self, i, keys, cap):
        self._check()
        op = self._ops[i]
        if type(cap) is not int or cap < 0 or len(keys) > op.batch_limit:
            raise ValueError("invalid scoped-record batch")
        if self._active_call is not None:
            raise ValueError("reentrant scoped-record dispatch")
        try:
            self._active_call = (i, tuple(keys), cap)
            answer = op.read_many(self._active_call[1], cap)
            if len(answer) != len(keys) or any(
                not isinstance(row, Unavailable)
                and (
                    not isinstance(row, (tuple, list))
                    or len(row) > cap
                    or any(not isinstance(h, Hit) for h in row)
                )
                for row in answer
            ):
                raise ValueError("incomplete scoped-record reply")
            self._validate()
            return answer
        except BaseException:
            self._revoke()
            raise
        finally:
            self._active_call = None

    def check(self):
        """Validate this same synchronous admission without issuing a native request."""
        self._check()

    def readers(self):
        self._check()
        # Do not close over `op` or `op.read_many`: the returned Readers must not
        # keep a raw callback if held past retirement.
        return tuple(replace(op, read_many=partial(self._dispatch, i)) for i, op in enumerate(self._ops))

    def _revoke(self):
        if self.closed:
            return
        self._closed = True
        parent, self._parent = self._parent, None
        parent._return_companion(self._revoke)
        self._ops = ()
        rec, self._record = self._record, None
        self._active_call = self._on_native = None
        pins, self._pins = self._pins, None
        try:
            if rec is not None:
                rec.issue = None
                try:
                    rec._retire_owned()
                finally:
                    self.final = dict(
                        pid=self.pid,
                        rc=rec.worker.proc.poll() if rec.worker is not None else None,
                        fd_closed=rec.f.closed,
                        start_ns=self.start_ns,
                        birth=self.birth,
                        work=dict(rec.stats),
                    )
                if (rec.worker is not None and self.final["rc"] != 0) or not self.final["fd_closed"]:
                    raise ValueError("unclean positional retirement")
        finally:
            if pins is not None:
                pins.close()

    def close(self):
        self._own()
        if not self.closed:
            try:
                self._validate()
            finally:
                self._revoke()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, typ, exc, tb):
        self._own()
        if typ is None:
            self.close()
        else:
            self._revoke()
