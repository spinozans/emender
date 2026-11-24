"""Bounded, sequential reuse of one trusted immutable m20 transport. Not an ACL or a daemon.

The mount owner supplies an opener returning a context manager for (verified_pages, revalidate).
It owns the actual pins, transport startup, and final reap. Neither the transport nor a populated
page cache is lent directly to callers. No client sessions, learned values or raw key cache here.
"""

from contextlib import contextmanager, ExitStack
import os
from threading import get_ident

from .m20 import MaterializedBranches
from .runtime import branch_reader


class _Identity:
    @property
    def source(self):
        return self._identity[0]

    @property
    def version(self):
        return self._identity[1]


class BranchGrant(_Identity):
    def __init__(self, owner, bank):
        self._owner = owner
        self._bank = bank
        self._identity = bank.source, bank.version
        self._companion = None
        self._companion_spent = False
        self.final = None

    def _checked_owner(self):
        owner = self._owner
        if owner is None:
            raise ValueError("expired branch grant")
        owner._check(self)
        return owner

    def check_descriptor(self, descriptor):
        """Check an owner's companion reader binding, without returning a descriptor or performing a lookup."""
        self._checked_owner()
        if (
            descriptor is None
            or getattr(self._bank.pages, "desc", None) is not descriptor
            or (descriptor.name, descriptor.version) != self._identity
        ):
            raise ValueError("not this grant's verified descriptor")

    def _claim_companion(self, descriptor, revoke):
        self.check_descriptor(descriptor)
        if self._companion_spent or not callable(revoke):
            raise ValueError("descriptor companion unavailable")
        self._companion_spent = True
        self._companion = revoke

    def _return_companion(self, revoke):
        if self._companion == revoke:
            self._companion = None

    def histograms(self, keys):
        owner = self._checked_owner()
        try:
            answers = self._bank.histograms(keys)
            owner._check(self)
            return answers
        except BaseException:
            owner._shutdown()
            raise

    def reader(self, *, structural_end=False):
        return branch_reader(self.source, self.version, self.histograms, structural_end=structural_end)

    def _expire(self):
        b = self._bank
        if b is not None:
            self.final = dict(rows=b.pages_fetched, batches=b.page_batches, cached_before_clear=len(b.cache))
            b.cache.clear()
        self._owner = self._bank = None
        revoke, self._companion = self._companion, None
        if revoke is not None:
            revoke()


class BranchDomain(_Identity):
    def __init__(self, source, version, open_verified, *, scope, max_grants):
        if scope != "whole":
            raise PermissionError("only an admitted whole archive")
        if (
            not all(type(x) is str and x for x in (source, version))
            or type(max_grants) is not int
            or not 1 <= max_grants <= 4096
        ):
            raise ValueError("invalid branch domain binding/limit")
        self._identity = source, version
        self._who = os.getpid(), get_ident()
        self._limit = max_grants
        self.grants = 0
        self.closed = False
        self._active = None
        with ExitStack() as stack:
            self._pages, self._validate = stack.enter_context(open_verified())
            self._validate()
            self._resources = stack.pop_all()

    @property
    def idle(self):
        return not self.closed and self._active is None

    def _owner(self):
        if (os.getpid(), get_ident()) != self._who:
            raise ValueError("foreign branch owner")

    def _check(self, grant):
        self._owner()
        if self.closed or grant is not self._active:
            raise ValueError("closed/foreign branch grant")
        try:
            self._validate()
        except BaseException:
            self._shutdown()
            raise

    @contextmanager
    def borrow(self, source, version, *, scope, excluded_docs=()):
        self._owner()
        if scope != "whole" or excluded_docs or (source, version) != (self.source, self.version):
            raise PermissionError("different scope or archive identity")
        if self.closed or self._active is not None or self.grants >= self._limit:
            raise ValueError("unavailable/exhausted branch domain")
        self._check(None)
        try:
            self._active = g = BranchGrant(
                self, MaterializedBranches(source, version, self._pages, scope=scope, cache_pages=256)
            )
        except BaseException:
            self._shutdown()
            raise
        self.grants += 1
        try:
            yield g
            self._check(g)
        except BaseException:
            self._shutdown()
            raise
        finally:
            try:
                g._expire()
            except BaseException:
                self._shutdown()
                raise
            finally:
                self._active = None

    def _shutdown(self):
        if self.closed:
            return
        self.closed = True
        active, self._active = self._active, None
        try:
            if active is not None:
                active._expire()
        finally:
            self._pages = self._validate = None
            self._resources.close()

    def abort(self):
        """Owner gives up an uncertain domain; invalidate even an outstanding cached grant."""
        self._owner()
        self._shutdown()

    def close(self):
        self._owner()
        if self._active is not None:
            raise ValueError("return active branch grant first")
        if not self.closed:
            try:
                self._check(None)
            finally:
                self._shutdown()

    def __enter__(self):
        self._check(None)
        return self

    def __exit__(self, typ, exc, tb):
        self._owner()
        try:
            if typ is None:
                self.close()
        finally:
            self._shutdown()
