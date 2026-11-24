"""One bounded, open document. Defer its *native index*, never its logical observations.

Both modes have the same private mirror and live-reader identities. No shared tape, archive,
background work, or cross-client answer cache. All reads and releases belong to the owner.
"""

from dataclasses import replace
import os
from pathlib import Path
from threading import get_ident
from time import perf_counter_ns
from uuid import uuid4

from ndm.memory_program.runtime import Refused
from .memory_program_mounts import LiveRecords
from .workbench_native import Live, check_binary


class PrivateTape(LiveRecords):
    def __init__(self, document, start, *, encode=None, byte_limit, demand):
        if (
            type(byte_limit) is not int
            or not 1 <= byte_limit <= 4096
            or type(demand) is not bool
            or any(type(i) is not int or i < 0 for i in (document, start))
        ):
            raise ValueError("invalid private tape admission")
        check_binary()
        self._owner = os.getpid(), get_ident()
        self._limit, self._demand = byte_limit, demand
        self._applied = 0
        self.native = None
        self.closed = False
        self.exit = None
        self._bind(document, start, uuid4().hex, self.watermark, encode)
        if not demand:
            try:
                self._open()
            except BaseException:
                self.close()
                raise

    def _open(self):
        if self.native is not None or self._applied:
            raise ValueError("second private native process")
        t = perf_counter_ns()
        self.native = Live()
        self.stats.update(native_starts=1, open_ns=perf_counter_ns() - t)

    def watermark(self):
        if self.closed or (os.getpid(), get_ident()) != self._owner:
            raise ValueError("closed or foreign private tape")
        n = len(self.data)
        if (
            n != self.stats["observed_bytes"]
            or not 0 <= self._applied <= n <= self._limit
            or (self.native is None and self._applied)
            or (
                self.native is not None
                and (self.native.visible != self._applied or self.native.proc.poll() is not None)
            )
        ):
            self.close()
            raise ValueError("ambiguous private tape frontier")
        return n

    def _sync(self):
        n = self.watermark()
        if self.native is None:
            self._open()
        if n != self._applied:
            missing = bytes(self.data[self._applied : n])
            t = perf_counter_ns()
            self.native.append(missing)  # private pipe; actual acknowledged append, never a future suffix
            if self.native.visible != n or self.native.tail != self.data[max(0, n - 8) :]:
                raise ValueError("unacknowledged private prefix")
            self._applied = n
            self.stats.update(native_appends=1, native_bytes=len(missing), append_ns=perf_counter_ns() - t)
        return self.native

    def _query_native(self):
        # Called only inside a selected, depth-checked histogram or locate operation.
        return self._sync()

    def advance(self, data):
        n = self.watermark()
        if type(data) is not bytes or not data:
            raise ValueError("nonempty observed bytes required")
        if n + len(data) > self._limit:
            raise Refused("private tape byte limit")
        try:
            self.data.extend(data)
            self.stats.update(observations=1, observed_bytes=len(data))
            if not self._demand:
                self._sync()
        except BaseException:
            self.close()
            raise

    def _read_checked(self, fun, *args):
        self.watermark()
        try:
            return fun(*args)
        except BaseException:
            self.close()
            raise

    def histogram(self, k):
        return self._read_checked(super().histogram, k)

    def locate(self, keys, cap):
        return self._read_checked(super().locate, keys, cap)

    def bounds(self, d):
        self.watermark()
        return super().bounds(d)

    def read(self, d, a, n):
        end = self.start + self.watermark()
        if d != self.document or not self.start <= a < a + n <= end:
            raise ValueError("outside private observed document")
        return super().read(d, a, n)

    def readers(self):
        self.watermark()

        def protect(read):
            # Check even a foreign span or an otherwise cacheable miss after release.
            return lambda keys, cap: self._read_checked(read, keys, cap)

        return tuple(replace(r, read_many=protect(r.read_many)) for r in super().readers())

    def close(self):
        if (os.getpid(), get_ident()) != self._owner:
            raise ValueError("foreign private tape owner")
        if self.closed:
            return
        self.closed = True
        self.stats["unsynced_at_close"] = len(self.data) - self._applied
        self.data.clear()  # do NOT flush a suffix nobody asked to query
        live, self.native = self.native, None
        if live is not None:
            p = live.proc
            hwm = 0
            try:
                if p.poll() is None:
                    status = Path(f"/proc/{p.pid}/status").read_text().splitlines()
                    hwm = int(next(line.split()[1] for line in status if line.startswith("VmHWM:")))
            finally:
                try:
                    live.close()
                finally:
                    live.tail = b""
                    self.exit = dict(pid=p.pid, rc=p.poll(), peak_kib=hwm)
            if p.returncode != 0:
                raise RuntimeError("unclean private native exit")
