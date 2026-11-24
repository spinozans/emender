"""Real whole-archive + single visible document capabilities for the generic memory executor."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import os

from ndm.memory_program.runtime import Field, Hit, Reader, Span, Unavailable, branch_reader
from ndm.memory_program.spans import SpanValues
from .b2_worker import PositionalWorker
from .workbench_native import DEPTH, EXE, Live, check_binary

SPAN_COLUMNS = (Field("occurrence", "span"),)


class ArchiveRecords:
    """Reuse a validated immutable descriptor; this object owns its positional worker and read fd.

    Refuse an overlarge raw interval, never take its first eight. An optional, same-version
    m20 parent histogram gives an exact lower bound on the *raw* interval for quick refusal.
    """

    def __init__(self, desc, file_receipt, *, scope, encode=None, branches=None):
        if scope != "whole":
            raise PermissionError("only admitted whole archives")
        if branches is not None and (branches.source, branches.version) != (desc.name, desc.version):
            raise ValueError("different m20 mount")
        check_binary()
        self.desc = desc
        self.branches = branches
        self.stats = Counter()
        self.identity = tuple(file_receipt[k] for k in ("inode", "size", "mtime_ns"))
        self.stack = ExitStack()
        try:
            self.f = self.stack.enter_context(desc.text.open("rb", buffering=0))
            self.verify()
            self.worker = PositionalWorker(desc, EXE)
            self.stack.callback(self.worker.close)
        except BaseException:
            self.stack.close()
            raise
        self.values = SpanValues(desc.name, desc.version, self.bounds, self.read, scope=scope, encode=encode)

    def verify(self):
        stat = os.fstat(self.f.fileno())
        path = self.desc.text.stat()
        if (
            (stat.st_ino, stat.st_size, stat.st_mtime_ns) != self.identity
            or not os.path.samestat(stat, path)
            or stat.st_size != self.desc.size
        ):
            raise ValueError("span value file changed")

    def close(self):
        try:
            self.verify()
        finally:
            self.stack.close()

    def bounds(self, d):
        starts = self.desc.starts
        if not 0 <= d < len(starts):
            return None
        return 0, (int(starts[d + 1]) if d + 1 < len(starts) else self.desc.size) - int(starts[d])

    def read(self, d, a, n):
        self.verify()
        self.stats["preads"] += 1
        self.stats["bytes"] += n
        return os.pread(self.f.fileno(), n, int(self.desc.starts[d]) + a)

    def locate(self, keys, cap):
        out = [None] * len(keys)
        active = []
        self.verify()
        for i, k in enumerate(keys):
            if type(k) is not bytes or not 1 <= len(k) <= 256:
                out[i] = Unavailable("positional key width")
            else:
                active.append(i)
        small = [i for i in active if len(keys[i]) <= 9]
        if self.branches is not None and small:
            parents = tuple(dict.fromkeys(keys[i][:-1] for i in small))
            known = dict(zip(parents, self.branches.histograms(parents), strict=True))
            for i in small:
                h = known[keys[i][:-1]]
                if not isinstance(h, Unavailable) and h[keys[i][-1]] > 8:
                    out[i] = Unavailable("raw interval exceeds 8")
                    self.stats["certified_overcap"] += 1
        active = [i for i in active if out[i] is None]
        if active:
            spans, rec = self.worker.request([keys[i] for i in active])
            self.stats["worker_batches"] += 1
            self.stats["worker_keys"] += len(active)
            self.stats["validated_ns"] += rec["validated_ns"]
            for i, found, report in zip(active, spans, rec["rows"], strict=True):
                raw = report["raw"]
                lf = report["lf"]
                if type(raw) is not int or raw < 0 or any(type(v) is not int or v < 0 for v in lf):
                    raise ValueError("invalid native work record")
                self.stats["lf_steps"] += sum(lf)
                self.stats["lf_rows"] += len(lf)
                for k in ("search_ns", "locate_ns", "verify_ns", "ranks"):
                    self.stats[k] += report[k]
                if report["status"] == "overcap":
                    out[i] = Unavailable("raw interval exceeds 8")
                    self.stats["native_overcap"] += 1
                elif len(found) > cap:
                    out[i] = Unavailable("occurrence row budget")
                else:
                    out[i] = tuple(Hit((Span(s.source, s.version, s.doc, s.start, s.stop),)) for s in found)
                    self.stats["verified_spans"] += len(found)
        return out

    def readers(self):
        return (
            Reader(
                self.desc.name,
                self.desc.version,
                "occurrences",
                "text",
                SPAN_COLUMNS,
                self.locate,
                batch_limit=16,
            ),
            self.values.reader(),
            self.values.reader(following=True),
        )


class LiveRecords:
    """Exactly one open native document. Mirror only observed bytes for span values and validation."""

    def __init__(self, document, start, *, encode=None):
        self.document, self.start = document, start
        self.native = Live()
        self.data = bytearray()
        self.stats = Counter()
        self.id = self.native.session_id
        self.binding = dict(session_id=self.id, watermark=lambda: self.native.visible)
        self.values = SpanValues(
            "tape", self.id, self.bounds, self.read, scope="whole", encode=encode, **self.binding
        )

    def close(self):
        self.native.close()

    def advance(self, data):
        self.native.append(data)
        self.data.extend(data)
        self.stats["appends"] += 1
        if len(self.data) != self.native.visible:
            raise ValueError("live visibility changed")

    def bounds(self, d):
        return (self.start, self.start + len(self.data)) if d == self.document else None

    def read(self, d, a, n):
        self.stats["value_reads"] += 1
        self.stats["bytes"] += n
        return bytes(self.data[a - self.start : a - self.start + n])

    def histogram(self, k):
        if not 0 < len(k) <= DEPTH:
            return Unavailable("live branch depth")
        self.stats["branch_rpcs"] += 1
        return self.native.branch(k)

    def locate(self, keys, cap):
        rows = []
        n = len(self.data)
        limit = min(cap, 1024)  # bound the line size in the native live adapter
        for k in keys:
            if not 0 < len(k) <= DEPTH:
                rows.append(Unavailable("live locate depth"))
                continue
            if n - len(k) + 1 > limit:
                # Native overcap is a terminal error. Prove capacity before issuing such a call.
                count = sum(self.histogram(k)) + int(self.data.endswith(k))  # open EOF doesn't vote
                if count > limit:
                    rows.append(Unavailable("live occurrence row budget"))
                    continue
            self.stats["locate_rpcs"] += 1
            reply = self.native._rpc(f"locate {k.hex()} 0 {n} {limit}")
            pos = reply["positions"]
            if not (
                isinstance(pos, list)
                and len(pos) <= limit
                and all(
                    type(i) is int and 0 <= i <= n - len(k) and self.data[i : i + len(k)] == k for i in pos
                )
                and pos == sorted(set(pos))
            ):
                raise ValueError("unverified live position")
            rows.append(
                tuple(
                    Hit((Span("tape", self.id, self.document, self.start + i, self.start + i + len(k)),))
                    for i in pos
                )
            )
        return rows

    def readers(self):
        return (
            branch_reader("tape", self.id, lambda keys: [self.histogram(k) for k in keys], **self.binding),
            Reader("tape", self.id, "occurrences", "text", SPAN_COLUMNS, self.locate, **self.binding),
            self.values.reader(),
            self.values.reader(following=True),
        )
