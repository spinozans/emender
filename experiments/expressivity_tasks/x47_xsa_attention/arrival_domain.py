"""Trusted local whole-archive serving domain. Two immutable workers per domain; one live per client."""

from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
import json
from ndm.memory_program import MaterializedBranches, Session
from ndm.memory_program.spans import mean_byte_values
from .memory_program_profile import measured
from .w3_data import HERE, Pages
from .memory_program_mounts import ArchiveRecords, LiveRecords


class ServingDomain:
    def __init__(self, manifest, model, limits, *, bank_type=MaterializedBranches, pages_factory=Pages):
        self.stack = ExitStack()
        self.closed = False
        self.limits = limits
        self.active = set()
        self.opened = 0
        self.reaped = []
        self.peak = 0
        self.provider = defaultdict(Counter)
        self.live_work = Counter()
        self.encode = lambda xs: mean_byte_values(xs, model.embed.weight)
        try:
            self.pages = pages_factory()
            self.stack.callback(self.pages.close)
            if (self.pages.exe_sha, self.pages.desc.version) != (manifest["exe_sha"], manifest["mount"]):
                raise ValueError("changed authoritatively bound P100 mount")
            self.bank = bank_type("P100", self.pages.desc.version, self.pages, scope="whole")
            conf = json.loads((HERE / "b1_mounts.json").read_text())["P100"]
            self.record = ArchiveRecords(
                self.pages.desc, conf["files"][1], scope="whole", encode=self.encode, branches=self.bank
            )
            self.stack.callback(self.record.close)
            self.shared = tuple(
                measured(r, self.provider) for r in (self.bank.reader(), *self.record.readers())
            )
        except BaseException:
            self.stack.close()
            raise

    def _session(self, row, live):
        return Session(
            self.shared + tuple(measured(r, self.provider) for r in live.readers()),
            self.limits,
            live.advance,
            document=row["doc"],
            start=row["start"],
            session_id=live.id,
        )

    @contextmanager
    def binding(self, row):
        if self.closed:
            raise ValueError("closed domain")
        self.record.verify()
        live = LiveRecords(row["doc"], row["start"], encode=self.encode)
        self.opened += 1
        pid = live.native.proc.pid
        self.active.add(pid)
        self.peak = max(self.peak, len(self.active))
        try:
            yield self._session(row, live), live
        finally:
            try:
                live.close()
            finally:
                self.live_work.update(live.stats)
                live.data.clear()
                live.native.tail = b""
                self.reaped.append((pid, live.native.proc.poll()))
                self.active.remove(pid)

    def close(self):
        if self.active:
            raise ValueError("live clients still using immutable domain")
        if not self.closed:
            self.closed = True
            self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
