"""Retain verified P100 file metadata across bounded waves, never a worker or a query result.

Trusted Linux mount owner only. A metadata view is not permission to start a checked worker.
Each wave spends a one-use launch, performs the real native handshake, and owns one BranchDomain
with one grant. The owner opens any positional/live clients separately and closes them in-wave.
"""

from contextlib import closing, contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
from threading import get_ident

import numpy as np
from ndm.memory_program.branch_domain import BranchDomain
from . import pinned_branch as pin
from .b1_mount import ArchiveDescriptor
from .serving_verify import require
from .w3_data import OUT, Pages
from .workbench_run import sha


@dataclass(frozen=True, slots=True, eq=False)
class _Metadata:
    name: str
    version: str
    index: Path
    text: Path
    size: int
    _offsets: bytes = field(repr=False)

    @property
    def starts(self):
        # Fresh array header, immutable backing storage. A stale adapter cannot change
        # either the offsets or the next adapter's shape by modifying its NumPy view.
        return np.frombuffer(self._offsets, "<i8")


class _Start:
    def __init__(self, image):
        self._image = image
        self.spent = False

    def claim(self):
        im = self._image
        if im is None or im._launch is not self or self.spent:
            raise ValueError("expired or spent image launch")
        im._verify()
        self.spent = True
        return im._binary, im._meta, (im._pins.fd(0), im._pins.fd(1))


class ImagePages(Pages):
    def __init__(self, *, launch):
        if type(launch) is not _Start:
            raise ValueError("owned image launch required")
        self.exe_sha, self.desc, fds = launch.claim()
        self.attest_ns = 0  # once, at image creation, outside this native open
        self._start_workspace(fds)


class P100Image:
    def __init__(self, manifest, *, scope, max_waves):
        conf, version = pin.whole_p100(manifest, scope)
        if type(max_waves) is not int or not 1 <= max_waves <= 4096:
            raise ValueError("invalid image wave budget")
        self._who = os.getpid(), get_ident()
        self._limit = max_waves
        self._version = version
        self._launch = self._meta = self._domain = None
        self.checks = self.waves = self.started = self.reaped = 0
        self.attestations = 0
        self._pins = pin.FilePins([pin.EXE, *(pin.ROOT / f["path"] for f in conf["files"])])
        try:
            self._binary = sha(Path(f"/proc/self/fd/{self._pins.fd(0)}"))
            require(
                self._binary == pin.BINARY == (OUT / "binary.sha").read_text().split()[0],
                "wrong pinned executable",
            )
            desc = ArchiveDescriptor("P100", conf, pin.ROOT)
            require(desc.version == version, "wrong retained descriptor")
            self._meta = _Metadata(
                desc.name,
                desc.version,
                desc.index,
                desc.text,
                desc.size,
                desc.starts.astype("<i8", copy=False).tobytes(),
            )
            self.starts_bytes = len(self._meta._offsets)
            require(
                self.starts_bytes == conf["docs"] * 8
                and not self._meta.starts.flags.writeable
                and np.array_equal(self._meta.starts, desc.starts),
                "different boundary image",
            )
            self.offsets_checked = len(desc.starts)
            self._verify()
            self.attestations += 1
        except BaseException:
            self._drop()
            raise

    @property
    def closed(self):
        return self._pins.closed

    def _owner(self):
        if (os.getpid(), get_ident()) != self._who:
            raise ValueError("foreign image owner")

    def _verify(self):
        self._owner()
        self._pins.check()
        self.checks += 1

    def _drop(self):
        self._meta = None
        self._pins.close()

    @contextmanager
    def wave(self, source, version, *, scope, excluded_docs=(), pages_type=ImagePages):
        self._owner()
        if (source, version, scope) != ("P100", self._version, "whole") or excluded_docs:
            raise PermissionError("different image scope or mount")
        if self.closed or self._launch is not None or self.waves >= self._limit:
            raise ValueError("unavailable/exhausted image")
        self.waves += 1
        self._launch = ticket = _Start(self)

        @contextmanager
        def transport():
            with closing(pages_type(launch=ticket)) as p:
                self.started += 1
                require(
                    ticket.spent and p.desc is self._meta and p.exe_sha == self._binary,
                    "unbound image worker",
                )

                def check():
                    self._verify()
                    pin.check_worker(self._pins, p)

                yield p, check
            require(p.proc.returncode == 0, "image worker did not exit cleanly")
            self.reaped += 1

        try:
            self._verify()
            with BranchDomain(source, version, transport, scope=scope, max_grants=1) as domain:
                self._domain = domain
                yield domain, self._meta
                # A caller catching an ambiguous read inside this context cannot turn the
                # poisoned domain into a successful wave and borrow a new process.
                require(domain.idle and domain.grants == 1, "unclean or unused image wave")
            require(self.reaped == self.started == self.waves, "unfinished image wave")
            self._verify()
        except BaseException:
            self._drop()
            raise
        finally:
            ticket._image = None
            self._launch = self._domain = None

    def close(self):
        self._owner()
        if self._launch is not None:
            raise ValueError("finish active image wave first")
        if not self.closed:
            try:
                self._verify()
            finally:
                self._drop()

    def __enter__(self):
        self._owner()
        try:
            self._verify()
        except BaseException:
            self._drop()
            raise
        return self

    def __exit__(self, typ, exc, tb):
        self._owner()
        if self._launch is not None:
            # Unbalanced manual context entry must not leave its pipes alive after the pins.
            try:
                if self._domain is not None:
                    self._domain.abort()
            finally:
                self._launch._image = None
                self._drop()
            if typ is None:
                raise ValueError("outstanding image wave at owner exit")
        elif typ is not None:
            self._drop()
        else:
            self.close()
