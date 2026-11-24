"""Linux implementation of a trusted, whole-P100 immutable worker domain. No client cache here."""

from contextlib import closing, contextmanager, ExitStack
from hashlib import sha256
import json
import os
from stat import S_ISREG

from ndm.memory_program.branch_domain import BranchDomain
from .b1_mount import HERE, ROOT
from .serving_verify import require
from .w3_data import EXE, Pages

BINARY = "12af5d707155e9301f1c092e7b75bad224b0b36dff2e6befd4789d33e8a9ec40"


def stamp(s):
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class FilePins:
    """Pins detect replacement and ordinary in-place change, not malicious kernel/owner interference."""

    def __init__(self, paths):
        self._entries = []
        self._stack = ExitStack()
        self.closed = False
        try:
            for path in paths:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
                self._stack.callback(os.close, fd)
                s = os.fstat(fd)
                require(S_ISREG(s.st_mode), "branch pin requires a regular file")
                self._entries.append((path, fd, stamp(s)))
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        if self.closed:
            raise ValueError("closed branch pins")
        for path, fd, expected in self._entries:
            require(
                stamp(os.stat(path, follow_symlinks=False)) == expected == stamp(os.fstat(fd)),
                "immutable branch file changed",
            )

    def fd(self, index):
        if self.closed:
            raise ValueError("closed branch pins")
        return self._entries[index][1]

    def close(self):
        self.closed = True
        self._entries.clear()
        self._stack.close()


def whole_p100(manifest, scope):
    if scope != "whole":
        raise PermissionError("partial P100 mount unavailable")
    conf = json.loads((HERE / "b1_mounts.json").read_text())["P100"]
    version = "span8-v1:" + sha256(json.dumps(conf, sort_keys=True).encode()).hexdigest()
    require((manifest["mount"], manifest["exe_sha"]) == (version, BINARY), "unbound P100 manifest")
    return conf, version


def check_worker(pins, p):
    require(
        p.proc.poll() is None
        and os.path.samestat(os.fstat(pins.fd(0)), os.stat(f"/proc/{p.proc.pid}/exe")),
        "wrong or dead pinned worker",
    )


@contextmanager
def p100_domain(manifest, *, scope, max_grants, pages_type=Pages):
    """Domain owner also gets the verified descriptor for opening *separate* positional readers.

    A predictor only sees domain.borrow(...). The owner must close any separate fd/worker and
    all private sessions before returning a grant. No model-dependent encoded values here.
    """
    conf, version = whole_p100(manifest, scope)
    opened = []

    @contextmanager
    def transport():
        paths = [EXE, *(ROOT / r["path"] for r in conf["files"])]
        with (
            closing(FilePins(paths)) as pins,
            closing(pages_type(pinned_fds=(pins.fd(0), pins.fd(1)))) as p,
        ):
            require(
                p.exe_sha == BINARY and p.desc.version == version and p.desc.name == "P100",
                "wrong native mount",
            )
            p.desc.starts.flags.writeable = False

            def verify():
                pins.check()
                check_worker(pins, p)

            opened.append(p.desc)
            try:
                yield p, verify
            finally:
                opened.clear()

    with BranchDomain("P100", version, transport, scope=scope, max_grants=max_grants) as domain:
        yield domain, opened[0]
