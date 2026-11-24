from contextlib import closing, contextmanager
from concurrent.futures import ThreadPoolExecutor
import os
import time

import numpy as np
import pytest

from ndm.memory_program.branch_domain import BranchDomain
from experiments.expressivity_tasks.x47_xsa_attention.pinned_branch import FilePins


class Transport:
    root, nodes = 1, 2

    def __init__(self):
        self.calls = []
        self.closed = 0
        self.verified = 0
        self.live = True
        self.bad = False

    def rows(self, ids):
        self.calls.append(tuple(ids))
        if self.bad:
            raise RuntimeError("ambiguous row")
        out = []
        for i in ids:
            h = np.zeros(257, np.uint64)
            child = np.zeros(256, np.uint32)
            if i == 1:
                h[0] = 3
                child[0] = 2
            else:
                h[255] = 2
                h[256] = 1
            out.append(dict(h=h, child=child))
        return out

    @contextmanager
    def open(self):
        def check():
            self.verified += 1
            if not self.live:
                raise ValueError("dead native")

        try:
            yield self, check
        finally:
            self.closed += 1


def test_retired_reader_cannot_use_cached_page_or_follow_next_grant():
    p = Transport()
    with BranchDomain("A", "V", p.open, scope="whole", max_grants=2) as domain:
        old = None
        for j in range(2):
            with domain.borrow("A", "V", scope="whole") as g:
                if old is not None:
                    with pytest.raises(ValueError, match="expired"):
                        old.read_many((b"\0",), 256)
                r = g.reader()
                assert r.read_many((b"\0",), 256)[0][0].keys == (b"\xff", b"\x00\xff")
                before = p.verified
                r.read_many((b"\0",), 256)
                assert p.verified >= before + 2 and len(p.calls) == 2 * (j + 1)
                old = r
            assert g.final == dict(rows=2, batches=2, cached_before_clear=2) and g._bank is None
        with pytest.raises(ValueError, match="exhausted"), domain.borrow("A", "V", scope="whole"):
            pass
    assert p.closed == 1 and p.calls == [(1,), (2,)] * 2
    with pytest.raises(ValueError):
        old.read_many((b"\0",), 256)


def test_grants_bound_to_one_owner_scope_and_nonoverlapping_lifetime():
    p = Transport()
    with pytest.raises(PermissionError):
        BranchDomain("A", "V", lambda: pytest.fail("opened"), scope="partial", max_grants=1)
    with BranchDomain("A", "V", p.open, scope="whole", max_grants=1) as d:
        for a, v, scope, ex in [
            ("B", "V", "whole", ()),
            ("A", "X", "whole", ()),
            ("A", "V", "part", ()),
            ("A", "V", "whole", (0,)),
        ]:
            with pytest.raises(PermissionError), d.borrow(a, v, scope=scope, excluded_docs=ex):
                pass
        with d.borrow("A", "V", scope="whole") as g:
            with pytest.raises(ValueError), d.borrow("A", "V", scope="whole"):
                pass
            with pytest.raises(ValueError):
                d.close()
            with ThreadPoolExecutor(1) as worker, pytest.raises(ValueError, match="foreign"):
                worker.submit(g.histograms, [b""]).result()
            assert not p.calls and g.histograms([b""])[0][0] == 3


@pytest.mark.parametrize("fault", ["dead", "bad_frame"])
def test_cached_hit_revalidates_and_fault_revokes_whole_transport(fault):
    p = Transport()
    d = BranchDomain("A", "V", p.open, scope="whole", max_grants=3)
    with pytest.raises((ValueError, RuntimeError)), d.borrow("A", "V", scope="whole") as g:
        r = g.reader()
        r.read_many((b"",), 256)
        assert p.calls == [(1,)]
        if fault == "dead":
            p.live = False
            r.read_many((b"",), 256)
        else:
            p.bad = True
            r.read_many((b"\0",), 256)
    assert p.closed == 1 and d.closed and g._bank is None
    with pytest.raises(ValueError):
        r.read_many((b"",), 256)
    with pytest.raises(ValueError), d.borrow("A", "V", scope="whole"):
        pass
    d.close()
    assert p.closed == 1


@pytest.mark.parametrize("fault", ["replace", "in_place"])
def test_file_pin_rejects_same_size_same_mtime_drift_and_releases_fd(tmp_path, fault):
    path = tmp_path / "file"
    path.write_bytes(b"abcd")
    st = path.stat()
    with closing(FilePins([path])) as pins:
        fd = pins.fd(0)
        assert os.pread(fd, 4, 0) == b"abcd"
        if fault == "replace":
            other = tmp_path / "other"
            other.write_bytes(b"ABCD")
            other.replace(path)
        else:
            # Separate the ctime ticks on Linux filesystems with a coarse current_time().
            time.sleep(0.025)
            path.write_bytes(b"ABCD")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert path.stat().st_size == 4
        with pytest.raises(ValueError, match="immutable"):
            pins.check()
    with pytest.raises(OSError):
        os.fstat(fd)
    with pytest.raises(ValueError):
        pins.check()


def test_open_failure_reaps_transport_without_lending_any_grant():
    p = Transport()
    p.live = False
    with pytest.raises(ValueError, match="dead native"):
        BranchDomain("A", "V", p.open, scope="whole", max_grants=1)
    assert p.closed == 1 and not p.calls
