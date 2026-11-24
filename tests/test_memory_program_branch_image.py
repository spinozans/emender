"""Tiny real descriptor/FD fixture and mocked pipes; not a claim about P100 wall or RSS."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
import zlib

import numpy as np
import pytest
from experiments.expressivity_tasks.x47_xsa_attention import (
    branch_image as bi,
    pinned_branch as pin,
    w3_data,
)


def digest(data):
    return sha256(data).hexdigest()


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    idx, text, seps, last = paths = [tmp_path / s for s in ("index", "text", "seps", "manifest")]
    text.write_bytes(b"ab\xff\x00x\x1exyz")
    members = b"".join(
        struct.pack("<IIQQQII", n, 0, 64, 0, 0, 0, 0) for n in (1, 4, 5, 7, 10, 11, 12, 13, 15, 20)
    )
    h = bytearray(64)
    h[:8] = b"SXI2\n\0\0\0"
    struct.pack_into("<Q", h, 8, 10)
    struct.pack_into("<IIQ", h, 32, 10, 464, 464)
    struct.pack_into("<I", h, 56, zlib.crc32(h + members))
    idx.write_bytes(h + members)
    seps.write_bytes(struct.pack("<4sIIIQQQQIIIQ", b"XSEP", 1, 0, 64, 3, 9, 1, 0, 0, 3, 2, 0))
    last.write_bytes(b"checked metadata")
    files = []
    for p in paths:
        s = p.stat()
        files.append(dict(path=p.name, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns))
    files[0]["directory_sha256"] = digest(idx.read_bytes())
    for rec, p in zip(files[2:], paths[2:], strict=True):
        rec["sha256"] = digest(p.read_bytes())
    conf = dict(files=files, docs=3)
    (tmp_path / "b1_mounts.json").write_text(json.dumps({"P100": conf}))
    exe = Path("/proc/self/exe").resolve()
    binary = digest(exe.read_bytes())
    (tmp_path / "binary.sha").write_text(binary + "\n")
    monkeypatch.setattr(pin, "ROOT", tmp_path)
    monkeypatch.setattr(pin, "HERE", tmp_path)
    monkeypatch.setattr(pin, "EXE", exe)
    monkeypatch.setattr(pin, "BINARY", binary)
    monkeypatch.setattr(w3_data, "EXE", exe)
    monkeypatch.setattr(bi, "OUT", tmp_path)
    row = np.zeros(1, dtype=w3_data.ROW)
    row["h"][0, 0] = 3
    header = b"W3BP0001" + struct.pack("<II", 1, 21707038)
    state = SimpleNamespace(
        actors=[], reply=header + row.tobytes(), index=idx, root=tmp_path, last=last, fds=[]
    )

    class Actor:
        def __init__(self, argv, *, pass_fds, stdin, stdout):
            assert len(pass_fds) == 2 and argv == [
                f"/proc/self/fd/{pass_fds[0]}",
                "branch-of",
                f"/proc/self/fd/{pass_fds[1]}",
                "--workspace",
            ]
            assert all(
                os.path.samestat(os.fstat(fd), p.stat()) for fd, p in zip(pass_fds, (exe, idx), strict=True)
            )
            self.pid, self.returncode = os.getpid(), None
            self.stdin, self.stdout = io.BytesIO(), io.BytesIO(state.reply)
            assert all(a.returncode == 0 for a in state.actors)
            state.actors.append(self)

        def poll(self):
            return self.returncode

        def wait(self, *, timeout):
            self.returncode = 0
            return 0

    OriginalPins = pin.FilePins

    class Pins(OriginalPins):
        def __init__(self, paths):
            super().__init__(paths)
            state.fds.extend(self.fd(i) for i in range(len(paths)))

    monkeypatch.setattr(pin, "FilePins", Pins)
    monkeypatch.setattr(w3_data.subprocess, "Popen", Actor)
    state.version = "span8-v1:" + digest(json.dumps(conf, sort_keys=True).encode())
    state.manifest = dict(mount=state.version, exe_sha=binary)
    return state


def image(t, count=2):
    return bi.P100Image(t.manifest, scope="whole", max_waves=count)


def wave(im, t, **kw):
    return im.wave("P100", t.version, scope="whole", **kw)


def loan(d, t):
    return d.borrow("P100", t.version, scope="whole")


def assert_closed(t):
    assert len(t.fds) == 5
    for fd in t.fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert all(p.returncode == 0 and p.stdin.closed and p.stdout.closed for p in t.actors)


def test_separate_worker_hello_and_cache_per_wave_immutable_metadata_only(tiny):
    t = tiny
    stale = None
    with image(t) as im:
        for i in range(2):
            with wave(im, t) as (d, meta), loan(d, t) as g:
                assert len(t.actors) == i + 1 and meta.starts.tolist() == [0, 3, 5]
                if stale is not None:
                    with pytest.raises(ValueError, match="expired"):
                        stale.read_many((b"",), 256)
                with pytest.raises(FrozenInstanceError):
                    meta.size = 4
                arr = meta.starts
                with pytest.raises(ValueError):
                    arr.flags.writeable = True
                arr.shape = (1, 3)
                assert meta.starts.shape == (3,)
                stale = g.reader()
                for _ in range(2):
                    assert g.histograms([b""])[0][0] == 3
                assert t.actors[-1].stdin.getvalue() == b"R" + struct.pack("<II", 1, 1)
            assert g.final["rows"] == 1 and t.actors[-1].returncode == 0
    assert im.reaped == im.started == 2 and im.offsets_checked == 3 and im.attestations == 1
    assert_closed(t)


def test_refuse_foreign_overlap_scope_exhaustion_and_close(tiny):
    t = tiny
    with pytest.raises(PermissionError):
        bi.P100Image(t.manifest, scope="part", max_waves=1)
    with pytest.raises(ValueError):
        image(t, 0)
    with image(t, 1) as im:
        for src, ver, scope, ex in [
            ("P100", "other", "whole", ()),
            ("X", t.version, "whole", ()),
            ("P100", t.version, "whole", (0,)),
            ("P100", t.version, "partial", ()),
        ]:
            with pytest.raises(PermissionError), im.wave(src, ver, scope=scope, excluded_docs=ex):
                pass
        assert not t.actors
        with wave(im, t) as (d, _), loan(d, t):
            with pytest.raises(ValueError), wave(im, t):
                pass
            with pytest.raises(ValueError, match="active"):
                im.close()
            with ThreadPoolExecutor(1) as ex, pytest.raises(ValueError, match="foreign"):
                ex.submit(im.close).result()
        with pytest.raises(ValueError), wave(im, t):
            pass
    with pytest.raises(ValueError), wave(im, t):
        pass
    assert len(t.actors) == 1
    assert_closed(t)


def test_start_is_not_a_descriptor_or_fd_and_can_be_spent_only_once(tiny):
    t = tiny
    permits = []

    def factory(*, launch):
        with ThreadPoolExecutor(1) as ex, pytest.raises(ValueError, match="foreign"):
            ex.submit(bi.ImagePages, launch=launch).result()
        permits.append(launch)
        p = bi.ImagePages(launch=launch)
        with pytest.raises(ValueError, match="spent"):
            bi.ImagePages(launch=launch)
        return p

    with image(t) as im:
        with wave(im, t, pages_type=factory) as (d, meta), loan(d, t):
            for not_permit in (meta, tuple(t.fds[:2])):
                with pytest.raises(ValueError, match="launch"):
                    bi.ImagePages(launch=not_permit)
        with wave(im, t) as (d, _), loan(d, t), pytest.raises(ValueError, match="expired"):
            bi.ImagePages(launch=permits[0])
    assert len(t.actors) == 2
    assert_closed(t)


def test_replace_between_waves_never_launch_second_worker(tiny):
    t = tiny
    with image(t) as im:
        with wave(im, t) as (d, _), loan(d, t):
            pass
        other = t.root / "replacement"
        other.write_bytes(t.index.read_bytes())
        other.replace(t.index)
        with pytest.raises(ValueError, match="immutable"), wave(im, t):
            pass
        assert im.closed and len(t.actors) == 1
    assert_closed(t)


def test_caught_warm_hit_fault_still_poisons_entire_image(tiny):
    t = tiny
    with image(t) as im:
        with pytest.raises(ValueError, match="unclean"), wave(im, t) as (d, _):
            with pytest.raises(ValueError, match="immutable"), loan(d, t) as g:
                g.histograms([b""])
                t.last.write_bytes(b"bad")
                g.histograms([b""])
            assert d.closed
        assert im.closed
        with pytest.raises(ValueError, match="unavailable"), wave(im, t):
            pass
    assert_closed(t)


@pytest.mark.parametrize("reply", [b"W3BP", b"W3BP0001" + struct.pack("<II", 1, 2)])
def test_bad_native_handshake_reaps_and_revokes_image(tiny, reply):
    t = tiny
    t.reply = reply
    with image(t) as im, pytest.raises(ValueError), wave(im, t):
        pass
    assert im.closed and len(t.actors) == 1
    assert_closed(t)


@pytest.mark.parametrize("corrupt", ["binary", "sidecar"])
def test_attestation_failure_closes_pins_before_worker(tiny, corrupt):
    t = tiny
    if corrupt == "binary":
        (t.root / "binary.sha").write_text("wrong\n")
    else:
        st = t.last.stat()
        t.last.write_bytes(bytes(st.st_size))
        os.utime(t.last, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ValueError, match="executable|sidecar identity"):
        image(t)
    assert not t.actors
    assert_closed(t)


def test_owner_exit_aborts_unbalanced_wave_and_grant(tiny):
    t = tiny
    with pytest.raises(ValueError, match="outstanding"), image(t) as im:
        pending_wave = wave(im, t)
        d, _ = pending_wave.__enter__()
        pending_grant = loan(d, t)
        reader = pending_grant.__enter__().reader()
    assert_closed(t)
    with pytest.raises(ValueError, match="expired"):
        reader.read_many((b"",), 256)
    with pytest.raises(ValueError):
        pending_grant.__exit__(None, None, None)
    with pytest.raises(ValueError):
        pending_wave.__exit__(None, None, None)
    assert d.closed and im.closed


def test_return_with_outstanding_grant_reaps_and_expires_it(tiny):
    t = tiny
    im = image(t)
    with pytest.raises(ValueError, match="unclean"), wave(im, t) as (d, _):
        pending = loan(d, t)
        stale = pending.__enter__().reader()
    with pytest.raises(ValueError, match="expired"):
        stale.read_many((b"",), 256)
    with pytest.raises(ValueError):
        pending.__exit__(None, None, None)
    assert im.closed and d.closed
    assert_closed(t)
