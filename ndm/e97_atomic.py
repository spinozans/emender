"""Small durable no-replace publication primitives for immutable receipts."""
from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import stat
import tempfile

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def publish_directory_no_replace(stage: Path, destination: Path) -> None:
    """Atomically publish a fully-fsynced authority directory without replacement.

    Linux ``renameat2(RENAME_NOREPLACE)`` is the authority boundary.  There is
    intentionally no destination existence precheck: unsupported kernels or
    filesystems fail closed rather than degrading to replace semantics.
    """

    if os.name != "posix" or not hasattr(ctypes, "CDLL"):
        raise RuntimeError("directory no-replace publication requires Linux renameat2")
    try:
        stage_stat = os.lstat(stage)
    except FileNotFoundError as exc:
        raise ValueError("directory publication stage is missing") from exc
    if not stat.S_ISDIR(stage_stat.st_mode) or stat.S_ISLNK(stage_stat.st_mode):
        raise ValueError("directory publication stage is not a directory")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("directory no-replace publication requires libc renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD, os.fsencode(stage), _AT_FDCWD, os.fsencode(destination), _RENAME_NOREPLACE,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError("immutable directory publication conflicts with existing authority")
        if code in {errno.ENOSYS, errno.EINVAL, errno.EXDEV, errno.EOPNOTSUPP}:
            raise RuntimeError("directory no-replace publication is unsupported")
        raise OSError(code, os.strerror(code), str(destination))
    fsync_directory(destination.parent)


def fsync_directory(path: Path) -> None:
    """Durably persist directory-entry changes on Linux filesystems."""

    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_bytes_no_replace(path: Path, payload: bytes, *, mode: int = 0o600) -> bool:
    """Atomically publish immutable bytes.

    A hard link supplies no-replace semantics.  An identical retry is accepted;
    a pre-existing different payload is an authority conflict.  Returns whether
    this call created the target.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if not stat.S_ISREG(existing.st_mode) or stat.S_ISLNK(existing.st_mode):
            raise ValueError("immutable publication target is not a regular file")
        if path.read_bytes() != payload:
            raise ValueError("immutable publication conflicts with existing bytes")
        return False

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            created = True
        except FileExistsError:
            existing = os.lstat(path)
            if not stat.S_ISREG(existing.st_mode) or stat.S_ISLNK(existing.st_mode):
                raise ValueError("immutable publication target is not a regular file")
            if path.read_bytes() != payload:
                raise ValueError("immutable publication conflicts with existing bytes")
            created = False
        fsync_directory(path.parent)
        return created
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
