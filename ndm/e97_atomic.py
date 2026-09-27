"""Small durable no-replace publication primitives for immutable receipts."""
from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
# A pre-existing FIFO must not block an immutable publisher before its type is
# rejected.  O_NONBLOCK has no effect on regular files.
_FILE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _relative_parts(path: Path, *, name: str) -> tuple[str, tuple[str, ...]]:
    """Split a path without resolving or following any component."""

    parts = path.parts
    if not parts:
        raise ValueError(f"{name} path is invalid")
    root = "/" if path.is_absolute() else "."
    names = tuple(parts[1:] if path.is_absolute() else parts)
    if not names or any(part in {"", ".", ".."} for part in names):
        raise ValueError(f"{name} path is invalid")
    return root, names


def open_directory_no_follow(path: Path | str) -> int:
    """Open a directory through component-relative ``O_NOFOLLOW`` traversals.

    The caller owns the returned descriptor.  This intentionally never uses
    ``Path.resolve``/``stat`` so a checked component cannot be exchanged before
    it is opened.
    """

    candidate = Path(path)
    if str(candidate) in {".", "/"}:
        return os.open(str(candidate), _DIRECTORY_FLAGS)
    root, names = _relative_parts(candidate, name="directory")
    descriptors: list[int] = []
    success = False
    try:
        current = os.open(root, _DIRECTORY_FLAGS)
        descriptors.append(current)
        for part in names:
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            descriptors.append(next_fd)
            current = next_fd
        for descriptor in reversed(descriptors[:-1]):
            os.close(descriptor)
        success = True
        return current
    except FileNotFoundError as exc:
        raise ValueError("directory is missing") from exc
    except OSError as exc:
        raise ValueError("directory cannot be opened safely") from exc
    finally:
        # On success all ancestors were closed above and only ``current`` is
        # retained.  On failure every acquired descriptor remains ours.
        if not success:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def durable_directory(path: Path | str, *, mode: int = 0o700) -> None:
    """Create a directory chain without links and persist every component entry.

    The fsync is deliberately unconditional after a child descriptor is opened.
    Another publisher may have created that child but not yet flushed its parent
    directory entry; returning from this publisher must still make that entry
    durable before any authority below it can be reported durable.
    """

    candidate = Path(path)
    if str(candidate) in {".", "/"}:
        return
    root, names = _relative_parts(candidate, name="directory")
    descriptors: list[int] = []
    try:
        current = os.open(root, _DIRECTORY_FLAGS)
        descriptors.append(current)
        for part in names:
            try:
                next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=mode, dir_fd=current)
                except FileExistsError:
                    pass
                next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            # ``next_fd`` proves the component is a directory reached without
            # following a link.  Flush its *containing* directory whether this
            # process or a concurrent publisher created the entry.
            os.fsync(current)
            descriptors.append(next_fd)
            current = next_fd
    except OSError as exc:
        raise ValueError("directory cannot be created safely") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _open_parent_no_follow(path: Path) -> tuple[int, str]:
    """Durably create then open the parent directory and return final basename."""

    root, names = _relative_parts(path, name="publication")
    final = names[-1]
    parent_names = names[:-1]
    descriptors: list[int] = []
    success = False
    try:
        current = os.open(root, _DIRECTORY_FLAGS)
        descriptors.append(current)
        for part in parent_names:
            try:
                next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
                next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            # See durable_directory: this also closes the concurrent-creator
            # crash window for publication parents.
            os.fsync(current)
            descriptors.append(next_fd)
            current = next_fd
        for descriptor in reversed(descriptors[:-1]):
            os.close(descriptor)
        success = True
        return current, final
    except OSError as exc:
        raise ValueError("publication parent cannot be opened safely") from exc
    finally:
        if not success:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _read_regular_at(parent_fd: int, name: str, *, maximum: int | None = None) -> bytes | None:
    """Read one regular file by descriptor without a pathname race.

    ``maximum`` bounds receipt/recovery reads before allocating their content.
    The nonblocking file open makes a FIFO or device rejection immediate.
    """

    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0):
        raise ValueError("regular-file read bound is invalid")
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("immutable publication target is not a regular file") from exc
        raise ValueError("immutable publication target cannot be opened safely") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("immutable publication target is not a regular file")
        if maximum is not None and info.st_size > maximum:
            raise ValueError("immutable publication target exceeds the bounded read limit")
        remaining = info.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(64 << 10, remaining))
            if not chunk:
                raise ValueError("immutable publication target changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("immutable publication target changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_regular_file_no_follow(path: Path | str, *, maximum: int) -> bytes:
    """Read a bounded regular file through no-follow descriptors at every level.

    Unlike :meth:`Path.read_bytes`, both all parent components and the final
    component are opened relative to already pinned descriptors.  A missing file
    raises ``FileNotFoundError``; malformed, linked, special, or oversized files
    raise ``ValueError``.
    """

    candidate = Path(path)
    root, names = _relative_parts(candidate, name="regular file")
    descriptors: list[int] = []
    try:
        current = os.open(root, _DIRECTORY_FLAGS)
        descriptors.append(current)
        for part in names[:-1]:
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            descriptors.append(next_fd)
            current = next_fd
        payload = _read_regular_at(current, names[-1], maximum=maximum)
        if payload is None:
            raise FileNotFoundError(str(candidate))
        return payload
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError("regular file cannot be opened safely") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _copy_regular_file(source_parent_fd: int, name: str, target_parent_fd: int) -> None:
    """Copy one source regular inode into an API-owned directory entry."""

    try:
        source_fd = os.open(name, _FILE_FLAGS, dir_fd=source_parent_fd)
    except OSError as exc:
        raise ValueError("directory publication source file cannot be opened safely") from exc
    try:
        source_info = os.fstat(source_fd)
        if not stat.S_ISREG(source_info.st_mode):
            raise ValueError("directory publication source contains a non-regular file")
        try:
            target_fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=target_parent_fd,
            )
        except OSError as exc:
            raise ValueError("directory publication staging copy cannot be created") from exc
        try:
            remaining = source_info.st_size
            while remaining:
                chunk = os.read(source_fd, min(64 << 10, remaining))
                if not chunk:
                    raise ValueError("directory publication source changed while copying")
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    if written <= 0:  # pragma: no cover - regular-file write contract
                        raise OSError("directory publication staging write failed")
                    view = view[written:]
                remaining -= len(chunk)
            if os.read(source_fd, 1):
                raise ValueError("directory publication source changed while copying")
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
        # The private directory, not the source basename, now owns this copy.
        os.fsync(target_parent_fd)
    finally:
        os.close(source_fd)


def _copy_directory_snapshot(source_fd: int, target_fd: int) -> None:
    """Copy a descriptor-safe directory snapshot into a private directory."""

    for name in sorted(os.listdir(source_fd)):
        try:
            info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError as exc:
            raise ValueError("directory publication source changed while copying") from exc
        if stat.S_ISDIR(info.st_mode):
            try:
                source_child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=source_fd)
            except OSError as exc:
                raise ValueError("directory publication source directory cannot be opened safely") from exc
            try:
                if not stat.S_ISDIR(os.fstat(source_child_fd).st_mode):  # pragma: no cover - open contract
                    raise ValueError("directory publication source changed type")
                try:
                    os.mkdir(name, mode=0o700, dir_fd=target_fd)
                    # Persist the private child entry before adding its contents.
                    os.fsync(target_fd)
                    target_child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=target_fd)
                except OSError as exc:
                    raise ValueError("directory publication staging directory cannot be created") from exc
                try:
                    _copy_directory_snapshot(source_child_fd, target_child_fd)
                    os.fsync(target_child_fd)
                finally:
                    os.close(target_child_fd)
            finally:
                os.close(source_child_fd)
        elif stat.S_ISREG(info.st_mode):
            _copy_regular_file(source_fd, name, target_fd)
        else:
            raise ValueError("directory publication source contains a link or special file")
    os.fsync(target_fd)


def _remove_private_tree(directory_fd: int) -> None:
    """Descriptor-safely remove API-owned staging contents without link traversal."""

    for name in os.listdir(directory_fd):
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(info.st_mode):
            child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            try:
                _remove_private_tree(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _snapshot_private_tree(directory_fd: int, *, prefix: str = "") -> tuple[dict[str, bytes], set[str]]:
    """Return every regular payload and directory name from an owned tree.

    This runs only after the source has been copied into API-owned private
    staging.  Comparing it with a caller-approved payload mapping immediately
    before ``renameat2`` closes child file/directory exchanges during copying.
    """

    payloads: dict[str, bytes] = {}
    directories: set[str] = set()
    for name in sorted(os.listdir(directory_fd)):
        relative = f"{prefix}{name}"
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise ValueError("directory publication private snapshot changed") from exc
        if stat.S_ISREG(info.st_mode):
            payload = _read_regular_at(directory_fd, name)
            if payload is None:  # pragma: no cover - list/open race is rejected above
                raise ValueError("directory publication private snapshot changed")
            payloads[relative] = payload
        elif stat.S_ISDIR(info.st_mode):
            try:
                child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            except OSError as exc:
                raise ValueError("directory publication private snapshot directory is unsafe") from exc
            try:
                child_payloads, child_directories = _snapshot_private_tree(
                    child_fd, prefix=f"{relative}/")
            finally:
                os.close(child_fd)
            directories.add(relative)
            payloads.update(child_payloads)
            directories.update(child_directories)
        else:
            raise ValueError("directory publication private snapshot contains a link or special file")
    return payloads, directories


def _validate_private_snapshot(directory_fd: int, expected_payloads: Mapping[str, bytes]) -> None:
    """Require the copied private tree to equal a validated regular-file map."""

    if not isinstance(expected_payloads, Mapping):
        raise TypeError("directory publication requires an expected payload mapping")
    expected: dict[str, bytes] = {}
    for relative, payload in expected_payloads.items():
        path = Path(relative) if isinstance(relative, str) else None
        if (path is None or path.is_absolute() or not path.parts or ".." in path.parts
                or relative != path.as_posix() or not isinstance(payload, bytes)):
            raise ValueError("directory publication expected payload mapping is invalid")
        expected[relative] = payload
    actual, directories = _snapshot_private_tree(directory_fd)
    expected_directories = {
        ancestor.as_posix()
        for relative in expected
        for ancestor in Path(relative).parents
        if ancestor.as_posix() != "."
    }
    if actual != expected or directories != expected_directories:
        raise ValueError("directory publication private snapshot differs from approved authority")


def cleanup_directory_best_effort(stage: Path | str) -> None:
    """Best-effort, descriptor-safe caller-stage cleanup after publication.

    A successful immutable destination is authoritative.  In particular, a
    racer replacing the caller-owned stage basename with a symlink must not turn
    that successful operation into a reported failure or trigger recursive link
    traversal.
    """

    candidate = Path(stage)
    try:
        parent_fd, name = _open_parent_no_follow(candidate)
    except (ValueError, OSError):
        return
    try:
        try:
            stage_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except OSError:
            # A linked or already removed basename can safely be unlinked as a
            # single entry; never recursively follow it.
            try:
                os.unlink(name, dir_fd=parent_fd)
            except OSError:
                return
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
            return
        try:
            _remove_private_tree(stage_fd)
        except OSError:
            return
        finally:
            try:
                os.close(stage_fd)
            except OSError:
                pass
        try:
            os.rmdir(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError:
            pass
    finally:
        try:
            os.close(parent_fd)
        except OSError:
            pass


def _rename_no_replace(source_parent_fd: int, source_name: str, destination_parent_fd: int, destination_name: str) -> None:
    """Run Linux renameat2 no-replace without a destination precheck."""

    if os.name != "posix" or not hasattr(ctypes, "CDLL"):
        raise RuntimeError("directory no-replace publication requires Linux renameat2")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("directory no-replace publication requires libc renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent_fd, os.fsencode(source_name), destination_parent_fd,
        os.fsencode(destination_name), _RENAME_NOREPLACE,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError("immutable directory publication conflicts with existing authority")
        if code in {errno.ENOSYS, errno.EINVAL, errno.EXDEV, errno.EOPNOTSUPP}:
            raise RuntimeError("directory no-replace publication is unsupported")
        raise OSError(code, os.strerror(code), destination_name)


def publish_directory_no_replace(
    stage: Path, destination: Path, *, expected_payloads: Mapping[str, bytes],
) -> None:
    """Publish a caller-approved tree snapshot without replacement.

    The caller must supply the exact payload map already semantically validated
    for ``stage``.  The API copies a descriptor-pinned source into private
    staging and verifies that private copy against that map immediately before
    ``renameat2``.  Thus neither a top-level nor a child exchange can publish
    bytes that were not approved by the caller.
    """

    destination = Path(destination)
    durable_directory(destination.parent)
    source_fd = open_directory_no_follow(stage)
    destination_parent_fd = open_directory_no_follow(destination.parent)
    private_parent_fd: int | None = None
    private_parent_name: str | None = None
    private_child_name = "authority"
    durable = False
    try:
        for _ in range(32):
            candidate = f".{destination.name}.publish.{secrets.token_hex(16)}"
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=destination_parent_fd)
            except FileExistsError:
                continue
            private_parent_name = candidate
            private_parent_fd = os.open(candidate, _DIRECTORY_FLAGS, dir_fd=destination_parent_fd)
            # Persist the API-owned parent entry before it can contain a child
            # that will become immutable publication authority.
            os.fsync(destination_parent_fd)
            break
        else:  # pragma: no cover - cryptographic collision is not practical
            raise RuntimeError("could not reserve directory publication staging parent")
        os.mkdir(private_child_name, mode=0o700, dir_fd=private_parent_fd)
        os.fsync(private_parent_fd)
        private_child_fd = os.open(private_child_name, _DIRECTORY_FLAGS, dir_fd=private_parent_fd)
        try:
            _copy_directory_snapshot(source_fd, private_child_fd)
            _validate_private_snapshot(private_child_fd, expected_payloads)
            os.fsync(private_child_fd)
        finally:
            os.close(private_child_fd)
        _rename_no_replace(
            private_parent_fd, private_child_name, destination_parent_fd,
            destination.name,
        )
        os.fsync(destination_parent_fd)
        durable = True
    finally:
        try:
            if private_parent_fd is not None:
                if not durable:
                    try:
                        _remove_private_tree(private_parent_fd)
                    except FileNotFoundError:
                        pass
                try:
                    os.close(private_parent_fd)
                except OSError:
                    if not durable:
                        raise
                private_parent_fd = None
            if private_parent_name is not None:
                try:
                    os.rmdir(private_parent_name, dir_fd=destination_parent_fd)
                except FileNotFoundError:
                    if not durable:
                        raise
                except OSError:
                    if not durable:
                        raise
                else:
                    # This post-publication temporary removal is durable only
                    # as cleanup.  Once the destination-parent fsync above
                    # succeeds, its fsync and every descriptor close below are
                    # intentionally unable to recast success as failure.
                    try:
                        os.fsync(destination_parent_fd)
                    except OSError:
                        if not durable:
                            raise
        finally:
            for descriptor in (destination_parent_fd, source_fd):
                try:
                    os.close(descriptor)
                except OSError:
                    if not durable:
                        raise


def fsync_directory(path: Path) -> None:
    """Durably persist directory-entry changes on Linux filesystems."""

    fd = open_directory_no_follow(path)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_bytes_atomically(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Atomically replace one regular output with the exact supplied bytes.

    Unlike immutable receipt publication this is deliberately for a designated
    rebuilt artifact (the checked-in source archive).  It never reopens the
    output pathname after producing ``payload`` and flushes the replacement
    directory entry before returning.
    """

    if not isinstance(payload, bytes):
        raise TypeError("atomic publication payload must be bytes")
    parent_fd, final = _open_parent_no_follow(Path(path))
    temporary: str | None = None
    durable = False
    try:
        for _ in range(32):
            candidate = f".{final}.{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    mode, dir_fd=parent_fd,
                )
                temporary = candidate
                break
            except FileExistsError:
                continue
        else:  # pragma: no cover - cryptographic collision is not practical
            raise RuntimeError("could not reserve atomic publication temporary")
        try:
            view = memoryview(payload)
            while view:
                count = os.write(descriptor, view)
                if count <= 0:  # pragma: no cover - regular-file write contract
                    raise OSError("atomic publication write failed")
                view = view[count:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, final, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        temporary = None
        os.fsync(parent_fd)
        durable = True
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                if not durable:
                    raise
            except OSError:
                if not durable:
                    raise
            else:
                try:
                    os.fsync(parent_fd)
                except OSError:
                    if not durable:
                        raise
        try:
            os.close(parent_fd)
        except OSError:
            if not durable:
                raise


def publish_bytes_no_replace(path: Path, payload: bytes, *, mode: int = 0o600) -> bool:
    """Atomically publish immutable bytes through a no-follow parent descriptor.

    A hard link supplies no-replace semantics.  An identical retry is accepted;
    a pre-existing different payload is an authority conflict.  Every opened
    parent component is fsynced before publication and every final outcome
    fsyncs its containing directory.  After that durable authority boundary,
    temporary hard-link cleanup and its directory fsync are best effort: a
    cleanup failure cannot recast a successful immutable publication.
    """

    if not isinstance(payload, bytes):
        raise TypeError("immutable publication payload must be bytes")
    parent_fd, final = _open_parent_no_follow(Path(path))
    temporary: str | None = None
    durable = False
    try:
        existing = _read_regular_at(parent_fd, final)
        if existing is not None:
            if existing != payload:
                raise ValueError("immutable publication conflicts with existing bytes")
            os.fsync(parent_fd)
            durable = True
            return False
        for _ in range(32):
            candidate = f".{final}.{secrets.token_hex(16)}.tmp"
            try:
                fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                             mode, dir_fd=parent_fd)
                temporary = candidate
                break
            except FileExistsError:
                continue
        else:  # pragma: no cover - cryptographic collision is not practical
            raise RuntimeError("could not reserve immutable publication temporary")
        try:
            written = 0
            while written < len(payload):
                count = os.write(fd, payload[written:])
                if count <= 0:  # pragma: no cover - regular-file write contract
                    raise OSError("immutable publication write failed")
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temporary, final, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                    follow_symlinks=False)
            created = True
        except FileExistsError:
            existing = _read_regular_at(parent_fd, final)
            if existing != payload:
                raise ValueError("immutable publication conflicts with existing bytes")
            created = False
        os.fsync(parent_fd)
        durable = True
        return created
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                if not durable:
                    raise
            except OSError:
                if not durable:
                    raise
            else:
                # Best-effort post-durable cleanup: flush the temporary-name
                # removal when possible, but never recast final authority if
                # this non-authoritative cleanup fsync fails.
                try:
                    os.fsync(parent_fd)
                except OSError:
                    if not durable:
                        raise
        try:
            os.close(parent_fd)
        except OSError:
            if not durable:
                raise
