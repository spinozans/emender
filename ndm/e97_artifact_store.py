"""Immutable content-addressed artifacts used by E97 correction authorities."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from ndm.e97_atomic import publish_bytes_no_replace
from ndm.e97_onpolicy_records import canonical_json, completion_marker_relative_path

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 1 << 30


class ArtifactStoreError(ValueError):
    """An artifact reference is malformed, missing, or has mutated bytes."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ArtifactStoreError(f"{name} must be a lowercase SHA-256 digest")
    return value


def artifact_relative_path(digest: str, suffix: str) -> str:
    """Return the sole conventional path for a stored immutable artifact."""

    digest = _digest(digest, "artifact digest")
    if not isinstance(suffix, str) or not suffix.startswith(".") or "/" in suffix:
        raise ArtifactStoreError("artifact suffix is invalid")
    return f"artifacts/{digest[:2]}/{digest}{suffix}"


def receipt_relative_path(digest: str) -> str:
    """Return the content-evidence path; it is not completion authority alone."""

    return f"receipts/{_digest(digest, 'receipt digest')}.json"


def completed_receipt_relative_path(task_id: str, digest: str) -> str:
    """Return the task-specific finalized marker required for completion authority."""

    try:
        return completion_marker_relative_path(task_id, digest)
    except ValueError as exc:
        raise ArtifactStoreError(str(exc)) from exc


def artifact_reference(digest: str, suffix: str) -> dict[str, str]:
    return {"sha256": _digest(digest, "artifact digest"), "path": artifact_relative_path(digest, suffix)}


class ArtifactStore:
    """A local immutable store with verified conventional relative references."""

    def __init__(self, root: Path | str) -> None:
        supplied = Path(root)
        supplied.mkdir(parents=True, exist_ok=True)
        try:
            root_info = os.lstat(supplied)
        except FileNotFoundError as exc:
            raise ArtifactStoreError("artifact root is missing") from exc
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise ArtifactStoreError("artifact root must be a real directory")
        self.root = supplied.resolve()

    def put_bytes(self, payload: bytes, *, suffix: str) -> dict[str, str]:
        if not isinstance(payload, bytes) or len(payload) > _MAX_ARTIFACT_BYTES:
            raise ArtifactStoreError("artifact payload exceeds the bounded store limit")
        digest = sha256_bytes(payload)
        relative = artifact_relative_path(digest, suffix)
        publish_bytes_no_replace(self.root / relative, payload)
        return {"sha256": digest, "path": relative}

    def put_json(self, value: Any) -> dict[str, str]:
        return self.put_bytes(canonical_json(value).encode("utf-8"), suffix=".json")

    @staticmethod
    def _read_descriptor(descriptor: int, *, name: str) -> bytes:
        """Read one already-open bounded regular-file descriptor."""

        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size < 0 or info.st_size > _MAX_ARTIFACT_BYTES:
            raise ArtifactStoreError(f"{name} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 << 10, remaining))
            if not chunk:
                raise ArtifactStoreError(f"{name} changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ArtifactStoreError(f"{name} grew while reading")
        return b"".join(chunks)

    @classmethod
    def _read_regular_payload(cls, path: Path, *, name: str) -> bytes:
        """Read an external regular file without following its final symlink."""

        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as exc:
            raise ArtifactStoreError(f"{name} is missing") from exc
        except OSError as exc:
            raise ArtifactStoreError(f"{name} cannot be opened safely") from exc
        try:
            return cls._read_descriptor(descriptor, name=name)
        finally:
            os.close(descriptor)

    def _read_store_relative(self, relative: str, *, name: str) -> bytes:
        """Open every store path component with openat/O_NOFOLLOW."""

        parts = Path(relative).parts
        if not parts or Path(relative).is_absolute() or ".." in parts:
            raise ArtifactStoreError(f"{name} path is invalid")
        directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        descriptors: list[int] = []
        try:
            current = os.open(self.root, directory_flags)
            descriptors.append(current)
            for part in parts[:-1]:
                current = os.open(part, directory_flags, dir_fd=current)
                descriptors.append(current)
            descriptor = os.open(parts[-1], file_flags, dir_fd=current)
            descriptors.append(descriptor)
            return self._read_descriptor(descriptor, name=name)
        except FileNotFoundError as exc:
            raise ArtifactStoreError(f"{name} is missing") from exc
        except OSError as exc:
            raise ArtifactStoreError(f"{name} cannot be opened safely") from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def put_file(self, source: Path, *, suffix: str) -> dict[str, str]:
        return self.put_bytes(self._read_regular_payload(source, name="source artifact"), suffix=suffix)

    def resolve_bytes(self, reference: Mapping[str, Any], *, suffix: str) -> bytes:
        if not isinstance(reference, Mapping) or set(reference) != {"sha256", "path"}:
            raise ArtifactStoreError("artifact reference fields mismatch")
        digest = _digest(reference["sha256"], "artifact reference digest")
        expected = artifact_relative_path(digest, suffix)
        if reference["path"] != expected:
            raise ArtifactStoreError("artifact reference path is not content-addressed")
        payload = self._read_store_relative(expected, name="artifact")
        if sha256_bytes(payload) != digest:
            raise ArtifactStoreError("artifact bytes do not match reference")
        return payload

    def resolve_json(self, reference: Mapping[str, Any]) -> Any:
        payload = self.resolve_bytes(reference, suffix=".json")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ArtifactStoreError("artifact JSON is invalid") from exc
        return value

    def resolve_published_receipt(self, reference: Mapping[str, Any], *, task_id: str) -> Any:
        """Resolve a receipt only when content evidence and its task marker agree."""

        if not isinstance(reference, Mapping) or set(reference) != {"sha256", "path"}:
            raise ArtifactStoreError("receipt reference fields mismatch")
        digest = _digest(reference["sha256"], "receipt reference digest")
        expected_marker = completed_receipt_relative_path(task_id, digest)
        if reference["path"] != expected_marker:
            raise ArtifactStoreError("receipt reference is not the task-specific finalized marker")
        marker_payload = self._read_store_relative(expected_marker, name="completed receipt marker")
        content_payload = self._read_store_relative(
            receipt_relative_path(digest), name="published completion receipt",
        )
        if marker_payload != content_payload or sha256_bytes(marker_payload) != digest:
            raise ArtifactStoreError("completed receipt marker does not bind exact content evidence")
        try:
            value = json.loads(marker_payload)
        except json.JSONDecodeError as exc:
            raise ArtifactStoreError("published completion receipt is invalid JSON") from exc
        if not isinstance(value, Mapping) or marker_payload != canonical_json(value).encode("utf-8"):
            raise ArtifactStoreError("published completion receipt is not canonical bytes")
        if value.get("task_id") != task_id:
            raise ArtifactStoreError("completed receipt marker task identity mismatch")
        return value
