"""Deterministic, checkout-verifiable source archives for first-party generators."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Any, Mapping

_SOURCE_MANIFEST_SCHEMA = "emender-e97-firstparty-generator-manifest-v3"

# This is deliberately a source-closure allowlist, not a manifest assertion.
# It covers only first-party generation, protected-overlap checking, replay, and
# validation plus their local dependencies.  Correction packing, leases, and
# artifact-store code are separate authorities and must not silently enter this
# generator archive.
EXPECTED_GENERATOR_COMPONENT_PATHS = frozenset({
    "configs/pi/e97-firstparty-authorization-license-v1.json",
    "configs/pi/e97-firstparty-cpu-environment-v1.json",
    "configs/pi/e97-firstparty-overlap-firewall-audit-v1.json",
    "ndm/e97_acquisition_controller.py",
    "ndm/e97_agent_protocol.py",
    "ndm/e97_atomic.py",
    "ndm/e97_first_party_read_observe.py",
    "ndm/e97_first_party_source_archive.py",
    "ndm/e97_onpolicy_records.py",
    "ndm/e97_protected_overlap.py",
    "ndm/e97_task_lake.py",
    "scripts/build_e97_first_party_source_archive.py",
    "scripts/check_e97_protected_overlap.py",
    "scripts/e97_first_party_validator.py",
    "scripts/validate_e97_first_party_task.py",
})


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _manifest_components(manifest_path: Path) -> list[dict[str, str]]:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("source component manifest is unreadable") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != _SOURCE_MANIFEST_SCHEMA:
        raise ValueError("source component manifest schema is invalid")
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("source component manifest must contain components")
    normalized: list[dict[str, str]] = []
    names: set[str] = set()
    for raw in components:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise ValueError("source component is invalid")
        name, digest = raw["path"], raw["sha256"]
        path = Path(name) if isinstance(name, str) else None
        if (not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or path is None or path.is_absolute() or ".." in path.parts
                or name in names or name != path.as_posix()):
            raise ValueError("source component path or SHA-256 is invalid")
        names.add(name)
        normalized.append({"path": name, "sha256": digest})
    if normalized != sorted(normalized, key=lambda component: component["path"]):
        raise ValueError("source component manifest must be path-sorted")
    if {component["path"] for component in normalized} != EXPECTED_GENERATOR_COMPONENT_PATHS:
        raise ValueError("source component manifest does not match the closed generator authority")
    return normalized


def verify_checkout_components(manifest_path: Path, *, checkout_root: Path) -> list[dict[str, str]]:
    """Verify the closed component list against the current checkout bytes."""

    components = _manifest_components(manifest_path)
    for component in components:
        path = checkout_root / component["path"]
        if not path.is_file() or path.is_symlink() or sha256_path(path) != component["sha256"]:
            raise ValueError(f"source component bytes changed: {component['path']}")
    return components


def build_source_archive(manifest_path: Path, output: Path, *, checkout_root: Path) -> str:
    """Write an uncompressed deterministic USTAR over exactly manifest components."""

    components = verify_checkout_components(manifest_path, checkout_root=checkout_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w", format=tarfile.USTAR_FORMAT) as archive:
        for component in components:
            payload = (checkout_root / component["path"]).read_bytes()
            info = tarfile.TarInfo(component["path"])
            info.size = len(payload)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(payload))
    verify_source_archive(manifest_path, output, checkout_root=checkout_root)
    return sha256_path(output)


def verify_archive_members(manifest_path: Path, archive_path: Path) -> str:
    """Verify normalized archive members against a copied component manifest."""

    components = _manifest_components(manifest_path)
    expected = {component["path"]: component["sha256"] for component in components}
    try:
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if names != [component["path"] for component in components]:
                raise ValueError("source archive members are not exactly manifest components")
            for member in members:
                if (not member.isreg() or member.mode != 0o644 or member.uid != 0 or member.gid != 0
                        or member.uname or member.gname or member.mtime != 0):
                    raise ValueError("source archive member metadata is not normalized")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source archive regular member is unreadable")
                payload = stream.read()
                if sha256_bytes(payload) != expected[member.name]:
                    raise ValueError("source archive member digest mismatch")
    except tarfile.TarError as exc:
        raise ValueError("source archive is invalid") from exc
    return sha256_path(archive_path)


def extract_verified_source_member(
    manifest_path: Path, archive_path: Path, member_path: str, destination: Path,
) -> str:
    """Extract one manifest-bound regular member into a private fresh path."""

    components = {component["path"]: component["sha256"] for component in _manifest_components(manifest_path)}
    expected = components.get(member_path)
    if expected is None:
        raise ValueError("requested source member is not in the closed manifest")
    try:
        with tarfile.open(archive_path, "r:") as archive:
            member = archive.getmember(member_path)
            if not member.isreg() or member.issym() or member.islnk():
                raise ValueError("requested source member is not a regular archive member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("requested source member is unreadable")
            payload = stream.read()
    except (KeyError, tarfile.TarError) as exc:
        raise ValueError("requested source member is absent from archive") from exc
    if sha256_bytes(payload) != expected:
        raise ValueError("requested source member digest does not match manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError as exc:
        raise ValueError("requested source extraction destination already exists") from exc
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return expected


def verify_source_archive(manifest_path: Path, archive_path: Path, *, checkout_root: Path) -> str:
    """Verify normalized archive members against both manifest and checkout bytes."""

    components = verify_checkout_components(manifest_path, checkout_root=checkout_root)
    archive_digest = verify_archive_members(manifest_path, archive_path)
    for component in components:
        try:
            with tarfile.open(archive_path, "r:") as archive:
                stream = archive.extractfile(component["path"])
                if stream is None or stream.read() != (checkout_root / component["path"]).read_bytes():
                    raise ValueError("source archive member differs from checkout")
        except tarfile.TarError as exc:
            raise ValueError("source archive is invalid") from exc
    return archive_digest
