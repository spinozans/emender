"""Sealed deterministic first-party read-observe fixture generation and replay."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import stat
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ndm.e97_acquisition_controller import ACQUISITION_CONTROLLER_SCHEMA, READ_OBSERVE_TOOLS, WorkspaceToolExecutor
from ndm.e97_atomic import (
    cleanup_directory_best_effort,
    durable_directory,
    publish_directory_no_replace,
    read_regular_file_no_follow,
)
from ndm.e97_onpolicy_records import (
    ActionProgressReceipt, NoProgressDetector, _validate_terminal_completion_usage,
    action_fingerprint, progress_fingerprint,
)
from ndm.e97_onpolicy_records import canonical_json, sha256_json, sha256_text, task_identity
from ndm.e97_agent_protocol import serialize_pi_messages
from ndm.e97_task_lake import TASK_BUNDLE_SCHEMA, canonical_intent_digest, validate_task_collection, validate_source_registry
from ndm.e97_first_party_source_archive import (
    source_archive_payload_verification,
    verified_loaded_module_closure_sha256,
    verified_source_member_payload,
    verify_archive_members,
)
from ndm.e97_protected_overlap import (
    validate_overlap_authorization, validate_authorized_overlap_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_REGISTRY = ROOT / "configs/pi/e97-onpolicy-source-registry-v1.json"
GENERATOR_MANIFEST = ROOT / "configs/pi/e97-firstparty-generator-manifest-v1.json"
GENERATOR_SOURCE_ARCHIVE = ROOT / "configs/pi/e97-firstparty-source-v1.tar"
ENVIRONMENT_DESCRIPTOR = ROOT / "configs/pi/e97-firstparty-cpu-environment-v1.json"
OVERLAP_AUDIT = ROOT / "configs/pi/e97-firstparty-overlap-firewall-audit-v1.json"
AUTHORIZATION_LICENSE = ROOT / "configs/pi/e97-firstparty-authorization-license-v1.json"
VALIDATOR_PROGRAM = ROOT / "scripts/e97_first_party_validator.py"
_CONTROLLER_SOURCE_MEMBER = "ndm/e97_acquisition_controller.py"
_VALIDATOR_SOURCE_MEMBER = "scripts/e97_first_party_validator.py"
VALIDATOR_LOGICAL_RUNTIME = "@runtime-python"
VALIDATOR_LOGICAL_PROGRAM = "@generator-source/scripts/e97_first_party_validator.py"
_REQUIRED_PANELS = {
    "ef481c637fde5916b8b0fe1f80cc2b4f0a6b88262088cbb33aefe0fed6bd6d09",
    "8d0d5d350a39c9f5c007d98e4a0010f4a06f39e5a7fb89ecd9ccc4e37e66b8d8",
    "939bcd66768884a1e5ec44bcf11fdf5d09602ebdabe86d4caefffd4ace8dbb60",
}


def _sha_path(path: Path) -> str:
    """Hash an executable or local generated file at its caller-owned path."""

    # Interpreter executables may deliberately be symlinks in a virtualenv.
    # Candidate/quarantine authority reads use ``_read_snapshot_file`` instead.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_snapshot_file(path: Path, *, name: str, maximum: int = 64 << 20) -> bytes:
    """Read one bounded regular file through no-follow descriptors at every level."""

    try:
        return read_regular_file_no_follow(path, maximum=maximum)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{name} cannot be opened safely") from exc


def validator_logical_argv(mode: str) -> list[str]:
    if mode not in {"focused", "regression"}:
        raise ValueError("validator mode is invalid")
    return [VALIDATOR_LOGICAL_RUNTIME, VALIDATOR_LOGICAL_PROGRAM, "--mode", mode]


def _digest(seed: str, domain: str) -> str:
    return hashlib.sha256(("e97-first-party-read-observe-v1\0" + seed + "\0" + domain).encode()).hexdigest()


def _tree_digest(root: Path) -> str:
    """Return the canonical observable identity of every fixture entry.

    Directories are explicit because ``list_files`` exposes them, including an
    otherwise empty directory.  A file-only digest could therefore approve an
    archive whose observable workspace differs from the sealed fixture.
    """

    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif path.is_file():
            entries.append({"path": relative, "type": "file", "sha256": _sha_path(path)})
        else:
            raise ValueError("fixture tree contains a non-regular non-directory entry")
    return sha256_json(entries)


def _archive_root_sha256(tasks: list[Mapping[str, Any]], authority_root: Path) -> str:
    """Bind every task archive path, bytes, and digest as one collection root."""

    entries = []
    for task in sorted(tasks, key=lambda item: item["fixture"]["artifact_path"]):
        archive = authority_root / task["fixture"]["artifact_path"]
        entries.append({
            "path": task["fixture"]["artifact_path"],
            "bytes": archive.stat().st_size,
            "sha256": _sha_path(archive),
        })
    return sha256_text(canonical_json(entries))


def _archive(root: Path, output: Path) -> None:
    """Write a deterministic tar with normalized ownership, timestamps, and modes."""
    with tarfile.open(output, "w", format=tarfile.USTAR_FORMAT) as archive:
        for path in sorted(root.rglob("*")):
            info = archive.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644 if path.is_file() else 0o755
            if path.is_file():
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)


def _verified_fixture_archive_payload(root: Path, archive_path: Path, *, tree_digest: str, disk_limit: int) -> bytes:
    """Retain an archive then prove it expands to its declared pre-publication tree."""

    payload = _read_snapshot_file(archive_path, name="generated fixture archive", maximum=256 << 20)
    with tempfile.TemporaryDirectory(prefix="e97-first-party-fixture-verify-") as temporary:
        extracted = Path(temporary) / "fixture"
        extracted.mkdir()
        safe_extract_fixture_archive(
            payload,
            extracted,
            expected_sha256=_snapshot_digest(payload),
            expected_tree_digest=tree_digest,
            disk_limit=disk_limit,
        )
    return payload


def safe_extract_fixture_archive(
    archive_payload: bytes,
    destination: Path,
    *,
    expected_sha256: str,
    expected_tree_digest: str,
    disk_limit: int,
) -> int:
    """Extract retained archive bytes without reopening an already-hashed pathname."""

    if not isinstance(archive_payload, bytes):
        raise ValueError("fixture archive snapshot is invalid")
    if hashlib.sha256(archive_payload).hexdigest() != expected_sha256:
        raise ValueError("fixture archive SHA-256 does not match sealed bundle")
    if disk_limit <= 0:
        raise ValueError("fixture disk limit is invalid")
    expanded = 0
    names: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as stream:
            members = stream.getmembers()
            if len(members) > 4096:
                raise ValueError("fixture archive member count exceeds limit")
            for member in members:
                relative = Path(member.name)
                if (member.name.startswith("/") or ".." in relative.parts or relative == Path(".")
                        or member.name != relative.as_posix() or member.name in names):
                    raise ValueError("fixture archive member path is unsafe")
                names.add(member.name)
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=False)
                    continue
                if not member.isreg() or member.issym() or member.islnk():
                    raise ValueError("fixture archive has non-regular member")
                expanded += member.size
                if expanded > disk_limit:
                    raise ValueError("fixture archive expanded bytes exceed disk limit")
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = stream.extractfile(member)
                if payload is None:
                    raise ValueError("fixture archive member is unreadable")
                with target.open("xb") as output:
                    shutil.copyfileobj(payload, output)
                os.chmod(target, 0o644)
    except tarfile.TarError as exc:
        raise ValueError("fixture archive is invalid") from exc
    if _tree_digest(destination) != expected_tree_digest:
        raise ValueError("fixture archive expanded tree does not match sealed bundle")
    return expanded


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        elif path.is_dir():
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _source_revision() -> str:
    """Require a clean committed checkout before minting a generation receipt."""

    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("first-party generation requires a readable committed checkout") from exc
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision) or status:
        raise ValueError("first-party generation requires a clean committed checkout")
    return revision


@dataclass(frozen=True)
class CheckedSourceAuthority:
    """Exact source authority bytes consumed by first-party generation."""

    registry: dict[str, Any]
    registry_payload: bytes
    manifest_payload: bytes
    archive_payload: bytes
    environment_payload: bytes
    overlap_payload: bytes
    license_payload: bytes
    source_members: Mapping[str, bytes]
    manifest_sha256: str
    archive_sha256: str


def _verify_environment(payload: bytes) -> None:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CPU environment receipt is invalid") from exc
    runtime = value.get("runtime") if isinstance(value, Mapping) else None
    observed = {"interpreter_sha256": _sha_path(Path(sys.executable)), "implementation": platform.python_implementation(),
                "python_version": platform.python_version(), "platform_system": platform.system(), "machine": platform.machine()}
    if value.get("schema") != "emender-e97-firstparty-cpu-environment-v2" or value.get("purpose") is None or runtime != observed:
        raise ValueError("CPU environment receipt does not match current runtime")


def _verify_overlap(payload: bytes) -> None:
    """Verify static checker authorization from one retained payload."""

    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("protected overlap audit is invalid") from exc
    validate_overlap_authorization(value)


def _verify_license(payload: bytes) -> None:
    """Require the retained first-party fixture license descriptor exactly once."""

    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("first-party authorization license is invalid") from exc
    if (not isinstance(value, Mapping)
            or set(value) != {"schema", "authorization", "license", "scope"}
            or value.get("schema") != "emender-e97-firstparty-authorization-license-v1"
            or any(not isinstance(value.get(field), str) or not value[field]
                   for field in ("authorization", "license", "scope"))):
        raise ValueError("first-party authorization license is invalid")


def _checked_registry(path: Path, digest: str, policy_sha256: str) -> tuple[dict[str, Any], bytes]:
    if path.absolute() != CHECKED_IN_REGISTRY.absolute():
        raise ValueError("registry path must be the checked-in registry")
    payload = _read_snapshot_file(path, name="checked-in registry")
    if _snapshot_digest(payload) != digest:
        raise ValueError("registry SHA-256 does not match checked-in bytes")
    try:
        registry = validate_source_registry(json.loads(payload))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("checked-in registry is invalid") from exc
    if registry["policy_sha256"] != policy_sha256:
        raise ValueError("policy SHA-256 does not match checked-in registry")
    if not _REQUIRED_PANELS.issubset({item["manifest_sha256"] for item in registry["protected_evaluation"]}):
        raise ValueError("registry is missing required protected panels")
    return registry, payload


def _checked_source_authority(
    registry_path: Path, registry_sha256: str, policy_sha256: str,
) -> CheckedSourceAuthority:
    """Hash, parse, verify, and retain every external generation authority once."""

    registry, registry_payload = _checked_registry(registry_path, registry_sha256, policy_sha256)
    manifest_payload, archive_payload, manifest_sha256, archive_sha256 = source_archive_payload_verification(
        GENERATOR_MANIFEST, GENERATOR_SOURCE_ARCHIVE, checkout_root=ROOT)
    environment_payload = _read_snapshot_file(ENVIRONMENT_DESCRIPTOR, name="CPU environment descriptor")
    overlap_payload = _read_snapshot_file(OVERLAP_AUDIT, name="protected overlap audit")
    license_payload = _read_snapshot_file(AUTHORIZATION_LICENSE, name="authorization license")
    _verify_environment(environment_payload)
    _verify_overlap(overlap_payload)
    _verify_license(license_payload)
    source_members = {}
    for member_path in (_CONTROLLER_SOURCE_MEMBER, _VALIDATOR_SOURCE_MEMBER):
        member_payload, _member_sha256 = verified_source_member_payload(
            manifest_payload, archive_payload, member_path)
        source_members[member_path] = member_payload
    receipt_payloads = {
        "source_archive_sha256": archive_payload,
        "license_sha256": license_payload,
        "environment_sha256": environment_payload,
        "overlap_sha256": overlap_payload,
    }
    for source in registry["sources"]:
        if source["status"] == "admitted" and source["kind"] == "first-party":
            for name, payload in receipt_payloads.items():
                if source["receipts"][name] != _snapshot_digest(payload):
                    raise ValueError("checked-in first-party registry receipt does not hash its artifact")
    return CheckedSourceAuthority(
        registry, registry_payload, manifest_payload, archive_payload,
        environment_payload, overlap_payload, license_payload, source_members,
        manifest_sha256, archive_sha256,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def generate(output: Path, *, seed: str, registry_path: Path = CHECKED_IN_REGISTRY,
             registry_sha256: str | None = None, policy_sha256: str | None = None) -> dict[str, Path]:
    """Create a sealed CPU-only task authority, publishing only by atomic rename."""
    if registry_sha256 is None or policy_sha256 is None:
        raise ValueError("checked-in registry SHA-256 and policy SHA-256 are required")
    authority = _checked_source_authority(registry_path, registry_sha256, policy_sha256)
    registry = authority.registry
    admitted = {item["id"]: item for item in registry["sources"] if item["status"] == "admitted" and item["kind"] == "first-party"}
    if not {"e97-firstparty-train", "e97-firstparty-development"}.issubset(admitted):
        raise ValueError("checked-in registry lacks pre-admitted first-party split sources")
    source_revision = _source_revision()
    generator_manifest_sha256 = authority.manifest_sha256
    generator_source_archive_sha256 = authority.archive_sha256
    parent = output.parent
    durable_directory(parent)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=parent))
    fixture_temporary = tempfile.TemporaryDirectory(prefix="e97-first-party-fixtures-")
    try:
        # Fixture build roots never enter the publication stage.  Only their
        # sealed archive derivatives are written below, so exact-payload
        # publication cannot be poisoned by generator scratch trees.
        fixtures = Path(fixture_temporary.name)
        private, archives = stage / "private_validators", stage / "archives"
        private.mkdir(); archives.mkdir()
        # Every copied source artifact is the retained payload just checked
        # above; generation never reopens a verified authority pathname.
        (stage / "source-registry.json").write_bytes(authority.registry_payload)
        (stage / "generator-manifest.json").write_bytes(authority.manifest_payload)
        (stage / "source-archive.tar").write_bytes(authority.archive_payload)
        (stage / "environment-descriptor.json").write_bytes(authority.environment_payload)
        (stage / "overlap-firewall-audit.json").write_bytes(authority.overlap_payload)
        (stage / "authorization-license.json").write_bytes(authority.license_payload)
        # These are members of the retained, canonical verified source archive
        # above, never a second mutable checkout read.
        controller_payload = authority.source_members[_CONTROLLER_SOURCE_MEMBER]
        runtime = {"schema_digest": sha256_text(ACQUISITION_CONTROLLER_SCHEMA), "tool_schema_digest": sha256_json(READ_OBSERVE_TOOLS),
                   "controller_digest": _loaded_replay_digest_from_archive(
                       authority.manifest_payload, authority.archive_payload),
                   "sandbox_image_digest": _snapshot_digest(authority.environment_payload),
                   "system_prompt_sha256": sha256_text("Use only the provided read-observe tools and ground the final in observations.")}
        validator_program_digest = _snapshot_digest(authority.source_members[_VALIDATOR_SOURCE_MEMBER])
        tasks: list[dict[str, Any]] = []
        for split in ("train", "development"):
            for kind in ("direct", "opaque"):
                family = f"firstparty-{split}-{kind}-read-v1"
                token, stem = _digest(seed, f"{split}-{kind}-gold")[:20], _digest(seed, f"{split}-{kind}-path")[:18]
                folder = f"archive-{stem[:8]}" if kind == "opaque" else f"document-{stem[:8]}"
                filename, relative = f"{stem[8:16]}.txt", f"{folder}/{stem[8:16]}.txt"
                root = fixtures / f"{split}-{kind}"; (root / folder).mkdir(parents=True)
                (root / folder / filename).write_text("token=" + token + "\n")
                if kind == "opaque":
                    (root / (_digest(seed, f"{split}-{kind}-stale-hint")[:12] + ".md")).write_text("The previous location is retired. Enumerate the workspace before reading.\n")
                prompt = (f"Read the exact token in `{relative}` and answer only with that token." if kind == "direct" else
                          "A stale workspace hint may name a retired file. Discover the one token-bearing text file by listing the workspace, then read it and answer only with its token.")
                tree, intent = _tree_digest(root), canonical_intent_digest(prompt)
                namespace = f"e97-{'train' if split == 'train' else 'dev'}-{family}"
                identity = task_identity(namespace=namespace, family_id=family, generator_source_digest=generator_source_archive_sha256, fixture_tree_digest=tree, intent_digest=intent)
                # Archive outside the publication stage, retain its exact bytes,
                # then independently expand those bytes before they can be copied
                # into a generation authority.  A mutation between tree digest and
                # tar construction therefore fails before publication.
                archive_scratch = fixtures / f"{identity}.tar"
                _archive(root, archive_scratch)
                archive_payload = _verified_fixture_archive_payload(
                    root, archive_scratch, tree_digest=tree, disk_limit=1 << 20)
                archive = archives / f"{identity}.tar"
                archive.write_bytes(archive_payload)
                spec = {"schema": "emender-e97-first-party-validator-v2", "task_identity": identity, "fixture_tree_digest": tree,
                        "archive_sha256": _snapshot_digest(archive_payload), "expected_token": token, "required_read_path": relative,
                        "program_sha256": validator_program_digest, "interpreter_sha256": _sha_path(Path(sys.executable)),
                        "minefield": {"allowed_tools": ["list_files", "read"], "forbidden_paths": ["/", ".."]}}
                _write_json(private / f"{identity}.json", spec)
                source = admitted[f"e97-firstparty-{split}"]
                tasks.append({"schema": TASK_BUNDLE_SCHEMA, "split": split,
                    "task": {"namespace": namespace, "family_id": family, "identity": identity, "generator_source_digest": generator_source_archive_sha256, "fixture_tree_digest": tree, "intent_digest": intent, "prompt": prompt, "difficulty": 1 if kind == "direct" else 2},
                    "source": {"registry_id": source["id"], "kind": "first-party", "repository": f"firstparty/{split}-read-observe", "revision": source_revision, "source_record_digest": sha256_json({"source": source["id"], "family": family, "generator": generator_source_archive_sha256}), "license_receipt_digest": source["receipts"]["license_sha256"]},
                    "fixture": {"artifact_path": f"archives/{archive.name}", "artifact_bytes": len(archive_payload), "artifact_sha256": spec["archive_sha256"], "tree_digest": tree}, "runtime": runtime,
                    "limits": {"turns": 12, "seconds": 60, "completion_tokens": 512, "output_bytes": 16384, "disk_bytes": 1 << 20, "processes": 1},
                    "validator": {"spec_digest": sha256_json(spec), "focused_argv": validator_logical_argv("focused"), "regression_argv": validator_logical_argv("regression"), "milestone_digest": sha256_text("required-read\0" + relative), "minefield_digest": sha256_json(spec["minefield"])}})
        tasks = validate_task_collection(tasks, registry=registry)
        task_path = stage / "tasks.jsonl"; task_path.write_text("".join(canonical_json(task) + "\n" for task in sorted(tasks, key=lambda x: x["task"]["identity"])))
        archive_root_sha256 = _archive_root_sha256(tasks, stage)
        receipt = {
            "schema": "emender-e97-first-party-generation-receipt-v6",
            "state": "quarantined-pending-protected-overlap",
            "seed_sha256": sha256_text(seed),
            "generator_component_manifest_sha256": generator_manifest_sha256,
            "generator_source_archive_sha256": generator_source_archive_sha256,
            "tasks_sha256": _sha_path(task_path),
            "archive_root_sha256": archive_root_sha256,
            "registry_sha256": registry_sha256,
            "policy_sha256": policy_sha256,
            "registry_copy_sha256": _sha_path(stage / "source-registry.json"),
            "controller_source_sha256": _snapshot_digest(controller_payload),
            "source_revision": source_revision,
            "environment_descriptor_sha256": runtime["sandbox_image_digest"],
            "overlap_firewall_audit_sha256": _snapshot_digest(authority.overlap_payload),
            "authorization_license_sha256": _snapshot_digest(authority.license_payload),
            "protected_manifest_sha256s": sorted(_REQUIRED_PANELS),
        }
        _write_json(stage / "generation-receipt.json", receipt)
        _write_json(stage / "authority-state.json", {
            "schema": "emender-e97-first-party-authority-state-v2",
            "state": "quarantined-pending-protected-overlap",
            "generation_receipt_sha256": _sha_path(stage / "generation-receipt.json"),
        })
        # Fixture scratch is outside ``stage`` and is removed before the
        # exact approved payload map is constructed and published.
        fixture_temporary.cleanup()
        approved = dict(validate_generated_quarantine(stage).payloads)
        _fsync_tree(stage)
        publish_directory_no_replace(stage, output, expected_payloads=approved)
        cleanup_directory_best_effort(stage)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        fixture_temporary.cleanup()
    return {"root": output, "registry": output / "source-registry.json", "tasks": output / "tasks.jsonl", "private": output / "private_validators", "receipt": output / "generation-receipt.json"}


@dataclass(frozen=True)
class ValidatedQuarantine:
    """One descriptor-safe generated authority snapshot retained for admission."""

    registry: dict[str, Any]
    tasks: list[dict[str, Any]]
    generation: dict[str, Any]
    payloads: Mapping[str, bytes]


def _snapshot_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_archive_root(tasks: list[Mapping[str, Any]], payloads: Mapping[str, bytes]) -> str:
    entries = []
    for task in sorted(tasks, key=lambda item: item["fixture"]["artifact_path"]):
        relative = task["fixture"]["artifact_path"]
        payload = payloads.get(relative)
        if payload is None:
            raise ValueError("quarantine archive snapshot is missing")
        entries.append({"path": relative, "bytes": len(payload), "sha256": _snapshot_digest(payload)})
    return sha256_text(canonical_json(entries))


def _snapshot_quarantine(root: Path) -> dict[str, bytes]:
    """Retain every authority component before parsing or admission copying."""

    required = (
        "source-registry.json", "tasks.jsonl", "generation-receipt.json", "authority-state.json",
        "generator-manifest.json", "source-archive.tar", "environment-descriptor.json",
        "overlap-firewall-audit.json", "authorization-license.json",
    )
    return {relative: _read_snapshot_file(root / relative, name=f"quarantine {relative}") for relative in required}


def _validate_quarantine_snapshot(payloads: dict[str, bytes]) -> ValidatedQuarantine:
    """Validate retained source bytes without reopening mutable quarantine paths."""

    try:
        registry = validate_source_registry(json.loads(payloads["source-registry.json"]))
        lines = payloads["tasks.jsonl"].decode("utf-8").splitlines()
        if not lines or any(not line for line in lines):
            raise ValueError("generated tasks are not strict JSONL")
        tasks = validate_task_collection([json.loads(line) for line in lines], registry=registry)
        generation = json.loads(payloads["generation-receipt.json"])
        state = json.loads(payloads["authority-state.json"])
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
        raise ValueError("generated quarantine is malformed") from exc
    expected = {
        "schema", "state", "seed_sha256", "generator_component_manifest_sha256",
        "generator_source_archive_sha256", "tasks_sha256", "archive_root_sha256", "registry_sha256",
        "policy_sha256", "registry_copy_sha256", "controller_source_sha256", "source_revision",
        "environment_descriptor_sha256", "overlap_firewall_audit_sha256", "authorization_license_sha256", "protected_manifest_sha256s",
    }
    if (set(generation) != expected or generation["schema"] != "emender-e97-first-party-generation-receipt-v6"
            or generation["state"] != "quarantined-pending-protected-overlap"):
        raise ValueError("generated quarantine receipt schema/state is invalid")
    for field in expected - {"schema", "state", "protected_manifest_sha256s", "source_revision"}:
        if not isinstance(generation[field], str) or len(generation[field]) != 64:
            raise ValueError("generated quarantine receipt digest is invalid")
    if (not isinstance(generation["source_revision"], str)
            or len(generation["source_revision"]) not in {40, 64}
            or any(char not in "0123456789abcdef" for char in generation["source_revision"])):
        raise ValueError("generated quarantine source revision is invalid")
    controller_member_payload, _ = verified_source_member_payload(
        payloads["generator-manifest.json"], payloads["source-archive.tar"], _CONTROLLER_SOURCE_MEMBER)
    validator_member_payload, _ = verified_source_member_payload(
        payloads["generator-manifest.json"], payloads["source-archive.tar"], _VALIDATOR_SOURCE_MEMBER)
    if generation.get("controller_source_sha256") != _snapshot_digest(controller_member_payload):
        raise ValueError("generated quarantine controller digest does not bind source archive")
    for task in tasks:
        archive_relative = task["fixture"]["artifact_path"]
        private_relative = f"private_validators/{task['task']['identity']}.json"
        if archive_relative not in payloads or private_relative not in payloads:
            raise ValueError("generated quarantine snapshot lacks task authority")
        archive_payload, spec_payload = payloads[archive_relative], payloads[private_relative]
        if (_snapshot_digest(archive_payload) != task["fixture"]["artifact_sha256"]
                or len(archive_payload) != task["fixture"]["artifact_bytes"]):
            raise ValueError("generated quarantine archive snapshot does not bind task")
        # Every retained archive must independently reconstruct the declared
        # tree before a quarantine is accepted or published.
        with tempfile.TemporaryDirectory(prefix="e97-quarantine-fixture-verify-") as temporary:
            extracted = Path(temporary) / "fixture"
            extracted.mkdir()
            safe_extract_fixture_archive(
                archive_payload,
                extracted,
                expected_sha256=task["fixture"]["artifact_sha256"],
                expected_tree_digest=task["fixture"]["tree_digest"],
                disk_limit=task["limits"]["disk_bytes"],
            )
        try:
            spec = json.loads(spec_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("generated quarantine private validator is invalid") from exc
        if (sha256_json(spec) != task["validator"]["spec_digest"]
                or spec.get("program_sha256") != _snapshot_digest(validator_member_payload)):
            raise ValueError("generated quarantine private validator does not bind task")
    _verify_environment(payloads["environment-descriptor.json"])
    _verify_overlap(payloads["overlap-firewall-audit.json"])
    _verify_license(payloads["authorization-license.json"])
    with tempfile.TemporaryDirectory(prefix="e97-quarantine-source-") as temporary:
        temporary_root = Path(temporary)
        manifest_path, archive_path = temporary_root / "manifest.json", temporary_root / "source.tar"
        manifest_path.write_bytes(payloads["generator-manifest.json"])
        archive_path.write_bytes(payloads["source-archive.tar"])
        verify_archive_members(manifest_path, archive_path)
    if (generation["tasks_sha256"] != _snapshot_digest(payloads["tasks.jsonl"])
            or generation["archive_root_sha256"] != _snapshot_archive_root(tasks, payloads)
            or generation["registry_copy_sha256"] != _snapshot_digest(payloads["source-registry.json"])
            or generation["registry_sha256"] != _snapshot_digest(payloads["source-registry.json"])
            or generation["generator_component_manifest_sha256"] != _snapshot_digest(payloads["generator-manifest.json"])
            or generation["generator_source_archive_sha256"] != _snapshot_digest(payloads["source-archive.tar"])
            or generation["environment_descriptor_sha256"] != _snapshot_digest(payloads["environment-descriptor.json"])
            or generation["overlap_firewall_audit_sha256"] != _snapshot_digest(payloads["overlap-firewall-audit.json"])
            or generation["authorization_license_sha256"] != _snapshot_digest(payloads["authorization-license.json"])
            or generation["protected_manifest_sha256s"] != sorted(item["manifest_sha256"] for item in registry["protected_evaluation"])):
        raise ValueError("generated quarantine receipt binding mismatch")
    for source in registry["sources"]:
        if source["status"] == "admitted" and source["kind"] == "first-party":
            expected_receipts = {
                "source_archive_sha256": generation["generator_source_archive_sha256"],
                "license_sha256": generation["authorization_license_sha256"],
                "environment_sha256": generation["environment_descriptor_sha256"],
                "overlap_sha256": _snapshot_digest(payloads["overlap-firewall-audit.json"]),
            }
            if source["receipts"] != expected_receipts:
                raise ValueError("generated quarantine source receipts do not bind retained authority")
    if state != {
            "schema": "emender-e97-first-party-authority-state-v2",
            "state": "quarantined-pending-protected-overlap",
            "generation_receipt_sha256": _snapshot_digest(payloads["generation-receipt.json"]),
    }:
        raise ValueError("generated authority is not an immutable quarantine")
    return ValidatedQuarantine(registry, tasks, generation, dict(payloads))


def validate_generated_quarantine(root: Path) -> ValidatedQuarantine:
    """Snapshot then validate a quarantine; no later admission step rereads it."""

    payloads = _snapshot_quarantine(root)
    try:
        registry = validate_source_registry(json.loads(payloads["source-registry.json"]))
        lines = payloads["tasks.jsonl"].decode("utf-8").splitlines()
        tasks = validate_task_collection([json.loads(line) for line in lines if line], registry=registry)
        for task in tasks:
            for relative in (task["fixture"]["artifact_path"], f"private_validators/{task['task']['identity']}.json"):
                if relative in {"", ".", ".."} or Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("generated quarantine authority path is invalid")
                payloads[relative] = _read_snapshot_file(root / relative, name=f"quarantine {relative}")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("generated quarantine is malformed") from exc
    return _validate_quarantine_snapshot(payloads)


def _write_snapshot_tree(stage: Path, payloads: Mapping[str, bytes]) -> None:
    for relative, payload in sorted(payloads.items()):
        if relative == "authority-state.json":
            continue
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _allowlist_descriptor(payload: bytes) -> dict[str, Any]:
    return {
        "path": "collection-authorization-allowlist.json", "bytes": len(payload),
        "sha256": _snapshot_digest(payload),
    }


def _validate_admitted_stage(
    stage: Path, source: ValidatedQuarantine, overlap_payload: bytes, authorization: Mapping[str, Any],
) -> dict[str, bytes]:
    """Reopen and verify the final staged authority before its one-way publish."""

    # Re-snapshot every copied component from the stage.  Substitute the
    # retained quarantine state only for this semantic revalidation; the real
    # admitted state is checked below as a separate final authority.
    staged_payloads = {
        relative: (_read_snapshot_file(stage / relative, name=f"admitted {relative}")
                   if relative != "authority-state.json" else payload)
        for relative, payload in source.payloads.items()
    }
    staged = _validate_quarantine_snapshot(staged_payloads)
    for relative, payload in source.payloads.items():
        if relative != "authority-state.json" and staged.payloads.get(relative) != payload:
            raise ValueError("admitted stage differs from the validated quarantine snapshot")
    # Inspect every final-only artifact from one safe read.
    final_payloads = {
        name: _read_snapshot_file(stage / name, name=f"admitted {name}")
        for name in ("protected-overlap-receipt.json", "collection-authorization-allowlist.json",
                     "authority-state.json", "admission-receipt.json")
    }
    if (final_payloads["protected-overlap-receipt.json"] != overlap_payload
            or final_payloads["collection-authorization-allowlist.json"] != authorization["allowlist_payload"]):
        raise ValueError("admitted stage final evidence differs from retained snapshots")
    descriptor = _allowlist_descriptor(authorization["allowlist_payload"])
    state, admission = json.loads(final_payloads["authority-state.json"]), json.loads(final_payloads["admission-receipt.json"])
    expected_state = {
        "schema": "emender-e97-first-party-authority-state-v3", "state": "admitted",
        "generation_receipt_sha256": _snapshot_digest(source.payloads["generation-receipt.json"]),
        "protected_overlap_receipt_sha256": _snapshot_digest(overlap_payload),
        "collection_authorization_sha256": authorization["authorization_sha256"],
        "collection_authorization_allowlist": descriptor,
        "collection_authorization": authorization["authorization"],
    }
    expected_admission = {
        "schema": "emender-e97-first-party-admission-receipt-v3",
        "registry_sha256": _snapshot_digest(source.payloads["source-registry.json"]),
        "generation_receipt_sha256": _snapshot_digest(source.payloads["generation-receipt.json"]),
        "protected_overlap_receipt_sha256": _snapshot_digest(overlap_payload),
        "tasks_sha256": source.generation["tasks_sha256"], "archive_root_sha256": source.generation["archive_root_sha256"],
        "protected_manifest_sha256s": source.generation["protected_manifest_sha256s"],
        "collection_authorization_sha256": authorization["authorization_sha256"],
        "collection_authorization_allowlist": descriptor,
        "collection_authorization": authorization["authorization"],
    }
    if state != expected_state or admission != expected_admission:
        raise ValueError("admitted stage authorization/state binding is invalid")
    # This exact map is the caller-approved tree manifest for immutable
    # directory publication.  It includes every copied source payload and each
    # final admission authority, so a child exchange during copying fails.
    return {**{
        relative: payload for relative, payload in source.payloads.items()
        if relative != "authority-state.json"
    }, **final_payloads}


def admit_generated_collection(
    quarantine: Path, overlap_receipt: Path, output: Path, *, diagnostic_cpu_system_gate: bool = False,
) -> dict[str, Path]:
    """Publish only the exact validated quarantine snapshots after authorization."""

    source = validate_generated_quarantine(quarantine)
    generation = source.generation
    if generation["protected_manifest_sha256s"] != sorted(_REQUIRED_PANELS):
        raise ValueError("generated quarantine does not bind the fixed protected panel set")
    overlap_payload = _read_snapshot_file(overlap_receipt, name="protected overlap receipt")
    try:
        overlap = json.loads(overlap_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("protected overlap receipt is malformed") from exc
    authorization = validate_authorized_overlap_receipt(
        overlap, receipt_sha256=_snapshot_digest(overlap_payload),
        registry_sha256=generation["registry_sha256"],
        generation_receipt_sha256=_snapshot_digest(source.payloads["generation-receipt.json"]),
        tasks_sha256=generation["tasks_sha256"], archive_root_sha256=generation["archive_root_sha256"],
        generator_manifest_sha256=generation["generator_component_manifest_sha256"],
        source_archive_sha256=generation["generator_source_archive_sha256"],
        source_revision=generation["source_revision"], controller_source_sha256=generation["controller_source_sha256"],
        diagnostic_cpu_system_gate=diagnostic_cpu_system_gate,
    )
    parent = output.parent
    durable_directory(parent)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=parent))
    try:
        _write_snapshot_tree(stage, source.payloads)
        (stage / "protected-overlap-receipt.json").write_bytes(overlap_payload)
        (stage / "collection-authorization-allowlist.json").write_bytes(authorization["allowlist_payload"])
        descriptor = _allowlist_descriptor(authorization["allowlist_payload"])
        _write_json(stage / "authority-state.json", {
            "schema": "emender-e97-first-party-authority-state-v3", "state": "admitted",
            "generation_receipt_sha256": _snapshot_digest(source.payloads["generation-receipt.json"]),
            "protected_overlap_receipt_sha256": _snapshot_digest(overlap_payload),
            "collection_authorization_sha256": authorization["authorization_sha256"],
            "collection_authorization_allowlist": descriptor,
            "collection_authorization": authorization["authorization"],
        })
        _write_json(stage / "admission-receipt.json", {
            "schema": "emender-e97-first-party-admission-receipt-v3",
            "registry_sha256": _snapshot_digest(source.payloads["source-registry.json"]),
            "generation_receipt_sha256": _snapshot_digest(source.payloads["generation-receipt.json"]),
            "protected_overlap_receipt_sha256": _snapshot_digest(overlap_payload),
            "tasks_sha256": generation["tasks_sha256"], "archive_root_sha256": generation["archive_root_sha256"],
            "protected_manifest_sha256s": generation["protected_manifest_sha256s"],
            "collection_authorization_sha256": authorization["authorization_sha256"],
            "collection_authorization_allowlist": descriptor,
            "collection_authorization": authorization["authorization"],
        })
        approved = _validate_admitted_stage(stage, source, overlap_payload, authorization)
        _fsync_tree(stage)
        publish_directory_no_replace(stage, output, expected_payloads=approved)
        # Successful publication is authoritative even if a stage-exchange
        # racer leaves behind a symlink or cleanup otherwise fails.
        cleanup_directory_best_effort(stage)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"root": output, "registry": output / "source-registry.json", "tasks": output / "tasks.jsonl",
            "private": output / "private_validators", "receipt": output / "generation-receipt.json",
            "overlap": output / "protected-overlap-receipt.json"}


@dataclass
class _ReplayState:
    """One live sealed workspace continuation and its reconstructed controller state."""

    executor: WorkspaceToolExecutor
    detector: NoProgressDetector
    ledger: dict[str, Any]
    acquired: dict[str, Mapping[str, Any]]


def _assistant_serialization(message: Mapping[str, Any]) -> str:
    """Return the exact native assistant body, retaining function argument bytes."""

    serialized = serialize_pi_messages([message], append_assistant_header=False)
    prefix = "Assistant:\n"
    if not serialized.startswith(prefix):  # pragma: no cover - serializer contract
        raise ValueError("native assistant serialization is invalid")
    return serialized[len(prefix):]


def _validate_terminal_limits(bundle: Mapping[str, Any], terminal: Mapping[str, Any]) -> None:
    """Require the stored terminal to attest and remain within sealed limits."""

    base_fields = {"status", "turns", "elapsed_seconds", "messages", "actions", "metadata", "error"}
    corrective_fields = base_fields | {
        "schema", "failed_terminal_sha256", "correction_start_message_index", "prefix_sha256",
    }
    if set(terminal) not in (base_fields, corrective_fields):
        raise ValueError("terminal fields are not closed")
    limits = bundle.get("limits")
    metadata = terminal.get("metadata")
    if not isinstance(limits, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("terminal limits metadata is missing")
    turns, elapsed, actions, messages = terminal["turns"], terminal["elapsed_seconds"], terminal["actions"], terminal["messages"]
    if (isinstance(turns, bool) or not isinstance(turns, int) or turns < 0
            or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0
            or not isinstance(actions, list) or not isinstance(messages, list)):
        raise ValueError("terminal counts or elapsed time are invalid")
    assistant_turns = sum(isinstance(message, Mapping) and message.get("role") == "assistant" for message in messages)
    if (turns != assistant_turns or turns > limits["turns"] or len(actions) > limits["turns"]
            or elapsed > limits["seconds"] or metadata.get("configured_limits") != dict(limits)
            or metadata.get("turn_count") != turns or metadata.get("action_count") != len(actions)
            or metadata.get("elapsed_seconds") != elapsed):
        raise ValueError("terminal does not attest sealed turn/time/token/observation limits")
    usage = metadata.get("completion_usage")
    assistants = [message for message in messages if isinstance(message, Mapping) and message.get("role") == "assistant"]
    token_limit = limits["completion_tokens"]
    if not isinstance(usage, list) or len(usage) != len(assistants):
        raise ValueError("terminal completion usage does not cover every assistant turn")
    total_completion_tokens = 0
    for index, (entry, assistant) in enumerate(zip(usage, assistants)):
        if (not isinstance(entry, Mapping) or set(entry) != {"sequence", "completion_tokens"}
                or entry.get("sequence") != index or isinstance(entry.get("completion_tokens"), bool)
                or not isinstance(entry.get("completion_tokens"), int)
                or not 1 <= entry["completion_tokens"] <= token_limit):
            raise ValueError("terminal completion usage receipt is invalid")
        try:
            body = _assistant_serialization(assistant)
        except (ValueError, TypeError) as exc:
            raise ValueError("terminal assistant completion serialization is invalid") from exc
        if len(body.encode("utf-8")) > token_limit * 8:
            raise ValueError("terminal assistant body exceeds deterministic token byte ceiling")
        total_completion_tokens += entry["completion_tokens"]
    if metadata.get("completion_tokens") != total_completion_tokens:
        raise ValueError("terminal completion token total is invalid")
    mechanical = metadata.get("mechanical_suffix") is not None
    _validate_terminal_completion_usage(
        terminal, limits, runtime=bundle.get("runtime"),
        allow_mechanical_suffix=mechanical,
    )
    for index, action in enumerate(actions):
        observation = action.get("effective_observation") if isinstance(action, Mapping) else None
        if (not isinstance(observation, str) or len(observation.encode("utf-8")) > limits["output_bytes"]
                or action.get("completion_tokens") != usage[index]["completion_tokens"]):
            raise ValueError("terminal action observation or completion usage exceeds sealed limit")


def _replay_actions(
    bundle: Mapping[str, Any], fixture_root: Path, terminal: Mapping[str, Any], state: _ReplayState | None = None,
) -> _ReplayState:
    """Verify actions against one executor, returning its retained continuation state."""

    _validate_terminal_limits(bundle, terminal)
    actions = terminal.get("actions")
    messages = terminal.get("messages")
    completion_usage = terminal["metadata"]["completion_usage"]
    if not isinstance(actions, list) or not isinstance(messages, list):
        raise ValueError("terminal transcript/actions are required")
    if (len(messages) < 2 or not isinstance(messages[0], Mapping) or not isinstance(messages[1], Mapping)
            or set(messages[0]) != {"role", "content"} or set(messages[1]) != {"role", "content"}
            or messages[0].get("role") != "system" or messages[1].get("role") != "user"
            or not isinstance(messages[0].get("content"), str) or not isinstance(messages[1].get("content"), str)):
        raise ValueError("terminal must begin with exact system/user messages")
    terminated = terminal.get("status") == "no_progress_terminated"
    expected_message_count = 2 + len(actions) * 2 - 1 if terminated else 2 + len(actions) * 2 + 1
    if len(messages) != expected_message_count:
        raise ValueError("terminal has extra, missing, or reordered messages")
    if terminated and (not actions or not isinstance(actions[-1], Mapping)
                       or actions[-1].get("decision") != "terminate"):
        raise ValueError("no-progress terminal must end in a terminating action")
    if state is None:
        state = _ReplayState(
            WorkspaceToolExecutor(
                fixture_root,
                max_output_bytes=bundle["limits"]["output_bytes"],
                max_tree_bytes=bundle["limits"]["disk_bytes"],
            ),
            NoProgressDetector(),
            {},
            {},
        )
    intervention_count = 0
    for index, claimed in enumerate(actions):
        if (not isinstance(claimed, Mapping) or claimed.get("sequence") != index
                or not isinstance(claimed.get("tool_name"), str)
                or not isinstance(claimed.get("arguments"), Mapping)):
            raise ValueError("terminal action identity/order is invalid")
        assistant_index = 2 + index * 2
        assistant = messages[assistant_index]
        if (not isinstance(assistant, Mapping) or set(assistant) != {"role", "content", "tool_calls"}
                or assistant.get("role") != "assistant" or assistant.get("content") is not None):
            raise ValueError("terminal action assistant ordering mismatch")
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("terminal has invalid tool call serialization")
        call = calls[0]
        if (not isinstance(call, Mapping) or set(call) != {"id", "type", "function"}
                or call.get("type") != "function" or claimed.get("tool_call_id") != call.get("id")):
            raise ValueError("terminal tool_call_id linkage mismatch")
        function = call.get("function")
        if not isinstance(function, Mapping) or set(function) != {"name", "arguments"}:
            raise ValueError("terminal function metadata is invalid")
        raw_arguments = function.get("arguments")
        if (not isinstance(raw_arguments, str) or raw_arguments != claimed.get("arguments_json")
                or json.loads(raw_arguments) != claimed.get("arguments")):
            raise ValueError("terminal original function.arguments bytes do not bind receipt")
        tool_name = function.get("name")
        if tool_name != claimed.get("tool_name") or _assistant_serialization(assistant) != (
                f"Action: {tool_name}\nArguments: {raw_arguments}"):
            raise ValueError("terminal assistant serialization does not bind receipt")
        if index != len(actions) - 1 or not terminated:
            tool = messages[assistant_index + 1]
            is_intervention = isinstance(tool, Mapping) and tool.get("controller_intervention") is True
            expected_tool_fields = {"role", "tool_call_id", "content", "controller_intervention"} if is_intervention else {"role", "tool_call_id", "content"}
            if (not isinstance(tool, Mapping) or set(tool) != expected_tool_fields
                    or tool.get("role") != "tool" or tool.get("tool_call_id") != call.get("id")
                    or tool.get("content") != claimed.get("effective_observation")
                    or is_intervention != (claimed.get("decision") == "terminate")):
                raise ValueError("terminal model-facing effective observation linkage mismatch")
            intervention_count += int(is_intervention)
        execution = state.executor.execute(tool_name, claimed["arguments"])
        workspace_state = state.executor.workspace_state()
        raw_digest = sha256_json(execution.raw_observation)
        state.ledger.update(dict(execution.source_ledger))
        key = sha256_json({"tool": tool_name, "raw_observation": execution.raw_observation})
        state.acquired.setdefault(key, {"tool": tool_name, "raw_observation_sha256": raw_digest})
        observations = [state.acquired[key] for key in sorted(state.acquired)]
        decision = state.detector.observe(ActionProgressReceipt(tool_name, claimed["arguments"], workspace_state, state.ledger, observations))
        effective = decision.observation if decision.observation is not None else execution.effective_observation
        expected = {"raw_observation": execution.raw_observation, "raw_observation_sha256": raw_digest,
                    "effective_observation": effective, "observation_sha256": sha256_text(effective),
                    "is_error": execution.is_error, "workspace_state": workspace_state,
                    "source_ledger": state.ledger, "acquired_observations": observations,
                    "action_fingerprint": action_fingerprint(tool_name, claimed["arguments"]),
                    "progress_fingerprint": progress_fingerprint(workspace_state=workspace_state, source_ledger=state.ledger, acquired_observations=observations),
                    "completion_tokens": completion_usage[index]["completion_tokens"],
                    "decision": decision.kind}
        for field, value in expected.items():
            observed = list(claimed.get(field, ())) if field == "acquired_observations" else claimed.get(field)
            if observed != value:
                raise ValueError(f"terminal action {index} replay mismatch: {field}")
    if not terminated:
        final = messages[-1]
        if (not isinstance(final, Mapping) or set(final) != {"role", "content"}
                or final.get("role") != "assistant" or not isinstance(final.get("content"), str)
                or not final["content"].startswith("Final:") or intervention_count > 1):
            raise ValueError("terminal final ordering mismatch")
    return state


def replay_corrective_suffix_from_failure(
    bundle: Mapping[str, Any], failed_terminal: Mapping[str, Any], fixture_root: Path,
    proposed_actions: list[Mapping[str, Any]], final_text: str, final_completion_tokens: int,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Continue exactly one verified failure executor through a corrective suffix."""

    if (failed_terminal.get("status") != "no_progress_terminated" or not isinstance(final_text, str)
            or not final_text.startswith("Final:") or isinstance(final_completion_tokens, bool)
            or not isinstance(final_completion_tokens, int)):
        raise ValueError("sealed failed terminal and Final suffix are required")
    state = _replay_actions(bundle, fixture_root, failed_terminal)
    failed_actions = failed_terminal["actions"]
    limits = bundle["limits"]
    if len(failed_actions) + len(proposed_actions) + 1 > limits["turns"]:
        state.executor.close()
        raise ValueError("corrective replay would exceed combined sealed turn limit")
    remaining_seconds = limits["seconds"] - failed_terminal["elapsed_seconds"]
    if remaining_seconds < 0:
        state.executor.close()
        raise ValueError("failed terminal exceeds sealed deadline")
    started = clock()
    deadline = started + remaining_seconds
    messages = [dict(message) for message in failed_terminal["messages"]]
    try:
        # The final terminating receipt is authentic but deliberately was not model-facing.
        intervention = failed_actions[-1]
        messages.append({"role": "tool", "tool_call_id": intervention["tool_call_id"],
                         "content": intervention["effective_observation"], "controller_intervention": True})
        correction_start = len(messages)
        combined_actions = [dict(action) for action in failed_actions]
        completion_usage = [dict(item) for item in failed_terminal["metadata"]["completion_usage"]]
        for sequence, proposed in enumerate(proposed_actions, start=len(combined_actions)):
            if clock() >= deadline:
                raise ValueError("corrective replay exceeded one sealed bundle deadline")
            if set(proposed) != {"tool_name", "arguments", "arguments_json", "completion_tokens"}:
                raise ValueError("proposed corrective action fields are invalid")
            name, arguments, raw_arguments, completion_tokens = (
                proposed["tool_name"], proposed["arguments"], proposed["arguments_json"], proposed["completion_tokens"]
            )
            if (not isinstance(raw_arguments, str) or json.loads(raw_arguments) != arguments
                    or not isinstance(name, str) or isinstance(completion_tokens, bool)
                    or not isinstance(completion_tokens, int)
                    or not 1 <= completion_tokens <= limits["completion_tokens"]):
                raise ValueError("proposed arguments or completion_tokens are invalid")
            call_id = f"correction-{sequence}"
            assistant = {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": raw_arguments}}]}
            assistant_body = _assistant_serialization(assistant)
            if (assistant_body != f"Action: {name}\nArguments: {raw_arguments}"
                    or len(assistant_body.encode("utf-8")) > limits["completion_tokens"] * 8):
                raise ValueError("corrective assistant serialization or body limit is invalid")
            messages.append(assistant)
            execution = state.executor.execute(name, arguments)
            workspace = state.executor.workspace_state()
            raw_digest = sha256_json(execution.raw_observation)
            state.ledger.update(dict(execution.source_ledger))
            key = sha256_json({"tool": name, "raw_observation": execution.raw_observation})
            state.acquired.setdefault(key, {"tool": name, "raw_observation_sha256": raw_digest})
            observations = [state.acquired[key] for key in sorted(state.acquired)]
            decision = state.detector.observe(ActionProgressReceipt(name, arguments, workspace, state.ledger, observations))
            if decision.kind == "terminate":
                raise ValueError("corrective suffix entered no-progress termination")
            effective = decision.observation if decision.observation is not None else execution.effective_observation
            if not isinstance(effective, str) or len(effective.encode("utf-8")) > limits["output_bytes"]:
                raise ValueError("corrective replay effective observation exceeds sealed output limit")
            action = {"sequence": sequence, "tool_call_id": call_id, "tool_name": name,
                      "arguments": dict(arguments), "arguments_json": raw_arguments,
                      "raw_observation": execution.raw_observation, "raw_observation_sha256": raw_digest,
                      "effective_observation": effective, "observation_sha256": sha256_text(effective),
                      "is_error": execution.is_error, "workspace_state": workspace,
                      "source_ledger": dict(state.ledger), "acquired_observations": observations,
                      "action_fingerprint": action_fingerprint(name, arguments),
                      "progress_fingerprint": progress_fingerprint(workspace_state=workspace, source_ledger=state.ledger, acquired_observations=observations),
                      "completion_tokens": completion_tokens, "decision": decision.kind}
            combined_actions.append(action)
            completion_usage.append({"sequence": len(completion_usage), "completion_tokens": completion_tokens})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": effective})
        if clock() > deadline:
            raise ValueError("corrective replay exceeded one sealed bundle deadline")
        if (not 1 <= final_completion_tokens <= limits["completion_tokens"]
                or len(final_text.encode("utf-8")) > limits["completion_tokens"] * 8):
            raise ValueError("corrective final completion_tokens or body exceeds sealed limit")
        messages.append({"role": "assistant", "content": final_text})
        completion_usage.append({"sequence": len(completion_usage), "completion_tokens": final_completion_tokens})
        prefix = messages[:correction_start]
        elapsed = failed_terminal["elapsed_seconds"] + (clock() - started)
        metadata = dict(failed_terminal["metadata"])
        metadata.update({
            "turn_count": len(combined_actions) + 1,
            "action_count": len(combined_actions),
            "elapsed_seconds": elapsed,
            "completion_usage": completion_usage,
            "completion_tokens": sum(item["completion_tokens"] for item in completion_usage),
            "mechanical_suffix": {
                "schema": "emender-e97-mechanical-cpu-system-gate-v1",
                "scope": "cpu-system-gate-mechanical",
                "training_eligible": False,
            },
        })
        return {"schema": "emender-e97-corrective-terminal-v2", "status": "success",
                "failed_terminal_sha256": sha256_json(failed_terminal),
                "correction_start_message_index": correction_start, "prefix_sha256": sha256_json(prefix),
                "turns": len(combined_actions) + 1, "elapsed_seconds": elapsed,
                "messages": messages, "actions": combined_actions, "metadata": metadata, "error": None}
    finally:
        state.executor.close()

class _PinnedValidatorArgv(list[str]):
    """Physical validator argv backed by verified inherited executable fds."""

    def __init__(self, interpreter_fd: int, program_fd: int, mode: str) -> None:
        super().__init__([
            f"/proc/self/fd/{interpreter_fd}", f"/proc/self/fd/{program_fd}", "--mode", mode,
        ])
        self.interpreter_fd = interpreter_fd
        self.program_fd = program_fd

    def close(self) -> None:
        """Close each owned descriptor exactly once, including failure paths."""

        for name in ("program_fd", "interpreter_fd"):
            descriptor = getattr(self, name)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, name, -1)


def _open_private_validator_payload(payload: bytes, *, label: str, mode: int) -> int:
    """Pin one retained validator input in an unlinked inherited descriptor."""

    if not isinstance(payload, bytes):
        raise ValueError(f"validator {label} payload is invalid")
    try:
        descriptor = os.memfd_create(f"e97-private-validator-{label}", os.MFD_CLOEXEC)
    except (AttributeError, OSError):
        descriptor, path = tempfile.mkstemp(prefix=f"e97-private-validator-{label}-")
        try:
            os.unlink(path)
        except BaseException:
            os.close(descriptor)
            raise
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:  # pragma: no cover - regular-file write contract
                raise OSError(f"validator private {label} write failed")
            view = view[count:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_private_validator_program(payload: bytes) -> int:
    """Pin verified program bytes in an unlinked private executable descriptor."""

    return _open_private_validator_payload(payload, label="program", mode=0o500)


@dataclass
class _PinnedValidatorInputs:
    """The exact spec/terminal bytes inherited by one validator child."""

    spec_fd: int
    terminal_fd: int
    spec_sha256: str
    terminal_sha256: str

    @classmethod
    def from_paths(cls, spec_path: Path, terminal_path: Path) -> "_PinnedValidatorInputs":
        spec_payload = _read_snapshot_file(spec_path, name="validator spec", maximum=1 << 20)
        terminal_payload = _read_snapshot_file(terminal_path, name="validator terminal", maximum=16 << 20)
        spec_fd = _open_private_validator_payload(spec_payload, label="spec", mode=0o400)
        try:
            terminal_fd = _open_private_validator_payload(terminal_payload, label="terminal", mode=0o400)
        except BaseException:
            os.close(spec_fd)
            raise
        return cls(
            spec_fd, terminal_fd, _snapshot_digest(spec_payload), _snapshot_digest(terminal_payload))

    def close(self) -> None:
        for name in ("terminal_fd", "spec_fd"):
            descriptor = getattr(self, name)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, name, -1)


def _hash_and_rewind(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _open_verified_interpreter(expected_sha256: str) -> int:
    """Pin the runtime executable by fd, hash it, and retain it for exec."""

    try:
        # Resolving a virtualenv launcher is acceptable only to select a final
        # candidate: execution is through the no-follow descriptor below.
        candidate = Path(sys.executable).resolve(strict=True)
        descriptor = os.open(candidate, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError("validator interpreter cannot be opened safely") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("validator interpreter is not a regular file")
        if _hash_and_rewind(descriptor) != expected_sha256:
            raise ValueError("validator program/interpreter identity mismatch")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _archived_validator_payload() -> tuple[bytes, str]:
    """Read the canonical archived validator member from one source snapshot."""

    manifest_payload = _read_snapshot_file(
        GENERATOR_MANIFEST, name="validator source manifest")
    archive_payload = _read_snapshot_file(
        GENERATOR_SOURCE_ARCHIVE, name="validator source archive", maximum=256 << 20)
    return verified_source_member_payload(
        manifest_payload, archive_payload, _VALIDATOR_SOURCE_MEMBER)


def _loaded_replay_digest_from_archive(manifest_payload: bytes, archive_payload: bytes) -> str:
    """Hash the two authoritative replay roots from one retained archive."""

    roots = (
        "ndm.e97_first_party_read_observe",
        "ndm.e97_acquisition_controller",
    )
    digests = [verified_loaded_module_closure_sha256(
        manifest_payload, archive_payload, root_module=root_module,
    ) for root_module in roots]
    return sha256_text(canonical_json({"roots": list(roots), "digests": digests}))


def _verified_loaded_replay_digest(expected_archive_sha256: str) -> str:
    """Bind every loaded replay implementation root to retained archive code.

    Replay executes both the workspace controller and this module's replay
    functions.  Checking only the former would leave an in-memory replacement
    of ``_replay_actions`` outside the sealed loaded-code closure.
    """

    manifest_payload = _read_snapshot_file(
        GENERATOR_MANIFEST, name="replay source manifest")
    archive_payload = _read_snapshot_file(
        GENERATOR_SOURCE_ARCHIVE, name="replay source archive", maximum=256 << 20)
    if _snapshot_digest(archive_payload) != expected_archive_sha256:
        raise ValueError("replay source archive does not match the task bundle")
    return _loaded_replay_digest_from_archive(manifest_payload, archive_payload)


# Backward-compatible private spelling used by focused callers/tests.  The
# returned identity now covers replay itself as well as its controller.
def _verified_loaded_controller_digest(expected_archive_sha256: str) -> str:
    return _verified_loaded_replay_digest(expected_archive_sha256)


def _validator_execution_argv(
    bundle: Mapping[str, Any], spec: Mapping[str, Any], *, mode: str,
    validator_program: Path | None = None, trusted_validator_sha256: str | None = None,
) -> tuple[list[str], _PinnedValidatorArgv]:
    """Resolve logical IDs to retained self-contained validator bytes and fds.

    Normal replay always executes the canonical source-archive member.  The
    explicit path/digest pair is retained for isolated private test injection;
    it is snapshotted once and still executes only its inherited descriptor.
    """

    logical = bundle["validator"].get(f"{mode}_argv") if isinstance(bundle.get("validator"), Mapping) else None
    expected_logical = validator_logical_argv(mode)
    if logical != expected_logical:
        raise ValueError("validator argv is not the sealed logical validator identity")
    if validator_program is None:
        program_payload, expected_program_sha256 = _archived_validator_payload()
    else:
        if trusted_validator_sha256 is None:
            raise ValueError("private validator injection requires a trusted digest")
        program_payload = _read_snapshot_file(Path(validator_program), name="trusted validator program")
        expected_program_sha256 = trusted_validator_sha256
    if (not isinstance(expected_program_sha256, str) or len(expected_program_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_program_sha256)):
        raise ValueError("trusted validator digest is invalid")
    if spec.get("program_sha256") != expected_program_sha256 or _snapshot_digest(program_payload) != expected_program_sha256:
        raise ValueError("validator program/interpreter identity mismatch")
    interpreter_fd = _open_verified_interpreter(str(spec.get("interpreter_sha256", "")))
    try:
        program_fd = _open_private_validator_program(program_payload)
        return expected_logical, _PinnedValidatorArgv(interpreter_fd, program_fd, mode)
    except BaseException:
        os.close(interpreter_fd)
        raise


def _run_validator(
    logical_argv: list[str], execution_argv: _PinnedValidatorArgv, spec_path: Path, terminal_path: Path,
    *, mode: str, limit: int, timeout: int, processes: int, expected_action_count: int,
) -> dict[str, Any]:
    """Run physical argv while recording only its stable logical identity."""

    import resource

    logical = logical_argv + [
        "--spec-fd", "<inherited-spec-fd>",
        "--terminal-fd", "<inherited-terminal-fd>",
    ]

    def limit_process() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(timeout)), max(1, int(timeout))))
        resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))

    process: subprocess.Popen[bytes] | None = None
    inputs: _PinnedValidatorInputs | None = None
    try:
        # Snapshot both evidence payloads before constructing child argv.  The
        # child receives only these unlinked descriptors, never their mutable
        # caller pathnames, and the returned receipt records their exact bytes.
        inputs = _PinnedValidatorInputs.from_paths(spec_path, terminal_path)
        with tempfile.NamedTemporaryFile(mode="w+b") as stdout, tempfile.NamedTemporaryFile(mode="w+b") as stderr:
            # All executable and evidence paths are inherited descriptors.  The
            # child never opens a validator/spec/terminal pathname, and its
            # self-contained program imports only the standard library.
            actual_argv = [
                *execution_argv,
                "--spec-fd", str(inputs.spec_fd),
                "--terminal-fd", str(inputs.terminal_fd),
            ]
            try:
                process = subprocess.Popen(
                    actual_argv,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
                    preexec_fn=limit_process,
                    start_new_session=True,
                    pass_fds=(
                        execution_argv.interpreter_fd, execution_argv.program_fd,
                        inputs.spec_fd, inputs.terminal_fd,
                    ),
                )
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise ValueError("validator timed out") from exc
            finally:
                if process is not None and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            stdout.seek(0)
            stderr.seek(0)
            out, err = stdout.read(limit + 1), stderr.read(limit + 1)
    finally:
        if inputs is not None:
            inputs.close()
        execution_argv.close()
    if len(out) > limit or len(err) > limit or code != 0:
        raise ValueError("validator failed")
    try:
        decoded = json.loads(out.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("validator stdout is not exact JSON") from exc
    expected_fields = {"mode", "status", "action_count"}
    if (not isinstance(decoded, Mapping) or set(decoded) != expected_fields
            or decoded.get("mode") != mode
            or decoded.get("status") != "pass" or isinstance(decoded.get("action_count"), bool)
            or decoded.get("action_count") != expected_action_count
            or out != (canonical_json(dict(decoded)) + "\n").encode("utf-8")):
        raise ValueError("validator did not emit its exact pass attestation")
    bound_argv = [
        "@verified-interpreter-fd", "@private-verified-validator", "--mode", mode,
        "--spec-fd", "<inherited-spec-fd>",
        "--terminal-fd", "<inherited-terminal-fd>",
    ]
    return {
        "logical_argv": logical,
        "logical_argv_sha256": sha256_json(logical),
        "bound_argv": bound_argv,
        "bound_argv_sha256": sha256_json(bound_argv),
        "stdout_sha256": sha256_text(out.decode()),
        "stderr_sha256": sha256_text(err.decode()),
        "spec_payload_sha256": inputs.spec_sha256,
        "terminal_payload_sha256": inputs.terminal_sha256,
        "output": dict(decoded),
    }


def validate_replay(
    bundle: Mapping[str, Any], spec: Mapping[str, Any], fixture_root: Path, terminal: Mapping[str, Any], *,
    runtime_schema_digest: str, validator_program: Path | None = None,
    trusted_validator_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay a full terminal and run its exact bounded focused/regression validators."""

    expanded_bytes = sum(path.stat().st_size for path in fixture_root.rglob("*") if path.is_file())
    if (expanded_bytes > bundle["limits"]["disk_bytes"]
            or _tree_digest(fixture_root) != bundle["task"]["fixture_tree_digest"]
            or spec.get("task_identity") != bundle["task"]["identity"]):
        raise ValueError("fixture/task/disk binding mismatch")
    metadata = terminal.get("metadata")
    loaded_controller_digest = _verified_loaded_controller_digest(
        bundle["task"]["generator_source_digest"])
    if (not isinstance(metadata, Mapping)
            or metadata.get("controller_build_sha256") != bundle["runtime"]["controller_digest"]
            or loaded_controller_digest != bundle["runtime"]["controller_digest"]
            or metadata.get("tool_schema_sha256") != bundle["runtime"]["tool_schema_digest"]
            or metadata.get("system_prompt_sha256") != bundle["runtime"]["system_prompt_sha256"]
            or metadata.get("configured_limits") != bundle["limits"]
            or runtime_schema_digest != bundle["runtime"]["schema_digest"]):
        raise ValueError("task/runtime/limits binding mismatch")
    validator_argvs: dict[str, tuple[list[str], _PinnedValidatorArgv]] = {}
    try:
        for mode in ("focused", "regression"):
            validator_argvs[mode] = _validator_execution_argv(
                bundle,
                spec,
                mode=mode,
                validator_program=validator_program,
                trusted_validator_sha256=trusted_validator_sha256,
            )
        state = _replay_actions(bundle, fixture_root, terminal)
        try:
            with tempfile.TemporaryDirectory(prefix="e97-validator-") as temporary:
                root = Path(temporary)
                spec_path, terminal_path = root / "spec.json", root / "terminal.json"
                spec_payload = (canonical_json(dict(spec)) + "\n").encode("utf-8")
                terminal_payload = (canonical_json(dict(terminal)) + "\n").encode("utf-8")
                spec_path.write_bytes(spec_payload)
                terminal_path.write_bytes(terminal_payload)
                results = {}
                for name in ("focused", "regression"):
                    logical_argv, execution_argv = validator_argvs[name]
                    result = _run_validator(
                        logical_argv,
                        execution_argv,
                        spec_path,
                        terminal_path,
                        mode=name,
                        limit=bundle["limits"]["output_bytes"],
                        timeout=min(30, int(bundle["limits"]["seconds"])),
                        processes=bundle["limits"]["processes"],
                        expected_action_count=len(terminal["actions"]),
                    )
                    if (result["spec_payload_sha256"] != _snapshot_digest(spec_payload)
                            or result["terminal_payload_sha256"] != _snapshot_digest(terminal_payload)):
                        raise ValueError("validator did not attest its inherited evidence payloads")
                    results[name] = result
        finally:
            state.executor.close()
    finally:
        # ``_run_validator`` also closes its argument, but this encompassing
        # idempotent cleanup covers partial construction, replay, temp-file,
        # and focused-validator failures before regression can start.
        for _logical, execution_argv in validator_argvs.values():
            execution_argv.close()
    return {
        "schema": "emender-e97-first-party-validator-receipt-v4",
        "status": "pass",
        "task_identity": bundle["task"]["identity"],
        "bundle_sha256": sha256_json(bundle),
        "fixture_tree_digest": bundle["task"]["fixture_tree_digest"],
        "archive_expanded_bytes": expanded_bytes,
        "configured_limits": dict(bundle["limits"]),
        "terminal_sha256": sha256_json(terminal),
        "validator_spec_digest": sha256_json(spec),
        "validator_spec_payload_sha256": _snapshot_digest(spec_payload),
        "validator_terminal_payload_sha256": _snapshot_digest(terminal_payload),
        "runtime_schema_digest": runtime_schema_digest,
        "validators": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--seed", required=True); parser.add_argument("--registry", type=Path, required=True); parser.add_argument("--registry-sha256", required=True); parser.add_argument("--policy-sha256", required=True)
    args = parser.parse_args(); print(json.dumps({k: str(v) for k, v in generate(args.output, seed=args.seed, registry_path=args.registry, registry_sha256=args.registry_sha256, policy_sha256=args.policy_sha256).items()}, sort_keys=True))

if __name__ == "__main__": main()
