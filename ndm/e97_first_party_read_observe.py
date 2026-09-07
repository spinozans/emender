"""Sealed deterministic first-party read-observe fixture generation and replay."""
from __future__ import annotations

import argparse
import hashlib
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
from typing import Any, Mapping

from ndm.e97_acquisition_controller import ACQUISITION_CONTROLLER_SCHEMA, READ_OBSERVE_TOOLS, WorkspaceToolExecutor
from ndm.e97_atomic import publish_directory_no_replace
from ndm.e97_onpolicy_records import ActionProgressReceipt, NoProgressDetector, action_fingerprint, progress_fingerprint
from ndm.e97_onpolicy_records import canonical_json, sha256_json, sha256_text, task_identity
from ndm.e97_agent_protocol import serialize_pi_messages
from ndm.e97_task_lake import TASK_BUNDLE_SCHEMA, canonical_intent_digest, validate_task_collection, validate_source_registry
from ndm.e97_first_party_source_archive import verify_archive_members, verify_source_archive
from ndm.e97_protected_overlap import validate_overlap_authorization, validate_overlap_receipt

ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_REGISTRY = ROOT / "configs/pi/e97-onpolicy-source-registry-v1.json"
GENERATOR_MANIFEST = ROOT / "configs/pi/e97-firstparty-generator-manifest-v1.json"
GENERATOR_SOURCE_ARCHIVE = ROOT / "configs/pi/e97-firstparty-source-v1.tar"
ENVIRONMENT_DESCRIPTOR = ROOT / "configs/pi/e97-firstparty-cpu-environment-v1.json"
VALIDATOR_PROGRAM = ROOT / "scripts/e97_first_party_validator.py"
VALIDATOR_LOGICAL_RUNTIME = "@runtime-python"
VALIDATOR_LOGICAL_PROGRAM = "@generator-source/scripts/e97_first_party_validator.py"
_REQUIRED_PANELS = {
    "ef481c637fde5916b8b0fe1f80cc2b4f0a6b88262088cbb33aefe0fed6bd6d09",
    "8d0d5d350a39c9f5c007d98e4a0010f4a06f39e5a7fb89ecd9ccc4e37e66b8d8",
    "939bcd66768884a1e5ec44bcf11fdf5d09602ebdabe86d4caefffd4ace8dbb60",
}


def _sha_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validator_logical_argv(mode: str) -> list[str]:
    if mode not in {"focused", "regression"}:
        raise ValueError("validator mode is invalid")
    return [VALIDATOR_LOGICAL_RUNTIME, VALIDATOR_LOGICAL_PROGRAM, "--mode", mode]


def _digest(seed: str, domain: str) -> str:
    return hashlib.sha256(("e97-first-party-read-observe-v1\0" + seed + "\0" + domain).encode()).hexdigest()


def _tree_digest(root: Path) -> str:
    return sha256_json([{"path": p.relative_to(root).as_posix(), "sha256": _sha_path(p)} for p in sorted(root.rglob("*")) if p.is_file()])


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


def safe_extract_fixture_archive(
    archive: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_tree_digest: str,
    disk_limit: int,
) -> int:
    """Extract only regular relative fixture members and verify the expanded tree."""

    if _sha_path(archive) != expected_sha256:
        raise ValueError("fixture archive SHA-256 does not match sealed bundle")
    if disk_limit <= 0:
        raise ValueError("fixture disk limit is invalid")
    expanded = 0
    names: set[str] = set()
    try:
        with tarfile.open(archive, "r:") as stream:
            for member in stream.getmembers():
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


def _verify_manifest() -> tuple[str, str]:
    """Verify both source component metadata and its immutable byte archive."""

    manifest_sha256 = _sha_path(GENERATOR_MANIFEST)
    source_archive_sha256 = verify_source_archive(
        GENERATOR_MANIFEST,
        GENERATOR_SOURCE_ARCHIVE,
        checkout_root=ROOT,
    )
    return manifest_sha256, source_archive_sha256


def _verify_environment() -> None:
    value = json.loads(ENVIRONMENT_DESCRIPTOR.read_text())
    runtime = value.get("runtime") if isinstance(value, Mapping) else None
    observed = {"interpreter_sha256": _sha_path(Path(sys.executable)), "implementation": platform.python_implementation(),
                "python_version": platform.python_version(), "platform_system": platform.system(), "machine": platform.machine()}
    if value.get("schema") != "emender-e97-firstparty-cpu-environment-v2" or value.get("purpose") is None or runtime != observed:
        raise ValueError("CPU environment receipt does not match current runtime")


def _verify_overlap(registry: Mapping[str, Any]) -> None:
    """Verify static checker authorization, never a fabricated semantic pass."""

    value = json.loads((ROOT / "configs/pi/e97-firstparty-overlap-firewall-audit-v1.json").read_text())
    validate_overlap_authorization(value)


def _checked_registry(path: Path, digest: str, policy_sha256: str) -> dict[str, Any]:
    if path.resolve() != CHECKED_IN_REGISTRY.resolve():
        raise ValueError("registry path must be the checked-in registry")
    if _sha_path(path) != digest:
        raise ValueError("registry SHA-256 does not match checked-in bytes")
    registry = validate_source_registry(json.loads(path.read_text()))
    if registry["policy_sha256"] != policy_sha256:
        raise ValueError("policy SHA-256 does not match checked-in registry")
    if not _REQUIRED_PANELS.issubset({item["manifest_sha256"] for item in registry["protected_evaluation"]}):
        raise ValueError("registry is missing required protected panels")
    return registry


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def generate(output: Path, *, seed: str, registry_path: Path = CHECKED_IN_REGISTRY,
             registry_sha256: str | None = None, policy_sha256: str | None = None) -> dict[str, Path]:
    """Create a sealed CPU-only task authority, publishing only by atomic rename."""
    if registry_sha256 is None or policy_sha256 is None:
        raise ValueError("checked-in registry SHA-256 and policy SHA-256 are required")
    registry = _checked_registry(registry_path, registry_sha256, policy_sha256)
    generator_manifest_sha256, generator_source_archive_sha256 = _verify_manifest()
    _verify_environment()
    _verify_overlap(registry)
    admitted = {item["id"]: item for item in registry["sources"] if item["status"] == "admitted" and item["kind"] == "first-party"}
    if not {"e97-firstparty-train", "e97-firstparty-development"}.issubset(admitted):
        raise ValueError("checked-in registry lacks pre-admitted first-party split sources")
    receipt_artifacts = {
        "source_archive_sha256": GENERATOR_SOURCE_ARCHIVE,
        "license_sha256": ROOT / "configs/pi/e97-firstparty-authorization-license-v1.json",
        "environment_sha256": ENVIRONMENT_DESCRIPTOR,
        "overlap_sha256": ROOT / "configs/pi/e97-firstparty-overlap-firewall-audit-v1.json",
    }
    for source in admitted.values():
        for name, artifact in receipt_artifacts.items():
            if source["receipts"][name] != _sha_path(artifact):
                raise ValueError("checked-in first-party registry receipt does not hash its artifact")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=parent))
    try:
        fixtures, private, archives = stage / "fixtures", stage / "private_validators", stage / "archives"
        fixtures.mkdir(); private.mkdir(); archives.mkdir()
        shutil.copyfile(registry_path, stage / "source-registry.json")
        shutil.copyfile(GENERATOR_MANIFEST, stage / "generator-manifest.json")
        shutil.copyfile(GENERATOR_SOURCE_ARCHIVE, stage / "source-archive.tar")
        runtime = {"schema_digest": sha256_text(ACQUISITION_CONTROLLER_SCHEMA), "tool_schema_digest": sha256_json(READ_OBSERVE_TOOLS),
                   "controller_digest": _sha_path(ROOT / "ndm/e97_acquisition_controller.py"),
                   "sandbox_image_digest": _sha_path(ENVIRONMENT_DESCRIPTOR),
                   "system_prompt_sha256": sha256_text("Use only the provided read-observe tools and ground the final in observations.")}
        validator_program_digest = _sha_path(VALIDATOR_PROGRAM)
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
                archive = archives / f"{identity}.tar"; _archive(root, archive)
                spec = {"schema": "emender-e97-first-party-validator-v1", "task_identity": identity, "fixture_tree_digest": tree,
                        "archive_sha256": _sha_path(archive), "expected_token": token, "required_read_path": relative,
                        "program_sha256": validator_program_digest, "interpreter_sha256": _sha_path(Path(sys.executable)),
                        "minefield": {"allowed_tools": ["list_files", "read"], "forbidden_paths": ["/", ".."]}}
                _write_json(private / f"{identity}.json", spec)
                source = admitted[f"e97-firstparty-{split}"]
                tasks.append({"schema": TASK_BUNDLE_SCHEMA, "split": split,
                    "task": {"namespace": namespace, "family_id": family, "identity": identity, "generator_source_digest": generator_source_archive_sha256, "fixture_tree_digest": tree, "intent_digest": intent, "prompt": prompt, "difficulty": 1 if kind == "direct" else 2},
                    "source": {"registry_id": source["id"], "kind": "first-party", "repository": f"firstparty/{split}-read-observe", "revision": source["revision"], "source_record_digest": sha256_json({"source": source["id"], "family": family, "generator": generator_source_archive_sha256}), "license_receipt_digest": source["receipts"]["license_sha256"]},
                    "fixture": {"artifact_path": f"archives/{archive.name}", "artifact_bytes": archive.stat().st_size, "artifact_sha256": spec["archive_sha256"], "tree_digest": tree}, "runtime": runtime,
                    "limits": {"turns": 12, "seconds": 60, "completion_tokens": 512, "output_bytes": 16384, "disk_bytes": 1 << 20, "processes": 1},
                    "validator": {"spec_digest": sha256_json(spec), "focused_argv": validator_logical_argv("focused"), "regression_argv": validator_logical_argv("regression"), "milestone_digest": sha256_text("required-read\0" + relative), "minefield_digest": sha256_json(spec["minefield"])}})
        tasks = validate_task_collection(tasks, registry=registry)
        task_path = stage / "tasks.jsonl"; task_path.write_text("".join(canonical_json(task) + "\n" for task in sorted(tasks, key=lambda x: x["task"]["identity"])))
        archive_root_sha256 = _archive_root_sha256(tasks, stage)
        receipt = {
            "schema": "emender-e97-first-party-generation-receipt-v3",
            "state": "quarantined-pending-protected-overlap",
            "seed_sha256": sha256_text(seed),
            "generator_component_manifest_sha256": generator_manifest_sha256,
            "generator_source_archive_sha256": generator_source_archive_sha256,
            "tasks_sha256": _sha_path(task_path),
            "archive_root_sha256": archive_root_sha256,
            "registry_sha256": registry_sha256,
            "policy_sha256": policy_sha256,
            "registry_copy_sha256": _sha_path(stage / "source-registry.json"),
            "controller_source_sha256": runtime["controller_digest"],
            "environment_descriptor_sha256": runtime["sandbox_image_digest"],
            "protected_manifest_sha256s": sorted(_REQUIRED_PANELS),
        }
        _write_json(stage / "generation-receipt.json", receipt)
        _write_json(stage / "authority-state.json", {
            "schema": "emender-e97-first-party-authority-state-v1",
            "state": "quarantined-pending-protected-overlap",
            "generation_receipt_sha256": _sha_path(stage / "generation-receipt.json"),
        })
        _fsync_tree(stage)
        publish_directory_no_replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"root": output, "registry": output / "source-registry.json", "tasks": output / "tasks.jsonl", "private": output / "private_validators", "receipt": output / "generation-receipt.json"}


def validate_generated_quarantine(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate an unadmitted generated collection before its private gate runs."""

    try:
        registry = validate_source_registry(json.loads((root / "source-registry.json").read_text()))
        tasks = [json.loads(line) for line in (root / "tasks.jsonl").read_text().splitlines() if line.strip()]
        tasks = validate_task_collection(tasks, registry=registry)
        receipt = json.loads((root / "generation-receipt.json").read_text())
        state = json.loads((root / "authority-state.json").read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("generated quarantine is malformed") from exc
    expected = {
        "schema", "state", "seed_sha256", "generator_component_manifest_sha256",
        "generator_source_archive_sha256", "tasks_sha256", "archive_root_sha256", "registry_sha256",
        "policy_sha256", "registry_copy_sha256", "controller_source_sha256",
        "environment_descriptor_sha256", "protected_manifest_sha256s",
    }
    if set(receipt) != expected or receipt["schema"] != "emender-e97-first-party-generation-receipt-v3" or receipt["state"] != "quarantined-pending-protected-overlap":
        raise ValueError("generated quarantine receipt schema/state is invalid")
    for field in expected - {"schema", "state", "protected_manifest_sha256s"}:
        if not isinstance(receipt[field], str) or len(receipt[field]) != 64:
            raise ValueError("generated quarantine receipt digest is invalid")
    verify_archive_members(root / "generator-manifest.json", root / "source-archive.tar")
    if (receipt["tasks_sha256"] != _sha_path(root / "tasks.jsonl")
            or receipt["archive_root_sha256"] != _archive_root_sha256(tasks, root)
            or receipt["registry_copy_sha256"] != _sha_path(root / "source-registry.json")
            or receipt["registry_sha256"] != _sha_path(root / "source-registry.json")
            or receipt["generator_component_manifest_sha256"] != _sha_path(root / "generator-manifest.json")
            or receipt["generator_source_archive_sha256"] != _sha_path(root / "source-archive.tar")
            or receipt["protected_manifest_sha256s"] != sorted(item["manifest_sha256"] for item in registry["protected_evaluation"])):
        raise ValueError("generated quarantine receipt binding mismatch")
    if (not isinstance(state, Mapping) or state != {
            "schema": "emender-e97-first-party-authority-state-v1",
            "state": "quarantined-pending-protected-overlap",
            "generation_receipt_sha256": _sha_path(root / "generation-receipt.json"),
    }):
        raise ValueError("generated authority is not an immutable quarantine")
    return registry, tasks, receipt


def admit_generated_collection(quarantine: Path, overlap_receipt: Path, output: Path) -> dict[str, Path]:
    """Copy a quarantine into a new admitted authority only after a bound pass receipt."""

    registry, _, generation = validate_generated_quarantine(quarantine)
    try:
        overlap = json.loads(overlap_receipt.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("protected overlap receipt is unreadable") from exc
    if generation["protected_manifest_sha256s"] != sorted(_REQUIRED_PANELS):
        raise ValueError("generated quarantine does not bind the fixed protected panel set")
    validate_overlap_receipt(
        overlap,
        candidate_collection_sha256=generation["tasks_sha256"],
        candidate_archive_root_sha256=generation["archive_root_sha256"],
    )
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=parent))
    try:
        shutil.copytree(quarantine, stage, dirs_exist_ok=True)
        shutil.copyfile(overlap_receipt, stage / "protected-overlap-receipt.json")
        _write_json(stage / "authority-state.json", {
            "schema": "emender-e97-first-party-authority-state-v1",
            "state": "admitted",
            "generation_receipt_sha256": _sha_path(stage / "generation-receipt.json"),
            "protected_overlap_receipt_sha256": _sha_path(stage / "protected-overlap-receipt.json"),
        })
        _write_json(stage / "admission-receipt.json", {
            "schema": "emender-e97-first-party-admission-receipt-v1",
            "registry_sha256": _sha_path(stage / "source-registry.json"),
            "generation_receipt_sha256": _sha_path(stage / "generation-receipt.json"),
            "protected_overlap_receipt_sha256": _sha_path(stage / "protected-overlap-receipt.json"),
            "tasks_sha256": generation["tasks_sha256"],
            "archive_root_sha256": generation["archive_root_sha256"],
            "protected_manifest_sha256s": generation["protected_manifest_sha256s"],
        })
        _fsync_tree(stage)
        publish_directory_no_replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "root": output,
        "registry": output / "source-registry.json",
        "tasks": output / "tasks.jsonl",
        "private": output / "private_validators",
        "receipt": output / "generation-receipt.json",
        "overlap": output / "protected-overlap-receipt.json",
    }


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
    started = time.monotonic()
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
            if time.monotonic() >= deadline:
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
        if time.monotonic() > deadline:
            raise ValueError("corrective replay exceeded one sealed bundle deadline")
        if (not 1 <= final_completion_tokens <= limits["completion_tokens"]
                or len(final_text.encode("utf-8")) > limits["completion_tokens"] * 8):
            raise ValueError("corrective final completion_tokens or body exceeds sealed limit")
        messages.append({"role": "assistant", "content": final_text})
        completion_usage.append({"sequence": len(completion_usage), "completion_tokens": final_completion_tokens})
        prefix = messages[:correction_start]
        elapsed = failed_terminal["elapsed_seconds"] + (time.monotonic() - started)
        metadata = dict(failed_terminal["metadata"])
        metadata.update({
            "turn_count": len(combined_actions) + 1,
            "action_count": len(combined_actions),
            "elapsed_seconds": elapsed,
            "completion_usage": completion_usage,
            "completion_tokens": sum(item["completion_tokens"] for item in completion_usage),
        })
        return {"schema": "emender-e97-corrective-terminal-v1", "status": "success",
                "failed_terminal_sha256": sha256_json(failed_terminal),
                "correction_start_message_index": correction_start, "prefix_sha256": sha256_json(prefix),
                "turns": len(combined_actions) + 1, "elapsed_seconds": elapsed,
                "messages": messages, "actions": combined_actions, "metadata": metadata, "error": None}
    finally:
        state.executor.close()

def _validator_execution_argv(
    bundle: Mapping[str, Any], spec: Mapping[str, Any], *, mode: str,
    validator_program: Path | None = None, trusted_validator_sha256: str | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve sealed logical validator IDs to one digest-verified physical argv."""

    logical = bundle["validator"].get(f"{mode}_argv") if isinstance(bundle.get("validator"), Mapping) else None
    expected_logical = validator_logical_argv(mode)
    if logical != expected_logical:
        raise ValueError("validator argv is not the sealed logical validator identity")
    interpreter = Path(sys.executable)
    program = VALIDATOR_PROGRAM if validator_program is None else Path(validator_program)
    expected_program_sha256 = _sha_path(VALIDATOR_PROGRAM) if trusted_validator_sha256 is None else trusted_validator_sha256
    if (not isinstance(expected_program_sha256, str) or len(expected_program_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_program_sha256)):
        raise ValueError("trusted validator digest is invalid")
    try:
        program_stat = os.stat(program, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("trusted validator program is unavailable") from exc
    if not stat.S_ISREG(program_stat.st_mode) or program.is_symlink():
        raise ValueError("trusted validator program is not a regular file")
    if (spec.get("program_sha256") != expected_program_sha256
            or _sha_path(program) != expected_program_sha256
            or spec.get("interpreter_sha256") != _sha_path(interpreter)):
        raise ValueError("validator program/interpreter identity mismatch")
    return expected_logical, [str(interpreter), str(program), "--mode", mode]


def _run_validator(
    logical_argv: list[str], execution_argv: list[str], spec_path: Path, terminal_path: Path,
    *, mode: str, limit: int, timeout: int, processes: int, expected_action_count: int,
) -> dict[str, Any]:
    """Run physical argv while recording only its stable logical identity."""

    import resource

    logical = logical_argv + ["--spec", "<spec>", "--terminal", "<terminal>"]

    def limit_process() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(timeout)), max(1, int(timeout))))
        resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))

    with tempfile.NamedTemporaryFile(mode="w+b") as stdout, tempfile.NamedTemporaryFile(mode="w+b") as stderr:
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                execution_argv + ["--spec", str(spec_path), "--terminal", str(terminal_path)],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT), "LC_ALL": "C", "LANG": "C"},
                preexec_fn=limit_process,
                start_new_session=True,
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
    return {
        "logical_argv": logical,
        "logical_argv_sha256": sha256_json(logical),
        "stdout_sha256": sha256_text(out.decode()),
        "stderr_sha256": sha256_text(err.decode()),
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
    if (not isinstance(metadata, Mapping)
            or metadata.get("controller_build_sha256") != bundle["runtime"]["controller_digest"]
            or metadata.get("tool_schema_sha256") != bundle["runtime"]["tool_schema_digest"]
            or metadata.get("system_prompt_sha256") != bundle["runtime"]["system_prompt_sha256"]
            or metadata.get("configured_limits") != bundle["limits"]
            or runtime_schema_digest != bundle["runtime"]["schema_digest"]):
        raise ValueError("task/runtime/limits binding mismatch")
    validator_argvs = {
        mode: _validator_execution_argv(
            bundle,
            spec,
            mode=mode,
            validator_program=validator_program,
            trusted_validator_sha256=trusted_validator_sha256,
        )
        for mode in ("focused", "regression")
    }
    state = _replay_actions(bundle, fixture_root, terminal)
    state.executor.close()
    with tempfile.TemporaryDirectory(prefix="e97-validator-") as temporary:
        root = Path(temporary)
        spec_path, terminal_path = root / "spec.json", root / "terminal.json"
        _write_json(spec_path, spec)
        _write_json(terminal_path, terminal)
        results = {}
        for name in ("focused", "regression"):
            logical_argv, execution_argv = validator_argvs[name]
            results[name] = _run_validator(
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
    return {
        "schema": "emender-e97-first-party-validator-receipt-v3",
        "status": "pass",
        "task_identity": bundle["task"]["identity"],
        "bundle_sha256": sha256_json(bundle),
        "fixture_tree_digest": bundle["task"]["fixture_tree_digest"],
        "archive_expanded_bytes": expanded_bytes,
        "configured_limits": dict(bundle["limits"]),
        "terminal_sha256": sha256_json(terminal),
        "validator_spec_digest": sha256_json(spec),
        "runtime_schema_digest": runtime_schema_digest,
        "validators": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--seed", required=True); parser.add_argument("--registry", type=Path, required=True); parser.add_argument("--registry-sha256", required=True); parser.add_argument("--policy-sha256", required=True)
    args = parser.parse_args(); print(json.dumps({k: str(v) for k, v in generate(args.output, seed=args.seed, registry_path=args.registry, registry_sha256=args.registry_sha256, policy_sha256=args.policy_sha256).items()}, sort_keys=True))

if __name__ == "__main__": main()
