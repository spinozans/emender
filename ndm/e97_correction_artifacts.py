"""Resolution and executable verification of artifact-backed correction records."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from ndm.e97_artifact_store import ArtifactStore, ArtifactStoreError
from ndm.e97_first_party_read_observe import safe_extract_fixture_archive, validate_replay
from ndm.e97_first_party_source_archive import extract_verified_source_member, verify_archive_members
from ndm.e97_onpolicy_records import canonical_json, sha256_json, validate_recovery_record
from ndm.e97_protected_overlap import OverlapError, validate_overlap_receipt
from ndm.e97_task_lake import validate_source_registry, validate_task_bundle, validate_task_collection


class CorrectionArtifactError(ValueError):
    """A correction record is not rooted in its declared immutable artifacts."""


def _json_bytes_equal(store: ArtifactStore, reference: Mapping[str, Any], inline: Any, name: str) -> Any:
    """Resolve a JSON artifact and require its literal canonical bytes equal inline bytes."""

    payload = store.resolve_bytes(reference, suffix=".json")
    expected = canonical_json(inline).encode("utf-8")
    if payload != expected:
        raise CorrectionArtifactError(f"inline {name} does not byte-equal its artifact")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - exact canonical bytes above
        raise CorrectionArtifactError(f"{name} artifact is not JSON") from exc


def _tasks_jsonl(payload: bytes) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
        if not lines or any(not line for line in lines):
            raise ValueError
        tasks = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CorrectionArtifactError("tasks collection is not strict JSONL") from exc
    if payload != ("".join(canonical_json(task) + "\n" for task in tasks)).encode("utf-8"):
        raise CorrectionArtifactError("tasks collection bytes are not canonical JSONL")
    return tasks


def validate_artifact_backed_recovery_record(value: Any, *, artifact_root: Path | str) -> dict[str, Any]:
    """Resolve, replay, and cross-check every authority artifact before packing SFT."""

    record = validate_recovery_record(value)
    binding = record["terminal_binding"]
    references = binding["artifacts"]
    store = ArtifactStore(artifact_root)
    try:
        registry = store.resolve_json(references["registry"])
        registry = validate_source_registry(registry)
        # Generation receipts and source components are retained as exact source
        # file bytes; unlike inline correction JSON they are never recanonicalized.
        generation_payload = store.resolve_bytes(references["generation_receipt"], suffix=".json")
        generation = json.loads(generation_payload)
        generator_manifest_payload = store.resolve_bytes(references["generator_manifest"], suffix=".json")
        source_archive_payload = store.resolve_bytes(references["source_archive"], suffix=".tar")
        overlap = store.resolve_json(references["overlap_receipt"])
        task_payload = store.resolve_bytes(references["tasks_collection"], suffix=".jsonl")
        tasks = validate_task_collection(_tasks_jsonl(task_payload), registry=registry)
        bundle = _json_bytes_equal(store, references["bundle"], binding["bundle"], "bundle")
        if bundle != validate_task_bundle(bundle, registry=registry):
            raise CorrectionArtifactError("bundle schema is not closed")
        if bundle not in tasks or sha256_json(bundle) != binding["bundle_sha256"]:
            raise CorrectionArtifactError("bundle is not in its sealed task collection")
        failed = _json_bytes_equal(store, references["failed_terminal"], binding["failed_terminal"], "failed terminal")
        corrective = _json_bytes_equal(
            store, references["corrective_terminal"], binding["corrective_terminal"], "corrective terminal",
        )
        execution = _json_bytes_equal(
            store, references["validator_execution"], binding["validator_execution"], "validator execution",
        )
        completion = store.resolve_published_receipt(
            references["completion_receipt"], task_id=record["task"]["identity"],
        )
        if (completion != binding["completion_receipt"]
                or sha256_json(completion) != binding["completion_receipt_sha256"]):
            raise CorrectionArtifactError("published completion receipt differs from inline artifact")
        private_spec = store.resolve_json(references["private_spec"])
        archive = store.resolve_bytes(references["archive"], suffix=".tar")
    except (ArtifactStoreError, ValueError) as exc:
        raise CorrectionArtifactError(str(exc)) from exc

    expected_generation_fields = {
        "schema", "state", "seed_sha256", "generator_component_manifest_sha256",
        "generator_source_archive_sha256", "tasks_sha256", "archive_root_sha256", "registry_sha256",
        "policy_sha256", "registry_copy_sha256", "controller_source_sha256",
        "environment_descriptor_sha256", "protected_manifest_sha256s",
    }
    if (not isinstance(generation, Mapping) or set(generation) != expected_generation_fields
            or generation["schema"] != "emender-e97-first-party-generation-receipt-v3"
            or generation["state"] != "quarantined-pending-protected-overlap"
            or generation["tasks_sha256"] != references["tasks_collection"]["sha256"]
            or generation["registry_sha256"] != references["registry"]["sha256"]
            or generation["registry_copy_sha256"] != references["registry"]["sha256"]
            or generation["generator_component_manifest_sha256"] != references["generator_manifest"]["sha256"]
            or generation["generator_source_archive_sha256"] != references["source_archive"]["sha256"]
            or generation["generator_source_archive_sha256"] != bundle["task"]["generator_source_digest"]):
        raise CorrectionArtifactError("generation receipt does not bind registry, collection, and source archive")
    if (not isinstance(generation["archive_root_sha256"], str)
            or len(generation["archive_root_sha256"]) != 64):
        raise CorrectionArtifactError("generation archive root digest is invalid")
    with tempfile.TemporaryDirectory(prefix="e97-generator-source-") as temporary:
        temporary_root = Path(temporary)
        manifest_path = temporary_root / "generator-manifest.json"
        source_archive_path = temporary_root / "source-archive.tar"
        manifest_path.write_bytes(generator_manifest_payload)
        source_archive_path.write_bytes(source_archive_payload)
        try:
            verified_source_archive_sha256 = verify_archive_members(manifest_path, source_archive_path)
        except ValueError as exc:
            raise CorrectionArtifactError("generator source archive does not match its component manifest") from exc
    if verified_source_archive_sha256 != generation["generator_source_archive_sha256"]:
        raise CorrectionArtifactError("generator source archive digest does not match generation receipt")
    try:
        validate_overlap_receipt(
            overlap,
            candidate_collection_sha256=generation["tasks_sha256"],
            candidate_archive_root_sha256=generation["archive_root_sha256"],
        )
    except OverlapError as exc:
        raise CorrectionArtifactError(str(exc)) from exc
    if (references["archive"]["sha256"] != binding["archive_sha256"]
            or len(archive) != bundle["fixture"]["artifact_bytes"]
            or references["archive"]["sha256"] != bundle["fixture"]["artifact_sha256"]
            or sha256_json(private_spec) != bundle["validator"]["spec_digest"]):
        raise CorrectionArtifactError("archive or private validator spec does not bind bundle")

    with tempfile.TemporaryDirectory(prefix="e97-correction-artifact-") as temporary:
        temporary_root = Path(temporary)
        fixture_root = temporary_root / "fixture"
        fixture_root.mkdir()
        archive_path = temporary_root / "fixture.tar"
        archive_path.write_bytes(archive)
        source_manifest_path = temporary_root / "generator-manifest.json"
        source_archive_path = temporary_root / "generator-source.tar"
        validator_program_path = temporary_root / "e97_first_party_validator.py"
        source_manifest_path.write_bytes(generator_manifest_payload)
        source_archive_path.write_bytes(source_archive_payload)
        try:
            validator_program_sha256 = extract_verified_source_member(
                source_manifest_path,
                source_archive_path,
                "scripts/e97_first_party_validator.py",
                validator_program_path,
            )
        except ValueError as exc:
            raise CorrectionArtifactError("archived first-party validator cannot be extracted") from exc
        safe_extract_fixture_archive(
            archive_path,
            fixture_root,
            expected_sha256=binding["archive_sha256"],
            expected_tree_digest=bundle["task"]["fixture_tree_digest"],
            disk_limit=bundle["limits"]["disk_bytes"],
        )
        replayed = validate_replay(
            bundle,
            private_spec,
            fixture_root,
            corrective,
            runtime_schema_digest=bundle["runtime"]["schema_digest"],
            validator_program=validator_program_path,
            trusted_validator_sha256=validator_program_sha256,
        )
    if replayed != execution:
        raise CorrectionArtifactError("stored validator execution is not exact deterministic replay output")
    if completion["receipt"]["accepted_terminal_sha256"] != binding["corrective_terminal_sha256"]:
        raise CorrectionArtifactError("completion receipt does not bind accepted corrective terminal")
    if sha256_json(failed) != binding["failed_terminal_sha256"]:
        raise CorrectionArtifactError("failed terminal artifact hash mismatch")
    return record
