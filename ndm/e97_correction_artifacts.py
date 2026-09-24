"""Resolution and executable verification of artifact-backed correction records."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from ndm.e97_artifact_store import ArtifactStore, ArtifactStoreError
from ndm.e97_first_party_read_observe import (
    _verify_environment, safe_extract_fixture_archive, validate_replay,
)
from ndm.e97_first_party_source_archive import extract_verified_source_member, verify_archive_members
from ndm.e97_onpolicy_records import canonical_json, sha256_json, validate_recovery_record
from ndm.e97_protected_overlap import (
    OverlapError, validate_authorized_overlap_receipt, validate_overlap_authorization,
)
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


def validate_artifact_backed_recovery_record(
    value: Any, *, artifact_root: Path | str, allow_diagnostic_cpu_system_gate: bool = False,
) -> dict[str, Any]:
    """Resolve, replay, and cross-check every authority artifact before packing SFT."""

    record = validate_recovery_record(
        value, allow_diagnostic_cpu_system_gate=allow_diagnostic_cpu_system_gate,
    )
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
        environment_payload = store.resolve_bytes(references["environment_descriptor"], suffix=".json")
        overlap_audit_payload = store.resolve_bytes(references["overlap_firewall_audit"], suffix=".json")
        license_payload = store.resolve_bytes(references["authorization_license"], suffix=".json")
        overlap_payload = store.resolve_bytes(references["overlap_receipt"], suffix=".json")
        overlap = json.loads(overlap_payload)
        admission = store.resolve_json(references["admission_receipt"])
        authority_state = store.resolve_json(references["authority_state"])
        allowlist_payload = store.resolve_bytes(references["collection_authorization_allowlist"], suffix=".json")
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
        "policy_sha256", "registry_copy_sha256", "controller_source_sha256", "source_revision",
        "environment_descriptor_sha256", "overlap_firewall_audit_sha256",
        "authorization_license_sha256", "protected_manifest_sha256s",
    }
    if (not isinstance(generation, Mapping) or set(generation) != expected_generation_fields
            or generation["schema"] != "emender-e97-first-party-generation-receipt-v6"
            or generation["state"] != "quarantined-pending-protected-overlap"
            or generation["tasks_sha256"] != references["tasks_collection"]["sha256"]
            or generation["registry_sha256"] != references["registry"]["sha256"]
            or generation["registry_copy_sha256"] != references["registry"]["sha256"]
            or generation["generator_component_manifest_sha256"] != references["generator_manifest"]["sha256"]
            or generation["generator_source_archive_sha256"] != references["source_archive"]["sha256"]
            or generation["environment_descriptor_sha256"] != references["environment_descriptor"]["sha256"]
            or generation["overlap_firewall_audit_sha256"] != references["overlap_firewall_audit"]["sha256"]
            or generation["authorization_license_sha256"] != references["authorization_license"]["sha256"]
            or generation["protected_manifest_sha256s"] != sorted(
                item["manifest_sha256"] for item in registry["protected_evaluation"])
            or generation["generator_source_archive_sha256"] != bundle["task"]["generator_source_digest"]):
        raise CorrectionArtifactError("generation receipt does not bind registry, collection, and source archive")
    if (not isinstance(generation["archive_root_sha256"], str)
            or len(generation["archive_root_sha256"]) != 64
            or not isinstance(generation["source_revision"], str)
            or len(generation["source_revision"]) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in generation["source_revision"])
            or not isinstance(generation["protected_manifest_sha256s"], list)
            or any(not isinstance(item, str) or len(item) != 64 for item in generation["protected_manifest_sha256s"])):
        raise CorrectionArtifactError("generation receipt digest bindings are invalid")
    try:
        _verify_environment(environment_payload)
        audit = json.loads(overlap_audit_payload)
        license_value = json.loads(license_payload)
        validate_overlap_authorization(audit)
    except (UnicodeDecodeError, json.JSONDecodeError, OverlapError, ValueError) as exc:
        raise CorrectionArtifactError("retained overlap audit is invalid") from exc
    if (not isinstance(license_value, Mapping)
            or set(license_value) != {"schema", "authorization", "license", "scope"}
            or license_value.get("schema") != "emender-e97-firstparty-authorization-license-v1"
            or any(not isinstance(license_value.get(field), str) or not license_value[field]
                   for field in ("authorization", "license", "scope"))):
        raise CorrectionArtifactError("retained authorization license is invalid")
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
        authorized = validate_authorized_overlap_receipt(
            overlap,
            receipt_sha256=references["overlap_receipt"]["sha256"],
            registry_sha256=generation["registry_sha256"],
            generation_receipt_sha256=references["generation_receipt"]["sha256"],
            tasks_sha256=generation["tasks_sha256"],
            archive_root_sha256=generation["archive_root_sha256"],
            generator_manifest_sha256=generation["generator_component_manifest_sha256"],
            source_archive_sha256=generation["generator_source_archive_sha256"],
            source_revision=generation["source_revision"],
            controller_source_sha256=generation["controller_source_sha256"],
            diagnostic_cpu_system_gate=allow_diagnostic_cpu_system_gate,
        )
    except OverlapError as exc:
        raise CorrectionArtifactError(str(exc)) from exc
    if allowlist_payload != authorized["allowlist_payload"]:
        raise CorrectionArtifactError("retained operator allowlist differs from the authorized snapshot")
    allowlist_descriptor = {
        "path": "collection-authorization-allowlist.json", "bytes": len(allowlist_payload),
        "sha256": references["collection_authorization_allowlist"]["sha256"],
    }
    if allowlist_descriptor["sha256"] != authorized["allowlist_sha256"]:
        raise CorrectionArtifactError("artifact-backed operator allowlist digest is invalid")
    expected_admission = {
        "schema": "emender-e97-first-party-admission-receipt-v3",
        "registry_sha256": references["registry"]["sha256"],
        "generation_receipt_sha256": references["generation_receipt"]["sha256"],
        "protected_overlap_receipt_sha256": references["overlap_receipt"]["sha256"],
        "tasks_sha256": generation["tasks_sha256"], "archive_root_sha256": generation["archive_root_sha256"],
        "protected_manifest_sha256s": generation["protected_manifest_sha256s"],
        "collection_authorization_sha256": authorized["authorization_sha256"],
        "collection_authorization_allowlist": allowlist_descriptor,
        "collection_authorization": authorized["authorization"],
    }
    expected_state = {
        "schema": "emender-e97-first-party-authority-state-v3", "state": "admitted",
        "generation_receipt_sha256": references["generation_receipt"]["sha256"],
        "protected_overlap_receipt_sha256": references["overlap_receipt"]["sha256"],
        "collection_authorization_sha256": authorized["authorization_sha256"],
        "collection_authorization_allowlist": allowlist_descriptor,
        "collection_authorization": authorized["authorization"],
    }
    if admission != expected_admission or authority_state != expected_state:
        raise CorrectionArtifactError("admission/state receipts do not bind canonical collection authorization")
    if (references["archive"]["sha256"] != binding["archive_sha256"]
            or len(archive) != bundle["fixture"]["artifact_bytes"]
            or references["archive"]["sha256"] != bundle["fixture"]["artifact_sha256"]
            or sha256_json(private_spec) != bundle["validator"]["spec_digest"]):
        raise CorrectionArtifactError("archive or private validator spec does not bind bundle")

    with tempfile.TemporaryDirectory(prefix="e97-correction-artifact-") as temporary:
        temporary_root = Path(temporary)
        fixture_root = temporary_root / "fixture"
        fixture_root.mkdir()
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
            archive,
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
