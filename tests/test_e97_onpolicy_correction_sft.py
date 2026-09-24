import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from ndm.e97_acquisition_controller import AcquisitionController, WorkspaceToolExecutor
from ndm.e97_artifact_store import ArtifactStore
from ndm.e97_correction_artifacts import CorrectionArtifactError, validate_artifact_backed_recovery_record
from ndm.e97_first_party_read_observe import (
    admit_generated_collection,
    generate,
    replay_corrective_suffix_from_failure,
    safe_extract_fixture_archive,
    validate_replay,
)
from ndm.e97_first_party_source_archive import build_source_archive
from ndm.e97_onpolicy_records import (
    CANONICAL_NO_PROGRESS_OBSERVATION,
    CONSUMED_V3_MANIFEST_SHA256,
    CONSUMED_V4_MANIFEST_SHA256,
    RECOVERY_RECORD_SCHEMA,
    canonical_action,
    canonical_json,
    completion_marker_relative_path,
    sha256_json,
    sha256_text,
)
from ndm.e97_task_leases import TaskLeases
from scripts import build_e97_onpolicy_correction_sft as builder


def test_correction_source_jsonl_uses_one_nofollow_snapshot_for_hash_and_parse(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    original, replacement = b'{"record":"sealed"}\n', b'{"record":"substituted"}\n'
    source.write_bytes(original)
    reader = builder.read_regular_file_no_follow

    def substitute_after_snapshot(path, *, maximum):
        payload = reader(path, maximum=maximum)
        if Path(path) == source:
            source.write_bytes(replacement)
        return payload

    monkeypatch.setattr(builder, "read_regular_file_no_follow", substitute_after_snapshot)
    assert builder.load_source_jsonl_snapshot(source, hashlib.sha256(original).hexdigest()) == ['{"record":"sealed"}']
    assert source.read_bytes() == replacement


@pytest.mark.parametrize("kind", ("symlink", "fifo"))
def test_correction_source_jsonl_snapshot_rejects_link_and_fifo(tmp_path, kind):
    source = tmp_path / "source.jsonl"
    if kind == "symlink":
        outside = tmp_path / "outside.jsonl"
        outside.write_text('{"record":"outside"}\n')
        source.symlink_to(outside)
    else:
        os.mkfifo(source)
    with pytest.raises(ValueError, match="cannot be snapshotted safely"):
        builder.load_source_jsonl_snapshot(source, "0" * 64)


class _ScriptedCompletion:
    """Fork-safe synthetic completion source for a non-production failed rollout."""

    def __init__(self, assistants, attestation, model_id):
        self.assistants = assistants
        self.attestation = attestation
        self.model_id = model_id
        self.position = 0

    def complete(self, request, *, deadline):
        del deadline
        assistant = self.assistants[self.position]
        self.position += 1
        identity = {
            "messages_sha256": sha256_text(canonical_json(request["messages"])),
            "system_prompt_sha256": sha256_text(request["messages"][0]["content"]),
            "tool_schema_sha256": sha256_json(request["tools"]),
            "request_sha256": sha256_json(request),
        }
        return {
            "model": self.model_id,
            "choices": [{"message": assistant}],
            "usage": {"completion_tokens": 1},
            "emender_request_identity": identity,
            "emender_service_attestation": self.attestation,
        }


def _sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _assistant_call(call_id, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "read", "arguments": canonical_json(arguments)},
        }],
    }


def _service_attestation(bundle, checkpoint_sha256, model_id):
    runtime = bundle["runtime"]
    return {
        "schema": "emender-e97-agent-service-attestation-v1",
        "checkpoint_path": "synthetic://private-checkpoint",
        "checkpoint_sha256": checkpoint_sha256,
        "args_json_sha256": sha256_text("private-synthetic-args"),
        "config_sha256": sha256_text("private-synthetic-config"),
        "weight_mode": "saved",
        "tokenizer": "private-synthetic-tokenizer",
        "model_id": model_id,
        "server_build_sha256": sha256_text("private-synthetic-server"),
        "controller_build_sha256": runtime["controller_digest"],
        "device": "cpu",
        "dtype": "float32",
        "use_triton": False,
        "ingest_mode": "tokenwise",
        "runtime_image_path": "synthetic://private-runtime",
        "runtime_image_sha256": runtime["sandbox_image_digest"],
        "tool_schema_sha256": runtime["tool_schema_digest"],
        "system_prompt_override_sha256": sha256_text(""),
        "runtime_identity_schema": "emender-e97-runtime-identity-v1",
        "max_output_tokens": bundle["limits"]["completion_tokens"],
        "max_sessions": 1,
        "python_implementation": "CPython",
        "python_version": "synthetic",
        "torch_version": "synthetic",
        "cuda_runtime": "",
        "cuda_available": False,
        "platform": "synthetic",
        "machine": "synthetic",
    }


def _private_overlap_receipt(first_party, overlap, tasks, quarantine):
    """Build only structural, private authorization evidence; no protected input is read."""

    candidates = overlap.candidate_records(tasks, quarantine)
    domains = overlap._domains(candidates)
    return {
        "schema": "emender-e97-protected-overlap-receipt-v1",
        "status": "pass",
        "candidate_collection_sha256": _sha_path(quarantine / "tasks.jsonl"),
        "candidate_archive_root_sha256": json.loads(
            (quarantine / "generation-receipt.json").read_text())["archive_root_sha256"],
        "checker_source_sha256": _sha_path(overlap.__file__),
        "protected_manifest_sha256s": sorted(
            item["manifest_sha256"]
            for item in json.loads((quarantine / "source-registry.json").read_text())["protected_evaluation"]
        ),
        # The fixed panel record identities are schema constants.  This private
        # test intentionally never opens the protected record files.
        "protected_records_sha256s": sorted(overlap._EXPECTED_RECORD_SHA256S.values()),
        "field_domain_counts": {
            field: {"candidate": len(values), "protected": 0}
            for field, values in sorted(domains.items())
        },
        "collision_counts": {field: 0 for field in sorted(domains)},
    }


def _admitted_private_authority(tmp_path, monkeypatch):
    """Generate and admit a current-schema authority under a temp-only allowlist."""

    import ndm.e97_first_party_read_observe as first_party
    import ndm.e97_protected_overlap as overlap

    private = tmp_path / "private-authority"
    private.mkdir()
    manifest = private / "generator-manifest.json"
    manifest.write_bytes(first_party.GENERATOR_MANIFEST.read_bytes())
    source_archive = private / "generator-source.tar"
    build_source_archive(manifest, source_archive, checkout_root=Path("."))
    environment = private / "environment.json"
    audit = private / "overlap-audit.json"
    license_path = private / "license.json"
    environment.write_bytes(first_party.ENVIRONMENT_DESCRIPTOR.read_bytes())
    audit.write_bytes(first_party.OVERLAP_AUDIT.read_bytes())
    license_path.write_bytes(first_party.AUTHORIZATION_LICENSE.read_bytes())

    registry = json.loads(first_party.CHECKED_IN_REGISTRY.read_text())
    registry["policy_sha256"] = _sha_path(
        "docs/EMENDER_E97_4B_ONPOLICY_TASK_LAKE_EXECUTION_PLAN.md")
    receipts = {
        "source_archive_sha256": _sha_path(source_archive),
        "license_sha256": _sha_path(license_path),
        "environment_sha256": _sha_path(environment),
        "overlap_sha256": _sha_path(audit),
    }
    registry["sources"] = [
        {
            "id": f"e97-firstparty-{split}",
            "kind": "first-party",
            "status": "admitted",
            "url": "https://example.invalid/private-e97-first-party",
            "revision": "a" * 40,
            "framework_license": "Private synthetic test fixture",
            "underlying_repository_policy": "per-repository-audit-required",
            "task_count_claim": 2,
            "receipts": receipts,
            "notes": "non-production private integration test authority",
        }
        for split in ("train", "development")
    ]
    registry_path = private / "source-registry.json"
    _write_json(registry_path, registry)

    monkeypatch.setattr(first_party, "CHECKED_IN_REGISTRY", registry_path)
    monkeypatch.setattr(first_party, "GENERATOR_MANIFEST", manifest)
    monkeypatch.setattr(first_party, "GENERATOR_SOURCE_ARCHIVE", source_archive)
    monkeypatch.setattr(first_party, "ENVIRONMENT_DESCRIPTOR", environment)
    monkeypatch.setattr(first_party, "OVERLAP_AUDIT", audit)
    monkeypatch.setattr(first_party, "AUTHORIZATION_LICENSE", license_path)
    # This fixture intentionally swaps immutable path globals to its private
    # authority.  Runtime closure attestation correctly binds those globals,
    # so keep the real closure coverage in its dedicated tests and inject the
    # synthetic receipt identity here rather than claiming this altered process
    # is the checked-in archive runtime.
    monkeypatch.setattr(first_party, "_loaded_replay_digest_from_archive", lambda *_args: "c" * 64)

    # The producer requires a clean committed checkout.  Keep its loaded
    # function intact for archive-closure attestation and inject only the two
    # local Git observations needed by this synthetic private authority.
    def clean_git_observation(command, *, text, stderr):
        del text, stderr
        if command[-2:] == ["rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if "status" in command:
            return ""
        raise AssertionError(f"unexpected synthetic Git command: {command}")

    monkeypatch.setattr(first_party.subprocess, "check_output", clean_git_observation)
    quarantine = tmp_path / "quarantine"
    generate(
        quarantine,
        seed="private-nonproduction-artifact-recovery",
        registry_path=registry_path,
        registry_sha256=_sha_path(registry_path),
        policy_sha256=registry["policy_sha256"],
    )
    generation = json.loads((quarantine / "generation-receipt.json").read_text())
    assert generation["schema"] == "emender-e97-first-party-generation-receipt-v6"
    tasks = [json.loads(line) for line in (quarantine / "tasks.jsonl").read_text().splitlines()]
    overlap_receipt = _private_overlap_receipt(first_party, overlap, tasks, quarantine)
    overlap_path = private / "overlap-receipt.json"
    _write_json(overlap_path, overlap_receipt)

    authorization = {
        "scope": "non-production-cpu-system-gate",
        "registry_sha256": generation["registry_sha256"],
        "generation_receipt_sha256": _sha_path(quarantine / "generation-receipt.json"),
        "protected_overlap_receipt_sha256": _sha_path(overlap_path),
        "tasks_sha256": generation["tasks_sha256"],
        "archive_root_sha256": generation["archive_root_sha256"],
        "generator_manifest_sha256": generation["generator_component_manifest_sha256"],
        "source_archive_sha256": generation["generator_source_archive_sha256"],
        "source_revision": generation["source_revision"],
        "controller_source_sha256": generation["controller_source_sha256"],
    }
    real_allowlist = overlap.COLLECTION_AUTHORIZATION_ALLOWLIST
    original_allowlist = real_allowlist.read_bytes()
    private_allowlist = private / "collection-allowlist.json"
    _write_json(private_allowlist, {
        "schema": "emender-e97-firstparty-collection-authorization-v1",
        "status": "authorized",
        "authorizations": [authorization],
    })
    monkeypatch.setattr(overlap, "COLLECTION_AUTHORIZATION_ALLOWLIST", private_allowlist)

    admitted = tmp_path / "admitted"
    admit_generated_collection(
        quarantine, overlap_path, admitted, diagnostic_cpu_system_gate=True)
    assert real_allowlist.read_bytes() == original_allowlist
    return admitted, private_allowlist


def _artifact_references(store, admitted, bundle, failed, corrective, execution):
    """Store the exact authority bytes and canonical inline terminal objects."""

    exact_json = {
        "registry": "source-registry.json",
        "generation_receipt": "generation-receipt.json",
        "generator_manifest": "generator-manifest.json",
        "environment_descriptor": "environment-descriptor.json",
        "overlap_firewall_audit": "overlap-firewall-audit.json",
        "authorization_license": "authorization-license.json",
        "overlap_receipt": "protected-overlap-receipt.json",
        "admission_receipt": "admission-receipt.json",
        "authority_state": "authority-state.json",
        "collection_authorization_allowlist": "collection-authorization-allowlist.json",
    }
    references = {
        name: store.put_bytes((admitted / relative).read_bytes(), suffix=".json")
        for name, relative in exact_json.items()
    }
    references["source_archive"] = store.put_bytes(
        (admitted / "source-archive.tar").read_bytes(), suffix=".tar")
    references["tasks_collection"] = store.put_bytes(
        (admitted / "tasks.jsonl").read_bytes(), suffix=".jsonl")
    references["bundle"] = store.put_json(bundle)
    references["archive"] = store.put_bytes(
        (admitted / bundle["fixture"]["artifact_path"]).read_bytes(), suffix=".tar")
    references["private_spec"] = store.put_bytes(
        (admitted / "private_validators" / f"{bundle['task']['identity']}.json").read_bytes(), suffix=".json")
    references["failed_terminal"] = store.put_json(failed)
    references["corrective_terminal"] = store.put_json(corrective)
    references["validator_execution"] = store.put_json(execution)
    return references


def _record_messages(failed, corrective):
    """Derive the exact zero-loss failure prefix and mechanical corrective suffix."""

    result = [
        {"role": "system", "text": failed["messages"][0]["content"], "loss": 0},
        {"role": "user", "text": failed["messages"][1]["content"], "loss": 0},
    ]
    for index, action in enumerate(failed["actions"]):
        text = f"Action: {action['tool_name']}\nArguments: {action['arguments_json']}"
        result.append({"role": "assistant", "text": text, "loss": 0})
        if index + 1 < len(failed["actions"]):
            result.append({"role": "tool", "text": action["effective_observation"], "loss": 0})
    result.append({
        "role": "tool",
        "text": failed["actions"][-1]["effective_observation"],
        "loss": 0,
    })
    for action in corrective["actions"][len(failed["actions"]):]:
        text = f"Action: {action['tool_name']}\nArguments: {action['arguments_json']}"
        result.append({"role": "assistant", "text": text, "loss": 1})
        result.append({"role": "tool", "text": action["effective_observation"], "loss": 0})
    result.append({"role": "assistant", "text": corrective["messages"][-1]["content"], "loss": 1})
    return result


def _completion_receipt(artifact_root, task_id, bundle_sha256, archive_sha256, failed_sha256,
                        corrective_sha256, execution_sha256):
    leases = TaskLeases(artifact_root)
    try:
        leases.initialize([{"task_id": task_id, "bundle_sha256": bundle_sha256}])
        lease = leases.claim("private-integration", lease_seconds=60)
        assert lease is not None
        digest = leases.complete(lease, {
            "task_id": task_id,
            "bundle_sha256": bundle_sha256,
            "archive_sha256": archive_sha256,
            "failed_terminal_sha256": failed_sha256,
            "accepted_terminal_sha256": corrective_sha256,
            "validator_receipt_sha256": execution_sha256,
            "validator_status": "pass",
        })
        marker = artifact_root / completion_marker_relative_path(task_id, digest)
        return json.loads(marker.read_text()), digest
    finally:
        leases.close()


def _complete_artifact_graph(tmp_path, monkeypatch):
    """Make one complete, non-trainable current-producer recovery graph."""

    admitted, private_allowlist = _admitted_private_authority(tmp_path, monkeypatch)
    bundles = [json.loads(line) for line in (admitted / "tasks.jsonl").read_text().splitlines()]
    bundle = next(item for item in bundles if item["split"] == "train")
    spec = json.loads((admitted / "private_validators" / f"{bundle['task']['identity']}.json").read_text())
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    archive_payload = (admitted / bundle["fixture"]["artifact_path"]).read_bytes()
    safe_extract_fixture_archive(
        archive_payload,
        fixture,
        expected_sha256=bundle["fixture"]["artifact_sha256"],
        expected_tree_digest=bundle["task"]["fixture_tree_digest"],
        disk_limit=bundle["limits"]["disk_bytes"],
    )

    checkpoint_sha256 = sha256_text("private-nonproduction-checkpoint")
    model_id = "private-nonproduction-model"
    attestation = _service_attestation(bundle, checkpoint_sha256, model_id)
    missing = {"path": "retired.txt", "offset": 1, "limit": 1}
    executor = WorkspaceToolExecutor(
        fixture,
        max_output_bytes=bundle["limits"]["output_bytes"],
        max_tree_bytes=bundle["limits"]["disk_bytes"],
    )
    try:
        controller = AcquisitionController.for_task_bundle(
            _ScriptedCompletion(
                [_assistant_call(f"failure-{index}", missing) for index in range(3)],
                attestation,
                model_id,
            ),
            executor,
            bundle=bundle,
            checkpoint_sha256=checkpoint_sha256,
            expected_service_attestation=attestation,
            controller_build_sha256=bundle["runtime"]["controller_digest"],
            model_id=model_id,
        )
        failed = json.loads(canonical_json(asdict(controller.run(
            system_prompt="Use only the provided read-observe tools and ground the final in observations.",
            user_prompt=bundle["task"]["prompt"],
        ))))
    finally:
        executor.close()
    assert failed["status"] == "no_progress_terminated"
    assert [action["decision"] for action in failed["actions"]] == [
        "continue", "inject_no_progress", "terminate",
    ]

    correct_arguments = {
        "path": spec["required_read_path"],
        "offset": 1,
        "limit": 1,
    }
    corrective = replay_corrective_suffix_from_failure(
        bundle,
        failed,
        fixture,
        [{
            "tool_name": "read",
            "arguments": correct_arguments,
            "arguments_json": canonical_json(correct_arguments),
            "completion_tokens": 1,
        }],
        "Final: " + spec["expected_token"],
        1,
    )
    assert corrective["metadata"]["mechanical_suffix"]["training_eligible"] is False
    execution = validate_replay(
        bundle,
        spec,
        fixture,
        corrective,
        runtime_schema_digest=bundle["runtime"]["schema_digest"],
    )
    assert execution["schema"] == "emender-e97-first-party-validator-receipt-v4"
    for mode in ("focused", "regression"):
        assert execution["validators"][mode]["spec_payload_sha256"] == execution["validator_spec_payload_sha256"]
        assert execution["validators"][mode]["terminal_payload_sha256"] == execution["validator_terminal_payload_sha256"]

    artifact_root = tmp_path / "artifact-root"
    store = ArtifactStore(artifact_root)
    references = _artifact_references(store, admitted, bundle, failed, corrective, execution)
    bundle_sha256 = sha256_json(bundle)
    archive_sha256 = references["archive"]["sha256"]
    failed_sha256 = sha256_json(failed)
    corrective_sha256 = sha256_json(corrective)
    execution_sha256 = sha256_json(execution)
    completion, completion_sha256 = _completion_receipt(
        artifact_root,
        bundle["task"]["identity"],
        bundle_sha256,
        archive_sha256,
        failed_sha256,
        corrective_sha256,
        execution_sha256,
    )
    references["completion_receipt"] = {
        "sha256": completion_sha256,
        "path": completion_marker_relative_path(bundle["task"]["identity"], completion_sha256),
    }

    messages = _record_messages(failed, corrective)
    record = {
        "schema": RECOVERY_RECORD_SCHEMA,
        "split": bundle["split"],
        "task": {
            name: bundle["task"][name]
            for name in ("namespace", "family_id", "identity", "generator_source_digest", "fixture_tree_digest", "intent_digest")
        },
        "student": {
            "rollout_identity": failed_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "decode": {"temperature": 0},
        },
        "teacher": {
            "tier": "mechanical-cpu-system-gate",
            "model_revision": "private-nonproduction-current-producer",
            "evidence": "mechanical-scripted-replay-v1",
        },
        "provenance": {
            "scope": "cpu-system-gate-mechanical",
            "training_eligible": False,
        },
        "runtime": bundle["runtime"],
        "first_divergence": {
            "message_index": 4,
            "target_start_message_index": 8,
            "class": "stale-action",
            "student_action_canonical": canonical_action(
                failed["actions"][1]["tool_name"], failed["actions"][1]["arguments"]),
            "pre_state_digest": failed["actions"][1]["progress_fingerprint"],
            "observation_digest": sha256_text(CANONICAL_NO_PROGRESS_OBSERVATION),
        },
        "messages": messages,
        "validator_receipt": {
            "validator_digest": bundle["validator"]["spec_digest"],
            "input_digest": bundle["task"]["fixture_tree_digest"],
            "postcondition_digest": execution_sha256,
            "action_graph_digest": sha256_json(corrective["actions"]),
            "passed": True,
            "cycle_free": True,
        },
        "source_provenance": {
            "source_digests": [bundle["task"]["generator_source_digest"]],
            "forbidden_panel_digests_checked": [
                CONSUMED_V3_MANIFEST_SHA256,
                CONSUMED_V4_MANIFEST_SHA256,
            ],
        },
        "terminal_binding": {
            "artifacts": references,
            "bundle": bundle,
            "bundle_sha256": bundle_sha256,
            "archive_sha256": archive_sha256,
            "failed_terminal": failed,
            "failed_terminal_sha256": failed_sha256,
            "corrective_terminal": corrective,
            "corrective_terminal_sha256": corrective_sha256,
            "validator_execution": execution,
            "validator_execution_sha256": execution_sha256,
            "completion_receipt": completion,
            "completion_receipt_sha256": completion_sha256,
            "lease_identity": completion["lease"]["identity"],
            "prefix_sha256": corrective["prefix_sha256"],
            "model_prefix_sha256": sha256_json(messages[:8]),
            "terminating_intervention": {
                "action_index": len(failed["actions"]) - 1,
                "effective_observation": failed["actions"][-1]["effective_observation"],
                "sent_to_model": False,
            },
        },
    }
    return record, artifact_root, private_allowlist


def _reseal_payload_hash_mutation(record, artifact_root, field):
    """Keep structural receipt links valid so replay must reject a forged payload hash."""

    mutated_root = artifact_root.parent / f"{artifact_root.name}-{field}"
    shutil.copytree(artifact_root / "artifacts", mutated_root / "artifacts")
    mutated = copy.deepcopy(record)
    execution = mutated["terminal_binding"]["validator_execution"]
    forged = sha256_text("forged-current-validator-" + field)
    execution[field] = forged
    for mode in ("focused", "regression"):
        execution["validators"][mode][field.removeprefix("validator_")] = forged
    execution_sha256 = sha256_json(execution)
    mutated["terminal_binding"]["validator_execution_sha256"] = execution_sha256
    mutated["validator_receipt"]["postcondition_digest"] = execution_sha256
    store = ArtifactStore(mutated_root)
    mutated["terminal_binding"]["artifacts"]["validator_execution"] = store.put_json(execution)

    binding = mutated["terminal_binding"]
    completion, completion_sha256 = _completion_receipt(
        mutated_root,
        mutated["task"]["identity"],
        binding["bundle_sha256"],
        binding["archive_sha256"],
        binding["failed_terminal_sha256"],
        binding["corrective_terminal_sha256"],
        execution_sha256,
    )
    binding["completion_receipt"] = completion
    binding["completion_receipt_sha256"] = completion_sha256
    binding["lease_identity"] = completion["lease"]["identity"]
    binding["artifacts"]["completion_receipt"] = {
        "sha256": completion_sha256,
        "path": completion_marker_relative_path(mutated["task"]["identity"], completion_sha256),
    }
    return mutated, mutated_root


def test_private_authorized_artifact_backed_recovery_uses_current_producers_end_to_end(tmp_path, monkeypatch):
    """P2: current generation/admission/replay/lease artifacts close one private CPU graph."""

    record, artifact_root, private_allowlist = _complete_artifact_graph(tmp_path, monkeypatch)
    assert validate_artifact_backed_recovery_record(
        record,
        artifact_root=artifact_root,
        allow_diagnostic_cpu_system_gate=True,
    ) == json.loads(canonical_json(record))
    assert (artifact_root / record["terminal_binding"]["artifacts"]["completion_receipt"]["path"]).is_file()
    assert private_allowlist.is_file()

    # v6's newly bound source revision and retained audit/license inputs cannot
    # be redirected merely by changing the generation artifact reference.
    original_generation = ArtifactStore(artifact_root).resolve_json(
        record["terminal_binding"]["artifacts"]["generation_receipt"])
    for field, replacement, match in (
        ("source_revision", "b" * 40, "not operator-authorized"),
        ("overlap_firewall_audit_sha256", "0" * 64, "generation receipt does not bind"),
        ("authorization_license_sha256", "0" * 64, "generation receipt does not bind"),
    ):
        mutated = copy.deepcopy(record)
        forged_generation = {**original_generation, field: replacement}
        mutated["terminal_binding"]["artifacts"]["generation_receipt"] = ArtifactStore(
            artifact_root).put_json(forged_generation)
        with pytest.raises(CorrectionArtifactError, match=match):
            validate_artifact_backed_recovery_record(
                mutated,
                artifact_root=artifact_root,
                allow_diagnostic_cpu_system_gate=True,
            )

    # These changes remain self-consistent at the receipt-schema layer; only
    # current replay of the exact private spec/terminal bytes rejects them.
    for field in ("validator_spec_payload_sha256", "validator_terminal_payload_sha256"):
        mutated, mutated_root = _reseal_payload_hash_mutation(record, artifact_root, field)
        with pytest.raises(CorrectionArtifactError, match="exact deterministic replay output"):
            validate_artifact_backed_recovery_record(
                mutated,
                artifact_root=mutated_root,
                allow_diagnostic_cpu_system_gate=True,
            )
