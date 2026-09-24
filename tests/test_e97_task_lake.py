import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from ndm.data.masked_sft_dataset import sha256

from ndm.e97_onpolicy_records import (
    CONSUMED_V3_MANIFEST_SHA256,
    CONSUMED_V4_MANIFEST_SHA256,
    sha256_text,
    task_identity,
)
from ndm.e97_task_lake import (
    SOURCE_REGISTRY_SCHEMA,
    TASK_BUNDLE_SCHEMA,
    canonical_intent_digest,
    source_registry_digest,
    validate_source_registry,
    validate_task_bundle,
    validate_task_collection,
)


def digest(label):
    return sha256_text(label)


def source(source_id, *, status="admitted", kind="first-party", revision=None):
    admitted = status == "admitted"
    return {
        "id": source_id,
        "kind": kind,
        "status": status,
        "url": f"https://example.invalid/{source_id}",
        "revision": revision or digest(source_id)[:40],
        "framework_license": "MIT",
        "underlying_repository_policy": "per-repository-audit-required",
        "task_count_claim": 10,
        "receipts": {
            "source_archive_sha256": digest(f"archive-{source_id}") if admitted else None,
            "license_sha256": digest(f"license-{source_id}") if admitted else None,
            "environment_sha256": digest(f"environment-{source_id}") if admitted else None,
            "overlap_sha256": digest(f"overlap-{source_id}") if admitted else None,
        },
        "notes": "test fixture",
    }


def protected(name, manifest):
    return {
        "name": name,
        "manifest_sha256": manifest,
        "repositories": [],
        "family_ids": [],
        "task_identities": [],
        "fixture_tree_digests": [],
        "intent_digests": [],
        "validator_spec_digests": [],
    }


REAL_REPO_HOLDOUT_MANIFEST = "939bcd66768884a1e5ec44bcf11fdf5d09602ebdabe86d4caefffd4ace8dbb60"


def registry():
    return {
        "schema": SOURCE_REGISTRY_SCHEMA,
        "created_at": "2026-09-06T00:00:00Z",
        "policy_sha256": digest("task-lake-plan-v1"),
        "protected_evaluation": [
            protected("consumed-v3", CONSUMED_V3_MANIFEST_SHA256),
            protected("consumed-v4", CONSUMED_V4_MANIFEST_SHA256),
            protected("real-repo-holdout-v1", REAL_REPO_HOLDOUT_MANIFEST),
        ],
        "sources": [source("train-repo"), source("dev-repo")],
    }


def task_bundle(*, split="train", source_id=None, family=None, repository=None):
    source_id = source_id or ("train-repo" if split == "train" else "dev-repo")
    family = family or f"opaque-read-{'train' if split == 'train' else 'dev'}-v1"
    repository = repository or f"example/{source_id}"
    prompt = f"Read the opaque value for the {split} task."
    fixture_tree = digest(f"fixture-{split}-{source_id}-{family}")
    registered = next(item for item in registry()["sources"] if item["id"] == source_id)
    generator = registered["receipts"]["source_archive_sha256"]
    intent = canonical_intent_digest(prompt)
    namespace = f"e97-{'train' if split == 'train' else 'dev'}-{family}"
    identity = task_identity(
        namespace=namespace,
        family_id=family,
        generator_source_digest=generator,
        fixture_tree_digest=fixture_tree,
        intent_digest=intent,
    )
    return {
        "schema": TASK_BUNDLE_SCHEMA,
        "split": split,
        "task": {
            "namespace": namespace,
            "family_id": family,
            "identity": identity,
            "generator_source_digest": generator,
            "fixture_tree_digest": fixture_tree,
            "intent_digest": intent,
            "prompt": prompt,
            "difficulty": 2,
        },
        "source": {
            "registry_id": source_id,
            "kind": registered["kind"],
            "repository": repository,
            "revision": registered["revision"],
            "source_record_digest": digest(f"record-{split}-{source_id}-{family}"),
            "license_receipt_digest": registered["receipts"]["license_sha256"],
        },
        "fixture": {
            "artifact_path": f"fixtures/{identity}.tar.zst",
            "artifact_bytes": 1024,
            "artifact_sha256": digest(f"artifact-{identity}"),
            "tree_digest": fixture_tree,
        },
        "runtime": {
            "schema_digest": digest("runtime-v1"),
            "tool_schema_digest": digest("tools-v1"),
            "controller_digest": digest("controller-v1"),
            "sandbox_image_digest": digest("sandbox-v1"),
            "system_prompt_sha256": digest("system-v1"),
        },
        "limits": {
            "turns": 12,
            "seconds": 300,
            "completion_tokens": 512,
            "output_bytes": 16384,
            "disk_bytes": 1 << 30,
            "processes": 32,
        },
        "validator": {
            "spec_digest": digest(f"validator-{split}-{family}"),
            "focused_argv": ["@runtime-python", "@generator-source/scripts/e97_first_party_validator.py", "--mode", "focused"],
            "regression_argv": ["@runtime-python", "@generator-source/scripts/e97_first_party_validator.py", "--mode", "regression"],
            "milestone_digest": digest(f"milestones-{split}-{family}"),
            "minefield_digest": digest(f"minefields-{split}-{family}"),
        },
    }


def test_registry_requires_consumed_panels_and_complete_admission_receipts():
    value = registry()
    assert validate_source_registry(value) == value
    assert source_registry_digest(value) == source_registry_digest(copy.deepcopy(value))

    missing_v3 = copy.deepcopy(value)
    missing_v3["protected_evaluation"] = missing_v3["protected_evaluation"][1:]
    with pytest.raises(ValueError, match="V3/V4 and real-repository"):
        validate_source_registry(missing_v3)

    missing_receipt = copy.deepcopy(value)
    missing_receipt["sources"][0]["receipts"]["license_sha256"] = None
    with pytest.raises(ValueError, match="lowercase SHA-256|every receipt"):
        validate_source_registry(missing_receipt)


def test_candidate_source_is_recorded_but_not_task_admissible():
    value = registry()
    value["sources"][0] = source("train-repo", status="candidate")
    assert validate_source_registry(value)["sources"][0]["status"] == "candidate"
    task = task_bundle()
    task["source"]["license_receipt_digest"] = digest("license-train-repo")
    with pytest.raises(ValueError, match="not admitted"):
        validate_task_bundle(task, registry=value)


def test_task_identity_binds_prompt_and_fixture():
    value = registry()
    task = task_bundle()
    assert validate_task_bundle(task, registry=value) == task

    changed_prompt = copy.deepcopy(task)
    changed_prompt["task"]["prompt"] += " Changed."
    with pytest.raises(ValueError, match="intent digest"):
        validate_task_bundle(changed_prompt, registry=value)

    changed_fixture = copy.deepcopy(task)
    changed_fixture["task"]["fixture_tree_digest"] = digest("other-tree")
    changed_fixture["fixture"]["tree_digest"] = digest("other-tree")
    with pytest.raises(ValueError, match="not bound"):
        validate_task_bundle(changed_fixture, registry=value)


def test_task_requires_admitted_revision_generator_archive_and_license_receipt():
    value = registry()
    task = task_bundle()
    bad_revision = copy.deepcopy(task)
    bad_revision["source"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="revision"):
        validate_task_bundle(bad_revision, registry=value)
    bad_generator = copy.deepcopy(task)
    bad_generator["task"]["generator_source_digest"] = digest("different-generator")
    bad_generator["task"]["identity"] = task_identity(
        namespace=bad_generator["task"]["namespace"],
        family_id=bad_generator["task"]["family_id"],
        generator_source_digest=bad_generator["task"]["generator_source_digest"],
        fixture_tree_digest=bad_generator["task"]["fixture_tree_digest"],
        intent_digest=bad_generator["task"]["intent_digest"],
    )
    with pytest.raises(ValueError, match="generator source digest"):
        validate_task_bundle(bad_generator, registry=value)
    bad_logical_validator = copy.deepcopy(task)
    bad_logical_validator["validator"]["focused_argv"][1] = "alternate-pass-emitter.py"
    with pytest.raises(ValueError, match="sealed logical"):
        validate_task_bundle(bad_logical_validator, registry=value)
    bad_license = copy.deepcopy(task)
    bad_license["source"]["license_receipt_digest"] = digest("different-license")
    with pytest.raises(ValueError, match="license receipt"):
        validate_task_bundle(bad_license, registry=value)


def test_protected_repository_and_semantic_identities_fail_closed():
    value = registry()
    task = task_bundle()
    value["protected_evaluation"].append({
        "name": "sealed-v5",
        "manifest_sha256": digest("v5-manifest"),
        "repositories": [task["source"]["repository"]],
        "family_ids": [],
        "task_identities": [],
        "fixture_tree_digests": [],
        "intent_digests": [],
        "validator_spec_digests": [],
    })
    with pytest.raises(ValueError, match="repository:sealed-v5"):
        validate_task_bundle(task, registry=value)

    value = registry()
    value["protected_evaluation"][0]["fixture_tree_digests"] = [
        task["task"]["fixture_tree_digest"]]
    with pytest.raises(ValueError, match="fixture_tree_digests"):
        validate_task_bundle(task, registry=value)


def test_collection_enforces_whole_family_and_repository_split():
    value = registry()
    train = task_bundle(split="train")
    development = task_bundle(split="development")
    assert len(validate_task_collection([train, development], registry=value)) == 2

    same_family = task_bundle(split="development", family=train["task"]["family_id"])
    with pytest.raises(ValueError, match="families must be disjoint"):
        validate_task_collection([train, same_family], registry=value)

    same_repo = task_bundle(split="development", repository=train["source"]["repository"])
    with pytest.raises(ValueError, match="repositories must be disjoint"):
        validate_task_collection([train, same_repo], registry=value)


def test_task_bundle_schema_and_limits_are_closed():
    value = registry()
    task = task_bundle()
    task["limits"]["unexpected"] = 1
    with pytest.raises(ValueError, match="limits fields mismatch"):
        validate_task_bundle(task, registry=value)

def test_shell_strings_are_not_validator_commands():
    value = registry()
    task = task_bundle()
    task["validator"]["focused_argv"] = "python checks/check_task.py"
    with pytest.raises(ValueError, match="argv array"):
        validate_task_bundle(task, registry=value)
    task = task_bundle()
    task["validator"]["focused_argv"] = ["bash", "-lc", "pytest -q"]
    with pytest.raises(ValueError, match="may not invoke a shell"):
        validate_task_bundle(task, registry=value)


def test_repository_identities_are_lowercase():
    value = registry()
    task = task_bundle(repository="Example/Train-Repo")
    with pytest.raises(ValueError, match="lowercase owner/name"):
        validate_task_bundle(task, registry=value)


def test_registry_round_trip_is_canonical_json():
    value = validate_source_registry(registry())
    assert json.loads(json.dumps(value, sort_keys=True)) == value


def test_validation_receipt_publication_is_no_replace_and_concurrent(tmp_path):
    from scripts.validate_e97_task_lake import atomic_write_json

    output = tmp_path / "receipt.json"
    value = {"status": "pass", "n": 1}
    outcomes = []
    failures = []

    def publish():
        try:
            outcomes.append(atomic_write_json(output, value))
        except Exception as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    workers = [threading.Thread(target=publish) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not failures
    assert len(outcomes) == 8
    assert json.loads(output.read_text()) == value
    atomic_write_json(output, value)
    with pytest.raises(ValueError, match="conflicts"):
        atomic_write_json(output, {"status": "fail", "n": 2})


def test_registry_validation_uses_one_descriptor_snapshot_for_hash_and_parse(tmp_path, monkeypatch):
    from scripts import validate_e97_task_lake as validator

    registry_path = tmp_path / "registry.json"
    payload = Path("configs/pi/e97-onpolicy-source-registry-v1.json").read_bytes()
    registry_path.write_bytes(payload)
    replacement = tmp_path / "replacement.json"
    replacement.write_text("{}")
    original_reader = validator.read_regular_file_no_follow

    def swap_after_snapshot(path, *, maximum):
        snapshot = original_reader(path, maximum=maximum)
        registry_path.unlink()
        registry_path.symlink_to(replacement)
        return snapshot

    monkeypatch.setattr(validator, "read_regular_file_no_follow", swap_after_snapshot)
    registry, observed = validator.load_registry_snapshot(registry_path, sha256(Path("configs/pi/e97-onpolicy-source-registry-v1.json")))
    assert observed == sha256(Path("configs/pi/e97-onpolicy-source-registry-v1.json"))
    assert [source["status"] for source in registry["sources"]] == [
        "candidate", "candidate", "candidate"]

    symlinked = tmp_path / "linked-registry.json"
    symlinked.symlink_to(replacement)
    with pytest.raises(ValueError, match="cannot be snapshotted safely"):
        validator.load_registry_snapshot(symlinked, sha256(replacement))


def test_task_jsonl_validation_uses_one_descriptor_snapshot_and_rejects_links_and_fifos(tmp_path, monkeypatch):
    from scripts import validate_e97_task_lake as validator

    tasks_path = tmp_path / "tasks.jsonl"
    original, replacement = b'{"task":"first"}\n', b'{"task":"replacement"}\n'
    tasks_path.write_bytes(original)
    actual_reader = validator.read_regular_file_no_follow

    def replace_after_snapshot(path, *, maximum):
        payload = actual_reader(path, maximum=maximum)
        if Path(path) == tasks_path:
            tasks_path.write_bytes(replacement)
        return payload

    monkeypatch.setattr(validator, "read_regular_file_no_follow", replace_after_snapshot)
    parsed, observed = validator.load_task_snapshot(
        tasks_path, hashlib.sha256(original).hexdigest())
    assert parsed == [{"task": "first"}]
    assert observed == hashlib.sha256(original).hexdigest()
    assert tasks_path.read_bytes() == replacement

    linked = tmp_path / "linked-tasks.jsonl"
    linked.symlink_to(tasks_path)
    with pytest.raises(ValueError, match="cannot be snapshotted safely"):
        validator.load_task_snapshot(linked, hashlib.sha256(replacement).hexdigest())
    fifo = tmp_path / "tasks.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="cannot be snapshotted safely"):
        validator.load_task_snapshot(fifo, "0" * 64)


def test_checked_in_candidate_registry_is_valid_and_policy_bound(tmp_path):
    path = Path("configs/pi/e97-onpolicy-source-registry-v1.json")
    value = validate_source_registry(json.loads(path.read_text()))
    assert [source["status"] for source in value["sources"]] == [
        "candidate", "candidate", "candidate"]
    policy = Path("docs/EMENDER_E97_4B_ONPOLICY_TASK_LAKE_EXECUTION_PLAN.md")
    assert value["policy_sha256"] == sha256(policy)
    output = tmp_path / "receipt.json"
    subprocess.run([
        sys.executable, "-m", "scripts.validate_e97_task_lake",
        "--registry", str(path),
        "--registry-sha256", sha256(path),
        "--output", str(output),
    ], check=True, capture_output=True, text=True)
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "pass"
    assert receipt["registry"]["candidate_sources"] == 3
    assert receipt["registry"]["admitted_sources"] == 0
    assert receipt["task_validation"] == "not-run"
    assert "tasks" not in receipt
    assert receipt["claims"] == [
        "schema-valid registry",
        "consumed V3/V4 manifests protected",
        "registry task-source policy requires admitted sources",
        "static protected metadata is schema-bound; semantic clearance requires a per-collection overlap receipt",
    ]
    assert output.stat().st_mode & 0o777 == 0o600
