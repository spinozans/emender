"""Fail-closed source registry and immutable task bundles for E97 acquisition."""
from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from ndm.e97_onpolicy_records import (
    CONSUMED_ID_PREFIXES,
    CONSUMED_PANEL_MANIFEST_SHA256S,
    canonical_json,
    sha256_text,
    task_identity,
)


SOURCE_REGISTRY_SCHEMA = "emender-e97-onpolicy-source-registry-v1"
TASK_BUNDLE_SCHEMA = "emender-e97-onpolicy-task-v1"
SOURCE_KINDS = {"swesmith", "swegym", "r2egym", "first-party", "openhands-replay"}
SOURCE_STATUSES = {"candidate", "admitted", "quarantined", "rejected"}
SPLITS = {"train", "development"}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _fields(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{name} fields mismatch: {observed}")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{name} is invalid")
    return value


def _strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicates")
    return list(value)


def canonical_intent_digest(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("task prompt must be non-empty")
    if len(prompt.encode("utf-8")) > 16 << 10:
        raise ValueError("task prompt is oversized")
    return sha256_text(prompt)


def _validate_receipts(value: Any, *, admitted: bool) -> dict[str, str | None]:
    receipt = _fields(value, {
        "source_archive_sha256", "license_sha256", "environment_sha256",
        "overlap_sha256",
    }, "source receipts")
    normalized: dict[str, str | None] = {}
    for key, item in receipt.items():
        if item is None and not admitted:
            normalized[key] = None
        else:
            normalized[key] = _digest(item, f"source receipt {key}")
    if admitted and any(value is None for value in normalized.values()):
        raise ValueError("admitted source requires every receipt")
    return normalized


def validate_source_registry(value: Any) -> dict[str, Any]:
    """Validate a registry. Candidate metadata is never task-admissible."""

    registry = _fields(value, {
        "schema", "created_at", "policy_sha256", "protected_evaluation", "sources",
    }, "source registry")
    if registry["schema"] != SOURCE_REGISTRY_SCHEMA:
        raise ValueError("unsupported source registry schema")
    if not isinstance(registry["created_at"], str) or not registry["created_at"]:
        raise ValueError("source registry created_at is required")
    _digest(registry["policy_sha256"], "source registry policy_sha256")

    if not isinstance(registry["protected_evaluation"], list):
        raise ValueError("protected_evaluation must be a list")
    protected = []
    for index, raw in enumerate(registry["protected_evaluation"]):
        panel = _fields(raw, {
            "name", "manifest_sha256", "repositories", "family_ids",
            "task_identities", "fixture_tree_digests", "intent_digests",
            "validator_spec_digests",
        }, f"protected evaluation {index}")
        _name(panel["name"], f"protected evaluation {index} name")
        _digest(panel["manifest_sha256"], f"protected evaluation {index} manifest")
        normalized_panel = dict(panel)
        repositories = _strings(panel["repositories"], "protected repositories")
        if any(repository != repository.lower() for repository in repositories):
            raise ValueError("protected repository identities must be lowercase")
        normalized_panel["repositories"] = repositories
        normalized_panel["family_ids"] = _strings(panel["family_ids"], "protected families")
        for key in ("task_identities", "fixture_tree_digests", "intent_digests",
                    "validator_spec_digests"):
            normalized_panel[key] = [
                _digest(item, f"protected {key}") for item in _strings(panel[key], f"protected {key}")
            ]
        protected.append(normalized_panel)
    protected_manifests = {panel["manifest_sha256"] for panel in protected}
    if not set(CONSUMED_PANEL_MANIFEST_SHA256S).issubset(protected_manifests):
        raise ValueError("registry must protect the consumed V3 and V4 manifests")

    if not isinstance(registry["sources"], list) or not registry["sources"]:
        raise ValueError("source registry must contain at least one source")
    sources, source_ids = [], set()
    for index, raw in enumerate(registry["sources"]):
        source = _fields(raw, {
            "id", "kind", "status", "url", "revision", "framework_license",
            "underlying_repository_policy", "task_count_claim", "receipts", "notes",
        }, f"source {index}")
        source_id = _name(source["id"], f"source {index} id")
        if source_id in source_ids:
            raise ValueError("duplicate source id")
        source_ids.add(source_id)
        if source["kind"] not in SOURCE_KINDS:
            raise ValueError(f"source {source_id} kind is invalid")
        if source["status"] not in SOURCE_STATUSES:
            raise ValueError(f"source {source_id} status is invalid")
        parsed = urlparse(source["url"] if isinstance(source["url"], str) else "")
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"source {source_id} requires an HTTPS URL")
        if not isinstance(source["revision"], str) or not _REVISION.fullmatch(source["revision"]):
            raise ValueError(f"source {source_id} requires an immutable hexadecimal revision")
        if not isinstance(source["framework_license"], str) or not source["framework_license"]:
            raise ValueError(f"source {source_id} framework license is required")
        if source["underlying_repository_policy"] != "per-repository-audit-required":
            raise ValueError(f"source {source_id} must require per-repository license audit")
        count = source["task_count_claim"]
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise ValueError(f"source {source_id} task count claim is invalid")
        receipts = _validate_receipts(source["receipts"], admitted=source["status"] == "admitted")
        archive = receipts["source_archive_sha256"]
        if archive is not None and archive in protected_manifests:
            raise ValueError(f"source {source_id} collides with protected evaluation")
        if not isinstance(source["notes"], str):
            raise ValueError(f"source {source_id} notes must be text")
        normalized_source = dict(source)
        normalized_source["receipts"] = receipts
        sources.append(normalized_source)

    normalized = dict(registry)
    normalized["protected_evaluation"] = protected
    normalized["sources"] = sources
    return json.loads(canonical_json(normalized))


def source_registry_digest(registry: Mapping[str, Any]) -> str:
    return sha256_text(f"{SOURCE_REGISTRY_SCHEMA}\0{canonical_json(validate_source_registry(registry))}")


def _validate_argv(value: Any, name: str) -> list[str]:
    if (not isinstance(value, list) or not value or len(value) > 64
            or any(not isinstance(item, str) or not item or "\x00" in item
                   or len(item.encode()) > 8192 for item in value)):
        raise ValueError(f"{name} must be a bounded non-empty argv array")
    executable = PurePosixPath(value[0]).name
    if executable in {"bash", "dash", "fish", "sh", "zsh"}:
        raise ValueError(f"{name} may not invoke a shell")
    return list(value)


def validate_task_bundle(value: Any, *, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one executable task against an admitted source registry entry."""

    normalized_registry = validate_source_registry(registry)
    sources = {source["id"]: source for source in normalized_registry["sources"]}
    task_bundle = _fields(value, {
        "schema", "split", "task", "source", "fixture", "runtime", "limits", "validator",
    }, "task bundle")
    if task_bundle["schema"] != TASK_BUNDLE_SCHEMA:
        raise ValueError("unsupported task bundle schema")
    split = task_bundle["split"]
    if split not in SPLITS:
        raise ValueError("task split must be train or development")

    task = _fields(task_bundle["task"], {
        "namespace", "family_id", "identity", "generator_source_digest",
        "fixture_tree_digest", "intent_digest", "prompt", "difficulty",
    }, "task")
    namespace = _name(task["namespace"], "task namespace")
    family = _name(task["family_id"], "task family_id")
    required_prefix = "e97-train-" if split == "train" else "e97-dev-"
    if not namespace.startswith(required_prefix):
        raise ValueError("task namespace does not match split")
    if any(namespace.startswith(prefix) or family.startswith(prefix)
           for prefix in CONSUMED_ID_PREFIXES):
        raise ValueError("consumed evaluation identity is inadmissible")
    generator_digest = _digest(task["generator_source_digest"], "generator source digest")
    fixture_digest = _digest(task["fixture_tree_digest"], "fixture tree digest")
    intent_digest = _digest(task["intent_digest"], "intent digest")
    if intent_digest != canonical_intent_digest(task["prompt"]):
        raise ValueError("intent digest does not bind the exact task prompt")
    if task["identity"] != task_identity(
        namespace=namespace,
        family_id=family,
        generator_source_digest=generator_digest,
        fixture_tree_digest=fixture_digest,
        intent_digest=intent_digest,
    ):
        raise ValueError("task identity is not bound to immutable task inputs")
    if (isinstance(task["difficulty"], bool) or not isinstance(task["difficulty"], int)
            or not 1 <= task["difficulty"] <= 7):
        raise ValueError("task difficulty must be an integer from 1 through 7")

    source = _fields(task_bundle["source"], {
        "registry_id", "kind", "repository", "revision", "source_record_digest",
        "license_receipt_digest",
    }, "task source")
    source_id = _name(source["registry_id"], "task source registry_id")
    registered = sources.get(source_id)
    if registered is None or registered["status"] != "admitted":
        raise ValueError("task source is not admitted")
    if source["kind"] != registered["kind"]:
        raise ValueError("task source kind does not match registry")
    if (not isinstance(source["repository"], str) or "/" not in source["repository"]
            or source["repository"].startswith("/") or source["repository"].endswith("/")
            or source["repository"] != source["repository"].lower()):
        raise ValueError("task source repository must be a lowercase owner/name identity")
    if source["revision"] != registered["revision"]:
        raise ValueError("task source revision does not match registry")
    _digest(source["source_record_digest"], "task source record digest")
    if source["license_receipt_digest"] != registered["receipts"]["license_sha256"]:
        raise ValueError("task source license receipt does not match registry")

    fixture = _fields(task_bundle["fixture"], {
        "artifact_path", "artifact_bytes", "artifact_sha256", "tree_digest",
    }, "fixture")
    if (not isinstance(fixture["artifact_path"], str) or not fixture["artifact_path"]
            or fixture["artifact_path"].startswith("/") or ".." in fixture["artifact_path"].split("/")):
        raise ValueError("fixture artifact path must be bounded and relative")
    if (isinstance(fixture["artifact_bytes"], bool) or not isinstance(fixture["artifact_bytes"], int)
            or not 0 < fixture["artifact_bytes"] <= 1 << 30):
        raise ValueError("fixture artifact byte count is invalid")
    _digest(fixture["artifact_sha256"], "fixture artifact digest")
    if fixture["tree_digest"] != fixture_digest:
        raise ValueError("fixture tree digest mismatch")

    runtime = _fields(task_bundle["runtime"], {
        "schema_digest", "tool_schema_digest", "controller_digest",
        "sandbox_image_digest", "system_prompt_sha256",
    }, "runtime")
    for key, item in runtime.items():
        _digest(item, f"runtime {key}")

    limits = _fields(task_bundle["limits"], {
        "turns", "seconds", "output_bytes", "disk_bytes", "processes",
    }, "limits")
    maxima = {"turns": 64, "seconds": 1800, "output_bytes": 1 << 20,
              "disk_bytes": 20 << 30, "processes": 256}
    for key, maximum in maxima.items():
        item = limits[key]
        if isinstance(item, bool) or not isinstance(item, int) or not 0 < item <= maximum:
            raise ValueError(f"task limit {key} is invalid")

    validator = _fields(task_bundle["validator"], {
        "spec_digest", "focused_argv", "regression_argv", "milestone_digest",
        "minefield_digest",
    }, "validator")
    for key in ("spec_digest", "milestone_digest", "minefield_digest"):
        _digest(validator[key], f"validator {key}")
    _validate_argv(validator["focused_argv"], "validator focused_argv")
    _validate_argv(validator["regression_argv"], "validator regression_argv")

    protected = normalized_registry["protected_evaluation"]
    collisions = []
    for panel in protected:
        if source["repository"] in panel["repositories"]:
            collisions.append(f"repository:{panel['name']}")
        for key, observed in (("family_ids", family), ("task_identities", task["identity"]),
                              ("fixture_tree_digests", fixture_digest),
                              ("intent_digests", intent_digest),
                              ("validator_spec_digests", validator["spec_digest"])):
            if observed in panel[key]:
                collisions.append(f"{key}:{panel['name']}")
    if collisions:
        raise ValueError("task collides with protected evaluation: " + ",".join(collisions))

    return json.loads(canonical_json(dict(task_bundle)))


def validate_task_collection(
    tasks: Sequence[Mapping[str, Any]], *, registry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate collection uniqueness and whole-family/repository split isolation."""

    if not tasks:
        raise ValueError("task collection is empty")
    normalized = [validate_task_bundle(task, registry=registry) for task in tasks]
    unique_fields = {
        "task identity": [task["task"]["identity"] for task in normalized],
        "fixture tree": [task["task"]["fixture_tree_digest"] for task in normalized],
        "source record": [task["source"]["source_record_digest"] for task in normalized],
        "validator input": [sha256_text(canonical_json({
            "fixture": task["task"]["fixture_tree_digest"],
            "spec": task["validator"]["spec_digest"],
            "prompt": task["task"]["intent_digest"],
        })) for task in normalized],
    }
    for name, values in unique_fields.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {name} identity")

    train = [task for task in normalized if task["split"] == "train"]
    development = [task for task in normalized if task["split"] == "development"]
    train_families = {task["task"]["family_id"] for task in train}
    dev_families = {task["task"]["family_id"] for task in development}
    if train_families.intersection(dev_families):
        raise ValueError("train and development families must be disjoint")
    train_repos = {task["source"]["repository"] for task in train}
    dev_repos = {task["source"]["repository"] for task in development}
    if train_repos.intersection(dev_repos):
        raise ValueError("train and development repositories must be disjoint")
    return normalized
