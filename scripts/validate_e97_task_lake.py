#!/usr/bin/env python3
"""Validate an immutable E97 task-lake registry and optional task JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ndm.e97_atomic import publish_bytes_no_replace, read_regular_file_no_follow
from ndm.e97_task_lake import (
    source_registry_digest,
    validate_source_registry,
    validate_task_collection,
)


def atomic_write_json(path: Path, value: object) -> None:
    """Publish one immutable validation receipt; identical retries are harmless."""

    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    publish_bytes_no_replace(path, payload)


def load_registry_snapshot(path: Path, expected_sha256: str) -> tuple[dict[str, object], str]:
    """Hash and parse exactly one bounded, descriptor-safe registry snapshot."""

    try:
        payload = read_regular_file_no_follow(path, maximum=8 << 20)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("source registry cannot be snapshotted safely") from exc
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("source registry SHA-256 mismatch")
    try:
        return validate_source_registry(json.loads(payload)), observed_sha256
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid source registry: {exc}") from exc


def load_task_snapshot(path: Path, expected_sha256: str) -> tuple[list[object], str]:
    """Hash and parse exactly one bounded task JSONL snapshot before validation."""

    try:
        payload = read_regular_file_no_follow(path, maximum=64 << 20)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("task JSONL cannot be snapshotted safely") from exc
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("task JSONL SHA-256 mismatch")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("task JSONL is not UTF-8") from exc
    if not lines:
        raise ValueError("task JSONL is empty")
    parsed: list[object] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank task record at line {line_number}")
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid task JSON at line {line_number}: {exc}") from exc
    return parsed, observed_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--tasks-jsonl", type=Path)
    parser.add_argument("--tasks-sha256")
    parser.add_argument("--overlap-receipt", type=Path)
    parser.add_argument("--archive-root-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.tasks_jsonl is None) != (args.tasks_sha256 is None):
        raise SystemExit("tasks-jsonl and tasks-sha256 must be supplied together")
    if args.tasks_jsonl is not None and (args.overlap_receipt is None or args.archive_root_sha256 is None):
        raise SystemExit("task collection validation requires a protected overlap receipt and archive root SHA-256")
    if args.tasks_jsonl is None and (args.overlap_receipt is not None or args.archive_root_sha256 is not None):
        raise SystemExit("overlap evidence is only valid with a task collection")
    try:
        registry, registry_sha256 = load_registry_snapshot(
            args.registry, args.registry_sha256)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.tasks_jsonl is not None:
        try:
            parsed, _task_source_sha256 = load_task_snapshot(
                args.tasks_jsonl, args.tasks_sha256)
            validate_task_collection(parsed, registry=registry)
        except ValueError as exc:
            raise SystemExit(f"invalid task collection: {exc}") from exc
        # This generic registry validator has no generation/admission/operator
        # tuple and is therefore intentionally unable to turn structural overlap
        # JSON into collection admission.  The canonical admission operation is
        # the sole reader of the checked-in authorization allowlist.
        raise SystemExit("collection admission requires the canonical operator authorization tuple")

    sources = registry["sources"]
    report = {
        "schema": "emender-e97-task-lake-validation-v1",
        "status": "pass",
        "registry": {
            "path": str(args.registry),
            "file_sha256": registry_sha256,
            "identity_sha256": source_registry_digest(registry),
            "sources": len(sources),
            "candidate_sources": sum(source["status"] == "candidate" for source in sources),
            "admitted_sources": sum(source["status"] == "admitted" for source in sources),
            "protected_panels": len(registry["protected_evaluation"]),
        },
        "task_validation": "not-run",
        "claims": [
            "schema-valid registry",
            "consumed V3/V4 manifests protected",
            "registry task-source policy requires admitted sources",
            "static protected metadata is schema-bound; semantic clearance requires a per-collection overlap receipt",
        ],
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
