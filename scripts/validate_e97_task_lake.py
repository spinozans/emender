#!/usr/bin/env python3
"""Validate an immutable E97 task-lake registry and optional task JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ndm.data.masked_sft_dataset import sha256
from ndm.e97_atomic import publish_bytes_no_replace
from ndm.e97_protected_overlap import OverlapError, validate_overlap_receipt
from ndm.e97_task_lake import (
    source_registry_digest,
    validate_source_registry,
    validate_task_collection,
)


def atomic_write_json(path: Path, value: object) -> None:
    """Publish one immutable validation receipt; identical retries are harmless."""

    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    publish_bytes_no_replace(path, payload)


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
    if sha256(args.registry) != args.registry_sha256:
        raise SystemExit("source registry SHA-256 mismatch")
    try:
        registry = validate_source_registry(json.loads(args.registry.read_text()))
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid source registry: {exc}") from exc

    tasks = []
    task_source_sha256 = None
    if args.tasks_jsonl is not None:
        if sha256(args.tasks_jsonl) != args.tasks_sha256:
            raise SystemExit("task JSONL SHA-256 mismatch")
        parsed = []
        for line_number, line in enumerate(args.tasks_jsonl.read_text().splitlines(), start=1):
            if not line.strip():
                raise SystemExit(f"blank task record at line {line_number}")
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid task JSON at line {line_number}: {exc}") from exc
        try:
            tasks = validate_task_collection(parsed, registry=registry)
        except ValueError as exc:
            raise SystemExit(f"invalid task collection: {exc}") from exc
        task_source_sha256 = args.tasks_sha256
        try:
            overlap = json.loads(args.overlap_receipt.read_text())
            validate_overlap_receipt(
                overlap,
                candidate_collection_sha256=task_source_sha256,
                candidate_archive_root_sha256=args.archive_root_sha256,
            )
        except (OSError, json.JSONDecodeError, OverlapError) as exc:
            raise SystemExit(f"task collection overlap admission failed: {exc}") from exc

    sources = registry["sources"]
    report = {
        "schema": "emender-e97-task-lake-validation-v1",
        "status": "pass",
        "registry": {
            "path": str(args.registry.resolve()),
            "file_sha256": args.registry_sha256,
            "identity_sha256": source_registry_digest(registry),
            "sources": len(sources),
            "candidate_sources": sum(source["status"] == "candidate" for source in sources),
            "admitted_sources": sum(source["status"] == "admitted" for source in sources),
            "protected_panels": len(registry["protected_evaluation"]),
        },
        "tasks": {
            "path": str(args.tasks_jsonl.resolve()) if args.tasks_jsonl is not None else None,
            "file_sha256": task_source_sha256,
            "records": len(tasks),
            "train": sum(task["split"] == "train" for task in tasks),
            "development": sum(task["split"] == "development" for task in tasks),
            "families": sorted({task["task"]["family_id"] for task in tasks}),
            "repositories": sorted({task["source"]["repository"] for task in tasks}),
        },
        "claims": [
            "schema-valid registry",
            "consumed V3/V4 manifests protected",
            "only admitted sources may back tasks",
            "derived task identities and exact prompt digests verified",
            "static protected metadata is schema-bound; semantic clearance requires a per-collection overlap receipt",
            "whole-family and whole-repository split isolation passed",
        ],
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
