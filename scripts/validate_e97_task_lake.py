#!/usr/bin/env python3
"""Validate an immutable E97 task-lake registry and optional task JSONL."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from ndm.data.masked_sft_dataset import sha256
from ndm.e97_task_lake import (
    source_registry_digest,
    validate_source_registry,
    validate_task_collection,
)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing receipt: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--tasks-jsonl", type=Path)
    parser.add_argument("--tasks-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.tasks_jsonl is None) != (args.tasks_sha256 is None):
        raise SystemExit("tasks-jsonl and tasks-sha256 must be supplied together")
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
            "protected identity collision checks passed",
            "whole-family and whole-repository split isolation passed",
        ],
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
