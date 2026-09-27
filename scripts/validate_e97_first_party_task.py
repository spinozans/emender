#!/usr/bin/env python3
"""Replay one generated first-party task with exact archive and runtime binding."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from ndm.e97_atomic import publish_bytes_no_replace, read_regular_file_no_follow
from ndm.e97_first_party_read_observe import safe_extract_fixture_archive, validate_replay
from ndm.e97_onpolicy_records import canonical_json, sha256_text
from ndm.e97_task_lake import validate_task_bundle, validate_source_registry


def _snapshot_input(path: Path, *, name: str, maximum: int) -> bytes:
    """Retain one descriptor-safe input before hashing, parsing, or extraction."""

    try:
        return read_regular_file_no_follow(path, maximum=maximum)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{name} cannot be snapshotted safely") from exc


def _require_sha256(payload: bytes, expected: str, name: str) -> None:
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{name} digest verification failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True); parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--registry", type=Path, required=True); parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True); parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--private-spec", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--runtime-schema-digest", required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        # No later step reopens these supplied authorities.  In particular,
        # archive extraction consumes ``archive_payload`` rather than hashing
        # one pathname and opening it again through tarfile.
        bundle_payload = _snapshot_input(args.bundle, name="bundle", maximum=8 << 20)
        registry_payload = _snapshot_input(args.registry, name="registry", maximum=8 << 20)
        archive_payload = _snapshot_input(args.archive, name="fixture archive", maximum=256 << 20)
        private_spec_payload = _snapshot_input(args.private_spec, name="private spec", maximum=8 << 20)
        receipt_payload = _snapshot_input(args.receipt, name="validator receipt", maximum=64 << 20)
        _require_sha256(bundle_payload, args.bundle_sha256, "bundle")
        _require_sha256(registry_payload, args.registry_sha256, "registry")
        _require_sha256(archive_payload, args.archive_sha256, "fixture archive")
        registry = validate_source_registry(json.loads(registry_payload))
        bundle = validate_task_bundle(json.loads(bundle_payload), registry=registry)
        if bundle["fixture"]["artifact_sha256"] != args.archive_sha256:
            raise ValueError("archive does not bind task")
        spec = json.loads(private_spec_payload)
        if sha256_text(canonical_json(spec)) != bundle["validator"]["spec_digest"]:
            raise ValueError("private validator spec digest mismatch")
        terminal = json.loads(receipt_payload)
        with tempfile.TemporaryDirectory(prefix="e97-first-party-replay-") as temporary:
            root = Path(temporary) / "fixture"
            root.mkdir()
            safe_extract_fixture_archive(
                archive_payload,
                root,
                expected_sha256=args.archive_sha256,
                expected_tree_digest=bundle["task"]["fixture_tree_digest"],
                disk_limit=bundle["limits"]["disk_bytes"],
            )
            result = validate_replay(bundle, spec, root, terminal, runtime_schema_digest=args.runtime_schema_digest)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation failed: {exc}") from exc
    try:
        publish_bytes_no_replace(args.output, (canonical_json(result) + "\n").encode("utf-8"))
    except ValueError as exc:
        raise SystemExit(f"validation failed: {exc}") from exc
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
