#!/usr/bin/env python3
"""Replay one generated first-party task with exact archive and runtime binding."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from ndm.e97_atomic import publish_bytes_no_replace
from ndm.e97_first_party_read_observe import safe_extract_fixture_archive, validate_replay
from ndm.e97_onpolicy_records import canonical_json, sha256_text
from ndm.e97_task_lake import validate_task_bundle, validate_source_registry


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True); parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--registry", type=Path, required=True); parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True); parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--private-spec", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--runtime-schema-digest", required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha(args.bundle) != args.bundle_sha256 or _sha(args.registry) != args.registry_sha256 or _sha(args.archive) != args.archive_sha256:
        raise SystemExit("exact input digest verification failed")
    try:
        registry = validate_source_registry(json.loads(args.registry.read_text()))
        bundle = validate_task_bundle(json.loads(args.bundle.read_text()), registry=registry)
        if bundle["fixture"]["artifact_sha256"] != args.archive_sha256: raise ValueError("archive does not bind task")
        spec = json.loads(args.private_spec.read_text())
        if sha256_text(canonical_json(spec)) != bundle["validator"]["spec_digest"]: raise ValueError("private validator spec digest mismatch")
        terminal = json.loads(args.receipt.read_text())
        with tempfile.TemporaryDirectory(prefix="e97-first-party-replay-") as temporary:
            root = Path(temporary) / "fixture"
            root.mkdir()
            safe_extract_fixture_archive(
                args.archive,
                root,
                expected_sha256=args.archive_sha256,
                expected_tree_digest=bundle["task"]["fixture_tree_digest"],
                disk_limit=bundle["limits"]["disk_bytes"],
            )
            result = validate_replay(bundle, spec, root, terminal, runtime_schema_digest=args.runtime_schema_digest)
    except (ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        raise SystemExit(f"validation failed: {exc}") from exc
    try:
        publish_bytes_no_replace(args.output, (canonical_json(result) + "\n").encode("utf-8"))
    except ValueError as exc:
        raise SystemExit(f"validation failed: {exc}") from exc
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
