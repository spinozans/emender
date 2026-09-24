#!/usr/bin/env python3
"""Run the sealed protected-panel overlap gate without emitting protected content."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ndm.e97_atomic import publish_bytes_no_replace
from ndm.e97_onpolicy_records import canonical_json
from ndm.e97_protected_overlap import OverlapError, _read_path_once, check_protected_overlap
from ndm.e97_task_lake import validate_source_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--candidate-collection", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--protected-panel-manifest", type=Path, action="append", required=True)
    parser.add_argument("--protected-panel-records", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        registry_payload = _read_path_once(args.registry, name="source registry")
    except OverlapError as exc:
        raise SystemExit(f"registry cannot be snapshotted safely: {exc}") from exc
    if hashlib.sha256(registry_payload).hexdigest() != args.registry_sha256:
        raise SystemExit("registry SHA-256 mismatch")
    if len(args.protected_panel_manifest) != len(args.protected_panel_records):
        raise SystemExit("each protected manifest requires one records file")
    try:
        # Hash and parse exactly the one descriptor-safe byte snapshot.
        registry = validate_source_registry(json.loads(registry_payload.decode("utf-8")))
        receipt = check_protected_overlap(
            registry=registry,
            candidate_collection=args.candidate_collection,
            candidate_root=args.candidate_root,
            protected_panels=zip(args.protected_panel_manifest, args.protected_panel_records),
        )
        payload = (canonical_json(receipt) + "\n").encode("utf-8")
        publish_bytes_no_replace(args.output, payload)
    except (OverlapError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"protected overlap check failed: {exc}") from exc
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "pass":
        raise SystemExit("protected overlap collision detected")


if __name__ == "__main__":
    main()
