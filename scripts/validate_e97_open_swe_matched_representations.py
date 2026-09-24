#!/usr/bin/env python3
"""Validate matched Open-SWE action-only/private-analysis authorities."""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
from typing import Any

import tiktoken

from ndm.data.masked_sft_dataset import RECORD_INDEX, sha256
from ndm.e97_agent_protocol import MAX_PRIVATE_ANALYSIS_BYTES, RS, parse_agent_turn
from ndm.e97_atomic import publish_bytes_no_replace

SCHEMA = "emender-e97-open-swe-matched-representation-validation-v1"
REQUIRED_SOURCES = {
    "builder_source", "action_normalizer_source", "codec_source",
    "protocol_source", "dataset_source",
}


def load_manifest(root: Path, representation: str) -> tuple[dict[str, Any], Path]:
    path = root / "manifest.json"
    value = json.loads(path.read_text())
    if value.get("status") != "complete" or value.get("training_eligible") is not False:
        raise ValueError(f"{representation} authority status is invalid")
    if value.get("representation") != representation:
        raise ValueError(f"{representation} manifest representation mismatch")
    if value.get("source_action_authority_manifest_sha256") != "382159f77d9075a99facf987a7b1f7a674fdf969762e21e0a862ec59ba303e60":
        raise ValueError(f"{representation} source authority mismatch")
    outputs = value.get("outputs")
    snapshots = value.get("source_snapshots")
    auxiliary = value.get("auxiliary_artifacts")
    if (not isinstance(outputs, dict) or set(outputs) != {"tokens", "mask", "index", "metadata"}
            or not isinstance(snapshots, dict) or set(snapshots) != REQUIRED_SOURCES
            or not isinstance(auxiliary, dict) or set(auxiliary) != {"trajectory_receipts"}):
        raise ValueError(f"{representation} output/source snapshot fields are incomplete")
    for group in (outputs, snapshots, auxiliary):
        for name, record in group.items():
            artifact = root / record["path"]
            if artifact.stat().st_size != record["bytes"] or sha256(artifact) != record["sha256"]:
                raise ValueError(f"{representation} artifact mismatch: {name}")
    return value, path


def receipt_identities(root: Path, manifest: dict[str, Any]) -> tuple[set[str], dict[str, int]]:
    receipt_path = root / manifest["auxiliary_artifacts"]["trajectory_receipts"]["path"]
    included: set[str] = set()
    status_counts: dict[str, int] = {}
    rows = 0
    with receipt_path.open() as handle:
        for line in handle:
            value = json.loads(line)
            rows += 1
            status = value.get("status")
            key = status if status == "included" else f"excluded:{value.get('reason')}"
            status_counts[key] = status_counts.get(key, 0) + 1
            identity = value.get("identity")
            if not isinstance(identity, str) or not identity:
                raise ValueError("trajectory receipt identity is invalid")
            if status == "included":
                if identity in included:
                    raise ValueError("duplicate included trajectory receipt")
                included.add(identity)
    if rows != manifest["counts"]["input_trajectories"]:
        raise ValueError("trajectory receipt count mismatch")
    if len(included) != manifest["counts"]["included_trajectories"]:
        raise ValueError("included trajectory count mismatch")
    return included, status_counts


def target_runs(mask: memoryview, start: int, length: int):
    position = start
    stop = start + length
    while position < stop:
        while position < stop and mask[position] == 0:
            position += 1
        run_start = position
        while position < stop and mask[position] == 1:
            position += 1
        if run_start < position:
            yield run_start, position


def validate_records(root: Path, manifest: dict[str, Any], representation: str) -> dict[str, Any]:
    outputs = manifest["outputs"]
    token_path = root / outputs["tokens"]["path"]
    mask_path = root / outputs["mask"]["path"]
    index_path = root / outputs["index"]["path"]
    metadata_path = root / outputs["metadata"]["path"]
    if token_path.stat().st_size % 4 or index_path.stat().st_size % RECORD_INDEX.size:
        raise ValueError(f"{representation} binary alignment is invalid")
    total_tokens = token_path.stat().st_size // 4
    if mask_path.stat().st_size != total_tokens:
        raise ValueError(f"{representation} token/mask lengths differ")
    record_count = index_path.stat().st_size // RECORD_INDEX.size
    metadata = [json.loads(line) for line in metadata_path.open()]
    if len(metadata) != record_count or record_count != manifest["counts"]["records"]:
        raise ValueError(f"{representation} record count mismatch")

    encoding = tiktoken.get_encoding("p50k_base")
    expected_offset = 0
    observed_targets = observed_units = analysis_tokens = 0
    identities: set[str] = set()
    trajectory_identities: set[str] = set()
    with token_path.open("rb") as token_file, mask_path.open("rb") as mask_file, index_path.open("rb") as index_file:
        with mmap.mmap(token_file.fileno(), 0, access=mmap.ACCESS_READ) as token_map, \
             mmap.mmap(mask_file.fileno(), 0, access=mmap.ACCESS_READ) as mask_map:
            mask_view = memoryview(mask_map)
            try:
                for record_index, record in enumerate(metadata):
                    offset, length, targets, split = RECORD_INDEX.unpack(index_file.read(RECORD_INDEX.size))
                    if offset != expected_offset or length <= 0 or length > manifest["segmentation_policy"]["max_record_tokens"]:
                        raise ValueError(f"{representation} record index continuity/bound is invalid")
                    if split not in {0, 1} or record.get("split") != split:
                        raise ValueError(f"{representation} split mismatch")
                    if any(record.get(key) != value for key, value in (
                        ("tokens", length), ("targets", targets))):
                        raise ValueError(f"{representation} metadata/index mismatch")
                    if record.get("identity") in identities:
                        raise ValueError(f"{representation} duplicate record identity")
                    identities.add(record["identity"])
                    trajectory_identities.add(record["trajectory_identity"])
                    actual_targets = sum(mask_map[offset:offset + length])
                    if actual_targets != targets:
                        raise ValueError(f"{representation} target mask count mismatch")
                    token_values = struct.unpack_from(f"<{length}I", token_map, offset * 4)
                    decoded = encoding.decode(list(token_values))
                    if not decoded.startswith("System:\n") or not decoded.endswith(RS):
                        raise ValueError(f"{representation} record framing is invalid")
                    units = 0
                    for run_start, run_stop in target_runs(mask_view, offset, length):
                        relative_start = run_start - offset
                        relative_stop = run_stop - offset
                        body = encoding.decode(list(token_values[relative_start:relative_stop]))
                        if body == RS:
                            continue
                        body = body.removesuffix(RS)
                        turn = parse_agent_turn(
                            body, private_analysis=representation == "private-analysis")
                        units += 1
                        if representation == "private-analysis":
                            reasoning = turn.private_analysis
                            if reasoning is None or len(reasoning.encode("utf-8")) > MAX_PRIVATE_ANALYSIS_BYTES:
                                raise ValueError("private analysis byte bound mismatch")
                            count = len(encoding.encode(reasoning, disallowed_special=()))
                            if count > manifest["analysis_policy"]["admission_token_cap"]:
                                raise ValueError("private analysis token bound mismatch")
                            analysis_tokens += count
                        elif turn.private_analysis is not None or body.startswith("Analysis:"):
                            raise ValueError("action-only target contains private analysis")
                    if units != record["target_units"]:
                        raise ValueError(f"{representation} complete target-unit count mismatch")
                    observed_units += units
                    observed_targets += targets
                    expected_offset += length
            finally:
                mask_view.release()
    if expected_offset != total_tokens or total_tokens != manifest["counts"]["tokens"]:
        raise ValueError(f"{representation} total token count mismatch")
    if observed_targets != manifest["counts"]["assistant_target_tokens"]:
        raise ValueError(f"{representation} total target count mismatch")
    if observed_units != manifest["counts"]["target_units"]:
        raise ValueError(f"{representation} total unit count mismatch")
    if representation == "private-analysis" and analysis_tokens != manifest["counts"]["private_analysis_tokens"]:
        raise ValueError("private analysis token total mismatch")
    return {
        "records": record_count,
        "tokens": total_tokens,
        "assistant_target_tokens": observed_targets,
        "target_units": observed_units,
        "private_analysis_tokens": analysis_tokens,
        "record_identity_sha256": hashlib.sha256(
            ("\n".join(sorted(identities)) + "\n").encode()).hexdigest(),
        "trajectory_identities": trajectory_identities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private_manifest, private_manifest_path = load_manifest(args.private_root, "private-analysis")
    action_manifest, action_manifest_path = load_manifest(args.action_root, "action-only")
    if private_manifest["source_snapshots"]["builder_source"]["sha256"] != action_manifest["source_snapshots"]["builder_source"]["sha256"]:
        raise SystemExit("representations were not built from identical builder bytes")
    private_receipts, private_statuses = receipt_identities(args.private_root, private_manifest)
    action_receipts, action_statuses = receipt_identities(args.action_root, action_manifest)
    if private_receipts != action_receipts or private_statuses != action_statuses:
        raise SystemExit("representation trajectory receipts are not exactly matched")
    private_records = validate_records(args.private_root, private_manifest, "private-analysis")
    action_records = validate_records(args.action_root, action_manifest, "action-only")
    if private_records.pop("trajectory_identities") != private_receipts:
        raise SystemExit("private record trajectories do not match included receipts")
    if action_records.pop("trajectory_identities") != action_receipts:
        raise SystemExit("action record trajectories do not match included receipts")
    if private_records["target_units"] != action_records["target_units"]:
        raise SystemExit("representation target logical units differ")
    identity_payload = "\n".join(sorted(private_receipts)) + "\n"
    result = {
        "schema": SCHEMA,
        "status": "passed",
        "training_eligible": False,
        "private_manifest_sha256": sha256(private_manifest_path),
        "action_manifest_sha256": sha256(action_manifest_path),
        "builder_source_sha256": private_manifest["source_snapshots"]["builder_source"]["sha256"],
        "validator_source_sha256": sha256(Path(__file__)),
        "included_trajectories": len(private_receipts),
        "included_trajectory_identity_sha256": hashlib.sha256(identity_payload.encode()).hexdigest(),
        "receipt_status_counts": private_statuses,
        "private_analysis": private_records,
        "action_only": action_records,
        "matched_target_units": private_records["target_units"],
        "policy": "candidate representation evidence only; no training promotion",
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    publish_bytes_no_replace(args.output, payload, mode=0o600)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(payload).hexdigest(), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
