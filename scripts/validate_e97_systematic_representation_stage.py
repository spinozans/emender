#!/usr/bin/env python3
"""Independently validate the sealed systematic representation stage."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import mmap
from pathlib import Path
from typing import Any

import numpy as np

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_atomic import publish_bytes_no_replace
from scripts.build_e97_systematic_representation_stage import (
    MATCHED_PACK_VALIDATION_SHA256, MATCHED_TRAJECTORY_SET_SHA256,
    MATCHED_VALIDATION_SHA256, SCHEMA, SOURCES, TARGETS,
)

VALIDATION_SCHEMA = "emender-e97-systematic-representation-stage-validation-v1"


def output_path(root: Path, descriptor: Any, name: str) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}:
        raise ValueError(f"invalid {name} descriptor")
    value = descriptor["path"]
    candidate = Path(value) if isinstance(value, str) else None
    if candidate is None or candidate.is_absolute() or len(candidate.parts) != 1:
        raise ValueError(f"invalid {name} publication path")
    path = root / candidate
    if not path.is_file() or path.stat().st_size != int(descriptor["bytes"]) or sha256(path) != descriptor["sha256"]:
        raise ValueError(f"{name} payload mismatch")
    return path


def validate_arm(root: Path, representation: str, expected_manifest: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if sha256(manifest_path) != expected_manifest:
        raise ValueError(f"{representation}: manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != AUTHORITY_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("training_eligible") is not True
            or manifest.get("stage_schema") != SCHEMA
            or manifest.get("representation") != representation):
        raise ValueError(f"{representation}: authority identity/status mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"tokens", "mask", "index", "metadata"}:
        raise ValueError(f"{representation}: output set mismatch")
    paths = {name: output_path(root, descriptor, f"{representation}:{name}")
             for name, descriptor in outputs.items()}
    records = np.memmap(paths["index"], mode="r", dtype=np.dtype([
        ("offset", "<u8"), ("tokens", "<u8"), ("targets", "<u8"),
        ("split", "u1"), ("pad", "V7")]))
    token_file = paths["tokens"].open("rb")
    mask_file = paths["mask"].open("rb")
    token_map = mmap.mmap(token_file.fileno(), 0, access=mmap.ACCESS_READ)
    mask_map = mmap.mmap(mask_file.fileno(), 0, access=mmap.ACCESS_READ)
    metadata_rows = 0
    offset = total_targets = 0
    unique_ids: set[str] = set()
    unique_source_records: set[tuple[str, int]] = set()
    shared: dict[str, list[tuple[int, int]]] = {name: [] for name in ("conversation", "compositional", "core-retention", "documents")}
    agent_trajectories: set[str] = set()
    agent_occurrences: Counter[int] = Counter()
    try:
        with paths["metadata"].open() as handle:
            for record_id, line in enumerate(handle):
                metadata_rows += 1
                value = json.loads(line)
                record = records[record_id]
                length = int(record["tokens"])
                targets = int(record["targets"])
                if int(record["offset"]) != offset or int(record["split"]) != 0:
                    raise ValueError(f"{representation}: record continuity/split mismatch")
                if not 1 < length <= 65_537 or targets <= 0 or targets > length:
                    raise ValueError(f"{representation}: record bounds mismatch")
                if value.get("tokens") != length or value.get("targets") != targets or value.get("split") != 0:
                    raise ValueError(f"{representation}: metadata/index mismatch")
                identity = value.get("identity_sha256")
                if not isinstance(identity, str) or len(identity) != 64 or identity in unique_ids:
                    raise ValueError(f"{representation}: invalid/duplicate record identity")
                unique_ids.add(identity)
                actual_targets = sum(mask_map[offset:offset + length])
                if actual_targets != targets or len(token_map[offset * 4:(offset + length) * 4]) != length * 4:
                    raise ValueError(f"{representation}: mask/token range mismatch")
                source = value.get("source")
                source_record_id = value.get("source_record_id")
                occurrence = value.get("source_occurrence")
                source_manifest = value.get("source_manifest_sha256")
                if not isinstance(source_record_id, int) or not isinstance(occurrence, int) or occurrence not in {0, 1}:
                    raise ValueError(f"{representation}: source occurrence identity invalid")
                unique_source_records.add((str(source_manifest), source_record_id))
                if source == "agent":
                    trajectory = value.get("trajectory_identity")
                    if not isinstance(trajectory, str) or not trajectory:
                        raise ValueError(f"{representation}: missing agent trajectory")
                    agent_trajectories.add(trajectory)
                    agent_occurrences[occurrence] += 1
                    expected_source = SOURCES[representation][1]
                elif source in shared:
                    if occurrence != 0 or value.get("trajectory_identity") is not None:
                        raise ValueError(f"{representation}: shared source occurrence mismatch")
                    shared[source].append((source_record_id, occurrence))
                    expected_source = SOURCES[source][1]
                else:
                    raise ValueError(f"{representation}: unknown source {source!r}")
                if source_manifest != expected_source:
                    raise ValueError(f"{representation}: source manifest mismatch")
                offset += length
                total_targets += targets
    finally:
        del records
        token_map.close(); mask_map.close(); token_file.close(); mask_file.close()
    counts = manifest["counts"]
    if (metadata_rows != int(counts["records"]) or offset != int(counts["tokens"])
            or total_targets != int(counts["assistant_target_tokens"])
            or int(counts["validation_records"]) != 0
            or len(unique_source_records) != int(counts["unique_source_records"])):
        raise ValueError(f"{representation}: aggregate counts mismatch")
    if not 49_900_000 <= total_targets <= 50_100_000:
        raise ValueError(f"{representation}: target horizon is outside frozen tolerance")
    if len(unique_source_records) < 10_000:
        raise ValueError(f"{representation}: fewer than 10K unique source records")
    if representation == "private-analysis" and set(agent_occurrences) != {0}:
        raise ValueError("private-analysis arm repeats agent records")
    if representation == "action-only" and not {0, 1}.issuperset(agent_occurrences):
        raise ValueError("action-only occurrence policy mismatch")
    return {
        "manifest_sha256": expected_manifest,
        "records": metadata_rows,
        "tokens": offset,
        "assistant_target_tokens": total_targets,
        "unique_source_records": len(unique_source_records),
        "agent_trajectories": agent_trajectories,
        "agent_occurrence_records": dict(agent_occurrences),
        "shared": shared,
        "record_identity_sha256": hashlib.sha256(("\n".join(sorted(unique_ids)) + "\n").encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.stage_root / "stage-receipt.json"
    stage_receipt = json.loads(receipt_path.read_text())
    if (stage_receipt.get("schema") != SCHEMA or stage_receipt.get("status") != "complete"
            or stage_receipt.get("training_eligible") is not True):
        raise SystemExit("stage receipt identity/status mismatch")
    recipe_sha256 = sha256(args.recipe)
    if stage_receipt.get("recipe_sha256") != recipe_sha256:
        raise SystemExit("stage recipe mismatch")
    selections = stage_receipt.get("selection_receipts", {})
    if (selections.get("matched_validation", {}).get("sha256") != MATCHED_VALIDATION_SHA256
            or selections.get("matched_pack_validation", {}).get("sha256") != MATCHED_PACK_VALIDATION_SHA256
            or selections.get("matched_trajectory_identity_sha256") != MATCHED_TRAJECTORY_SET_SHA256):
        raise SystemExit("stage matched-source evidence mismatch")
    private = validate_arm(
        args.stage_root / "private-analysis", "private-analysis",
        stage_receipt["private_analysis"]["manifest_sha256"])
    action = validate_arm(
        args.stage_root / "action-only", "action-only",
        stage_receipt["action_only"]["manifest_sha256"])
    if private["agent_trajectories"] != action["agent_trajectories"]:
        raise SystemExit("arms do not contain the same unique agent trajectories")
    if private["shared"] != action["shared"]:
        raise SystemExit("arms do not contain the exact same shared replay records")
    target_difference = action["assistant_target_tokens"] - private["assistant_target_tokens"]
    if abs(target_difference) > 50_000:
        raise SystemExit("arms exceed the 0.1% integral target-matching bound")
    for value in (private, action):
        value.pop("agent_trajectories")
        value.pop("shared")
    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed",
        "training_eligible": True,
        "stage_receipt_sha256": sha256(receipt_path),
        "recipe_sha256": recipe_sha256,
        "matched_validation_sha256": MATCHED_VALIDATION_SHA256,
        "matched_pack_validation_sha256": MATCHED_PACK_VALIDATION_SHA256,
        "selected_agent_trajectory_identity_sha256": selections["selected_agent_trajectory_identity_sha256"],
        "selected_agent_trajectories": selections["selected_agent_trajectories"],
        "private_analysis": private,
        "action_only": action,
        "assistant_target_difference": target_difference,
        "relative_target_difference": abs(target_difference) / private["assistant_target_tokens"],
        "validator_source_sha256": sha256(Path(__file__)),
        "claims": [
            "all authority manifest and payload hashes verified",
            "all record offsets, masks, counts, splits, and source bindings verified",
            "same unique Open-SWE trajectories and exact shared replay records",
            "at least 10K unique source records per arm",
            "approximately 50M assistant targets with less than 0.1% arm difference",
        ],
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    publish_bytes_no_replace(args.output, payload, mode=0o600)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(payload).hexdigest(), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
