#!/usr/bin/env python3
"""Independently validate a complete-record masked-SFT pack authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ndm.data.masked_sft_dataset import (
    AUTHORITY_SCHEMA, BOUNDARY_PACK_SCHEMA, PACK_INDEX, PACK_SCHEMA, RECORD_INDEX,
    sha256, snapshot_manifest,
)


def _payload(root: Path, descriptor: object, name: str) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}:
        raise SystemExit(f"{name} descriptor is invalid")
    relative = descriptor["path"]
    candidate = Path(relative) if isinstance(relative, str) else None
    if (candidate is None or candidate.is_absolute() or len(candidate.parts) != 1
            or candidate.name != relative or relative in {"", ".", ".."}):
        raise SystemExit(f"{name} path is not publication-relative")
    path = root / candidate
    if not path.is_file() or path.stat().st_size != descriptor["bytes"] or sha256(path) != descriptor["sha256"]:
        raise SystemExit(f"{name} integrity mismatch")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--authority-manifest-sha256", required=True)
    parser.add_argument("--pack-manifest-sha256", required=True)
    parser.add_argument("--diagnostic-cpu-system-gate", action="store_true")
    args = parser.parse_args()
    authority_path = args.authority_root / "manifest.json"
    pack_path = args.pack_root / "manifest.json"
    try:
        authority = snapshot_manifest(authority_path, args.authority_manifest_sha256, name="authority")
        packs = snapshot_manifest(pack_path, args.pack_manifest_sha256, name="pack")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if authority.get("schema") != AUTHORITY_SCHEMA or authority.get("status") != "complete":
        raise SystemExit("authority manifest is not complete")
    pack_schema = packs.get("schema")
    if pack_schema not in {PACK_SCHEMA, BOUNDARY_PACK_SCHEMA} or packs.get("status") != "complete":
        raise SystemExit("pack manifest is not complete")
    boundary_aware = pack_schema == BOUNDARY_PACK_SCHEMA
    if packs.get("authority_manifest_sha256") != args.authority_manifest_sha256:
        raise SystemExit("pack manifest does not bind the token authority")
    training_eligible = authority.get("training_eligible")
    pack_training_eligible = packs.get("training_eligible")
    if (not isinstance(training_eligible, bool)
            or not isinstance(pack_training_eligible, bool)
            or pack_training_eligible is not training_eligible):
        raise SystemExit("authority/pack training eligibility must be explicit matching booleans")
    if not training_eligible and not args.diagnostic_cpu_system_gate:
        raise SystemExit("non-trainable authority requires --diagnostic-cpu-system-gate")
    if not isinstance(packs.get("outputs"), dict):
        raise SystemExit("pack outputs are invalid")
    for name, info in packs["outputs"].items():
        _payload(args.pack_root, info, f"pack output {name}")

    outputs = authority.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"tokens", "mask", "index", "metadata"}:
        raise SystemExit("authority outputs are invalid")
    # Integrity-check tokens too even though this validator only reads index/mask.
    _payload(args.authority_root, outputs["tokens"], "token authority")
    record_path = _payload(args.authority_root, outputs["index"], "record authority")
    mask_path = _payload(args.authority_root, outputs["mask"], "target-mask authority")
    metadata_path = _payload(args.authority_root, outputs["metadata"], "record metadata")
    records = np.memmap(
        record_path, mode="r", dtype=np.dtype([
            ("offset", "<u8"), ("tokens", "<u8"), ("targets", "<u8"),
            ("split", "u1"), ("pad", "V7")]))
    masks = None
    if boundary_aware:
        masks = np.memmap(mask_path, mode="r", dtype="u1")
    source_filter = packs.get("source_filter")
    source_selected = np.ones(len(records), dtype=np.bool_)
    if source_filter is not None:
        included = source_filter.get("include_exact")
        if not isinstance(included, list) or not included or any(
                not isinstance(value, str) for value in included):
            raise SystemExit("invalid pack source filter")
        # metadata_path was verified together with every immutable authority payload.
        wanted = set(included)
        selected_values = []
        with metadata_path.open() as stream:
            for line in stream:
                selected_values.append(json.loads(line).get("source") in wanted)
        if len(selected_values) != len(records):
            raise SystemExit("record metadata/index count mismatch")
        source_selected = np.asarray(selected_values, dtype=np.bool_)
    ids_info = packs["outputs"]["pack_records"]
    record_ids = np.memmap(
        _payload(args.pack_root, ids_info, "pack records"), mode="r", dtype="<u4")
    sequence_tokens = int(packs["sequence_tokens"])
    observed_ids = []
    for split_name, split_value in (("train", 0), ("validation", 1)):
        info = packs["outputs"][f"{split_name}_index"]
        index_dtype = np.dtype([
            ("record_offset", "<u8"), ("record_count", "<u8"),
            ("tokens", "<u8"), ("targets", "<u8")])
        index_path = _payload(args.pack_root, info, f"{split_name} pack index")
        index = (np.memmap(index_path, mode="r", dtype=index_dtype)
                 if int(info["bytes"]) else np.empty((0,), dtype=index_dtype))
        counts = {"packs": len(index), "records": 0, "tokens": 0,
                  "assistant_target_tokens": 0}
        for pack in index:
            start = int(pack["record_offset"])
            stop = start + int(pack["record_count"])
            ids = record_ids[start:stop]
            selected = records[ids]
            if boundary_aware:
                selected_targets = sum(
                    int(masks[int(record["offset"]) + 1:
                              int(record["offset"]) + int(record["tokens"])].sum())
                    for record in selected)
            else:
                selected_targets = int(selected["targets"].sum())
            if (len(ids) == 0 or np.any(selected["split"] != split_value)
                    or int(selected["tokens"].sum()) != int(pack["tokens"])
                    or selected_targets != int(pack["targets"])
                    or int(pack["tokens"]) > sequence_tokens):
                raise SystemExit(f"invalid {split_name} pack descriptor")
            observed_ids.extend(int(value) for value in ids)
            counts["records"] += len(ids)
            counts["tokens"] += int(pack["tokens"])
            counts["assistant_target_tokens"] += int(pack["targets"])
        expected = packs["splits"][split_name]
        for field, value in counts.items():
            if value != expected[field]:
                raise SystemExit(f"{split_name} {field} accounting mismatch")
        eligible = np.flatnonzero(
            (records["split"] == split_value) & (records["tokens"] <= sequence_tokens)
            & source_selected)
        selected_ids = np.asarray(
            [value for value in observed_ids if int(records[value]["split"]) == split_value],
            dtype=np.int64)
        if not np.array_equal(np.sort(selected_ids), eligible):
            raise SystemExit(f"{split_name} eligible records are missing or duplicated")
    if len(set(observed_ids)) != len(observed_ids):
        raise SystemExit("record appears in more than one pack")
    receipt = {
        "schema": "emender-e97-sft-pack-validation-v1", "status": "pass",
        "authority_manifest_sha256": args.authority_manifest_sha256,
        "pack_manifest_sha256": args.pack_manifest_sha256,
        "training_eligible": training_eligible,
        "records_validated": len(observed_ids),
        "packs_validated": sum(value["packs"] for value in packs["splits"].values()),
        "output_sha256": {name: value["sha256"] for name, value in packs["outputs"].items()},
    }
    destination = args.pack_root / "validation.json"
    destination.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
