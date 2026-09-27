#!/usr/bin/env python3
"""Build deterministic complete-record pack descriptors for masked SFT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np

from ndm.data.masked_sft_dataset import (
    AUTHORITY_SCHEMA, BOUNDARY_PACK_SCHEMA, PACK_INDEX, PACK_SCHEMA,
    RECORD_INDEX, sha256, snapshot_manifest,
)


def _atomic(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def _entry(path: Path) -> dict:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def _authority_payload(root: Path, descriptor: object, name: str) -> Path:
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--authority-manifest-sha256", required=True)
    parser.add_argument("--include-source", action="append", default=[],
                        help="Exact metadata source to retain; repeat for a union")
    parser.add_argument(
        "--max-records-per-pack", type=int, default=0,
        help="Maximum complete records per pack; 0 keeps greedy packing unbounded")
    parser.add_argument(
        "--boundary-aware", action="store_true",
        help="Emit v2 packs with per-document resets and effective target counts")
    parser.add_argument(
        "--sampler-mode", choices=("hash-replacement", "epoch-permutation"),
        default="hash-replacement",
        help="Bind the intended counter sampler into the immutable pack manifest")
    parser.add_argument("--diagnostic-cpu-system-gate", action="store_true",
                        help="Allow only an explicitly non-trainable mechanical authority")
    args = parser.parse_args()
    if args.context_size <= 0:
        raise SystemExit("context-size must be positive")
    if args.max_records_per_pack < 0:
        raise SystemExit("max-records-per-pack must be nonnegative")
    authority_manifest_path = args.authority_root / "manifest.json"
    try:
        authority = snapshot_manifest(
            authority_manifest_path, args.authority_manifest_sha256, name="authority")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if authority.get("schema") != AUTHORITY_SCHEMA or authority.get("status") != "complete":
        raise SystemExit("input is not a complete masked-SFT authority")
    training_eligible = authority.get("training_eligible")
    if not isinstance(training_eligible, bool):
        raise SystemExit("authority training eligibility must be an explicit boolean")
    if not training_eligible and not args.diagnostic_cpu_system_gate:
        raise SystemExit("non-trainable authority requires --diagnostic-cpu-system-gate")
    outputs_info = authority.get("outputs")
    if not isinstance(outputs_info, dict) or set(outputs_info) != {"tokens", "mask", "index", "metadata"}:
        raise SystemExit("authority outputs are invalid")
    # Every immutable payload is verified before any record offsets are trusted.
    token_path = _authority_payload(args.authority_root, outputs_info["tokens"], "token payload")
    mask_path = _authority_payload(args.authority_root, outputs_info["mask"], "target mask")
    record_path = _authority_payload(args.authority_root, outputs_info["index"], "record index")
    metadata_path = _authority_payload(args.authority_root, outputs_info["metadata"], "record metadata")
    records = np.memmap(
        record_path, mode="r",
        dtype=np.dtype([("offset", "<u8"), ("tokens", "<u8"),
                        ("targets", "<u8"), ("split", "u1"), ("pad", "V7")]))
    masks = None
    if args.boundary_aware:
        masks = np.memmap(mask_path, mode="r", dtype="u1")
    include_sources = tuple(sorted(set(args.include_source)))
    included_record_ids = None
    if include_sources:
        # metadata_path was integrity-checked with all immutable authority payloads above.
        wanted = set(include_sources)
        included_record_ids = set()
        metadata_count = 0
        with metadata_path.open() as stream:
            for record_id, line in enumerate(stream):
                metadata_count += 1
                if json.loads(line).get("source") in wanted:
                    included_record_ids.add(record_id)
        if metadata_count != len(records):
            raise SystemExit("record metadata/index count mismatch")
        if not included_record_ids:
            raise SystemExit("source filter selected no records")
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pack_records": args.output_root / "pack_records.uint32.bin",
        "train_index": args.output_root / "train_packs.idx",
        "validation_index": args.output_root / "validation_packs.idx",
    }
    if any(path.exists() for path in (*outputs.values(), args.output_root / "manifest.json")):
        raise SystemExit("refusing to overwrite an existing pack authority")
    temporary = {name: _atomic(path) for name, path in outputs.items()}
    for path in temporary.values():
        path.unlink(missing_ok=True)

    sequence_tokens = args.context_size + 1
    split_receipts = {}
    record_cursor = 0
    try:
        with (temporary["pack_records"].open("wb", buffering=4 << 20) as record_out,
              temporary["train_index"].open("wb", buffering=4 << 20) as train_out,
              temporary["validation_index"].open("wb", buffering=1 << 20) as validation_out):
            for split_name, split_value, index_out in (
                    ("train", 0, train_out), ("validation", 1, validation_out)):
                current_ids = []
                current_tokens = 0
                current_targets = 0
                counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0,
                          "packs": 0, "excluded_oversize_records": 0,
                          "excluded_oversize_tokens": 0,
                          "excluded_oversize_assistant_target_tokens": 0}

                def flush() -> None:
                    nonlocal current_ids, current_tokens, current_targets, record_cursor
                    if not current_ids:
                        return
                    record_out.write(struct.pack(f"<{len(current_ids)}I", *current_ids))
                    index_out.write(PACK_INDEX.pack(
                        record_cursor, len(current_ids), current_tokens, current_targets))
                    record_cursor += len(current_ids)
                    counts["packs"] += 1
                    current_ids, current_tokens, current_targets = [], 0, 0

                for record_id, record in enumerate(records):
                    if (int(record["split"]) != split_value
                            or (included_record_ids is not None
                                and record_id not in included_record_ids)):
                        continue
                    token_count = int(record["tokens"])
                    if args.boundary_aware:
                        source = int(record["offset"])
                        # The first token in every independent record has no
                        # same-document predecessor and therefore cannot be a
                        # causal target in a packed example.
                        target_count = int(masks[source + 1:source + token_count].sum())
                    else:
                        target_count = int(record["targets"])
                    if token_count > sequence_tokens:
                        counts["excluded_oversize_records"] += 1
                        counts["excluded_oversize_tokens"] += token_count
                        counts["excluded_oversize_assistant_target_tokens"] += target_count
                        continue
                    if current_ids and (
                            current_tokens + token_count > sequence_tokens
                            or (args.max_records_per_pack > 0
                                and len(current_ids) >= args.max_records_per_pack)):
                        flush()
                    current_ids.append(record_id)
                    current_tokens += token_count
                    current_targets += target_count
                    counts["records"] += 1
                    counts["tokens"] += token_count
                    counts["assistant_target_tokens"] += target_count
                flush()
                split_receipts[split_name] = counts
        for name, path in outputs.items():
            temporary[name].replace(path)
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    del records
    if masks is not None:
        del masks

    manifest = {
        "schema": (BOUNDARY_PACK_SCHEMA if args.boundary_aware else PACK_SCHEMA),
        "status": "complete",
        "authority_manifest_sha256": args.authority_manifest_sha256,
        "training_eligible": training_eligible,
        "diagnostic_system_gate": ("cpu-system-gate" if not training_eligible else None),
        "context_size": args.context_size, "sequence_tokens": sequence_tokens,
        "packing": "stable-record-order greedy next-fit; no record splitting",
        "max_records_per_pack": (args.max_records_per_pack or None),
        "source_filter": ({"include_exact": list(include_sources)}
                          if include_sources else None),
        "sampler_mode": args.sampler_mode,
        "sampling": (
            "pack IDs traverse deterministic fixed-world epoch permutations"
            if args.sampler_mode == "epoch-permutation"
            else "pack IDs sampled with replacement by emender-record-pack-counter-v1"),
        "fields": ({
            "tokens": "uint32[context_size+1] token-aligned",
            "loss_mask": "bool[context_size] prediction-aligned",
            "valid_mask": "bool[context_size+1] token-aligned",
            "reset_before": "bool[context_size+1] token-aligned",
        } if args.boundary_aware else None),
        "boundary_semantics": ({
            "reset": "clear every recurrent layer before each independent record",
            "cross_document_target": "masked",
            "padding": "loss-masked recurrent identity transition",
        } if args.boundary_aware else None),
        "splits": split_receipts,
        "record_index_bytes": RECORD_INDEX.size, "pack_index_bytes": PACK_INDEX.size,
        "outputs": {name: _entry(path) for name, path in outputs.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
