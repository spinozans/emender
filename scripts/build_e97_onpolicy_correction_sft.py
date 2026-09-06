#!/usr/bin/env python3
"""Build a masked-SFT authority from verified student-state corrections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_onpolicy_records import (
    CONSUMED_PANEL_MANIFEST_SHA256S,
    CONSUMED_V3_MANIFEST_SHA256,
    CONSUMED_V4_MANIFEST_SHA256,
    RECOVERY_RECORD_SCHEMA,
    recovery_record_fingerprint,
    validate_recovery_record,
)
from scripts.build_e97_pi_finalization_repair_sft import serialize_live_aligned
from scripts.build_e97_pi_instruction_sft import ENCODING


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def accepted_set_digest(fingerprints: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(fingerprints)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-record-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.max_record_tokens <= 0:
        raise SystemExit("max-record-tokens must be positive")
    if sha256(args.source_jsonl) != args.source_sha256:
        raise SystemExit("source JSONL SHA-256 mismatch")

    rows = []
    for line_number, line in enumerate(args.source_jsonl.read_text().splitlines(), start=1):
        if not line.strip():
            raise SystemExit(f"blank source record at line {line_number}")
        try:
            rows.append(validate_recovery_record(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"invalid source record at line {line_number}: {exc}") from exc
    if not rows:
        raise SystemExit("source JSONL contains no records")

    fingerprints = [recovery_record_fingerprint(row) for row in rows]
    if len(fingerprints) != len(set(fingerprints)):
        raise SystemExit("duplicate correction record fingerprint")
    task_identities = [row["task"]["identity"] for row in rows]
    if len(task_identities) != len(set(task_identities)):
        raise SystemExit("at most one correction record per task identity is allowed in v1")
    fixture_identities = [row["task"]["fixture_tree_digest"] for row in rows]
    if len(fixture_identities) != len(set(fixture_identities)):
        raise SystemExit("duplicate fixture tree identity")
    validator_inputs = [row["validator_receipt"]["input_digest"] for row in rows]
    if len(validator_inputs) != len(set(validator_inputs)):
        raise SystemExit("duplicate validator input identity")

    # A single authority must bind one exact student and runtime.  Mixed versions
    # require an explicit higher-level mixture rather than silent co-mingling.
    identity_fields = {
        "checkpoint_sha256": {row["student"]["checkpoint_sha256"] for row in rows},
        "runtime_schema_digest": {row["runtime"]["schema_digest"] for row in rows},
        "controller_digest": {row["runtime"]["controller_digest"] for row in rows},
        "system_prompt_sha256": {row["runtime"]["system_prompt_sha256"] for row in rows},
        "tool_schema_digest": {row["runtime"]["tool_schema_digest"] for row in rows},
    }
    mixed = [name for name, values in identity_fields.items() if len(values) != 1]
    if mixed:
        raise SystemExit(f"mixed correction authority identities: {mixed}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    encoding = tiktoken.get_encoding(ENCODING)
    counts = {
        "records": 0,
        "tokens": 0,
        "assistant_target_tokens": 0,
        "zero_loss_context_tokens": 0,
        "train_records": 0,
        "validation_records": 0,
    }
    family_counts: dict[str, int] = {}
    teacher_counts: dict[str, int] = {}
    offset = 0
    with paths["tokens"].open("wb") as token_out, paths["mask"].open("wb") as mask_out, \
         paths["index"].open("wb") as index_out, paths["metadata"].open("w") as metadata_out:
        for row, fingerprint in zip(rows, fingerprints):
            messages = [(message["role"], message["text"]) for message in row["messages"]]
            targets = {index for index, message in enumerate(row["messages"])
                       if message["role"] == "assistant" and message["loss"] == 1}
            tokens, masks, complete = serialize_live_aligned(
                messages,
                encoding,
                target_mode="all-assistant",
                target_assistant_positions=targets,
                target_terminal_newline=True,
            )
            if len(tokens) > args.max_record_tokens:
                raise SystemExit(
                    f"record {fingerprint} has {len(tokens)} tokens; truncation is forbidden")
            target_count = sum(masks)
            split = int(row["split"] == "validation")
            token_out.write(struct.pack(f"<{len(tokens)}I", *tokens))
            mask_out.write(bytes(masks))
            index_out.write(RECORD_INDEX.pack(offset, len(tokens), target_count, split))
            metadata_out.write(json.dumps({
                "id": f"e97-correction-{fingerprint[:24]}",
                "record_fingerprint": fingerprint,
                "task_identity": row["task"]["identity"],
                "task_namespace": row["task"]["namespace"],
                "family_id": row["task"]["family_id"],
                "split": split,
                "teacher_tier": row["teacher"]["tier"],
                "student_checkpoint_sha256": row["student"]["checkpoint_sha256"],
                "tokens": len(tokens),
                "targets": target_count,
                "serialization_sha256": hashlib.sha256(complete.encode()).hexdigest(),
                "validator_receipt_sha256": hashlib.sha256(
                    json.dumps(row["validator_receipt"], sort_keys=True,
                               separators=(",", ":")).encode()).hexdigest(),
            }, sort_keys=True) + "\n")
            offset += len(tokens)
            counts["records"] += 1
            counts["tokens"] += len(tokens)
            counts["assistant_target_tokens"] += target_count
            counts["zero_loss_context_tokens"] += len(tokens) - target_count
            counts["validation_records" if split else "train_records"] += 1
            family = row["task"]["family_id"]
            family_counts[family] = family_counts.get(family, 0) + 1
            tier = row["teacher"]["tier"]
            teacher_counts[tier] = teacher_counts.get(tier, 0) + 1

    manifest = {
        "schema": AUTHORITY_SCHEMA,
        "status": "complete",
        "purpose": "verified E97 student-state correction with zero-loss failure prefixes",
        "source_record_schema": RECOVERY_RECORD_SCHEMA,
        "source_jsonl": str(args.source_jsonl.resolve()),
        "source_sha256": args.source_sha256,
        "tokenizer": ENCODING,
        "target_policy": "contiguous verified corrective assistant suffix only",
        "truncation_policy": "reject",
        "max_record_tokens": args.max_record_tokens,
        "accepted_record_set_sha256": accepted_set_digest(fingerprints),
        "identity": {name: next(iter(values)) for name, values in identity_fields.items()},
        "forbidden_evaluation": {
            "manifest_sha256s": list(CONSUMED_PANEL_MANIFEST_SHA256S),
            "v3_manifest_sha256": CONSUMED_V3_MANIFEST_SHA256,
            "v3": "diagnostic-only identities and source digest rejected",
            "v4_manifest_sha256": CONSUMED_V4_MANIFEST_SHA256,
            "v4": "diagnostic-only identities and source digest rejected",
        },
        "counts": counts,
        "family_counts": family_counts,
        "teacher_counts": teacher_counts,
        "outputs": {name: entry(path) for name, path in paths.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
