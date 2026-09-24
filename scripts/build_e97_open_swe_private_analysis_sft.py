#!/usr/bin/env python3
"""Build matched action-only or bounded-private-analysis Open-SWE SFT.

The converter is opt-in and leaves the legacy action-only builder unchanged.
Both arms share the stricter private-analysis admission gate. It excludes
complete trajectories on any analysis or 64K logical-unit overflow; no
reasoning, action, observation, or final is silently truncated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import struct
from typing import Any

import pyarrow.parquet as pq
import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_agent_protocol import (
    E97_PI_AGENT_ANALYSIS_SYSTEM_V1, E97_PI_AGENT_SYSTEM_V2,
    MAX_PRIVATE_ANALYSIS_BYTES,
)
from scripts import build_e97_tulu3_sft as codec
from scripts.build_e97_open_swe_sft import (
    ALLOWED_LICENSES,
    DATASET_CARD_SHA256,
    DATASET_ID,
    DATASET_REVISION,
    EXCLUDED_REPOSITORIES,
    action_from_call,
    normalize_text,
)

ROLE_PREFIX = {"system": "System:\n", "user": "User:\n", "assistant": "Assistant:\n", "tool": "Tool:\n"}
ANALYSIS_PROTOCOL = "e97-pi-agent-analysis-v1"
_ANALYSIS_ENCODING: Any = None
_MAX_RECORD_TOKENS = 65_536
_ANALYSIS_TOKEN_CAP = 2_048
_REPRESENTATION = "private-analysis"
_SOURCE_SNAPSHOTS = {
    "builder_source": Path(__file__).read_bytes(),
    "action_normalizer_source": Path(action_from_call.__code__.co_filename).read_bytes(),
    "codec_source": Path(codec.__file__).read_bytes(),
    "protocol_source": Path(__file__).parents[1].joinpath("ndm/e97_agent_protocol.py").read_bytes(),
    "dataset_source": Path(__file__).parents[1].joinpath("ndm/data/masked_sft_dataset.py").read_bytes(),
}


def worker_init() -> None:
    global _ANALYSIS_ENCODING
    codec._worker_init()
    _ANALYSIS_ENCODING = tiktoken.get_encoding("p50k_base")


def split(identity: str) -> int:
    digest = hashlib.sha256(f"open-swe-private-analysis-v1\0{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % 100 == 0


def _finish_from_call(call: dict[str, Any]) -> str:
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "finish":
        raise ValueError("invalid_finish_call")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_tool_arguments") from exc
    if not isinstance(arguments, dict):
        raise ValueError("non_object_tool_arguments")
    message = " ".join(str(arguments.get("message", "Task completed.")).split())
    if not message:
        raise ValueError("empty_finish")
    return f"Final: {message}"


def _reasoning_prefix(reasoning: str, *, encoding: Any = None) -> tuple[str, int]:
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("empty_private_analysis")
    if len(reasoning.encode("utf-8")) > MAX_PRIVATE_ANALYSIS_BYTES:
        raise ValueError("private_analysis_byte_cap")
    active_encoding = encoding or _ANALYSIS_ENCODING
    if active_encoding is None:
        raise RuntimeError("analysis tokenizer is not initialized")
    token_count = len(active_encoding.encode(reasoning, disallowed_special=()))
    if token_count > _ANALYSIS_TOKEN_CAP:
        raise ValueError("private_analysis_token_cap")
    encoded = json.dumps(reasoning, ensure_ascii=False, separators=(",", ":"))
    return f"Analysis: {encoded}\n", token_count


def normalize_private_messages(row: dict[str, Any], *, encoding: Any = None):
    """Return complete product-protocol messages and per-unit analysis counts."""

    source = row.get("messages")
    if not isinstance(source, list) or len(source) < 4:
        raise ValueError("missing_messages")
    user = next((normalize_text(str(item.get("content", ""))) for item in source if item.get("role") == "user"), "")
    if not user:
        raise ValueError("missing_user")
    normalized: list[tuple[str, str]] = [
        ("system", E97_PI_AGENT_ANALYSIS_SYSTEM_V1), ("user", user),
    ]
    unit_analysis_tokens: list[int] = []
    pending_reasoning: list[str] = []
    source_think_turns = 0
    index = 0
    while index < len(source):
        message = source[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            index += 1
            continue
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("assistant_requires_one_tool_call")
        call = calls[0]
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        reasoning = str(message.get("reasoning_content") or "")
        next_message = source[index + 1] if index + 1 < len(source) else None
        next_tool = (
            normalize_text(str(next_message.get("content", "")))
            if isinstance(next_message, dict) and next_message.get("role") == "tool"
            else None
        )
        if name == "think":
            if next_tool != "Your thought has been logged.":
                raise ValueError("noncanonical_think_observation")
            if not reasoning.strip():
                raise ValueError("empty_private_analysis")
            pending_reasoning.append(reasoning)
            source_think_turns += 1
            index += 2
            continue
        combined_reasoning = "\n\n".join([*pending_reasoning, reasoning]).strip()
        pending_reasoning.clear()
        prefix, reasoning_tokens = _reasoning_prefix(combined_reasoning, encoding=encoding)
        if name == "finish":
            normalized.append(("assistant", prefix + _finish_from_call(call)))
            unit_analysis_tokens.append(reasoning_tokens)
            index += 1
            break
        kind, action = action_from_call(call, next_tool)
        if kind != "action" or not isinstance(action, str):
            raise ValueError(f"unsupported_product_turn:{name}")
        if next_tool is None:
            raise ValueError("action_without_tool_result")
        normalized.append(("assistant", prefix + action))
        normalized.append(("tool", next_tool or "(no tool output)"))
        unit_analysis_tokens.append(reasoning_tokens)
        index += 2
    if pending_reasoning:
        raise ValueError("orphan_think_reasoning")
    if normalized[-1][0] != "assistant" or "\nFinal:" not in normalized[-1][1]:
        raise ValueError("missing_finish")
    if len(unit_analysis_tokens) < 2:
        raise ValueError("no_actions")
    return normalized, unit_analysis_tokens, source_think_turns


def action_only_messages(messages: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove only canonical analysis frames after private-branch admission."""

    converted = [("system", E97_PI_AGENT_SYSTEM_V2), messages[1]]
    for role, content in messages[2:]:
        if role != "assistant":
            converted.append((role, content))
            continue
        analysis_line, separator, body = content.partition("\n")
        if not separator or not analysis_line.startswith("Analysis: ") or not body:
            raise ValueError("invalid_private_analysis_frame")
        # Decode to prove that delimiter-looking reasoning cannot alter the body
        # chosen for the action-only arm.
        try:
            reasoning = json.loads(analysis_line[len("Analysis: "):])
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_private_analysis_frame") from exc
        if not isinstance(reasoning, str):
            raise ValueError("invalid_private_analysis_frame")
        converted.append((role, body))
    return converted


def encode_record(messages: list[tuple[str, str, bool]]):
    pieces = []
    for index, (role, content, targeted) in enumerate(messages):
        if index:
            pieces.append(("\n\n", False))
        pieces.append((ROLE_PREFIX[role], False))
        pieces.append((content, targeted))
    pieces.append((codec.RS, True))
    return codec._encode_pieces(pieces)


def segment_complete_units(messages: list[tuple[str, str]], max_tokens: int):
    base = [(role, content, False) for role, content in messages[:2]]
    if len(encode_record(base)[0]) >= max_tokens:
        raise ValueError("oversize_system_and_user")
    units: list[list[tuple[str, str, bool]]] = []
    index = 2
    while index < len(messages):
        role, content = messages[index]
        if role != "assistant":
            raise ValueError("invalid_normalized_role_order")
        unit = [(role, content, True)]
        if index + 1 < len(messages) and messages[index + 1][0] == "tool":
            tool_role, tool_content = messages[index + 1]
            unit.append((tool_role, tool_content, False))
            index += 1
        units.append(unit)
        index += 1
    records: list[list[tuple[str, str, bool]]] = []
    current = list(base)
    previous_context: list[tuple[str, str, bool]] | None = None
    for unit in units:
        if len(encode_record(current + unit)[0]) <= max_tokens:
            current += unit
            previous_context = [(role, content, False) for role, content, _ in unit]
            continue
        if any(targeted for _, _, targeted in current):
            records.append(current)
        candidate = list(base)
        if previous_context and len(encode_record(candidate + previous_context + unit)[0]) <= max_tokens:
            candidate += previous_context
        if len(encode_record(candidate + unit)[0]) > max_tokens:
            candidate = list(base)
        if len(encode_record(candidate + unit)[0]) > max_tokens:
            raise ValueError("oversize_complete_logical_unit")
        current = candidate + unit
        previous_context = [(role, content, False) for role, content, _ in unit]
    if any(targeted for _, _, targeted in current):
        records.append(current)
    if sum(sum(targeted for _, _, targeted in record) for record in records) != len(units):
        raise RuntimeError("segmentation duplicated or dropped a target unit")
    return records


def serialize_item(item):
    source_file, row_index, row = item
    identity = f"open-swe:{row.get('trajectory_id', source_file + ':' + str(row_index))}"
    receipt = {"identity": identity, "source_file": source_file, "source_row": row_index}
    repo = str(row.get("repo", "")).strip()
    license_name = str(row.get("license", "")).strip()
    if row.get("resolved") != 1:
        return {"receipt": {**receipt, "status": "excluded", "reason": "not_resolved"}}
    if repo.lower() in EXCLUDED_REPOSITORIES:
        return {"receipt": {**receipt, "status": "excluded", "reason": "holdout_repository"}}
    if license_name not in ALLOWED_LICENSES:
        return {"receipt": {**receipt, "status": "excluded", "reason": f"license:{license_name}"}}
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("git_hack_attempted"):
        return {"receipt": {**receipt, "status": "excluded", "reason": "git_hack_attempted"}}
    try:
        messages, analysis_tokens, think_turns = normalize_private_messages(row)
        # Private segmentation is always the admission gate, including for the
        # matched action-only arm. Thus a private-analysis overflow cannot
        # leave a trajectory present in only one representation.
        private_segments = segment_complete_units(messages, _MAX_RECORD_TOKENS)
        if _REPRESENTATION == "private-analysis":
            segments = private_segments
        else:
            segments = segment_complete_units(
                action_only_messages(messages), _MAX_RECORD_TOKENS)
    except ValueError as exc:
        return {"receipt": {**receipt, "status": "excluded", "reason": str(exc)}}
    encoded_records = []
    trajectory_split = int(split(identity))
    for segment_index, segment in enumerate(segments):
        tokens, masks, complete = encode_record(segment)
        encoded_records.append({
            "identity": f"{identity}:{_REPRESENTATION}-window:{segment_index:04d}",
            "trajectory_identity": identity,
            "source_file": source_file,
            "source_row": row_index,
            "repo": repo,
            "license": license_name,
            "split": trajectory_split,
            "tokens": len(tokens),
            "targets": sum(masks),
            "target_units": sum(targeted for _, _, targeted in segment),
            "token_bytes": struct.pack(f"<{len(tokens)}I", *tokens),
            "mask_bytes": bytes(masks),
            "serialization_sha256": hashlib.sha256(complete.encode()).hexdigest(),
        })
    return {
        "records": encoded_records,
        "receipt": {
            **receipt, "status": "included", "segments": len(encoded_records),
            "source_units": len(analysis_tokens), "source_think_turns_folded": think_turns,
            "private_analysis_tokens": sum(analysis_tokens),
            "max_private_analysis_tokens": max(analysis_tokens),
        },
    }


def rows(paths):
    for path in paths:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row_index, row in enumerate(batch.to_pylist(), offset):
                yield path.name, row_index, row
            offset += batch.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-record-tokens", type=int, default=65_536)
    parser.add_argument("--analysis-token-cap", type=int, default=2_048)
    parser.add_argument(
        "--representation", choices=("private-analysis", "action-only"),
        default="private-analysis",
    )
    args = parser.parse_args()
    global _MAX_RECORD_TOKENS, _ANALYSIS_TOKEN_CAP, _REPRESENTATION
    _MAX_RECORD_TOKENS = args.max_record_tokens
    _ANALYSIS_TOKEN_CAP = args.analysis_token_cap
    _REPRESENTATION = args.representation
    if _MAX_RECORD_TOKENS < 512 or _ANALYSIS_TOKEN_CAP <= 0:
        raise SystemExit("record and analysis token caps are invalid")
    cache_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    cache_object = Path(cache_root or "") / codec.TOKENIZER_CACHE_KEY
    if not cache_root or not cache_object.is_file() or sha256(cache_object) != codec.TOKENIZER_SHA256:
        raise SystemExit("verified p50k cache is required")
    worker_init()
    if sha256(args.input_root / "README.md") != DATASET_CARD_SHA256:
        raise SystemExit("Open-SWE dataset card mismatch")
    source_manifest_path = args.source_authority / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    paths = []
    for record in source_manifest.get("input_files", []):
        path = next(iter(args.input_root.rglob(record["name"])), None)
        if path is None or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise SystemExit(f"source mismatch: {record.get('name')}")
        paths.append(path)
    if len(paths) != 18:
        raise SystemExit(f"expected 18 pinned OpenHands shards, found {len(paths)}")

    args.output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    outputs = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    trajectory_receipts_path = args.output_root / "trajectory_receipts.jsonl"
    source_snapshot_paths = {}
    for name, payload in _SOURCE_SNAPSHOTS.items():
        path = args.output_root / f"{name}.py"
        path.write_bytes(payload)
        source_snapshot_paths[name] = path
    counts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    offset = 0
    iterator = rows(paths)
    if args.limit:
        iterator = itertools.islice(iterator, args.limit)
    with outputs["tokens"].open("xb") as token_out, outputs["mask"].open("xb") as mask_out, \
         outputs["index"].open("xb") as index_out, outputs["metadata"].open("x") as metadata_out, \
         trajectory_receipts_path.open("x") as receipt_out, \
         mp.Pool(args.workers, initializer=worker_init) as pool:
        for result in pool.imap(serialize_item, iterator, chunksize=2):
            counts["input_trajectories"] += 1
            receipt = result["receipt"]
            receipt_out.write(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n")
            if receipt["status"] != "included":
                exclusions[receipt["reason"]] += 1
                continue
            counts["included_trajectories"] += 1
            counts["source_units"] += receipt["source_units"]
            counts["source_think_turns_folded"] += receipt["source_think_turns_folded"]
            counts["private_analysis_tokens"] += receipt["private_analysis_tokens"]
            for record in result["records"]:
                token_out.write(record.pop("token_bytes"))
                mask_out.write(record.pop("mask_bytes"))
                index_out.write(RECORD_INDEX.pack(offset, record["tokens"], record["targets"], record["split"]))
                metadata_out.write(json.dumps(record, sort_keys=True) + "\n")
                offset += record["tokens"]
                counts["records"] += 1
                counts["tokens"] += record["tokens"]
                counts["assistant_target_tokens"] += record["targets"]
                counts["target_units"] += record["target_units"]
                counts["validation_records" if record["split"] else "train_records"] += 1
    manifest = {
        "schema": AUTHORITY_SCHEMA,
        "status": "complete",
        "training_eligible": False,
        "promotion_status": "candidate representation experiment only",
        "purpose": f"verified OpenHands {_REPRESENTATION} matched protocol comparison",
        "representation": _REPRESENTATION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_card_sha256": DATASET_CARD_SHA256,
        "source_action_authority_manifest_sha256": sha256(source_manifest_path),
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "pyarrow_version": importlib.metadata.version("pyarrow"),
            "tiktoken_version": importlib.metadata.version("tiktoken"),
        },
        "agent_protocol": (
            ANALYSIS_PROTOCOL if _REPRESENTATION == "private-analysis"
            else "e97-pi-agent-legacy-v1"),
        "system_prompt": (
            E97_PI_AGENT_ANALYSIS_SYSTEM_V1 if _REPRESENTATION == "private-analysis"
            else E97_PI_AGENT_SYSTEM_V2),
        "analysis_policy": {
            "tokenizer": "p50k_base", "admission_token_cap": _ANALYSIS_TOKEN_CAP,
            "admission_byte_cap": MAX_PRIVATE_ANALYSIS_BYTES,
            "included_in_assistant_targets": _REPRESENTATION == "private-analysis",
            "oversize": "exclude complete trajectory in both arms; never truncate or silently summarize",
            "think_turns": "fold reasoning into the next executable action; accept only canonical think acknowledgement",
        },
        "segmentation_policy": {
            "max_record_tokens": _MAX_RECORD_TOKENS,
            "boundary": "whole assistant-analysis-action/tool-observation units",
            "reset": "every emitted record is an independent recurrent document",
            "overlap": "at most one previous complete unit as untargeted context when it fits",
            "oversize": "exclude complete trajectory if any complete logical unit cannot fit",
        },
        "filters": {
            "resolved": 1, "licenses": sorted(ALLOWED_LICENSES),
            "excluded_repositories": sorted(EXCLUDED_REPOSITORIES), "git_hack_attempted": False,
        },
        "counts": dict(counts),
        "exclusions": dict(sorted(exclusions.items())),
        "input_files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths],
        "outputs": {
            name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "auxiliary_artifacts": {
            "trajectory_receipts": {
                "path": trajectory_receipts_path.name,
                "bytes": trajectory_receipts_path.stat().st_size,
                "sha256": sha256(trajectory_receipts_path),
            }
        },
        "source_snapshots": {
            name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in source_snapshot_paths.items()
        },
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
