#!/usr/bin/env python3
"""Build an immutable masked-SFT authority from the E97 real-human-chat intake.

Admits filtered English, non-flagged, schema-valid conversations from the
pinned LMSYS-Chat-1M and WildChat snapshots (real user-model conversation
register; operator priority lane) into the same conversation-rehearsal
render the prep builder consumes via read_conversation_slice
(schema emender-e97-tulu3-masked-sft-v1, records.jsonl/records.idx/
tokens.uint32.bin/assistant_mask.uint8.bin). Serialization follows
scripts/build_e97_smoltalk2_sft.py exactly: role-prefixed plain turns
("User:\n"/"Assistant:\n"), blank line between messages, trailing RS; all
and only assistant content is supervised (masked). Multi-turn conversations
are kept whole; records that cannot pack whole (>MAX_TOKENS) are dropped
and counted. CPU only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import struct

import pyarrow.parquet as pq

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from scripts import build_e97_tulu3_sft as codec

LMSYS_DATASET_ID = "lmsys/lmsys-chat-1m"
LMSYS_DATASET_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"
LMSYS_DATASET_CARD_SHA256 = None  # bound after the gated snapshot completes
WILDCHAT_DATASET_ID = "allenai/WildChat"
WILDCHAT_DATASET_REVISION = "f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4"
TOKENIZER = "p50k_base"
ROLES = {"user": "User", "assistant": "Assistant"}
MAX_TOKENS = 65537  # whole-record boundary-pack ceiling (CONTEXT_SIZE+1, read_conversation_slice limit)
# Responding-model selection is explicit per build: the operator-directed
# quality filter (documented in the intake report). Closed-provider assistant
# outputs are admitted only where the repo already has training precedent
# (the admitted Tulu 3 authority trains on tulu_v3.9_wildchat_100k
# GPT-4/GPT-3.5 outputs; the operator previously accepted lmsys/lmsys-chat-1m
# for the 50B instruction corpus). Open-weights arena models carry no closed
# provider terms on their outputs.
WILDCHAT_ALLOWED_MODELS = frozenset({"gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview"})


def split(identity: str) -> int:
    digest = hashlib.sha256(f"real-human-chat-v1\0{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % 100 == 0


def worker_init() -> None:
    codec._worker_init()


def _message_flagged(entry) -> bool:
    if not isinstance(entry, dict):
        return True  # missing annotation fails closed
    if entry.get("flagged"):
        return True
    categories = entry.get("categories")
    return isinstance(categories, dict) and any(categories.values())


def _normalize(content: str) -> str:
    return content.replace(codec.RS, " ").replace("\r\n", "\n").replace("\r", "\n").strip()


def serialize_item(item):
    """Filter + render one conversation. Errors are per-record and counted."""
    identity, corpus, source_file, row_index, row = (
        item["identity"], item["corpus"], item["source_file"],
        item["source_row"], item["row"])
    funnel = row.get("_funnel", {})
    result_base = {"identity": identity, "source": corpus,
                   "source_file": source_file, "source_row": row_index}
    messages = row.get("conversation") or []
    # trailing unanswered user turn(s) are truncated, not supervised content
    while messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
        messages = messages[:-1]
        funnel["trailing_user_turn_truncated"] = funnel.get("trailing_user_turn_truncated", 0) + 1
    if not messages:
        return {**result_base, "error": "no_answered_turns", **funnel}
    pieces = []
    roles = []
    for message in messages:
        if not isinstance(message, dict):
            return {**result_base, "error": "invalid_message", **funnel}
        role, content = message.get("role"), message.get("content")
        if role not in ROLES:
            return {**result_base, "error": f"unsupported_role:{role}", **funnel}
        if not isinstance(content, str) or not _normalize(content):
            return {**result_base, "error": "empty_content", **funnel}
        normalized = _normalize(content)
        if pieces:
            pieces.append(("\n\n", False))
        pieces.append((f"{ROLES[role]}:\n", False))
        pieces.append((normalized, role == "assistant"))
        roles.append(role)
    if roles[-1] != "assistant":
        return {**result_base, "error": "final_message_is_not_assistant", **funnel}
    pieces.append((codec.RS, True))
    try:
        tokens, masks, complete = codec._encode_pieces(pieces)
    except ValueError as error:
        return {**result_base, "error": str(error), **funnel}
    if len(tokens) > MAX_TOKENS:
        return {**result_base, "error": "exceeds_pack_ceiling", **funnel}
    return {
        "identity": identity, "source": corpus, "model": row.get("model"),
        "turns": row.get("turn"), "language": row.get("language"),
        "pii_redacted": bool(row.get("redacted")),
        "source_file": source_file, "source_row": row_index,
        "split": int(split(identity)),
        "tokens": len(tokens), "targets": sum(masks),
        "token_bytes": struct.pack(f"<{len(tokens)}I", *tokens),
        "mask_bytes": bytes(masks),
        "serialization_sha256": hashlib.sha256(complete.encode()).hexdigest(),
        **funnel,
    }


def _prefilter(row, corpus, lmsys_models):
    """Corpus-level admission funnel. Returns (reason|None, row').

    Order: language -> model selection -> moderation/toxicity -> annotation
    completeness -> message-structure sanity. Reasons are counted by the
    builder and recorded in the manifest funnel.
    """
    if row.get("language") != "English":
        return "not_english", None
    allowed = lmsys_models if corpus == "lmsys-chat-1m" else WILDCHAT_ALLOWED_MODELS
    model = row.get("model")
    if model not in allowed:
        return "model_not_selected", None
    messages = row.get("conversation") or []
    moderation = row.get("openai_moderation") or []
    if corpus == "wildchat":
        if row.get("toxic") or any(bool(m.get("toxic")) for m in messages if isinstance(m, dict)):
            return "toxic", None
    if len(moderation) != len(messages):
        return "moderation_annotation_incomplete", None
    for entry in moderation:
        if _message_flagged(entry):
            return "moderation_flagged", None
    if not messages:
        return "empty_conversation", None
    if isinstance(messages[0], dict) and not (messages[0].get("content") or "").strip():
        return "empty_first_user_turn", None
    return None, row


def rows(paths, corpus, lmsys_models):
    for path in paths:
        offset = 0
        reader = pq.ParquetFile(path).iter_batches(batch_size=256)
        for batch in reader:
            for row_index, row in enumerate(batch.to_pylist(), offset):
                identity = f"{corpus}:{row.get('conversation_id')}"
                reason, kept = _prefilter(row, corpus, lmsys_models)
                if reason is not None:
                    yield {"identity": identity, "reason": reason,
                           "source_file": path.name, "source_row": row_index}
                    continue
                kept = dict(kept)
                kept["_funnel"] = {}
                yield {"identity": identity, "reason": None, "corpus": corpus,
                       "source_file": path.name, "source_row": row_index, "row": kept}
            offset += batch.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmsys-root", type=Path, default=None)
    parser.add_argument("--lmsys-models", default=None,
                        help="comma-separated responding-model allowlist for lmsys-chat-1m "
                             "(required with --lmsys-root; the documented quality selection)")
    parser.add_argument("--wildchat-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-ids", type=Path, default=None,
                        help="file of record identities to exclude (shingle-collision exclusion rebuild)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cache_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_root:
        raise SystemExit("TIKTOKEN_CACHE_DIR must bind the verified p50k cache")
    cache_object = Path(cache_root) / codec.TOKENIZER_CACHE_KEY
    if not cache_object.is_file() or sha256(cache_object) != codec.TOKENIZER_SHA256:
        raise SystemExit("verified p50k cache object is missing or corrupt")
    worker_init()
    exclude_ids = None
    if args.exclude_ids is not None:
        exclude_ids = {line.strip() for line in args.exclude_ids.read_text().splitlines() if line.strip()}
        if not exclude_ids:
            raise SystemExit("--exclude-ids file is empty")
    if args.lmsys_root is not None and not args.lmsys_models:
        raise SystemExit("--lmsys-root requires the documented --lmsys-models selection")
    lmsys_models = frozenset(
        m.strip() for m in (args.lmsys_models or "").split(",") if m.strip())
    inputs = []
    for root, corpus, card_expected in (
            (args.lmsys_root, "lmsys-chat-1m", LMSYS_DATASET_CARD_SHA256),
            (args.wildchat_root, "wildchat", None)):
        if root is None:
            continue
        card_sha = sha256(root / "README.md")
        if card_expected is not None and card_sha != card_expected:
            raise SystemExit(f"{corpus} dataset card mismatch")
        paths = sorted((root / "data").glob("*.parquet"))
        if not paths:
            raise SystemExit(f"{corpus}: no parquet shards")
        inputs.append((corpus, root, paths, card_sha))
    if not inputs:
        raise SystemExit("no corpora given")
    args.output_root.mkdir(parents=True, exist_ok=False)
    outputs = {"tokens": args.output_root / "tokens.uint32.bin",
               "mask": args.output_root / "assistant_mask.uint8.bin",
               "index": args.output_root / "records.idx",
               "metadata": args.output_root / "records.jsonl"}
    counts = Counter()
    errors = Counter()
    funnel = Counter()
    sources = Counter()
    seen_serializations = set()
    offset = 0
    iterator = (item for corpus, _root, paths, _sha in inputs
                for item in rows(paths, corpus, lmsys_models))
    if args.limit:
        import itertools
        iterator = itertools.islice(iterator, args.limit)
    with outputs["tokens"].open("wb") as token_out, outputs["mask"].open("wb") as mask_out, \
         outputs["index"].open("wb") as index_out, outputs["metadata"].open("w") as metadata_out, \
         mp.Pool(args.workers, initializer=worker_init) as pool:
        for result in pool.imap(serialize_item,
                                (item for item in iterator
                                 if item["reason"] is None
                                 and (exclude_ids is None or item["identity"] not in exclude_ids)),
                                chunksize=16):
            counts["input_records"] += 1
            if "error" in result:
                errors[result.pop("error")] += 1
                continue
            digest = result["serialization_sha256"]
            if digest in seen_serializations:
                errors["duplicate_serialization"] += 1
                continue
            seen_serializations.add(digest)
            truncated = result.pop("trailing_user_turn_truncated", 0)
            if truncated:
                funnel["trailing_user_turn_truncated"] += truncated
            token_out.write(result.pop("token_bytes"))
            mask_out.write(result.pop("mask_bytes"))
            index_out.write(RECORD_INDEX.pack(
                offset, result["tokens"], result["targets"], result["split"]))
            metadata_out.write(json.dumps(result, sort_keys=True) + "\n")
            offset += result["tokens"]
            counts["records"] += 1
            counts["tokens"] += result["tokens"]
            counts["assistant_target_tokens"] += result["targets"]
            counts["validation_records" if result["split"] else "train_records"] += 1
            sources[result["source"]] += 1
    # pre-filter rejections and exclusions were filtered out of the pool
    # stream; count them in a second (untokenized) pass
    excluded_identity_count = 0
    iterator = (item for corpus, _root, paths, _sha in inputs
                for item in rows(paths, corpus, lmsys_models))
    if args.limit:
        import itertools
        iterator = itertools.islice(iterator, args.limit)
    for item in iterator:
        if exclude_ids is not None and item["identity"] in exclude_ids:
            funnel["excluded_shingle_collision"] += 1
            excluded_identity_count += 1
            continue
        if item["reason"] is not None:
            funnel[item["reason"]] += 1
            counts["prefiltered_records"] = counts.get("prefiltered_records", 0) + 1
    manifest = {
        "schema": AUTHORITY_SCHEMA, "status": "complete",
        "excluded_shingle_collision_identities": excluded_identity_count,
        "purpose": ("admitted real-human-chat conversation register (LMSYS-Chat-1M + "
                    "WildChat English, moderation-filtered) for the E97 chat cohort"),
        "tokenizer": TOKENIZER, "counts": dict(counts), "errors": dict(errors),
        "prefilter_funnel": dict(funnel),
        "source_counts": dict(sources),
        "max_tokens_per_record": MAX_TOKENS,
        "license_policy": (
            "lmsys/lmsys-chat-1m: gated LMSYS-Chat-1M Dataset License Agreement — "
            "training use granted for research and commercial purposes; no dataset "
            "redistribution (weights are not the dataset); right-to-request-deletion "
            "retained by keeping this authority deletable; model-specific-terms selection "
            "documented in the intake report. allenai/WildChat: ODC-BY — training and "
            "derivative distribution permitted with attribution (cite the WildChat paper "
            "and dataset in any model card)."),
        "selection_policy": (
            "English-only (corpus language field); conversations with any OpenAI-moderation "
            "flagged message or WildChat toxic field excluded; trailing unanswered user turns "
            "truncated; empty-content records dropped; exact serialization dedup across both "
            "corpora; 1% deterministic validation split; whole-record pack ceiling 65537 tokens (CONTEXT_SIZE+1). "
            "LMSYS responding-model selection (quality tier): "
            + (",".join(sorted(lmsys_models)) if lmsys_models else "<wildchat-only build>")
            + ". WildChat: " + ",".join(sorted(WILDCHAT_ALLOWED_MODELS)) + "."),
        "inputs": [],
        "outputs": {name: {"path": path.name, "bytes": path.stat().st_size,
                           "sha256": sha256(path)} for name, path in outputs.items()},
    }
    for corpus, root, paths, card_sha in inputs:
        manifest["inputs"].append({
            "corpus": corpus,
            "dataset_id": LMSYS_DATASET_ID if corpus == "lmsys-chat-1m" else WILDCHAT_DATASET_ID,
            "dataset_revision": (LMSYS_DATASET_REVISION if corpus == "lmsys-chat-1m"
                                 else WILDCHAT_DATASET_REVISION),
            "dataset_card_sha256": card_sha,
            "files": [{"name": path.name, "bytes": path.stat().st_size,
                       "sha256": sha256(path)} for path in paths],
        })
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
