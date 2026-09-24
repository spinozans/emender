#!/usr/bin/env python3
"""Raw-corpus audit for the E97 real-human-chat intake (LMSYS-Chat-1M + WildChat).

Reads the pinned local snapshots directly (parquet) and reports, per corpus:
language distribution, turn-count distribution, per-model conversation counts,
moderation/toxicity flag rates, PII-redaction flag rates, role-structure
validity, and the exact filter funnel counts the admission builder applies
(English, model selection, moderation, schema, packability). Shard-parallel
(CPU only); the receipt records pinned dataset-card and shard SHA-256s.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import pyarrow.parquet as pq

SUPPORTED_ROLES = {"user", "assistant"}
COLUMNS = ["conversation_id", "model", "conversation", "turn", "language",
           "openai_moderation", "toxic", "redacted"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flagged(entry) -> bool:
    """Moderation annotation entry: flagged or any category true. Fails closed."""
    if not isinstance(entry, dict):
        return True
    if entry.get("flagged"):
        return True
    categories = entry.get("categories")
    return isinstance(categories, dict) and any(categories.values())


def audit_shard(task):
    path, corpus = task
    counts = {
        "rows": 0, "english": 0, "redacted": 0, "toxic_field": 0,
        "moderation_flagged": 0, "first_message_empty": 0,
        "moderation_length": 0,
    }
    languages = Counter()
    models = Counter()
    turn_buckets = Counter()
    funnel = Counter()
    role_errors = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=COLUMNS):
        for row in batch.to_pylist():
            counts["rows"] += 1
            language = row.get("language")
            languages[language] += 1
            models[row.get("model")] += 1
            if language == "English":
                counts["english"] += 1
            if row.get("redacted"):
                counts["redacted"] += 1
            if corpus == "wildchat" and row.get("toxic"):
                counts["toxic_field"] += 1
            messages = row.get("conversation") or []
            moderation = row.get("openai_moderation") or []
            if len(moderation) != len(messages):
                funnel["moderation_annotation_incomplete"] += 1
            if corpus == "wildchat" and (row.get("toxic") or any(
                    isinstance(m, dict) and m.get("toxic") for m in messages)):
                funnel["toxic"] += 1
            if any(_flagged(entry) for entry in moderation):
                funnel["moderation_flagged"] += 1
                counts["moderation_flagged"] += 1
            if not messages:
                funnel["empty_conversation"] += 1
            elif isinstance(messages[0], dict) and not (messages[0].get("content") or "").strip():
                funnel["empty_first_user_turn"] += 1
                counts["first_message_empty"] += 1
            roles = [m.get("role") for m in messages if isinstance(m, dict)]
            if any(role not in SUPPORTED_ROLES for role in roles) or len(roles) != len(messages):
                funnel["unsupported_role"] += 1
            elif roles and roles[-1] != "assistant":
                funnel["trailing_unanswered_user_turn"] += 1
            if messages:
                repeats = sum(1 for i in range(len(roles) - 1) if roles[i] == roles[i + 1])
                if repeats:
                    funnel[f"consecutive_same_role_x{min(repeats, 3)}"] += 1
            turns = row.get("turn")
            turn_buckets[int(turns) if isinstance(turns, int) else -1] += 1
    return {"counts": counts, "languages": dict(languages), "models": dict(models),
            "turn_buckets": dict(turn_buckets), "funnel": dict(funnel),
            "role_errors": dict(role_errors)}


def audit_corpus(root: Path, corpus: str) -> dict:
    shards = sorted((root / "data").glob("*.parquet"))
    if not shards:
        raise SystemExit(f"{corpus}: no parquet shards under {root}/data")
    with mp.Pool(min(8, len(shards))) as pool:
        results = pool.map(audit_shard, [(path, corpus) for path in shards])
    languages = Counter()
    models = Counter()
    turn_buckets = Counter()
    funnel = Counter()
    counts = Counter()
    for result in results:
        for key, value in result["counts"].items():
            counts[key] += value
        languages.update(result["languages"])
        models.update(result["models"])
        turn_buckets.update({int(k): v for k, v in result["turn_buckets"].items()})
        funnel.update(result["funnel"])
    return {
        "corpus": corpus,
        "rows": counts["rows"],
        "english": counts["english"],
        "languages": dict(languages.most_common(40)),
        "models": dict(sorted(models.items(), key=lambda kv: -kv[1])),
        "turn_histogram": {str(k): v for k, v in sorted(turn_buckets.items()) if k >= 0},
        "funnel": dict(funnel),
        "moderation_flagged_conversations": counts["moderation_flagged"],
        "wildchat_toxic_field_conversations": counts["toxic_field"],
        "pii_redacted_conversations": counts["redacted"],
        "first_message_empty": counts["first_message_empty"],
        "shards": [{"name": path.name, "bytes": path.stat().st_size,
                    "sha256": sha256(path)} for path in shards],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmsys-root", type=Path, default=None)
    parser.add_argument("--wildchat-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    if args.lmsys_root:
        entry = audit_corpus(args.lmsys_root, "lmsys-chat-1m")
        entry["dataset_card_sha256"] = sha256(args.lmsys_root / "README.md")
        results.append(entry)
    if args.wildchat_root:
        entry = audit_corpus(args.wildchat_root, "wildchat")
        entry["dataset_card_sha256"] = sha256(args.wildchat_root / "README.md")
        results.append(entry)
    args.output.write_text(json.dumps({
        "schema": "emender-e97-real-human-chat-intake-audit-v1",
        "status": "survey-not-admission",
        "corpora": results,
        "checker_sha256": sha256(Path(__file__)),
    }, indent=2, sort_keys=True) + "\n")
    print("REAL_HUMAN_CHAT_INTAKE_AUDIT", [r["corpus"] for r in results],
          args.output)


if __name__ == "__main__":
    main()
