#!/usr/bin/env python3
"""Audit complete-trajectory effects of private-analysis message caps.

This reads only the pinned local Open-SWE source files.  It never emits source
reasoning text: oversize messages are identified by immutable coordinates,
lengths, and content digests.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import tiktoken

from scripts.build_e97_open_swe_sft import (
    ALLOWED_LICENSES,
    EXCLUDED_REPOSITORIES,
    normalize_messages,
)

SCHEMA = "emender-e97-open-swe-private-analysis-cap-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_eligible(row: dict[str, Any]) -> bool:
    repo = str(row.get("repo", "")).strip().lower()
    license_name = str(row.get("license", "")).strip()
    metadata = row.get("metadata") or {}
    return (
        repo not in EXCLUDED_REPOSITORIES
        and license_name in ALLOWED_LICENSES
        and not (isinstance(metadata, dict) and metadata.get("git_hack_attempted"))
    )


def add_trajectory(summary: dict[str, Counter], group: str, trajectory: dict[str, Any], cap: int) -> None:
    target = summary[group]
    excluded = trajectory["max_reasoning_tokens"] > cap
    target["trajectories"] += 1
    target["reasoning_messages"] += trajectory["reasoning_messages"]
    target["reasoning_tokens"] += trajectory["reasoning_tokens"]
    target["assistant_messages"] += trajectory["assistant_messages"]
    target["tool_calls"] += trajectory["tool_calls"]
    if excluded:
        target["excluded_trajectories"] += 1
        target["excluded_reasoning_messages"] += trajectory["reasoning_messages"]
        target["excluded_reasoning_tokens"] += trajectory["reasoning_tokens"]
        target["excluded_assistant_messages"] += trajectory["assistant_messages"]
        target["excluded_tool_calls"] += trajectory["tool_calls"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--action-authority", type=Path, required=True)
    parser.add_argument("--prior-token-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[1024, 2048, 4096])
    args = parser.parse_args()
    caps = sorted(set(args.caps))
    if not caps or caps[0] <= 0:
        raise SystemExit("caps must be positive")

    authority_manifest_path = args.action_authority / "manifest.json"
    authority = json.loads(authority_manifest_path.read_text())
    prior = json.loads(args.prior_token_audit.read_text())
    if prior.get("source_authority_manifest_sha256") != sha256_file(authority_manifest_path):
        raise SystemExit("prior audit does not bind the selected action authority")

    input_files = authority.get("input_files")
    if not isinstance(input_files, list) or not input_files:
        raise SystemExit("action authority lacks pinned input files")
    paths: list[Path] = []
    for record in input_files:
        path = args.source_root / record["name"]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise SystemExit(f"source mismatch: {path}")
        paths.append(path)

    encoding = tiktoken.get_encoding("p50k_base")
    trajectories: list[dict[str, Any]] = []
    oversize_2048: list[dict[str, Any]] = []
    normalizing_errors: Counter[str] = Counter()
    repository_exclusions: Counter[str] = Counter()
    language_exclusions: Counter[str] = Counter()

    for path in paths:
        offset = 0
        columns = ["resolved", "trajectory_id", "repo", "license", "language", "metadata", "messages"]
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=64):
            for row_index, row in enumerate(batch.to_pylist(), offset):
                if row.get("resolved") != 1:
                    continue
                messages = row.get("messages") or []
                lengths: list[int] = []
                assistant_messages = tool_calls = 0
                for message_index, message in enumerate(messages):
                    if not isinstance(message, dict) or message.get("role") != "assistant":
                        continue
                    assistant_messages += 1
                    calls = message.get("tool_calls") or []
                    if isinstance(calls, list):
                        tool_calls += len(calls)
                    reasoning = str(message.get("reasoning_content") or "")
                    token_count = len(encoding.encode(reasoning, disallowed_special=()))
                    lengths.append(token_count)
                    if token_count > 2048:
                        oversize_2048.append({
                            "source_file": path.name,
                            "source_row": row_index,
                            "trajectory_id": str(row.get("trajectory_id", "")),
                            "repository": str(row.get("repo", "")),
                            "language": str(row.get("language", "")),
                            "assistant_message_index": message_index,
                            "reasoning_tokens": token_count,
                            "reasoning_utf8_bytes": len(reasoning.encode("utf-8")),
                            "reasoning_sha256": sha256_text(reasoning),
                        })
                eligible = source_eligible(row)
                normalizable = False
                if eligible:
                    try:
                        normalize_messages(row)
                        normalizable = True
                    except ValueError as exc:
                        normalizing_errors[str(exc)] += 1
                trajectory = {
                    "source_eligible": eligible,
                    "action_normalizable": normalizable,
                    "reasoning_messages": len(lengths),
                    "reasoning_tokens": sum(lengths),
                    "max_reasoning_tokens": max(lengths, default=0),
                    "assistant_messages": assistant_messages,
                    "tool_calls": tool_calls,
                    "repository": str(row.get("repo", "")),
                    "language": str(row.get("language", "")),
                }
                trajectories.append(trajectory)
            offset += batch.num_rows

    effects: dict[str, Any] = {}
    for cap in caps:
        groups = {
            "all_resolved": Counter(),
            "source_eligible": Counter(),
            "action_normalizable": Counter(),
        }
        for trajectory in trajectories:
            add_trajectory(groups, "all_resolved", trajectory, cap)
            if trajectory["source_eligible"]:
                add_trajectory(groups, "source_eligible", trajectory, cap)
            if trajectory["action_normalizable"]:
                add_trajectory(groups, "action_normalizable", trajectory, cap)
            if cap == 2048 and trajectory["max_reasoning_tokens"] > cap:
                repository_exclusions[trajectory["repository"]] += 1
                language_exclusions[trajectory["language"]] += 1
        normalized_groups = {}
        for name, counter in groups.items():
            values = dict(counter)
            values["retained_trajectories"] = values.get("trajectories", 0) - values.get("excluded_trajectories", 0)
            values["complete_trajectory_policy"] = "exclude if any assistant reasoning message exceeds cap"
            normalized_groups[name] = values
        effects[str(cap)] = normalized_groups

    oversize_2048.sort(key=lambda item: (
        -item["reasoning_tokens"], item["source_file"], item["source_row"], item["assistant_message_index"]))
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "training_eligible": False,
        "source_authority_manifest_sha256": sha256_file(authority_manifest_path),
        "prior_token_audit_sha256": sha256_file(args.prior_token_audit),
        "tokenizer": "p50k_base",
        "caps": caps,
        "effects": effects,
        "oversize_2048_messages": oversize_2048,
        "oversize_2048_repository_trajectory_counts": dict(sorted(repository_exclusions.items())),
        "oversize_2048_language_trajectory_counts": dict(sorted(language_exclusions.items())),
        "normalization_errors": dict(sorted(normalizing_errors.items())),
        "policy": (
            "measurement only; no source reasoning text retained; no truncation or summarization authorized; "
            "complete trajectories are the exclusion unit"
        ),
        "inputs": [{
            "name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path),
        } for path in paths],
    }
    payload = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(args.output, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            args.output.unlink()
        except FileNotFoundError:
            pass
        raise
    print(json.dumps({
        "output": str(args.output), "sha256": hashlib.sha256(payload).hexdigest(),
        "oversize_2048_messages": len(oversize_2048),
        "effects_2048": effects.get("2048"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
