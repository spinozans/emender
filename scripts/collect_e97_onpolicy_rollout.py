#!/usr/bin/env python3
"""Ingest one authentic Pi JSONL rollout and explicit post-action receipts.

This is an ingestion/validation tool only.  It never starts Pi, a sandbox, or
GPU work; callers provide already captured immutable inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from ndm.e97_phase_c_collector import (
    PHASE_C_RECEIPT_SCHEMA,
    PiEventError,
    apply_no_progress,
    parse_pi_events,
    receipt_fingerprint,
    validate_collection_inputs,
)
from ndm.e97_onpolicy_records import canonical_json, sha256_text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PiEventError(f"invalid JSON input {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PiEventError(f"cannot read JSONL input {path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise PiEventError(f"blank JSONL line {path}:{number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PiEventError(f"invalid JSONL line {path}:{number}") from exc
        if not isinstance(value, dict):
            raise PiEventError(f"JSONL line {path}:{number} is not an object")
        rows.append(value)
    if not rows:
        raise PiEventError(f"JSONL input {path} is empty")
    return rows


def atomic_json_write(path: Path, value: Any) -> None:
    """Publish once, without overwrite; an interrupted write leaves no partial receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publish is atomic and fails instead of replacing an
        # already sealed receipt, including when another writer races us.
        os.link(temporary, path)
        os.unlink(temporary)
        temporary = ""
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def verify_file_pin(path: Path, expected: str, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise PiEventError(f"{name} SHA-256 pin must be lowercase hexadecimal")
    actual = file_sha256(path)
    if actual != expected:
        raise PiEventError(f"{name} SHA-256 mismatch")
    return actual


def collect(args: argparse.Namespace) -> dict[str, Any]:
    task_path = Path(args.task_bundle)
    registry_path = Path(args.source_registry)
    task_sha = file_sha256(task_path)
    registry_sha = file_sha256(registry_path)
    raw_events_path = Path(args.raw_events)
    state_receipts_path = Path(args.state_receipts)
    raw_events_sha = verify_file_pin(raw_events_path, args.raw_events_sha256, "raw events")
    state_receipts_sha = verify_file_pin(state_receipts_path, args.state_receipts_sha256, "state receipts")
    task, registry = validate_collection_inputs(
        task_bundle=read_json(task_path), source_registry=read_json(registry_path),
        task_bundle_sha256=args.task_bundle_sha256,
        source_registry_sha256=args.source_registry_sha256,
        actual_task_bundle_sha256=task_sha,
        actual_source_registry_sha256=registry_sha,
    )
    checkpoint = args.student_checkpoint_sha256
    if not isinstance(checkpoint, str) or len(checkpoint) != 64 or any(c not in "0123456789abcdef" for c in checkpoint):
        raise PiEventError("student checkpoint identity must be a lowercase SHA-256 digest")
    try:
        system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PiEventError(f"cannot read system prompt: {exc}") from exc
    expected_prompt_sha = task["runtime"]["system_prompt_sha256"]
    if sha256_text(system_prompt) != expected_prompt_sha:
        raise PiEventError("system prompt digest mismatch")

    event_rows = read_jsonl(raw_events_path)
    state_rows = read_jsonl(state_receipts_path)
    transcript = parse_pi_events(
        event_rows, system_prompt=system_prompt, user_prompt=task["task"]["prompt"])
    decisions = apply_no_progress(transcript, state_rows)
    observations = [
        {
            "sequence": index,
            "tool_call_id": action.tool_call_id,
            "tool_name": action.tool_name,
            "arguments": action.arguments,
            "result": action.observation_raw,
            "text": action.observation_text,
            "is_error": action.is_error,
        }
        for index, action in enumerate(transcript.actions)
    ]
    body: dict[str, Any] = {
        "schema": PHASE_C_RECEIPT_SCHEMA,
        "status": "complete",
        "task_bundle_sha256": task_sha,
        "source_registry_sha256": registry_sha,
        "task": {
            "identity": task["task"]["identity"],
            "namespace": task["task"]["namespace"],
            "source_registry_id": task["source"]["registry_id"],
            "source_record_digest": task["source"]["source_record_digest"],
        },
        "runtime": task["runtime"],
        "student": {"checkpoint_sha256": checkpoint},
        "inputs": {
            "raw_events_sha256": raw_events_sha,
            "state_receipts_sha256": state_receipts_sha,
            "event_count": transcript.event_count,
        },
        "terminal_status": transcript.terminal_status,
        "terminal_error": transcript.terminal_error,
        "messages": transcript.messages,
        "authentic_observations": observations,
        "state_receipts": state_rows,
        "no_progress": decisions,
    }
    # Bind the identity to every validated input without a circular field.
    body["student"]["rollout_identity"] = receipt_fingerprint(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-bundle", required=True)
    parser.add_argument("--task-bundle-sha256", required=True)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--source-registry-sha256", required=True)
    parser.add_argument("--raw-events", required=True)
    parser.add_argument("--raw-events-sha256", required=True)
    parser.add_argument("--state-receipts", required=True)
    parser.add_argument("--state-receipts-sha256", required=True)
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--student-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        receipt = collect(args)
        atomic_json_write(Path(args.output), receipt)
    except (PiEventError, OSError, ValueError) as exc:
        print(f"collect_e97_onpolicy_rollout: {exc}", flush=True)
        return 2
    print(json.dumps({"status": receipt["status"], "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
