#!/usr/bin/env python3
"""Typed, bounded first-party task validator invoked from bundle argv."""
from __future__ import annotations
import argparse
import errno
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

_MAX_SPEC_BYTES = 1 << 20
_MAX_TERMINAL_BYTES = 16 << 20

_SPEC_SCHEMA = "emender-e97-first-party-validator-v2"
_TOKEN_LINE = re.compile(r"^token=([^\r\n]+)\n?$")


def _spec(value: Any) -> Mapping[str, Any]:
    fields = {
        "schema", "task_identity", "fixture_tree_digest", "archive_sha256",
        "expected_token", "required_read_path", "program_sha256", "interpreter_sha256",
        "minefield",
    }
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != _SPEC_SCHEMA:
        raise ValueError("private validator spec schema is invalid")
    for name in ("task_identity", "fixture_tree_digest", "archive_sha256", "program_sha256", "interpreter_sha256"):
        item = value.get(name)
        if not isinstance(item, str) or len(item) != 64 or any(char not in "0123456789abcdef" for char in item):
            raise ValueError(f"private validator {name} is invalid")
    if (not isinstance(value.get("expected_token"), str) or not value["expected_token"]
            or not isinstance(value.get("required_read_path"), str) or not value["required_read_path"]):
        raise ValueError("private validator grounding fields are invalid")
    minefield = value.get("minefield")
    if (not isinstance(minefield, Mapping) or set(minefield) != {"allowed_tools", "forbidden_paths"}
            or minefield["allowed_tools"] != ["list_files", "read"]
            or not isinstance(minefield["forbidden_paths"], list)):
        raise ValueError("private validator minefield is invalid")
    return value


def _grounded_token(actions: list[Mapping[str, Any]], required_path: str) -> str:
    """Derive the expected token only from a successful model-facing read receipt."""

    observed: list[str] = []
    for action in actions:
        if (action.get("tool_name") != "read" or action.get("arguments", {}).get("path") != required_path
                or action.get("is_error") is not False):
            continue
        raw = action.get("raw_observation")
        effective = action.get("effective_observation")
        if not isinstance(raw, Mapping) or effective != json.dumps(raw, sort_keys=True, separators=(",", ":")):
            continue
        lines = raw.get("lines")
        if (raw.get("ok") is not True or not isinstance(lines, list) or len(lines) != 1
                or not isinstance(lines[0], Mapping) or not isinstance(lines[0].get("text"), str)):
            continue
        match = _TOKEN_LINE.fullmatch(lines[0]["text"])
        if match is not None:
            observed.append(match.group(1))
    if not observed or len(set(observed)) != 1:
        raise ValueError("successful grounded token reads are missing or disagree")
    return observed[0]


def _read_regular_descriptor(descriptor: int, *, maximum: int) -> bytes:
    """Read one already-open regular descriptor from byte zero under a strict bound."""

    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 0:
        raise ValueError("input descriptor is invalid")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ValueError("read bound is invalid")
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("input is not a regular file")
        if info.st_size < 0 or info.st_size > maximum:
            raise ValueError("input exceeds read limit")
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining, chunks = info.st_size, []
        while remaining:
            chunk = os.read(descriptor, min(64 << 10, remaining))
            if not chunk:
                raise ValueError("input changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("input changed while reading")
        return b"".join(chunks)
    except OSError as exc:
        raise ValueError("input cannot be opened safely") from exc


def _read_regular_file_no_follow(path: Path, *, maximum: int) -> bytes:
    """Self-contained bounded no-follow reader for private validator inputs."""

    candidate = Path(path)
    parts = candidate.parts
    if not parts:
        raise ValueError("path is invalid")
    root = "/" if candidate.is_absolute() else "."
    names = parts[1:] if candidate.is_absolute() else parts
    if not names or any(part in {"", ".", ".."} for part in names):
        raise ValueError("path is invalid")
    descriptors: list[int] = []
    try:
        current = os.open(root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(current)
        for component in names[:-1]:
            current = os.open(component, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=current)
            descriptors.append(current)
        descriptor = os.open(names[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=current)
        try:
            return _read_regular_descriptor(descriptor, maximum=maximum)
        finally:
            os.close(descriptor)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise FileNotFoundError(str(candidate)) from exc
        raise ValueError("input cannot be opened safely") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _snapshot_json(path: Path, *, name: str, maximum: int) -> Any:
    """Read one bounded CLI authority once without following a link or FIFO."""

    try:
        return json.loads(_read_regular_file_no_follow(path, maximum=maximum))
    except (FileNotFoundError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"validator {name} failed: {exc}") from exc


def _snapshot_json_fd(descriptor: int, *, name: str, maximum: int) -> Any:
    """Duplicate and snapshot an explicitly inherited authority descriptor once."""

    try:
        duplicate = os.dup(descriptor)
    except OSError as exc:
        raise ValueError(f"validator {name} failed: input descriptor cannot be duplicated") from exc
    try:
        return json.loads(_read_regular_descriptor(duplicate, maximum=maximum))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"validator {name} failed: {exc}") from exc
    finally:
        os.close(duplicate)


def _input_json(path: Path | None, descriptor: int | None, *, name: str, maximum: int) -> Any:
    """Require exactly one safe pathname or inherited descriptor authority."""

    if (path is None) == (descriptor is None):
        raise ValueError(f"validator {name} requires exactly one path or descriptor")
    if path is not None:
        return _snapshot_json(path, name=name, maximum=maximum)
    return _snapshot_json_fd(descriptor, name=name, maximum=maximum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("focused", "regression"), required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--spec-fd", type=int)
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--terminal-fd", type=int)
    args = parser.parse_args()
    try:
        spec = _spec(_input_json(args.spec, args.spec_fd, name="spec", maximum=_MAX_SPEC_BYTES))
        terminal = _input_json(args.terminal, args.terminal_fd, name="terminal", maximum=_MAX_TERMINAL_BYTES)
    except ValueError as exc:
        raise SystemExit(f"validator spec failed: {exc}") from exc
    actions = terminal.get("actions")
    if not isinstance(actions, list) or any(not isinstance(action, Mapping) for action in actions):
        raise SystemExit("terminal actions are required")
    allowed = spec["minefield"]["allowed_tools"]
    if any(item.get("tool_name") not in allowed for item in actions):
        raise SystemExit("minefield tool violation")
    if args.mode == "focused":
        final = next((item.get("content", "") for item in reversed(terminal.get("messages", []))
                      if isinstance(item, Mapping) and item.get("role") == "assistant"), "")
        try:
            observed_token = _grounded_token(actions, spec["required_read_path"])
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if observed_token != spec["expected_token"]:
            raise SystemExit("private expected token does not match authentic read observation")
        if terminal.get("status") != "success" or final != "Final: " + observed_token:
            raise SystemExit("focused outcome failed")
    else:
        if len(actions) > 12:
            raise SystemExit("regression action limit failed")
        for action in actions:
            path = str(action.get("arguments", {}).get("path", ""))
            if path.startswith("/") or ".." in path.split("/"):
                raise SystemExit("regression path containment failed")
    print(json.dumps({"mode": args.mode, "status": "pass", "action_count": len(actions)}, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
