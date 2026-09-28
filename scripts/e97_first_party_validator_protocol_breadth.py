#!/usr/bin/env python3
"""Typed, bounded first-party task validator invoked from bundle argv.

Era 4 (protocol-breadth curriculum, operator directive 2026-09-28): every
check is byte-for-byte the era-3 behavior (scripts/
e97_first_party_validator_first_action.py — era-2 checks plus the optional
required_first_action field) plus additive, optional private-spec fields for
mechanical OUTCOME verification of multi-step protocol-breadth families
(edit / lookup / recovery / sum / chat; freshly authored,
execution-suite-STYLE — the collapse fix for the bank-gate-v1 0/96
behavioral-overfitting finding):

  "expected_final": <exact required finish message>
  "required_grounded_reads": [{"path": <str>, "expected_text": <str>}, ...]
  "required_error_read": {"path": <str>}
  "required_workspace_writes": [{"path": <str>, "original_text": <str>,
                                 "expected_text": <str>}, ...]

When "expected_final" is declared (outcome mode), focused grading checks —
in addition to the era-3 minefield and required_first_action checks:

  * required_error_read (recovery): one read attempt on the declared broken
    path must end in a real error, and it must PRECEDE every successful
    grounded read of the required paths (the diagnose-first discipline);
  * required_grounded_reads: for every declared path, at least one
    successful read receipt whose verbatim lines equal the declared
    expected_text, and NO successful read of that path may show lines that
    contradict it (every receipt must be a contiguous line-slice of the
    expected text).  The receipts are the bank's faithful dense projection
    of real Pi tool observations, so this is a mechanical workspace-state
    check through the model's own verified read-back;
  * required_workspace_writes: the terminal must contain at least one
    non-error write/edit action on the declared path, and replaying the
    declared original_text through the terminal's own write/edit actions
    in order must produce exactly the declared expected_text (gym edit
    semantics: every edits[].oldText matches the pre-batch text exactly
    once, non-overlapping), AND the path must read back exactly
    expected_text through a successful read receipt;
  * the final assistant message must be exactly "Final: " + expected_final
    and the terminal status must be success.

Absent "expected_final": no outcome check and grading is identical to
era-3 (proven by tests/test_e97_first_party_validator_protocol_breadth.py:
era-2-, era-3-, and era-4-shaped specs without the new fields grade
identically across all three programs).

In outcome mode the era-2 fields expected_token / required_read_path
remain REQUIRED by the spec schema but are ledger/uniqueness bookkeeping
(the anti-fixture token identity and the primary grounding path); they are
not graded in outcome mode.  The minefield widens for protocol-breadth
families: allowed_tools must then be a subset of the declared gym
workspace-op surface (list_files, read, write, edit, bash, fffind, ffgrep)
and must contain "read" (every family grounds in observations); era-2
mode keeps the exact ["list_files", "read"] surface.

This program is the sealed archive member pinned by every NEW collection's
private validator spec (ndm.e97_first_party_read_observe.
_VALIDATOR_SOURCE_MEMBER). The repo's scripts/e97_first_party_validator.py
(era-2) and scripts/e97_first_party_validator_first_action.py (era-3)
intentionally remain their pinned bytes because in-flight bank pool tasks
byte-pin those absolute paths; all eras coexist under their own shas (the
bank's pinned-validator pattern, extended).
"""
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
_MAX_OUTCOME_TEXT = 8192
_MAX_GROUNDED_READS = 8
_MAX_WORKSPACE_WRITES = 4

_SPEC_SCHEMA = "emender-e97-first-party-validator-v2"
_TOKEN_LINE = re.compile(r"^token=([^\r\n]+)\n?$")
# The declared gym workspace-op tool surface for protocol-breadth families
# (the bank's Pi-native lane surface minus the non-workspace tools).
_WORKSPACE_TOOLS = frozenset({"list_files", "read", "write", "edit", "bash", "fffind", "ffgrep"})


def _spec(value: Any) -> Mapping[str, Any]:
    fields = {
        "schema", "task_identity", "fixture_tree_digest", "archive_sha256",
        "expected_token", "required_read_path", "program_sha256", "interpreter_sha256",
        "minefield", "required_first_action",
        "expected_final", "required_grounded_reads", "required_error_read",
        "required_workspace_writes",
    }
    if not isinstance(value, Mapping) or set(value) - fields or value.get("schema") != _SPEC_SCHEMA:
        raise ValueError("private validator spec schema is invalid")
    for name in ("task_identity", "fixture_tree_digest", "archive_sha256", "program_sha256", "interpreter_sha256"):
        item = value.get(name)
        if not isinstance(item, str) or len(item) != 64 or any(char not in "0123456789abcdef" for char in item):
            raise ValueError(f"private validator {name} is invalid")
    if (not isinstance(value.get("expected_token"), str) or not value["expected_token"]
            or not isinstance(value.get("required_read_path"), str) or not value["required_read_path"]):
        raise ValueError("private validator grounding fields are invalid")
    _required_first_action(value)
    outcome = _outcome_fields(value)
    minefield = value.get("minefield")
    if not isinstance(minefield, Mapping) or set(minefield) != {"allowed_tools", "forbidden_paths"}:
        raise ValueError("private validator minefield is invalid")
    allowed = minefield["allowed_tools"]
    if outcome is None:
        if allowed != ["list_files", "read"]:
            raise ValueError("private validator minefield is invalid")
    else:
        if (not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed)
                or not set(allowed) <= _WORKSPACE_TOOLS or "read" not in allowed):
            raise ValueError("private validator minefield is invalid")
    if not isinstance(minefield["forbidden_paths"], list):
        raise ValueError("private validator minefield is invalid")
    return value


def _required_first_action(spec: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Validate the optional first-action criterion declaration (era-3)."""

    if "required_first_action" not in spec:
        return None
    required = spec["required_first_action"]
    if (not isinstance(required, Mapping) or set(required) - {"tool", "arguments"}
            or not isinstance(required.get("tool"), str) or not required["tool"]):
        raise ValueError("private validator required_first_action is invalid")
    if "arguments" in required and not isinstance(required["arguments"], Mapping):
        raise ValueError("private validator required_first_action arguments are invalid")
    return required


def _text(value: Any, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value) or len(value) > _MAX_OUTCOME_TEXT:
        raise ValueError(f"private validator {name} is invalid")
    if "\x00" in value:
        raise ValueError(f"private validator {name} is invalid")
    return value


def _outcome_fields(spec: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Validate the optional era-4 outcome declarations; None = era-2/3 mode.

    Returns a normalized view of the declared outcome fields for grading;
    "expected_final" present means outcome mode (the era-2 token/final
    checks are replaced by the outcome checks below).
    """
    if "expected_final" not in spec:
        for stray in ("required_grounded_reads", "required_error_read",
                      "required_workspace_writes"):
            if stray in spec:
                raise ValueError(f"private validator {stray} requires expected_final")
        return None
    _text(spec["expected_final"], name="expected_final")
    reads = spec.get("required_grounded_reads")
    if reads is None:
        reads = []
    if not isinstance(reads, list) or len(reads) > _MAX_GROUNDED_READS:
        raise ValueError("private validator required_grounded_reads is invalid")
    seen_paths: set[str] = set()
    for entry in reads:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "expected_text"}:
            raise ValueError("private validator required_grounded_reads is invalid")
        path = _text(entry["path"], name="grounded read path")
        _text(entry["expected_text"], name="grounded read expected_text")
        if path in seen_paths:
            raise ValueError("private validator required_grounded_reads is invalid")
        seen_paths.add(path)
    error_read = spec.get("required_error_read")
    if error_read is not None:
        if not isinstance(error_read, Mapping) or set(error_read) != {"path"}:
            raise ValueError("private validator required_error_read is invalid")
        _text(error_read["path"], name="error read path")
    writes = spec.get("required_workspace_writes")
    if writes is None:
        writes = []
    if not isinstance(writes, list) or len(writes) > _MAX_WORKSPACE_WRITES:
        raise ValueError("private validator required_workspace_writes is invalid")
    write_paths: set[str] = set()
    for entry in writes:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "original_text", "expected_text"}:
            raise ValueError("private validator required_workspace_writes is invalid")
        path = _text(entry["path"], name="workspace write path")
        _text(entry["original_text"], name="workspace write original_text", allow_empty=True)
        _text(entry["expected_text"], name="workspace write expected_text")
        if path in write_paths:
            raise ValueError("private validator required_workspace_writes is invalid")
        write_paths.add(path)
    if not reads and not writes:
        raise ValueError("private validator outcome mode requires grounded reads or writes")
    return {"expected_final": spec["expected_final"], "required_grounded_reads": reads,
            "required_error_read": error_read, "required_workspace_writes": writes}


def _check_first_action(actions: list[Mapping[str, Any]],
                        required: Mapping[str, Any]) -> None:
    """Stage-B first-action criterion over the first emitted dense action."""

    if not actions:
        raise SystemExit("required first action missing: episode emitted no action")
    first = actions[0]
    if not isinstance(first, Mapping) or first.get("tool_name") != required["tool"]:
        raise SystemExit("required first action tool mismatch")
    arguments = first.get("arguments")
    if not isinstance(arguments, Mapping):
        raise SystemExit("required first action arguments are invalid")
    for key, expected in (required.get("arguments") or {}).items():
        if key not in arguments or arguments[key] != expected:
            raise SystemExit(f"required first action argument mismatch: {key}")


def _read_receipts(actions: list[Mapping[str, Any]], path: str,
                   *, errors: bool) -> list[tuple[int, list[str]]]:
    """Verbatim model-facing read receipts for one path (success or error)."""

    receipts: list[tuple[int, list[str]]] = []
    for action in actions:
        if (action.get("tool_name") != "read"
                or action.get("arguments", {}).get("path") != path
                or action.get("is_error") is not errors):
            continue
        if not errors:
            raw = action.get("raw_observation")
            effective = action.get("effective_observation")
            if not isinstance(raw, Mapping) or effective != json.dumps(raw, sort_keys=True, separators=(",", ":")):
                continue
            lines = raw.get("lines")
            if (raw.get("ok") is not True or not isinstance(lines, list)
                    or any(not isinstance(line, Mapping)
                           or not isinstance(line.get("text"), str) for line in lines)):
                continue
            receipts.append((action.get("sequence", -1), [line["text"] for line in lines]))
        else:
            receipts.append((action.get("sequence", -1), []))
    return receipts


def _check_grounded_read(actions: list[Mapping[str, Any]], path: str,
                         expected_text: str, *, min_sequence: int = -1) -> None:
    """One path must read back exactly expected_text, never contradicting it.

    Receipts are the bank's faithful verbatim projection of real tool
    observations (dense proof world and Pi-native bank world share the
    receipt shape), so equality here is a mechanical workspace-state check
    through the model's own read-back.  Partial reads are tolerated (any
    contiguous line-slice of the expected text) but no successful read of
    the path — at or after ``min_sequence`` (the last workspace mutation
    for write families; the episode start otherwise) — may show content
    outside the expected text.
    """
    receipts = [(sequence, lines) for sequence, lines
                in _read_receipts(actions, path, errors=False)
                if sequence >= min_sequence]
    if not receipts:
        raise SystemExit(f"required grounded read missing: {path}")
    expected_lines = expected_text.split("\n")
    matched = False
    for _, observed_lines in receipts:
        if observed_lines == expected_lines:
            matched = True
            continue
        width = len(observed_lines)
        if not any(expected_lines[index:index + width] == observed_lines
                   for index in range(len(expected_lines) - width + 1)):
            raise SystemExit(f"grounded read contradicts expected text: {path}")
    if not matched:
        raise SystemExit(f"required grounded read never observed expected text: {path}")


def _apply_edits(text: str, edits: Any) -> str:
    """Apply one gym-shaped edit batch: unique, non-overlapping oldTexts."""

    if (not isinstance(edits, list) or not edits
            or any(not isinstance(edit, Mapping) or set(edit) != {"oldText", "newText"}
                   or not isinstance(edit.get("oldText"), str) or not isinstance(edit.get("newText"), str)
                   or not edit["oldText"] for edit in edits)):
        raise _WriteMutationError("edit batch is invalid")
    spans: list[tuple[int, int, str]] = []
    for edit in edits:
        old = edit["oldText"]
        first = text.find(old)
        if first < 0 or text.find(old, first + 1) >= 0:
            raise _WriteMutationError("edit oldText does not match exactly once")
        spans.append((first, first + len(old), edit["newText"]))
    spans.sort()
    for (_, end_before, _), (start_after, _, _) in zip(spans, spans[1:]):
        if end_before > start_after:
            raise _WriteMutationError("edit batch spans overlap")
    result, cursor = [], 0
    for start, end, replacement in spans:
        result.append(text[cursor:start])
        result.append(replacement)
        cursor = end
    result.append(text[cursor:])
    return "".join(result)


class _WriteMutationError(Exception):
    pass


def _check_workspace_write(actions: list[Mapping[str, Any]], entry: Mapping[str, Any]) -> None:
    """A real write/edit action chain must produce exactly expected_text.

    The declared original_text is replayed through the terminal's own
    non-error write/edit actions on the path in order (gym edit semantics),
    the final virtual state must equal the declared expected_text, and the
    path must read back exactly expected_text through a successful read
    receipt AFTER the last mutation — the workspace-state check.  Reads
    before the last mutation may legitimately show the original text.
    """
    path = entry["path"]
    virtual = entry["original_text"]
    mutated = False
    last_mutation = -1
    for action in actions:
        if action.get("is_error") is not False:
            continue
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping) or arguments.get("path") != path:
            continue
        tool = action.get("tool_name")
        try:
            if tool == "write":
                content = arguments.get("content")
                if not isinstance(content, str):
                    raise _WriteMutationError("write content is invalid")
                virtual = content
            elif tool == "edit":
                virtual = _apply_edits(virtual, arguments.get("edits"))
            else:
                continue
        except _WriteMutationError as exc:
            raise SystemExit(f"workspace write action is not a valid mutation of {path}: {exc}")
        mutated = True
        last_mutation = max(last_mutation, action.get("sequence", -1))
    if not mutated:
        raise SystemExit(f"required workspace write missing: {path}")
    if virtual != entry["expected_text"]:
        raise SystemExit(f"workspace write chain does not produce expected text: {path}")
    # The write entry's expected_text is the RAW final file content (the
    # edit/write chain replays over raw text); the read-back receipt uses
    # the receipt currency (content minus exactly one trailing newline).
    readback_expected = entry["expected_text"][:-1] \
        if entry["expected_text"].endswith("\n") else entry["expected_text"]
    _check_grounded_read(actions, path, readback_expected,
                         min_sequence=last_mutation)


def _check_outcome(spec: Mapping[str, Any], outcome: Mapping[str, Any],
                   actions: list[Mapping[str, Any]],
                   terminal: Mapping[str, Any]) -> None:
    """Era-4 mechanical outcome verification (focused mode)."""
    error_read = outcome["required_error_read"]
    if error_read is not None:
        broken = _read_receipts(actions, error_read["path"], errors=True)
        if not broken:
            raise SystemExit(f"required error read missing: {error_read['path']}")
        grounded_sequences = [sequence for entry in outcome["required_grounded_reads"]
                              for sequence, _ in _read_receipts(actions, entry["path"], errors=False)]
        if grounded_sequences and min(sequence for sequence, _ in broken) > min(grounded_sequences):
            raise SystemExit("required error read must precede the grounded recovery reads")
    for entry in outcome["required_grounded_reads"]:
        _check_grounded_read(actions, entry["path"], entry["expected_text"])
    for entry in outcome["required_workspace_writes"]:
        _check_workspace_write(actions, entry)
    final = next((item.get("content", "") for item in reversed(terminal.get("messages", []))
                  if isinstance(item, Mapping) and item.get("role") == "assistant"), "")
    if terminal.get("status") != "success" or final != "Final: " + outcome["expected_final"]:
        raise SystemExit("focused outcome failed")


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
        required_first_action = _required_first_action(spec)
        if required_first_action is not None:
            _check_first_action(actions, required_first_action)
        outcome = _outcome_fields(spec)
        if outcome is not None:
            _check_outcome(spec, outcome, actions, terminal)
        else:
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
