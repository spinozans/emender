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
            # Receipt currency: both the dense controller and the bank's
            # gym projection build effective_observation with canonical_json
            # (ensure_ascii=False); token files are pure ASCII so era-2-mode
            # behavior is unchanged, but era-4 grounded reads span arbitrary
            # UTF-8 fixture text and must verify against the same canonical
            # form (an em-dash must not void the receipt).
            if not isinstance(raw, Mapping) or effective != json.dumps(
                    raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False):
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
        if not isinstance(raw, Mapping) or effective != json.dumps(
                raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False):
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



# Additive era8 sealed end-state kind; legacy validator functions above remain unchanged.
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

FAMILIES = ("terminal", "conversation")
MORPHS = ("entity_rename", "chain_depth", "distractor", "failure_inversion")
INSTRUCTIONS = {
    "terminal": """Author one newly invented Terminal-Bench-pattern terminal task, not a benchmark copy.
Use a natural seeded workspace (2-6 realistic text/code/config files, each
150-4096 bytes). Require multiple real bash/read/write operations: transform
files, repair a failing program/config, process data, set up an offline
environment or perform local git operations. Never use a token echo.
solution.actions is a 2-10 item reference plan using read/write/edit/bash,
with actual tool arguments; it must solve the stated goal through real Pi.
solution.assertions is a nonempty bounded list of end-state checks:
{"kind":"exact","path":relative,"content":text},
{"kind":"regex","path":relative,"pattern":fullmatch_regex},
{"kind":"absent","path":relative}, or
{"kind":"command","argv":[strings],"exit_code":integer,"timeout":1..10}.
Include at least one exact/regex check on an output that the plan changes.
All assertions must be true after the reference plan and at least one must
be false initially. State all requirements in the prompt, not just privately.
Do not access network, install/download packages, or use absolute paths.
""",
    "conversation": """Author one newly invented conversation-agent-pattern task.
Use a natural seeded workspace (2-6 files, each 150-4096 bytes). The agent
must ask for a missing parameter BEFORE acting, then produce constrained
responses and/or recover from a user correction. There are 2-4 user turns
INCLUDING the opening prompt. No free-form user simulator.
solution.user_script is a 1-3 item list. Each step has exactly:
{"on": {behavior_class: user_text}, "reply_pattern": fullmatch_regex,
 "allow_tools": boolean}. Behavior classes are question (reply ending ?),
json (valid JSON object/array), statement (other reply), acted (any workspace
tool since the prior user turn). Use explicit deterministic branches; any
missing branch fails closed. The first step must allow_tools=false and have
only a question branch, giving the initially missing parameter. A later turn
may correct it. reply_pattern must constrain the reply at that step.
solution.actions is the full 2-12 item reference plan using read/write/edit/bash
and finish; intermediate finish.message replies drive the script, last finish
ends the episode. solution.final_pattern constrains the last reply, not a token.
solution.assertions is the same end-state schema as terminal; include a real
exact/regex artifact reflecting the FULL exchange, not just the opening.
User-script text MUST NOT appear in public fixtures or be appended to the
opening prompt. Agent can see only the opening and successive user turns.
""",
}
JSON_HINT = ('Strict JSON: {"family": "terminal" or "conversation", '
             '"workspace":"natural", "prompt":str, "files":[{"path":str,"content":str}], '
             '"token":20-lowercase-hex-bookkeeping-only, "solution":{"actions":'
             '[{"tool":str,"arguments":object}], "assertions":[objects], '
             '"user_script":[objects] (conversation only), "final_pattern":str '
             '(conversation only)}, "notes":str}.')


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe_path(value: Any) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value)
            or Path(value).is_absolute() or ".." in Path(value).parts
            or value != Path(value).as_posix() or value == "."):
        raise ValueError("unsafe end-state path")
    return value


def pattern(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise ValueError("invalid bounded pattern")
    # Keep authored patterns simple: no backreferences, lookarounds or nested
    # repetition. This is a validator, not an arbitrary regex runtime.
    if "(" in value or "\\1" in value:
        raise ValueError("pattern groups are not supported")
    re.compile(value)
    return value


def assertions_schema(value: Any) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("end-state assertions missing or unbounded")
    file_paths = set()
    content_check = False
    for check in value:
        if not isinstance(check, dict):
            raise ValueError("invalid assertion")
        kind = check.get("kind")
        keys = {"exact": {"kind", "path", "content"},
                "regex": {"kind", "path", "pattern"},
                "absent": {"kind", "path"},
                "command": {"kind", "argv", "exit_code", "timeout"}}.get(kind)
        if keys is None or set(check) != keys:
            raise ValueError("invalid assertion fields")
        if kind == "command":
            argv = check["argv"]
            if (not isinstance(argv, list) or not 1 <= len(argv) <= 16
                    or any(not isinstance(a, str) or not a or len(a) > 1024 or "\x00" in a for a in argv)
                    or type(check["exit_code"]) is not int
                    or type(check["timeout"]) is not int or not 1 <= check["timeout"] <= 10):
                raise ValueError("invalid command check")
        else:
            path = safe_path(check["path"])
            if path in file_paths:
                raise ValueError("duplicate assertion path")
            file_paths.add(path)
            if kind == "exact":
                text = check["content"]
                if not isinstance(text, str) or not text or len(text.encode()) > 8192 or "\x00" in text:
                    raise ValueError("invalid expected content")
                if re.fullmatch(r'(?:token=[^\r\n]+|[0-9a-f]{20})\n?', text):
                    raise ValueError('end-state artifact cannot be a token echo')
                content_check = True
            if kind == "regex":
                pattern(check["pattern"])
                if 'token=' in check['pattern']:
                    raise ValueError('end-state regex cannot grade a token echo')
                content_check = True
    if not content_check:
        raise ValueError("end state needs an exact/regex artifact, not command/token only")
    return value


def script_schema(value: Any) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError("user script must have 2-4 user turns including opening")
    for step in value:
        if (not isinstance(step, dict) or set(step) != {"on", "reply_pattern", "allow_tools"}
                or type(step["allow_tools"]) is not bool or not isinstance(step["on"], dict)
                or not step["on"] or not set(step["on"]) <= {"question", "json", "statement", "acted"}
                or any(not isinstance(t, str) or not 1 <= len(t) <= 2048 for t in step["on"].values())):
            raise ValueError("invalid sealed user step")
        pattern(step["reply_pattern"])
    if value[0]["allow_tools"] or set(value[0]["on"]) != {"question"}:
        raise ValueError("script must require clarification before acting")
    return value


def behavior_class(reply: str, acted: bool) -> str:
    if acted:
        return "acted"
    if reply.strip().endswith("?"):
        return "question"
    try:
        if isinstance(json.loads(reply), (dict, list)):
            return "json"
    except ValueError:
        pass
    return "statement"


def user_turn(script: list[dict], index: int, reply: str, acted: bool) -> tuple[str, str]:
    step = script_schema(script)[index]
    kind = behavior_class(reply, acted)
    if acted and not step["allow_tools"]:
        raise ValueError("agent acted before clarification")
    if not re.fullmatch(step["reply_pattern"], reply):
        raise ValueError("conversation constrained reply mismatch")
    if kind not in step["on"]:
        raise ValueError("unscripted agent behavior")
    return kind, step["on"][kind]


def guards(spec: dict, common) -> None:
    if spec.get("workspace") != "natural":
        raise ValueError("era8 requires natural workspace")
    common(spec, set(), set(), minimum_files=2, token_in_files=False)
    for file in spec['files']:
        workspace_path(file['path'])
    solution = spec["solution"]
    if spec['token'] in canonical(solution):
        raise ValueError('bookkeeping token cannot enter reference outcome')
    required = {"actions", "assertions"}
    if spec["family"] == "conversation":
        required |= {"user_script", "final_pattern"}
        script_schema(solution["user_script"])
        pattern(solution["final_pattern"])
    if set(solution) != required:
        raise ValueError("era8 solution fields")
    assertions_schema(solution["assertions"])
    actions = solution["actions"]
    if not isinstance(actions, list) or not 2 <= len(actions) <= 12:
        raise ValueError("reference action bounds")
    for action in actions:
        if (not isinstance(action, dict) or set(action) != {"tool", "arguments"}
                or action["tool"] not in {"read", "write", "edit", "bash", "finish"}
                or not isinstance(action["arguments"], dict)):
            raise ValueError("invalid reference tool action")
        if "path" in action["arguments"]:
            safe_path(action["arguments"]["path"])
        action_allowed(action['tool'], action['arguments'])
    for check in solution['assertions']:
        if check['kind'] == 'command':
            assertion_argv_allowed(check['argv'])
    if spec["family"] == "terminal" and any(a["tool"] == "finish" for a in actions):
        raise ValueError("terminal reference finish is appended by proof")
    if spec["family"] == "conversation":
        index, acted = 0, False
        for action in actions:
            if action["tool"] != "finish":
                acted = True
            elif index < len(solution["user_script"]):
                user_turn(solution["user_script"], index, action["arguments"]["message"], acted)
                index += 1
                acted = False
            else:
                if not re.fullmatch(solution["final_pattern"], action["arguments"]["message"]):
                    raise ValueError("reference final mismatch")
        if index != len(solution["user_script"]) or actions[-1]["tool"] != "finish":
            raise ValueError("reference script incomplete")


def _read_beneath(root_fd: int, path: str) -> str | None:
    parts = safe_path(path).split("/")
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            if not stat.S_ISREG(os.fstat(leaf).st_mode):
                raise ValueError("end-state file is not regular")
            text = os.read(leaf, 8193)
            if len(text) > 8192:
                raise ValueError("end-state file exceeds bound")
            return text.decode("utf-8")
        finally:
            os.close(leaf)
    except FileNotFoundError:
        return None
    finally:
        os.close(fd)


def capture_end_state(workspace: Path, checks: list[dict], identity: str) -> dict:
    """Owner-produced receipt; never inferred from model read-backs/claims."""
    assertions_schema(checks)
    for check in checks:
        if check["kind"] == "command":
            assertion_argv_allowed(check["argv"])
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        files, commands = {}, []
        for check in checks:
            if check["kind"] != "command":
                files[check["path"]] = _read_beneath(fd, check["path"])
            else:
                cp = subprocess.run(check["argv"], cwd=f"/proc/self/fd/{fd}", pass_fds=(fd,),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=check["timeout"], check=False,
                                    env={**os.environ, 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_TEMPLATE_DIR': ''})
                commands.append({"argv": check["argv"], "exit_code": cp.returncode})
        return {"kind": "workspace-end-state-v1", "task_identity": identity,
                "files": files, "commands": commands}
    finally:
        os.close(fd)


def check_end_state(checks: list[dict], receipt: dict, identity: str) -> bool:
    assertions_schema(checks)
    paths = {c["path"] for c in checks if c["kind"] != "command"}
    commands = [c for c in checks if c["kind"] == "command"]
    if (not isinstance(receipt, dict) or set(receipt) != {"kind", "task_identity", "files", "commands"}
            or receipt["kind"] != "workspace-end-state-v1" or receipt["task_identity"] != identity
            or not isinstance(receipt["files"], dict) or set(receipt["files"]) != paths
            or not isinstance(receipt["commands"], list) or len(receipt["commands"]) != len(commands)):
        return False
    for check in checks:
        kind = check["kind"]
        if kind == "command":
            actual = receipt["commands"][commands.index(check)]
            if actual != {"argv": check["argv"], "exit_code": check["exit_code"]}:
                return False
        else:
            actual = receipt["files"][check["path"]]
            if kind == "absent" and actual is not None:
                return False
            if kind == "exact" and actual != check["content"]:
                return False
            if kind == "regex" and (not isinstance(actual, str) or len(actual) > 8192
                                     or re.fullmatch(check["pattern"], actual) is None):
                return False
    return True


def check_exchange(script: list[dict], final_pattern: str, messages: list[dict], actions: list[dict]) -> bool:
    """Replay sealed branches over the entire faithfully projected exchange."""
    script_schema(script)
    index, acted, last = 0, False, None
    finished = False
    for message in messages[2:]:  # projection starts system + opening user
        if finished or (last is not None and message.get('role') != 'user'):
            return False
        if message.get("role") == "tool":
            acted = True
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            content = message["content"]
            if not content.startswith("Final: "):
                return False
            reply = content[7:]
            if index < len(script):
                try:
                    _, last = user_turn(script, index, reply, acted)
                except ValueError:
                    return False
            else:
                last = None
                if not re.fullmatch(final_pattern, reply):
                    return False
                finished = True
        elif message.get("role") == "user":
            if last is None or message.get("content") != last:
                return False
            index += 1
            acted, last = False, None
    return index == len(script) and last is None and finished and bool(actions)

OFFLINE_DISCIPLINE = """
Mandatory offline grammar for both reference AND policy execution (no arbitrary interpreters):
- Commands: mkdir [-p] paths; rm [-f|-r|-rf] paths; cp/mv source dest; cat/sort/uniq/wc/head/tail/paste paths.
- awk exactly -F, '/^[a-z]+,[0-9]+$/{print $1 ":" $2*INTEGER}' path (also + or -); sed -n 'START,ENDp' path.
- printf '%s\\n' literal; test -f|-s|-d path; git init; git add paths; git status --porcelain; git log --oneline.
- Bounded semicolon/pipeline chains and > safe-relative-path redirects only. No shell expansion, substitutions, newline commands, globs, backgrounding, networking, absolute/traversal/.git-metadata paths or executable scripts.
- Command assertions use ONLY read-only commands from this grammar, not mkdir/rm/cp/mv/git init/add.
- Python/program execution, package installation and arbitrary environment setup are NOT authorized; create/repair text or code as artifacts, process data, and initialize local git using these permitted operations.
"""

import re
import shlex



def workspace_path(value):
    path = safe_path(value)
    if any(p.startswith('.git') for p in path.split('/')):
        raise ValueError('private git metadata path forbidden')
    return path


def _literal(value):
    if not isinstance(value, str) or any(c in value for c in ('$','`','\\','\x00','\n','\r')):
        raise ValueError('shell literal required')
    return value


def argv_allowed(argv):
    if not argv or len(argv) > 32:
        raise ValueError('bounded offline argv required')
    name, args = argv[0], argv[1:]
    if name in {'mkdir', 'rm', 'cp', 'mv', 'cat', 'sort', 'uniq', 'wc', 'head', 'tail', 'paste'}:
        flags = {'mkdir': {'-p'}, 'rm': {'-f', '-r', '-rf'}, 'cp': set(), 'mv': set(),
                 'cat': set(), 'sort': {'-n', '-r', '-u'}, 'uniq': {'-c'}, 'wc': {'-l', '-w', '-c'},
                 'head': set(), 'tail': set(), 'paste': set()}[name]
        paths = []
        for arg in args:
            if arg in flags:
                continue
            _literal(arg)
            paths.append(workspace_path(arg))
        if not paths or (name in {'cp', 'mv'} and len(paths) != 2):
            raise ValueError('offline file command paths missing')
    elif name == 'awk':
        # A deliberately small data-transform DSL: numeric CSV records,
        # print a field and bounded arithmetic; no arbitrary AWK program.
        if len(args) != 3 or args[0] != '-F,':
            raise ValueError('awk form outside offline grammar')
        if not re.fullmatch(r'/\^\[a-z\]\+,\[0-9\]\+\$/\{print \$1 ":" \$2[+*\-][0-9]{1,6}\}', args[1]):
            raise ValueError('awk program outside data-transform grammar')
        workspace_path(args[2])
    elif name == 'sed':
        if len(args) != 3 or args[0] != '-n' or not re.fullmatch(r'[0-9]{1,4}(,[0-9]{1,4})?p', args[1]):
            raise ValueError('sed form outside offline grammar')
        workspace_path(args[2])
    elif name == 'printf':
        if not args or len(args) > 8 or args[0] not in {'%s', '%s\n', '%s\\n'}:
            raise ValueError('printf format outside offline grammar')
        for arg in args[1:]:
            _literal(arg)
    elif name == 'test':
        if len(args) != 2 or args[0] not in {'-f','-s','-d'}:
            raise ValueError('test form outside offline grammar')
        workspace_path(args[1])
    elif name == 'git':
        # No commit, config, hooks, aliases, remote, clone, submodule, clean,
        # filters, textconv or external diff. New-family HOME/config are empty.
        if args == ['init'] or args in (['status', '--porcelain'], ['log', '--oneline']):
            return
        if args and args[0] == 'add' and len(args) >= 2:
            for arg in args[1:]:
                workspace_path(arg)
        else:
            raise ValueError('git form outside offline grammar')
    else:
        raise ValueError('command outside offline grammar')


def shell_allowed(command):
    if not isinstance(command, str) or not 1 <= len(command) <= 4096 or any(c in command for c in ('\x00', '\n', '\r')):
        raise ValueError('bounded offline shell required')
    lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>')
    lexer.whitespace_split = True
    lexer.commenters = ''
    tokens = list(lexer)
    if len(tokens) > 128:
        raise ValueError('shell token bound')
    command_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {';', '|'}:
            argv_allowed(command_tokens)
            command_tokens = []
        elif token == '>':
            argv_allowed(command_tokens)
            command_tokens = []
            i += 1
            if i >= len(tokens):
                raise ValueError('redirect target missing')
            workspace_path(tokens[i])
            if i+1 < len(tokens) and tokens[i+1] != ';':
                raise ValueError('redirect must end command')
            if i+1 < len(tokens):
                i += 1
        elif any(c in token for c in ('&', '<', '>')):
            raise ValueError('shell operator outside grammar')
        else:
            command_tokens.append(token)
        i += 1
    if command_tokens:
        argv_allowed(command_tokens)
    elif tokens and tokens[-1] in {';', '|'}:
        raise ValueError('trailing shell operator')
    if not tokens:
        raise ValueError('empty shell')


def assertion_argv_allowed(argv):
    argv_allowed(argv)
    if argv[0] not in {'test', 'cat', 'sort', 'uniq', 'wc', 'head', 'tail', 'paste', 'awk', 'sed'} and not (argv[0] == 'git' and argv[1:] in (['status', '--porcelain'], ['log', '--oneline'])):
        raise ValueError('assertion command must be read-only')


def action_allowed(tool, arguments):
    if tool in {'read','write','edit'}:
        workspace_path(arguments.get('path'))
    elif tool == 'bash':
        if set(arguments) - {'command', 'timeout'}:
            raise ValueError('bash arguments outside grammar')
        shell_allowed(arguments.get('command'))
        if 'timeout' in arguments and (type(arguments['timeout']) is not int or not 1 <= arguments['timeout'] <= 30):
            raise ValueError('bash timeout outside bounds')
    elif tool not in {'finish','think'}:
        raise ValueError('tool outside era8 offline surface')

def _diversity_spec(value):
    if not isinstance(value, Mapping) or "end_state_assertions" not in value:
        return _spec(value), False
    if set(value) & {"expected_final", "required_grounded_reads", "required_workspace_writes", "required_error_read", "required_first_action"}:
        raise ValueError("end-state kind cannot reinterpret terminal-receipt outcomes")
    extra = {"end_state_assertions", "user_script", "final_pattern"}
    clean = {k: v for k, v in value.items() if k not in extra}
    # Validate unchanged common identity/minefield fields with legacy schema.
    _spec({**clean, "expected_final": "done", "required_grounded_reads": [{"path": clean.get("required_read_path"), "expected_text": "schema-only"}]})
    assertions_schema(value["end_state_assertions"])
    for check in value["end_state_assertions"]:
        if check['kind'] == 'command':
            assertion_argv_allowed(check['argv'])
    if "user_script" in value:
        script_schema(value["user_script"])
        pattern(value.get("final_pattern"))
    elif "final_pattern" in value:
        raise ValueError("final pattern requires sealed user script")
    return value, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("focused", "regression"), required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--spec-fd", type=int)
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--terminal-fd", type=int)
    args = parser.parse_args()
    try:
        spec, diversity = _diversity_spec(_input_json(args.spec, args.spec_fd, name="spec", maximum=_MAX_SPEC_BYTES))
        terminal = _input_json(args.terminal, args.terminal_fd, name="terminal", maximum=_MAX_TERMINAL_BYTES)
    except ValueError as exc:
        raise SystemExit(f"validator spec failed: {exc}") from exc
    actions = terminal.get("actions")
    if not isinstance(actions, list) or any(not isinstance(action, Mapping) for action in actions):
        raise SystemExit("terminal actions are required")
    allowed = spec["minefield"]["allowed_tools"]
    if any(item.get("tool_name") not in allowed for item in actions):
        raise SystemExit("minefield tool violation")
    if diversity:
        try:
            for action in actions:
                action_allowed(action.get('tool_name'), action.get('arguments', {}))
        except ValueError as exc:
            raise SystemExit(f"sealed invalid-action outcome: {exc}")
    if diversity and args.mode == "focused":
        if terminal.get("status") != "success" or not check_end_state(
                spec["end_state_assertions"], terminal.get("sealed_end_state"), spec["task_identity"]):
            raise SystemExit("sealed end state failed")
        if "user_script" in spec and not check_exchange(spec["user_script"], spec["final_pattern"], terminal.get("messages", []), actions):
            raise SystemExit("sealed full exchange failed")
        if len(actions) < 2:
            raise SystemExit("multi-step end-state task needs real workspace actions")
    elif args.mode == "focused":
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
