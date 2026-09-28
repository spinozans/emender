"""Era-3 first-action validator: additive over era-2, sealed pins intact.

Operator directive 2026-09-28 (first-action criterion curriculum): the
era-3 program (scripts/e97_first_party_validator_first_action.py) is the
sealed archive member every NEW collection's private spec pins; the repo's
scripts/e97_first_party_validator.py intentionally remains the era-2 bytes
(72820d63…) because in-flight bank pool tasks byte-pin that absolute path.
These tests prove:

  * era-2 and era-3 grade every era-2-shaped spec IDENTICALLY (absent
    required_first_action = no check);
  * a declared required_first_action fires the Stage-B-mirrored check
    (exact tool + exact declared key arguments on the FIRST emitted
    action) in focused mode;
  * the era-2 program rejects extended specs (why they pin era-3);
  * the sealed logical argv identity is unchanged and the old admitted
    lake still re-validates under the current modules.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ndm.e97_first_party_source_archive import (
    verify_archive_members,
    verify_checkout_components,
)
from ndm.e97_first_party_read_observe import (
    _VALIDATOR_SOURCE_MEMBER,
    validator_logical_argv,
)
from ndm.e97_task_lake import (
    validate_source_registry,
    validate_task_collection,
)

REPO = Path(__file__).resolve().parents[1]
ERA2 = REPO / "scripts" / "e97_first_party_validator.py"
ERA3 = REPO / "scripts" / "e97_first_party_validator_first_action.py"
MANIFEST = REPO / "configs/pi/e97-firstparty-generator-manifest-v1.json"
ARCHIVE = REPO / "configs/pi/e97-firstparty-source-v1.tar"
REGISTRY = REPO / "configs/pi/e97-onpolicy-source-registry-v1.json"
OLD_LAKE = Path("/mnt/nvme2n1/erikg/task_lake/e97-firstparty-cpu-phase-bc-v1-admitted")
# The era-2 bytes the 300 in-flight bank pool tasks pin at the repo path.
ERA2_SHA256 = "72820d63ce20e60cebb2e5951275d48522e17b22eb2bd2695f5ef8202f8d8582"

TOKEN = "0123456789abcdef0123"


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _spec(required_first_action=None, expected_token=TOKEN) -> dict:
    spec = {
        "schema": "emender-e97-first-party-validator-v2",
        "task_identity": "a" * 64,
        "fixture_tree_digest": "b" * 64,
        "archive_sha256": "c" * 64,
        "expected_token": expected_token,
        "required_read_path": "depot/token.txt",
        "program_sha256": "d" * 64,
        "interpreter_sha256": "e" * 64,
        "minefield": {"allowed_tools": ["list_files", "read"],
                      "forbidden_paths": ["/", ".."]},
    }
    if required_first_action is not None:
        spec["required_first_action"] = required_first_action
    return spec


def _read_action(path: str = "depot/token.txt", arguments: dict | None = None) -> dict:
    arguments = {"path": path, "offset": 1, "limit": 64} if arguments is None else dict(arguments)
    raw = {"ok": True, "tool": "read", "path": path, "offset": 1,
           "lines": [{"line": 1, "text": f"token={TOKEN}"}], "truncated": False}
    return {"sequence": 0, "tool_name": "read", "arguments": arguments,
            "is_error": False, "raw_observation": raw,
            "effective_observation": _canonical(raw)}


def _list_action() -> dict:
    return {"sequence": 0, "tool_name": "list_files",
            "arguments": {"path": ".", "depth": 16, "limit": 1000},
            "is_error": False}


def _terminal(actions, *, final=f"Final: {TOKEN}", status="success") -> dict:
    messages = [{"role": "user", "content": "find the token"}]
    if final is not None:
        messages.append({"role": "assistant", "content": final})
    return {"status": status, "messages": messages, "actions": actions}


def _run(mode: str, program: Path, spec: dict, terminal: dict, tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_canonical(spec))
    terminal_path = tmp_path / "terminal.json"
    terminal_path.write_text(_canonical(terminal))
    completed = subprocess.run(
        [sys.executable, str(program), "--mode", mode,
         "--spec", str(spec_path), "--terminal", str(terminal_path)],
        capture_output=True, text=True, timeout=60)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


# ------------------------------------------------- era coexistence + identity
def test_repo_era2_bytes_pin_is_unchanged():
    assert hashlib.sha256(ERA2.read_bytes()).hexdigest() == ERA2_SHA256
    era3_sha = hashlib.sha256(ERA3.read_bytes()).hexdigest()
    assert era3_sha != ERA2_SHA256


def test_sealed_logical_argv_identity_is_unchanged():
    assert validator_logical_argv("focused") == [
        "@runtime-python", "@generator-source/scripts/e97_first_party_validator.py",
        "--mode", "focused"]
    assert validator_logical_argv("regression") == [
        "@runtime-python", "@generator-source/scripts/e97_first_party_validator.py",
        "--mode", "regression"]
    assert _VALIDATOR_SOURCE_MEMBER == \
        "scripts/e97_first_party_validator_protocol_breadth.py"  # era-4 re-pin


def test_generator_closure_and_registry_bind_current_artifacts():
    components = verify_checkout_components(MANIFEST, checkout_root=REPO)
    by_path = {component["path"]: component["sha256"] for component in components}
    assert by_path["scripts/e97_first_party_validator.py"] == ERA2_SHA256
    assert by_path["scripts/e97_first_party_validator_first_action.py"] == \
        hashlib.sha256(ERA3.read_bytes()).hexdigest()
    assert verify_archive_members(MANIFEST, ARCHIVE) == \
        hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    registry = validate_source_registry(json.loads(REGISTRY.read_text()))
    for source in registry["sources"]:
        if source["kind"] == "first-party":
            assert source["receipts"]["source_archive_sha256"] == \
                hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()


@pytest.mark.skipif(not OLD_LAKE.is_dir(), reason="old admitted lake not mounted")
def test_old_admitted_lake_revalidates_under_current_modules():
    """The bank's own refresh path re-validates the old lake with the CURRENT
    repo modules; the era-3 deployment must not disturb it."""
    registry = validate_source_registry(
        json.loads((OLD_LAKE / "source-registry.json").read_text()))
    tasks = [json.loads(line) for line in
             (OLD_LAKE / "tasks.jsonl").read_text().splitlines() if line.strip()]
    assert len(tasks) == 4
    assert len(validate_task_collection(tasks, registry=registry)) == 4


# -------------------------------------------- old specs grade identically
CASES = [
    ("pass", _spec(), [_read_action()], "success", f"Final: {TOKEN}"),
    ("wrong-final", _spec(), [_read_action()], "success", "Final: wrong"),
    ("failed-status", _spec(), [_read_action()], "failed", f"Final: {TOKEN}"),
    ("wrong-token", _spec(expected_token="f" * 20), [_read_action()], "success", f"Final: {TOKEN}"),
    ("no-actions", _spec(), [], "success", f"Final: {TOKEN}"),
    ("wrong-read-path", _spec(), [_read_action(path="depot/decoy.txt")], "success", f"Final: {TOKEN}"),
    ("minefield-violation", _spec(),
     [{**_read_action(), "tool_name": "bash"}], "success", f"Final: {TOKEN}"),
]


@pytest.mark.parametrize("name,spec,actions,status,final", CASES)
def test_old_specs_grade_identically_across_eras(name, spec, actions, status, final, tmp_path):
    terminal = _terminal(actions, final=final, status=status)
    for mode in ("focused", "regression"):
        era2 = _run(mode, ERA2, spec, terminal, tmp_path)
        era3 = _run(mode, ERA3, spec, terminal, tmp_path)
        assert era2 == era3, f"{name}/{mode}: era-2 {era2} vs era-3 {era3}"


# ------------------------------------------- first-action criterion firing
def test_first_action_solver_shape_arguments_pass(tmp_path):
    spec = _spec({"tool": "read", "arguments": {"path": "depot/token.txt"}})
    terminal = _terminal([_read_action()])  # dense proof world: {path, offset, limit}
    assert _run("focused", ERA3, spec, terminal, tmp_path)[0] == 0


def test_first_action_bank_natural_shape_arguments_pass(tmp_path):
    spec = _spec({"tool": "read", "arguments": {"path": "depot/token.txt"}})
    terminal = _terminal([_read_action(arguments={"path": "depot/token.txt"})])
    assert _run("focused", ERA3, spec, terminal, tmp_path)[0] == 0
    assert _run("regression", ERA3, spec, terminal, tmp_path)[0] == 0


def test_first_action_wrong_tool_fails_while_era2_unaffected(tmp_path):
    spec = _spec()
    extended = _spec({"tool": "read", "arguments": {"path": "depot/token.txt"}})
    terminal = _terminal([_list_action(), _read_action()])  # token still grounded
    assert _run("focused", ERA2, spec, terminal, tmp_path)[0] == 0
    assert _run("focused", ERA3, extended, terminal, tmp_path)[0] != 0


def test_first_action_wrong_declared_key_fails(tmp_path):
    extended = _spec({"tool": "read", "arguments": {"path": "depot/token.txt"}})
    terminal = _terminal([_read_action(path="depot/decoy.txt"), _read_action()])
    assert _run("focused", ERA3, extended, terminal, tmp_path)[0] != 0


def test_first_action_tool_only_declaration(tmp_path):
    extended = _spec({"tool": "read"})
    assert _run("focused", ERA3, extended, _terminal([_read_action()]), tmp_path)[0] == 0
    assert _run("focused", ERA3, extended,
               _terminal([_list_action(), _read_action()]), tmp_path)[0] != 0


def test_first_action_missing_when_declared_fails(tmp_path):
    extended = _spec({"tool": "read", "arguments": {"path": "depot/token.txt"}})
    assert _run("focused", ERA3, extended, _terminal([]), tmp_path)[0] != 0


def test_era2_rejects_extended_spec(tmp_path):
    extended = _spec({"tool": "read", "arguments": {"path": "depot/token.txt"}})
    terminal = _terminal([_read_action()])
    completed = _run("focused", ERA2, extended, terminal, tmp_path)
    assert completed[0] != 0
    assert "private validator spec schema is invalid" in completed[2]


@pytest.mark.parametrize("bad", [
    {"tool": "read", "arguments": "not-a-mapping"},
    {"arguments": {"path": "depot/token.txt"}},
    {"tool": "read", "bogus": True},
    "not-a-mapping",
])
def test_era3_rejects_malformed_first_action_declaration(bad, tmp_path):
    spec = _spec(bad)
    terminal = _terminal([_read_action()])
    completed = _run("focused", ERA3, spec, terminal, tmp_path)
    assert completed[0] != 0
    assert "required_first_action" in completed[2]
