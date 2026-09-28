"""Era-4 protocol-breadth validator: additive over era-3, sealed pins intact.

Operator directive 2026-09-28 (protocol-breadth curriculum — the collapse
fix for the bank-gate-v1 0/96 execution-suite finding): the era-4 program
(scripts/e97_first_party_validator_protocol_breadth.py) is the sealed
archive member every NEW collection's private spec pins; the repo's
scripts/e97_first_party_validator.py intentionally remains the era-2 bytes
(72820d63…) and scripts/e97_first_party_validator_first_action.py the
era-3 bytes, because in-flight bank pool tasks byte-pin those absolute
paths.  These tests prove:

  * era-2- and era-3-shaped specs (no expected_final) grade IDENTICALLY
    under era-2, era-3, and era-4 (additive-only extension);
  * outcome mode (expected_final declared) fires the era-4 mechanical
    outcome checks: exact-answer, verbatim grounded read-backs
    (workspace-state), error-read diagnosis ordering (recovery), and
    write/edit chain verification (edit family);
  * every negative control fails closed (missing/contradicting/unequal
    evidence never passes);
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
ERA4 = REPO / "scripts" / "e97_first_party_validator_protocol_breadth.py"
MANIFEST = REPO / "configs/pi/e97-firstparty-generator-manifest-v1.json"
ARCHIVE = REPO / "configs/pi/e97-firstparty-source-v1.tar"
REGISTRY = REPO / "configs/pi/e97-onpolicy-source-registry-v1.json"
OLD_LAKE = Path("/mnt/nvme2n1/erikg/task_lake/e97-firstparty-cpu-phase-bc-v1-admitted")
# The era-2 bytes the 300 in-flight bank pool tasks pin at the repo path.
ERA2_SHA256 = "72820d63ce20e60cebb2e5951275d48522e17b22eb2bd2695f5ef8202f8d8582"
# The era-3 bytes the in-flight first-action bank pool tasks pin.
ERA3_SHA256 = hashlib.sha256(ERA3.read_bytes()).hexdigest()

TOKEN = "0123456789abcdef0123"


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _base_spec() -> dict:
    return {
        "schema": "emender-e97-first-party-validator-v2",
        "task_identity": "a" * 64,
        "fixture_tree_digest": "b" * 64,
        "archive_sha256": "c" * 64,
        "expected_token": TOKEN,
        "required_read_path": "depot/token.txt",
        "program_sha256": "d" * 64,
        "interpreter_sha256": "e" * 64,
        "minefield": {"allowed_tools": ["list_files", "read"],
                      "forbidden_paths": ["/", ".."]},
    }


def _outcome_spec(**extra) -> dict:
    spec = _base_spec()
    spec["minefield"]["allowed_tools"] = ["list_files", "read", "write", "edit"]
    spec["expected_final"] = "done"
    spec["required_grounded_reads"] = []
    spec["required_workspace_writes"] = [
        {"path": "depot/state.txt", "original_text": "mode=alpha\nstatus=retired",
         "expected_text": "mode=alpha\nstatus=active"}]
    spec.update(extra)
    return spec


def _read_action(path: str = "depot/token.txt", text: str = f"token={TOKEN}",
                 sequence: int = 0, arguments: dict | None = None) -> dict:
    arguments = {"path": path, "offset": 1, "limit": 64} if arguments is None else dict(arguments)
    lines = [{"line": index, "text": line}
             for index, line in enumerate(text.split("\n"), start=1)]
    raw = {"ok": True, "tool": "read", "path": path, "offset": 1,
           "lines": lines, "truncated": False}
    return {"sequence": sequence, "tool_name": "read", "arguments": arguments,
            "is_error": False, "raw_observation": raw,
            "effective_observation": _canonical(raw)}


def _error_read_action(path: str, sequence: int) -> dict:
    return {"sequence": sequence, "tool_name": "read",
            "arguments": {"path": path, "offset": 1, "limit": 64},
            "is_error": True}


def _edit_action(path: str, edits: list, sequence: int) -> dict:
    return {"sequence": sequence, "tool_name": "edit",
            "arguments": {"path": path, "edits": edits}, "is_error": False}


def _write_action(path: str, content: str, sequence: int) -> dict:
    return {"sequence": sequence, "tool_name": "write",
            "arguments": {"path": path, "content": content}, "is_error": False}


def _list_action(sequence: int = 0) -> dict:
    return {"sequence": sequence, "tool_name": "list_files",
            "arguments": {"path": ".", "depth": 16, "limit": 1000},
            "is_error": False}


def _terminal(actions, *, final="Final: done", status="success") -> dict:
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "task"}]
    for action in actions:
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": f"e97-{action['sequence']}", "type": "function",
                                         "function": {"name": action["tool_name"],
                                                      "arguments": _canonical(action["arguments"])}}]})
        messages.append({"role": "tool", "tool_call_id": f"e97-{action['sequence']}",
                         "content": "observed"})
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
def test_repo_era2_and_era3_bytes_pins_are_unchanged():
    assert hashlib.sha256(ERA2.read_bytes()).hexdigest() == ERA2_SHA256
    assert hashlib.sha256(ERA3.read_bytes()).hexdigest() == ERA3_SHA256
    era4_sha = hashlib.sha256(ERA4.read_bytes()).hexdigest()
    assert era4_sha not in {ERA2_SHA256, ERA3_SHA256}


def test_sealed_logical_argv_identity_is_unchanged():
    assert validator_logical_argv("focused") == [
        "@runtime-python", "@generator-source/scripts/e97_first_party_validator.py",
        "--mode", "focused"]
    assert validator_logical_argv("regression") == [
        "@runtime-python", "@generator-source/scripts/e97_first_party_validator.py",
        "--mode", "regression"]
    assert _VALIDATOR_SOURCE_MEMBER == \
        "scripts/e97_first_party_validator_protocol_breadth.py"


def test_generator_closure_and_registry_bind_current_artifacts():
    components = verify_checkout_components(MANIFEST, checkout_root=REPO)
    by_path = {component["path"]: component["sha256"] for component in components}
    assert by_path["scripts/e97_first_party_validator.py"] == ERA2_SHA256
    assert by_path["scripts/e97_first_party_validator_first_action.py"] == ERA3_SHA256
    assert by_path["scripts/e97_first_party_validator_protocol_breadth.py"] == \
        hashlib.sha256(ERA4.read_bytes()).hexdigest()
    assert verify_archive_members(MANIFEST, ARCHIVE) == \
        hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    registry = validate_source_registry(json.loads(REGISTRY.read_text()))
    for source in registry["sources"]:
        if source["kind"] == "first-party":
            assert source["receipts"]["source_archive_sha256"] == \
                hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()


@pytest.mark.skipif(not OLD_LAKE.is_dir(), reason="old admitted lake not mounted")
def test_old_admitted_lake_revalidates_under_current_modules():
    registry = validate_source_registry(
        json.loads((OLD_LAKE / "source-registry.json").read_text()))
    tasks = [json.loads(line) for line in
             (OLD_LAKE / "tasks.jsonl").read_text().splitlines() if line.strip()]
    assert len(tasks) == 4
    assert len(validate_task_collection(tasks, registry=registry)) == 4


# -------------------------------------------- old specs grade identically
ERA2_CASES = [
    ("pass", _base_spec(), [_read_action()], "success", f"Final: {TOKEN}"),
    ("wrong-final", _base_spec(), [_read_action()], "success", "Final: wrong"),
    ("failed-status", _base_spec(), [_read_action()], "failed", f"Final: {TOKEN}"),
    ("wrong-token", {**_base_spec(), "expected_token": "f" * 20}, [_read_action()],
     "success", f"Final: {TOKEN}"),
    ("no-actions", _base_spec(), [], "success", f"Final: {TOKEN}"),
    ("wrong-read-path", _base_spec(), [_read_action(path="depot/decoy.txt")],
     "success", f"Final: {TOKEN}"),
    ("minefield-violation", _base_spec(),
     [{**_read_action(), "tool_name": "bash"}], "success", f"Final: {TOKEN}"),
]


@pytest.mark.parametrize("name,spec,actions,status,final", ERA2_CASES)
def test_old_specs_grade_identically_across_eras(name, spec, actions, status, final, tmp_path):
    terminal = _terminal(actions, final=final, status=status)
    for mode in ("focused", "regression"):
        era2 = _run(mode, ERA2, spec, terminal, tmp_path)
        era3 = _run(mode, ERA3, spec, terminal, tmp_path)
        era4 = _run(mode, ERA4, spec, terminal, tmp_path)
        assert era2 == era3 == era4, f"{name}/{mode}: {era2} vs {era3} vs {era4}"


# era-3-shaped specs (required_first_action declared) grade identically under
# era-3 and era-4 (era-2 rejects the extended spec shape by design).
ERA3_CASES = [
    ("first-action-pass", {**_base_spec(),
                           "required_first_action": {"tool": "read",
                                                     "arguments": {"path": "depot/token.txt"}}},
     [_read_action()], "success", f"Final: {TOKEN}"),
    ("first-action-fail", {**_base_spec(),
                           "required_first_action": {"tool": "read",
                                                     "arguments": {"path": "depot/token.txt"}}},
     [_list_action(), _read_action(sequence=1)], "success", f"Final: {TOKEN}"),
]


@pytest.mark.parametrize("name,spec,actions,status,final", ERA3_CASES)
def test_era3_specs_grade_identically_under_era4(name, spec, actions, status, final, tmp_path):
    terminal = _terminal(actions, final=final, status=status)
    for mode in ("focused", "regression"):
        era3 = _run(mode, ERA3, spec, terminal, tmp_path)
        era4 = _run(mode, ERA4, spec, terminal, tmp_path)
        assert era3 == era4, f"{name}/{mode}: era-3 {era3} vs era-4 {era4}"


# ------------------------------------------------------ outcome mode: edit
def test_outcome_edit_chain_passes(tmp_path):
    spec = _outcome_spec()
    actions = [
        _read_action("depot/state.txt", "mode=alpha\nstatus=retired", 0),
        _edit_action("depot/state.txt",
                     [{"oldText": "status=retired", "newText": "status=active"}], 1),
        _read_action("depot/state.txt", "mode=alpha\nstatus=active", 2),
    ]
    assert _run("focused", ERA4, spec, _terminal(actions), tmp_path)[0] == 0
    assert _run("regression", ERA4, spec, _terminal(actions), tmp_path)[0] == 0


def test_outcome_write_full_content_passes(tmp_path):
    spec = _outcome_spec()
    actions = [
        _write_action("depot/state.txt", "mode=alpha\nstatus=active", 0),
        _read_action("depot/state.txt", "mode=alpha\nstatus=active", 1),
    ]
    assert _run("focused", ERA4, spec, _terminal(actions), tmp_path)[0] == 0


def test_outcome_missing_write_fails(tmp_path):
    spec = _outcome_spec()
    actions = [_read_action("depot/state.txt", "mode=alpha\nstatus=active", 0)]
    assert _run("focused", ERA4, spec, _terminal(actions), tmp_path)[0] != 0


def test_outcome_missing_readback_fails(tmp_path):
    spec = _outcome_spec()
    actions = [
        _edit_action("depot/state.txt",
                     [{"oldText": "status=retired", "newText": "status=active"}], 0),
    ]
    assert _run("focused", ERA4, spec, _terminal(actions), tmp_path)[0] != 0


def test_outcome_unedited_readback_fails(tmp_path):
    spec = _outcome_spec()
    actions = [
        _edit_action("depot/state.txt",
                     [{"oldText": "status=retired", "newText": "status=active"}], 0),
        _read_action("depot/state.txt", "mode=alpha\nstatus=retired", 1),
    ]
    assert _run("focused", ERA4, spec, _terminal(actions), tmp_path)[0] != 0


def test_outcome_wrong_edit_result_fails(tmp_path):
    spec = _outcome_spec()
    actions = [
        _edit_action("depot/state.txt",
                     [{"oldText": "status=retired", "newText": "status=archived"}], 0),
        _read_action("depot/state.txt", "mode=alpha\nstatus=archived", 1),
    ]
    assert _run("focused", ERA4, spec, _terminal(actions), tmp_path)[0] != 0


def test_outcome_ambiguous_oldtext_fails(tmp_path):
    spec = _outcome_spec()
    actions = [
        _edit_action("depot/state.txt", [{"oldText": "a", "newText": "z"}], 0),
        _read_action("depot/state.txt", "mode=alpha\nstatus=active", 1),
    ]
    assert _run("focused", ERA4, spec, _terminal(actions), tmp_path)[0] != 0


def test_outcome_wrong_final_fails(tmp_path):
    spec = _outcome_spec()
    actions = [
        _edit_action("depot/state.txt",
                     [{"oldText": "status=retired", "newText": "status=active"}], 0),
        _read_action("depot/state.txt", "mode=alpha\nstatus=active", 1),
    ]
    assert _run("focused", ERA4, spec, _terminal(actions),
                tmp_path)[0] == 0
    assert _run("focused", ERA4, spec,
                _terminal(actions, final="Final: done!"), tmp_path)[0] != 0


# ------------------------------------------- outcome mode: reads/recovery
def test_outcome_grounded_read_only_passes(tmp_path):
    spec = _base_spec()
    spec["expected_final"] = TOKEN
    spec["required_grounded_reads"] = [
        {"path": "depot/token.txt", "expected_text": f"token={TOKEN}"}]
    actions = [_read_action()]
    terminal = _terminal(actions, final=f"Final: {TOKEN}")
    assert _run("focused", ERA4, spec, terminal, tmp_path)[0] == 0
    assert _run("regression", ERA4, spec, terminal, tmp_path)[0] == 0


def test_outcome_partial_read_slice_is_tolerated(tmp_path):
    spec = _base_spec()
    spec["expected_final"] = TOKEN
    spec["required_grounded_reads"] = [
        {"path": "logs/night.txt", "expected_text": "night shift\naccess code below\ntoken-line"}]
    actions = [_read_action("logs/night.txt", "access code below", 0),
               _read_action("logs/night.txt", "night shift\naccess code below\ntoken-line", 1)]
    terminal = _terminal(actions, final=f"Final: {TOKEN}")
    assert _run("focused", ERA4, spec, terminal, tmp_path)[0] == 0


def test_outcome_contradicting_read_fails(tmp_path):
    spec = _base_spec()
    spec["expected_final"] = TOKEN
    spec["required_grounded_reads"] = [
        {"path": "logs/night.txt", "expected_text": "night shift"}]
    actions = [_read_action("logs/night.txt", "day shift", 0),
               _read_action("logs/night.txt", "night shift", 1)]
    terminal = _terminal(actions, final=f"Final: {TOKEN}")
    assert _run("focused", ERA4, spec, terminal, tmp_path)[0] != 0


def test_outcome_recovery_error_read_ordering(tmp_path):
    spec = _base_spec()
    spec["expected_final"] = TOKEN
    spec["required_grounded_reads"] = [
        {"path": "catalog.json", "expected_text": '{"active": "records/a.txt"}'},
        {"path": "records/a.txt", "expected_text": f"token={TOKEN}"}]
    spec["required_error_read"] = {"path": "missing.json"}
    actions = [_error_read_action("missing.json", 0),
              _read_action("catalog.json", '{"active": "records/a.txt"}', 1),
              _read_action("records/a.txt", f"token={TOKEN}", 2)]
    terminal = _terminal(actions, final=f"Final: {TOKEN}")
    assert _run("focused", ERA4, spec, terminal, tmp_path)[0] == 0
    late = [_read_action("catalog.json", '{"active": "records/a.txt"}', 0),
            _error_read_action("missing.json", 1),
            _read_action("records/a.txt", f"token={TOKEN}", 2)]
    assert _run("focused", ERA4, spec, _terminal(late, final=f"Final: {TOKEN}"),
                tmp_path)[0] != 0
    no_error = [action for action in actions if action["tool_name"] != "read" or not action["is_error"]]
    assert _run("focused", ERA4, spec, _terminal(no_error, final=f"Final: {TOKEN}"),
                tmp_path)[0] != 0


def test_outcome_first_action_composes(tmp_path):
    spec = _outcome_spec()
    spec["required_first_action"] = {"tool": "read", "arguments": {"path": "depot/state.txt"}}
    actions = [
        _read_action("depot/state.txt", "mode=alpha\nstatus=retired", 0),
        _edit_action("depot/state.txt",
                     [{"oldText": "status=retired", "newText": "status=active"}], 1),
        _read_action("depot/state.txt", "mode=alpha\nstatus=active", 2),
    ]
    assert _run("focused", ERA4, spec, _terminal(actions), tmp_path)[0] == 0
    wrong_first = [
        _list_action(0),
        _read_action("depot/state.txt", "mode=alpha\nstatus=retired", 1),
        _edit_action("depot/state.txt",
                     [{"oldText": "status=retired", "newText": "status=active"}], 2),
        _read_action("depot/state.txt", "mode=alpha\nstatus=active", 3),
    ]
    assert _run("focused", ERA4, spec, _terminal(wrong_first), tmp_path)[0] != 0


# ---------------------------------------------------- outcome mode: shape
@pytest.mark.parametrize("bad_spec,needle", [
    ({**_outcome_spec(), "expected_final": ""}, "expected_final"),
    ({**_outcome_spec(), "required_grounded_reads": [
        {"path": "a.txt", "expected_text": "x"}, {"path": "a.txt", "expected_text": "y"}]},
     "required_grounded_reads"),
    ({**_outcome_spec(), "required_error_read": {"path": "a.txt", "bogus": 1}},
     "required_error_read"),
    ({**_outcome_spec(), "required_workspace_writes": [{"path": "a.txt"}]},
     "required_workspace_writes"),
    ({**_outcome_spec(expected_final="x"), "required_grounded_reads": [],
      "required_workspace_writes": []}, "requires grounded reads"),
    ({**_base_spec(), "required_grounded_reads": [{"path": "a.txt", "expected_text": "x"}]},
     "requires expected_final"),
])
def test_era4_rejects_malformed_outcome_declarations(bad_spec, needle, tmp_path):
    completed = _run("focused", ERA4, bad_spec, _terminal([_read_action()]), tmp_path)
    assert completed[0] != 0
    assert needle in completed[2]


def test_outcome_mode_minefield_widens_but_stays_within_workspace_surface(tmp_path):
    spec = _outcome_spec()
    spec["minefield"]["allowed_tools"] = ["read", "edit", "bogus_tool"]
    completed = _run("focused", ERA4, spec, _terminal([_read_action()]), tmp_path)
    assert completed[0] != 0
    assert "minefield" in completed[2]
    spec["minefield"]["allowed_tools"] = ["edit", "write"]  # no read grounding
    assert _run("focused", ERA4, spec, _terminal([_read_action()]), tmp_path)[0] != 0


def test_era2_mode_minefield_stays_exact(tmp_path):
    spec = {**_base_spec(), "expected_final": "x",
            "required_grounded_reads": [{"path": "a.txt", "expected_text": "x"}]}
    spec["minefield"]["allowed_tools"] = ["list_files", "read", "write"]
    assert _run("focused", ERA4, spec, _terminal([_read_action()]), tmp_path)[0] != 0
